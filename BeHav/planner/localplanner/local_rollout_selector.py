#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
local_rollout_selector.py

CMU-like stable local planner by GROUP scoring.

Put it in:
  ~/depthmap/localplanner/local_rollout_selector.py

Inputs:
  /local_traversability_grid      nav_msgs/msg/OccupancyGrid
  /current_waypoint_local         geometry_msgs/msg/PoseStamped

Outputs:
  /local_selected_trajectory      nav_msgs/msg/Path
  /local_candidate_trajectories   nav_msgs/msg/Path, optional debug visualization

Core idea copied from CMU-style local planner:
  - Do NOT choose directly among every single trajectory.
  - Generate many candidate paths, but assign them into 7 direction groups.
  - Each safe candidate path votes for its group.
  - Select the best group.
  - Publish the representative path of that selected group.
  - If representative is unsafe, publish the safest backup inside that group.
  - Use group-level hysteresis, not single-path sticky locking.

This prevents left-right oscillation because the output decision space is group-level,
not every individual path competing every frame.

Map values expected from depth_grid_standalone.py:
  -1 = unknown
   0 = free
 100 = occupied

No vehicle control is published here.
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Bool


@dataclass
class GridInfo:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    frame_id: str


@dataclass
class Rollout:
    traj_id: int
    group_id: int              # 0..6 only, CMU-like path group
    variant_id: int
    family: str                # "direct" or "recovery"
    is_representative: bool
    length: float
    y_peak: float
    y_end: float
    points: np.ndarray         # N x 3: x, y, yaw


@dataclass
class PathEval:
    rollout: Rollout
    safe: bool
    path_score: float
    endpoint_dist: float
    footprint_penalty: float
    center_penalty: float
    reason: str


class LocalRolloutSelector(Node):
    def __init__(self, args):
        parameter_overrides = []
        if args.use_sim_time:
            parameter_overrides.append(Parameter("use_sim_time", Parameter.Type.BOOL, True))

        super().__init__("local_rollout_selector", parameter_overrides=parameter_overrides)
        self.args = args

        self.current_waypoint_local: Optional[Tuple[float, float]] = None
        self.last_waypoint_wall_time: float = 0.0
        self.goal_reached: bool = False

        self.selected_group_id: Optional[int] = None
        self.selected_rollout_id: Optional[int] = None
        self.last_selected_rollout: Optional[Rollout] = None
        self.no_valid_plan_count: int = 0
        self.group_scores_ema = np.zeros(7, dtype=np.float32)
        self.group_scores_ema_ready: bool = False

        self.last_switch_wall_time: float = 0.0
        self.plan_count: int = 0
        self.last_plan_wall_time: float = 0.0

        self.rollouts = self.generate_rollout_library()
        self.rollouts_by_group: Dict[int, List[Rollout]] = {}
        self.representative_by_group: Dict[int, Rollout] = {}
        for r in self.rollouts:
            self.rollouts_by_group.setdefault(r.group_id, []).append(r)
            if r.is_representative:
                self.representative_by_group[r.group_id] = r

        self.footprint_offsets = self.make_footprint_offsets()

        self.grid_sub = self.create_subscription(
            OccupancyGrid,
            args.grid_topic,
            self.grid_callback,
            10,
        )
        self.wp_sub = self.create_subscription(
            PoseStamped,
            args.current_waypoint_local_topic,
            self.waypoint_callback,
            10,
        )
        self.goal_reached_sub = self.create_subscription(
            Bool,
            args.far_goal_reached_topic,
            self.goal_reached_callback,
            10,
        )

        self.selected_pub = self.create_publisher(Path, args.selected_path_topic, 10)
        self.candidates_pub = self.create_publisher(Path, args.candidate_paths_topic, 10)
        self.local_ok_pub = self.create_publisher(Bool, args.local_planner_ok_topic, 10)

        self.get_logger().info("local_rollout_selector.py started")
        self.get_logger().info(f"grid_topic       : {args.grid_topic}")
        self.get_logger().info(f"waypoint_topic   : {args.current_waypoint_local_topic}")
        self.get_logger().info(f"selected_path    : {args.selected_path_topic}")
        self.get_logger().info(f"local_ok_topic   : {args.local_planner_ok_topic}")
        self.get_logger().info(f"group_count      : 7")
        self.get_logger().info(f"candidate count  : {len(self.rollouts)}")
        self.get_logger().info("selection        : CMU-like group score + representative path")
        self.get_logger().info("vehicle control  : disabled")

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def waypoint_callback(self, msg: PoseStamped) -> None:
        self.current_waypoint_local = (float(msg.pose.position.x), float(msg.pose.position.y))
        self.last_waypoint_wall_time = time.time()

    def goal_reached_callback(self, msg: Bool) -> None:
        self.goal_reached = bool(msg.data)
        if self.goal_reached:
            self.selected_group_id = None
            self.selected_rollout_id = None
            self.last_selected_rollout = None
            self.no_valid_plan_count = 0
            self.group_scores_ema_ready = False

    def grid_callback(self, msg: OccupancyGrid) -> None:
        now = time.time()
        if self.args.plan_hz > 0.0:
            min_dt = 1.0 / self.args.plan_hz
            if now - self.last_plan_wall_time < min_dt:
                return
            self.last_plan_wall_time = now

        width = int(msg.info.width)
        height = int(msg.info.height)
        if width <= 0 or height <= 0:
            self.get_logger().warn("Invalid OccupancyGrid size.")
            return

        data = np.asarray(msg.data, dtype=np.int16)
        if data.size != width * height:
            self.get_logger().warn(f"Grid data length mismatch: got {data.size}, expected {width * height}")
            return

        grid = data.reshape(height, width)
        info = GridInfo(
            width=width,
            height=height,
            resolution=float(msg.info.resolution),
            origin_x=float(msg.info.origin.position.x),
            origin_y=float(msg.info.origin.position.y),
            frame_id=msg.header.frame_id if msg.header.frame_id else self.args.local_frame_fallback,
        )

        if self.args.publish_candidates:
            self.publish_candidate_paths(msg.header.stamp, info.frame_id)

        if self.goal_reached:
            self.publish_empty_path(msg.header.stamp, info.frame_id)
            self.publish_local_ok(False)
            return

        selected = self.select_by_group_score(grid, info)
        if selected is None:
            self.get_logger().warn("No valid group/path selected. Publishing empty path.")
            self.publish_empty_path(msg.header.stamp, info.frame_id)
            self.publish_local_ok(False)
            return

        self.publish_rollout(selected, msg.header.stamp, info.frame_id)
        self.publish_local_ok(True)

        self.plan_count += 1
        if self.plan_count % max(1, self.args.print_every) == 0:
            wp_text = "none" if self.current_waypoint_local is None else f"({self.current_waypoint_local[0]:.2f},{self.current_waypoint_local[1]:.2f})"
            self.get_logger().info(
                f"selected group={selected.group_id}, family={selected.family}, rep={selected.is_representative}, "
                f"y_peak={selected.y_peak:.2f}, y_end={selected.y_end:.2f}, waypoint={wp_text}"
            )

    # ------------------------------------------------------------------
    # Rollout library
    # ------------------------------------------------------------------

    def generate_rollout_library(self) -> List[Rollout]:
        """Generate candidate paths assigned into 7 fixed groups.

        Group IDs:
          0: hard right
          1: mid right
          2: slight right
          3: center
          4: slight left
          5: mid left
          6: hard left

        In base_footprint:
          +x forward
          +y left
          -y right
        """
        L = self.args.preview_distance

        # Representative lateral endpoint for each group.
        group_y = {
            0: -2.1,
            1: -1.4,
            2: -0.7,
            3:  0.0,
            4:  0.7,
            5:  1.4,
            6:  2.1,
        }

        rollouts: List[Rollout] = []
        traj_id = 0

        for gid in range(7):
            gy = group_y[gid]

            # 1) Representative direct path for the group.
            points = self.make_direct_rollout(L, gy)
            rollouts.append(Rollout(
                traj_id=traj_id,
                group_id=gid,
                variant_id=0,
                family="direct",
                is_representative=True,
                length=L,
                y_peak=gy,
                y_end=gy,
                points=points,
            ))
            traj_id += 1

            # 2) Direct variants around the group representative.
            # Center group gets fewer duplicates.
            direct_offsets = [] if gid == 3 else [-0.35, 0.35]
            for off in direct_offsets:
                y_end = self.clamp(gy + off, -self.args.max_lateral_offset, self.args.max_lateral_offset)
                points = self.make_direct_rollout(L, y_end)
                rollouts.append(Rollout(
                    traj_id=traj_id,
                    group_id=gid,
                    variant_id=traj_id,
                    family="direct",
                    is_representative=False,
                    length=L,
                    y_peak=y_end,
                    y_end=y_end,
                    points=points,
                ))
                traj_id += 1

            # 3) Recovery variants: avoid in the group's direction, then come back.
            # For center group, include mild left/right recoveries.
            if gid == 3:
                recover_peaks = [-1.0, 1.0]
            else:
                recover_peaks = [gy, self.clamp(gy * 1.15, -self.args.max_lateral_offset, self.args.max_lateral_offset)]

            for peak in recover_peaks:
                y_end = self.args.recovery_end_ratio * peak
                points = self.make_recovery_rollout(L, peak, y_end)
                rollouts.append(Rollout(
                    traj_id=traj_id,
                    group_id=gid,
                    variant_id=traj_id,
                    family="recovery",
                    is_representative=False,
                    length=L,
                    y_peak=peak,
                    y_end=y_end,
                    points=points,
                ))
                traj_id += 1

        return rollouts

    def make_direct_rollout(self, length: float, y_end: float) -> np.ndarray:
        step = max(0.10, self.args.trajectory_sample_step)
        n = max(2, int(math.ceil(length / step)) + 1)
        xs = np.linspace(0.0, length, n, dtype=np.float32)
        t = xs / max(length, 1e-6)

        s = 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3
        ds_dt = 30.0 * t**4 - 60.0 * t**3 + 30.0 * t**2
        ys = y_end * s
        dy_dx = y_end * ds_dt / max(length, 1e-6)
        yaws = np.arctan2(dy_dx, np.ones_like(dy_dx))

        return np.stack([xs, ys, yaws], axis=1).astype(np.float32)

    def make_recovery_rollout(self, length: float, y_peak: float, y_end: float) -> np.ndarray:
        step = max(0.10, self.args.trajectory_sample_step)
        n = max(2, int(math.ceil(length / step)) + 1)
        xs = np.linspace(0.0, length, n, dtype=np.float32)
        t = xs / max(length, 1e-6)

        smoother = 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3
        ds_dt = 30.0 * t**4 - 60.0 * t**3 + 30.0 * t**2

        bump = y_peak - 0.5 * y_end
        sin_term = np.sin(np.pi * t)
        bump_shape = sin_term * sin_term
        dbump_dt = np.pi * np.sin(2.0 * np.pi * t)

        ys = y_end * smoother + bump * bump_shape
        dy_dt = y_end * ds_dt + bump * dbump_dt
        dy_dx = dy_dt / max(length, 1e-6)
        yaws = np.arctan2(dy_dx, np.ones_like(dy_dx))

        return np.stack([xs, ys, yaws], axis=1).astype(np.float32)

    def make_footprint_offsets(self) -> np.ndarray:
        half = 0.5 * self.args.vehicle_size
        step = max(0.20, self.args.footprint_sample_step)
        coords = np.arange(-half, half + 1e-6, step, dtype=np.float32)
        offsets = []
        for x in coords:
            for y in coords:
                offsets.append((float(x), float(y)))

        offsets.extend([(-half, -half), (-half, half), (half, -half), (half, half), (0.0, 0.0)])
        return np.asarray(offsets, dtype=np.float32)

    # ------------------------------------------------------------------
    # CMU-like group selection
    # ------------------------------------------------------------------

    def select_by_group_score(self, grid: np.ndarray, info: GridInfo) -> Optional[Rollout]:
        if self.current_waypoint_local is None:
            return None

        if time.time() - self.last_waypoint_wall_time > self.args.waypoint_timeout:
            self.get_logger().warn("current_waypoint_local timeout.")
            return None

        target_x, target_y = self.current_waypoint_local

        if math.hypot(target_x, target_y) <= self.args.local_waypoint_stop_radius:
            # The current intermediate waypoint is close enough. Publish an empty path
            # until farplanner either declares final goal reached or provides a new waypoint.
            return None

        group_scores = np.zeros(7, dtype=np.float32)
        group_safe_counts = np.zeros(7, dtype=np.int32)
        evals_by_group: Dict[int, List[PathEval]] = {i: [] for i in range(7)}
        reject_reasons = {
            "occupied": 0,
            "near_unknown": 0,
            "out_of_map": 0,
            "footprint": 0,
            "safe": 0,
        }

        for rollout in self.rollouts:
            ev = self.evaluate_path(rollout, grid, info, target_x, target_y)
            evals_by_group[rollout.group_id].append(ev)

            if ev.safe:
                reject_reasons["safe"] += 1
                group_safe_counts[rollout.group_id] += 1
                group_scores[rollout.group_id] += ev.path_score
            else:
                reject_reasons[ev.reason] = reject_reasons.get(ev.reason, 0) + 1

        # Discard groups with too few safe candidate paths.
        for gid in range(7):
            if group_safe_counts[gid] < self.args.min_safe_paths_per_group:
                group_scores[gid] = -1e9

        if np.max(group_scores) <= -1e8:
            self.debug_rejections(reject_reasons)
            self.no_valid_plan_count += 1

            # Do not instantly drop the trajectory on a single noisy grid frame.
            # Holding for a few planning frames prevents the tracker from doing
            # stop-go-stop when depth/unknown cells flicker.
            if (
                self.last_selected_rollout is not None
                and self.no_valid_plan_count <= self.args.max_hold_last_path_frames
            ):
                return self.last_selected_rollout

            self.selected_group_id = None
            self.selected_rollout_id = None
            self.last_selected_rollout = None
            self.group_scores_ema_ready = False
            return None

        smoothed_scores = self.smooth_group_scores(group_scores)
        raw_best_group = int(np.argmax(smoothed_scores))
        selected_group = self.apply_group_hysteresis(raw_best_group, smoothed_scores, group_safe_counts)

        selected_rollout = self.pick_representative_or_backup(selected_group, evals_by_group[selected_group])
        if selected_rollout is None:
            # This should be rare if group_safe_counts says the group is valid.
            self.no_valid_plan_count += 1
            if (
                self.last_selected_rollout is not None
                and self.no_valid_plan_count <= self.args.max_hold_last_path_frames
            ):
                return self.last_selected_rollout
            self.selected_group_id = None
            self.selected_rollout_id = None
            self.last_selected_rollout = None
            return None

        if selected_group != self.selected_group_id:
            if self.args.debug_switch:
                old = self.selected_group_id
                self.get_logger().info(
                    f"group switch {old} -> {selected_group}, scores={smoothed_scores.tolist()}, safe={group_safe_counts.tolist()}"
                )
            self.selected_group_id = selected_group
            self.last_switch_wall_time = time.time()

        self.no_valid_plan_count = 0
        self.last_selected_rollout = selected_rollout
        self.selected_rollout_id = selected_rollout.traj_id
        return selected_rollout

    def evaluate_path(
        self,
        rollout: Rollout,
        grid: np.ndarray,
        info: GridInfo,
        target_x: float,
        target_y: float,
    ) -> PathEval:
        ok, center_penalty, endpoint_dist, reason = self.score_centerline(rollout, grid, info, target_x, target_y)
        if not ok:
            return PathEval(rollout, False, -1e9, endpoint_dist, 1e18, center_penalty, reason)

        ok_fp, footprint_penalty = self.footprint_collision_penalty(rollout, grid, info)
        if not ok_fp:
            return PathEval(rollout, False, -1e9, endpoint_dist, footprint_penalty, center_penalty, "footprint")

        # CMU-like group voting score:
        # each safe path votes for its group. A group wins when many of its variants are good.
        target_heading = math.atan2(target_y, max(target_x, 1e-6))
        path_heading = math.atan2(rollout.y_end, rollout.length)
        heading_err = abs(self.normalize_angle(path_heading - target_heading))

        # Convert costs to a positive score. Keep the endpoint distance important, but not as a single-path winner.
        score = 0.0
        score += self.args.base_safe_score
        score -= self.args.endpoint_dist_weight * endpoint_dist
        score -= self.args.heading_error_weight * heading_err
        score -= self.args.center_penalty_weight * center_penalty
        score -= self.args.footprint_penalty_weight * footprint_penalty
        score -= self.args.shape_weight * abs(rollout.y_peak)

        if rollout.is_representative:
            score += self.args.representative_bonus
        if rollout.family == "recovery":
            score += self.args.recovery_bonus

        return PathEval(rollout, True, score, endpoint_dist, footprint_penalty, center_penalty, "safe")

    def smooth_group_scores(self, group_scores: np.ndarray) -> np.ndarray:
        """Low-pass group scores to reduce left/right flicker from noisy grids."""
        alpha = float(self.args.group_score_ema_alpha)
        alpha = self.clamp(alpha, 0.0, 1.0)

        if not self.group_scores_ema_ready:
            self.group_scores_ema = group_scores.copy()
            self.group_scores_ema_ready = True
            return self.group_scores_ema.copy()

        for gid in range(7):
            if group_scores[gid] <= -1e8:
                self.group_scores_ema[gid] = -1e9
                continue

            if self.group_scores_ema[gid] <= -1e8:
                self.group_scores_ema[gid] = group_scores[gid]
            else:
                self.group_scores_ema[gid] = (
                    alpha * group_scores[gid]
                    + (1.0 - alpha) * self.group_scores_ema[gid]
                )

        return self.group_scores_ema.copy()

    def apply_group_hysteresis(
        self,
        raw_best_group: int,
        group_scores: np.ndarray,
        group_safe_counts: np.ndarray,
    ) -> int:
        if self.selected_group_id is None:
            return raw_best_group

        old = int(self.selected_group_id)
        if old < 0 or old >= 7:
            return raw_best_group

        old_score = float(group_scores[old])
        new_score = float(group_scores[raw_best_group])

        # If current group no longer has enough valid paths, switch immediately.
        if group_safe_counts[old] < self.args.min_safe_paths_per_group:
            return raw_best_group

        if raw_best_group == old:
            return old

        # Hold time: do not switch too quickly unless new group is much better.
        held = time.time() - self.last_switch_wall_time
        improvement = new_score - old_score
        if held < self.args.min_group_hold_time and improvement < self.args.emergency_group_score_margin:
            return old

        # Require new group to beat old group by a clear margin.
        # Extra margin for switching across center, e.g. left to right.
        margin = self.args.group_score_margin
        if (old - 3) * (raw_best_group - 3) < 0:
            margin = max(margin, self.args.cross_center_score_margin)

        if improvement < margin:
            return old

        return raw_best_group

    def pick_representative_or_backup(self, group_id: int, evals: List[PathEval]) -> Optional[Rollout]:
        safe = [e for e in evals if e.safe]
        if not safe:
            return None

        safe.sort(key=lambda e: (-e.path_score, e.endpoint_dist, e.footprint_penalty))
        best = safe[0]

        # First keep the previously selected rollout inside the same group unless
        # a new safe rollout is clearly better. This removes same-group path twitching.
        old_eval = None
        for e in safe:
            if self.selected_rollout_id is not None and e.rollout.traj_id == self.selected_rollout_id:
                old_eval = e
                break

        if old_eval is not None:
            if best.path_score - old_eval.path_score < self.args.rollout_switch_margin:
                return old_eval.rollout

        # Prefer representative path if it is safe and not much worse than group alternatives.
        rep_eval = None
        for e in safe:
            if e.rollout.is_representative:
                rep_eval = e
                break

        if rep_eval is not None:
            if best.path_score - rep_eval.path_score < self.args.representative_keep_margin:
                return rep_eval.rollout

        # If representative is unsafe or much worse, choose safest/best backup inside the same group.
        return best.rollout

    # ------------------------------------------------------------------
    # Path scoring
    # ------------------------------------------------------------------

    def score_centerline(
        self,
        rollout: Rollout,
        grid: np.ndarray,
        info: GridInfo,
        target_x: float,
        target_y: float,
    ) -> Tuple[bool, float, float, str]:
        occupied_hits = 0
        unknown_near = 0
        unknown_far = 0
        out_hits = 0
        free_hits = 0

        stride = max(1, self.args.centerline_stride)
        for x, y, _yaw in rollout.points[::stride]:
            x = float(x)
            y = float(y)
            cell = self.world_to_grid(x, y, info)
            if cell is None:
                out_hits += 1
                continue

            ix, iy = cell
            val = int(grid[iy, ix])
            dist = math.hypot(x, y)

            if val >= self.args.occupied_threshold:
                occupied_hits += 1
            elif val < 0:
                if dist < self.args.near_unknown_distance:
                    unknown_near += 1
                else:
                    unknown_far += 1
            else:
                free_hits += 1

        if occupied_hits > self.args.max_occupied_hits:
            return False, 1e18, 1e18, "occupied"
        if unknown_near > self.args.max_near_unknown_hits:
            return False, 1e18, 1e18, "near_unknown"
        if out_hits > self.args.max_out_of_map_hits:
            return False, 1e18, 1e18, "out_of_map"

        end_x = float(rollout.points[-1, 0])
        end_y = float(rollout.points[-1, 1])
        endpoint_dist = math.hypot(end_x - target_x, end_y - target_y)

        penalty = 0.0
        penalty += self.args.occupied_weight * occupied_hits
        penalty += self.args.unknown_near_weight * unknown_near
        penalty += self.args.unknown_far_weight * unknown_far
        penalty += self.args.out_of_map_weight * out_hits
        penalty -= self.args.free_weight * free_hits

        return True, penalty, endpoint_dist, "ok"

    def footprint_collision_penalty(self, rollout: Rollout, grid: np.ndarray, info: GridInfo) -> Tuple[bool, float]:
        penalty = 0.0
        occupied_hits = 0
        near_unknown_hits = 0
        stride = max(1, self.args.footprint_check_stride)

        for p in rollout.points[::stride]:
            x = float(p[0])
            y = float(p[1])
            yaw = float(p[2])
            c = math.cos(yaw)
            s = math.sin(yaw)
            dist = math.hypot(x, y)

            for ox, oy in self.footprint_offsets:
                wx = x + c * float(ox) - s * float(oy)
                wy = y + s * float(ox) + c * float(oy)

                if self.args.allow_behind_origin and wx < info.origin_x:
                    continue

                cell = self.world_to_grid(wx, wy, info)
                if cell is None:
                    penalty += self.args.out_of_map_footprint_penalty
                    continue

                ix, iy = cell
                val = int(grid[iy, ix])
                if val >= self.args.occupied_threshold:
                    occupied_hits += 1
                    penalty += self.args.occupied_footprint_penalty
                elif val < 0:
                    if dist < self.args.near_unknown_distance:
                        near_unknown_hits += 1
                        penalty += self.args.unknown_near_footprint_penalty
                    else:
                        penalty += self.args.unknown_far_footprint_penalty

        if occupied_hits > self.args.max_footprint_occupied_hits:
            return False, 1e18
        if near_unknown_hits > self.args.max_footprint_near_unknown_hits:
            return False, 1e18
        return True, penalty

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def world_to_grid(self, x: float, y: float, info: GridInfo) -> Optional[Tuple[int, int]]:
        ix = int(math.floor((x - info.origin_x) / info.resolution))
        iy = int(math.floor((y - info.origin_y) / info.resolution))
        if ix < 0 or ix >= info.width or iy < 0 or iy >= info.height:
            return None
        return ix, iy

    def publish_rollout(self, rollout: Rollout, stamp, frame_id: str) -> None:
        msg = Path()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        for x, y, yaw in rollout.points:
            msg.poses.append(self.make_pose(float(x), float(y), float(yaw), frame_id, stamp))
        self.selected_pub.publish(msg)

    def publish_candidate_paths(self, stamp, frame_id: str) -> None:
        msg = Path()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        for rollout in self.rollouts:
            for x, y, yaw in rollout.points:
                msg.poses.append(self.make_pose(float(x), float(y), float(yaw), frame_id, stamp))
        self.candidates_pub.publish(msg)

    def publish_empty_path(self, stamp, frame_id: str) -> None:
        msg = Path()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        self.selected_pub.publish(msg)

    def publish_local_ok(self, ok: bool) -> None:
        msg = Bool()
        msg.data = bool(ok)
        self.local_ok_pub.publish(msg)

    def make_pose(self, x: float, y: float, yaw: float, frame_id: str, stamp) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.08
        qx, qy, qz, qw = self.yaw_to_quat(yaw)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg

    def debug_rejections(self, reasons: Dict[str, int]) -> None:
        if not self.args.debug_rejections:
            return
        self.get_logger().warn(
            "rollout status: "
            + ", ".join([f"{k}={v}" for k, v in reasons.items()])
        )

    @staticmethod
    def yaw_to_quat(yaw: float):
        half = 0.5 * yaw
        return 0.0, 0.0, math.sin(half), math.cos(half)

    @staticmethod
    def normalize_angle(a: float) -> float:
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    @staticmethod
    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))


def parse_args():
    parser = argparse.ArgumentParser(description="CMU-like group local rollout selector. No vehicle control.")

    parser.add_argument("--grid-topic", default="/local_traversability_grid")
    parser.add_argument("--current-waypoint-local-topic", default="/current_waypoint_local")
    parser.add_argument("--selected-path-topic", default="/local_selected_trajectory")
    parser.add_argument("--candidate-paths-topic", default="/local_candidate_trajectories")
    parser.add_argument("--local-planner-ok-topic", default="/local_planner_ok")
    parser.add_argument("--far-goal-reached-topic", default="/far_goal_reached")

    parser.add_argument("--use-sim-time", dest="use_sim_time", action="store_true", default=True)
    parser.add_argument("--no-sim-time", dest="use_sim_time", action="store_false")
    parser.add_argument("--local-frame-fallback", default="base_footprint")

    # Rollout library.
    parser.add_argument("--preview-distance", type=float, default=4.5)
    parser.add_argument("--trajectory-sample-step", type=float, default=0.30)
    parser.add_argument("--max-lateral-offset", type=float, default=2.5)
    parser.add_argument("--recovery-end-ratio", type=float, default=0.25)
    parser.add_argument("--plan-hz", type=float, default=5.0)

    # CMU-like group scoring / hysteresis.
    parser.add_argument("--min-safe-paths-per-group", type=int, default=1)
    parser.add_argument("--min-group-hold-time", type=float, default=2.5)
    parser.add_argument("--group-score-margin", type=float, default=3.0)
    parser.add_argument("--cross-center-score-margin", type=float, default=6.0)
    parser.add_argument("--emergency-group-score-margin", type=float, default=8.0)
    parser.add_argument("--representative-keep-margin", type=float, default=2.5)
    parser.add_argument("--rollout-switch-margin", type=float, default=2.5)
    parser.add_argument("--group-score-ema-alpha", type=float, default=0.35)
    parser.add_argument("--max-hold-last-path-frames", type=int, default=3)
    parser.add_argument("--waypoint-timeout", type=float, default=1.0)
    parser.add_argument("--local-waypoint-stop-radius", type=float, default=0.6)

    # Footprint / validity.
    parser.add_argument("--centerline-stride", type=int, default=1)
    parser.add_argument("--vehicle-size", type=float, default=1.0)
    parser.add_argument("--footprint-sample-step", type=float, default=0.50)
    parser.add_argument("--footprint-check-stride", type=int, default=1)
    parser.add_argument("--allow-behind-origin", action="store_true", default=True)

    # Occupancy semantics.
    parser.add_argument("--occupied-threshold", type=int, default=50)
    parser.add_argument("--near-unknown-distance", type=float, default=0.6)
    parser.add_argument("--max-occupied-hits", type=int, default=2)
    parser.add_argument("--max-near-unknown-hits", type=int, default=50)
    parser.add_argument("--max-out-of-map-hits", type=int, default=8)
    parser.add_argument("--max-footprint-occupied-hits", type=int, default=0)
    parser.add_argument("--max-footprint-near-unknown-hits", type=int, default=120)

    # Group path score.
    parser.add_argument("--base-safe-score", type=float, default=10.0)
    parser.add_argument("--endpoint-dist-weight", type=float, default=2.2)
    parser.add_argument("--heading-error-weight", type=float, default=1.5)
    parser.add_argument("--center-penalty-weight", type=float, default=0.25)
    parser.add_argument("--footprint-penalty-weight", type=float, default=0.20)
    parser.add_argument("--shape-weight", type=float, default=0.10)
    parser.add_argument("--representative-bonus", type=float, default=1.0)
    parser.add_argument("--recovery-bonus", type=float, default=0.2)

    # Raw penalty weights.
    parser.add_argument("--free-weight", type=float, default=0.04)
    parser.add_argument("--occupied-weight", type=float, default=20.0)
    parser.add_argument("--unknown-near-weight", type=float, default=0.5)
    parser.add_argument("--unknown-far-weight", type=float, default=0.05)
    parser.add_argument("--out-of-map-weight", type=float, default=1.5)

    # Footprint penalties.
    parser.add_argument("--occupied-footprint-penalty", type=float, default=15.0)
    parser.add_argument("--unknown-near-footprint-penalty", type=float, default=2.5)
    parser.add_argument("--unknown-far-footprint-penalty", type=float, default=0.5)
    parser.add_argument("--out-of-map-footprint-penalty", type=float, default=2.0)

    parser.add_argument("--publish-candidates", action="store_true")
    parser.add_argument("--debug-rejections", action="store_true")
    parser.add_argument("--debug-switch", action="store_true")
    parser.add_argument("--print-every", type=int, default=20)

    args, ros_args = parser.parse_known_args()
    return args, ros_args


def main():
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)
    node = LocalRolloutSelector(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
