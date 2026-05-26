#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from geometry_msgs.msg import PoseStamped, PoseArray
    from nav_msgs.msg import OccupancyGrid, Path
    from std_msgs.msg import Bool, String
except Exception:
    # Allows syntax checking outside a ROS2 environment.
    pass

try:
    from .local_types import GridInfo, Rollout, PathEval, CandidatePath, ArbiterEval
except ImportError:
    from local_types import GridInfo, Rollout, PathEval, CandidatePath, ArbiterEval

class GroupSelectorMixin:
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

        far_path_cost = self.far_path_alignment_cost(rollout)
        if far_path_cost is not None:
            score -= self.args.far_path_weight * far_path_cost

        if rollout.is_representative:
            score += self.args.representative_bonus
        if rollout.family == "recovery":
            score += self.args.recovery_bonus

        return PathEval(rollout, True, score, endpoint_dist, footprint_penalty, center_penalty, "safe")

    def far_path_alignment_cost(self, rollout: Rollout) -> Optional[float]:
        """Penalty for deviating from /far_local_plan.

        The far planner's path is a structural/global guide. The rollout is still
        collision-checked locally, but among safe rollouts we prefer the one that
        stays close to this guide. This fixes the common mismatch where the green
        far path and blue local rollout diverge.
        """
        if self.far_path_xy is None:
            return None
        if time.time() - self.last_far_path_wall_time > self.args.far_path_timeout:
            return None
        if len(self.far_path_xy) < 2:
            return None

        stride = max(1, self.args.far_path_rollout_stride)
        pts = rollout.points[::stride, :2].astype(np.float32)
        if pts.shape[0] == 0:
            return None

        guide = self.far_path_xy
        total = 0.0
        count = 0
        for p in pts:
            d = guide - p
            dist2 = np.sum(d * d, axis=1)
            total += math.sqrt(float(np.min(dist2)))
            count += 1

        mean_dist = total / max(count, 1)

        end = rollout.points[-1, :2].astype(np.float32)
        d_end = guide - end
        end_dist = math.sqrt(float(np.min(np.sum(d_end * d_end, axis=1))))

        # Heading consistency with the first segment of the far guide.
        g0 = guide[0]
        g1 = guide[min(len(guide) - 1, self.args.far_path_heading_index)]
        guide_heading = math.atan2(float(g1[1] - g0[1]), float(g1[0] - g0[0]))
        rollout_heading = math.atan2(float(rollout.y_end), float(rollout.length))
        heading_err = abs(self.normalize_angle(rollout_heading - guide_heading))

        return (
            mean_dist
            + self.args.far_path_endpoint_weight * end_dist
            + self.args.far_path_heading_weight * heading_err
        )

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

