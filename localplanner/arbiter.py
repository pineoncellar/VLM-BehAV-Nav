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

class ArbiterMixin:
    def select_by_arbiter(self, grid: np.ndarray, info: GridInfo) -> Optional[ArbiterEval]:
        candidates = self.build_candidate_pool()
        if not candidates:
            self.no_valid_plan_count += 1
            return self.hold_last_eval_if_allowed()

        evals = [self.evaluate_candidate_for_arbiter(c, grid, info) for c in candidates]
        safe = [e for e in evals if e.safe]

        if not safe:
            if self.args.debug_rejections:
                reasons = {}
                for e in evals:
                    reasons[e.reason] = reasons.get(e.reason, 0) + 1
                self.get_logger().warn("arbiter unsafe candidates: " + ", ".join([f"{k}={v}" for k, v in reasons.items()]))
            self.no_valid_plan_count += 1
            return self.hold_last_eval_if_allowed()

        # Hard hierarchy:
        # 1. If there are safe candidates inside the FAR route tube, only choose among them.
        # 2. Otherwise choose an AVOID_LOCAL candidate that maximizes progress and can rejoin.
        in_tube = [
            e for e in safe
            if e.mean_ref_dist <= self.args.route_tube_radius
            and e.max_ref_dist <= self.args.route_tube_max_deviation
            and e.progress_s >= self.args.min_route_progress
        ]

        if in_tube:
            pool = in_tube
            target_mode = "FOLLOW_REF"
        else:
            pool = safe
            target_mode = "AVOID_LOCAL"

        best = min(pool, key=lambda e: e.total_cost)

        # Sticky candidate hysteresis: do not switch unless clearly better.
        old_eval = None
        if self.selected_candidate_id is not None:
            for e in safe:
                if e.candidate.candidate_id == self.selected_candidate_id:
                    old_eval = e
                    break

        if old_eval is not None:
            old_allowed = True
            if target_mode == "FOLLOW_REF":
                old_allowed = (
                    old_eval.mean_ref_dist <= self.args.route_tube_max_deviation
                    and old_eval.progress_s >= self.args.min_route_progress
                )
            if old_allowed and (old_eval.total_cost - best.total_cost) <= self.args.arbiter_switch_margin:
                best = old_eval

        self.local_mode = target_mode
        self.no_valid_plan_count = 0
        self.selected_candidate_id = best.candidate.candidate_id
        self.last_selected_path_points = best.candidate.points.copy()
        self.last_selected_eval = best

        if best.candidate.rollout is not None:
            self.selected_rollout_id = best.candidate.rollout.traj_id
            self.selected_group_id = best.candidate.rollout.group_id
            self.last_selected_rollout = best.candidate.rollout
        else:
            self.selected_rollout_id = None
            self.last_selected_rollout = None

        return best

    def hold_last_eval_if_allowed(self) -> Optional[ArbiterEval]:
        if (
            self.last_selected_eval is not None
            and self.no_valid_plan_count <= self.args.max_hold_last_path_frames
        ):
            self.local_mode = "HOLD_LAST"
            return self.last_selected_eval
        self.selected_candidate_id = None
        self.last_selected_eval = None
        self.last_selected_path_points = None
        self.local_mode = "RECOVERY_STOP"
        return None

    def build_candidate_pool(self) -> List[CandidatePath]:
        candidates: List[CandidatePath] = []

        ref = self.get_active_reference_path()
        if ref is not None and len(ref) >= 2:
            ref_pts = self.prepare_reference_candidate(ref)
            if ref_pts is not None and len(ref_pts) >= 2:
                candidates.append(CandidatePath(
                    candidate_id="reference:0",
                    source="reference",
                    group_id=-1,
                    rollout=None,
                    points=ref_pts,
                ))

        for r in self.rollouts:
            candidates.append(CandidatePath(
                candidate_id=f"rollout:{r.traj_id}",
                source="rollout",
                group_id=r.group_id,
                rollout=r,
                points=r.points,
            ))

        return candidates

    def get_active_reference_path(self) -> Optional[np.ndarray]:
        if self.far_path_xy is None or len(self.far_path_xy) < 2:
            return None
        if time.time() - self.last_far_path_wall_time > self.args.far_path_timeout:
            return None
        return self.far_path_xy

    def get_active_waypoint_queue(self) -> Optional[np.ndarray]:
        if self.waypoint_queue_xy is not None and len(self.waypoint_queue_xy) > 0:
            if time.time() - self.last_waypoint_queue_wall_time <= self.args.waypoint_queue_timeout:
                return self.waypoint_queue_xy

        # Fallback: current subgoal or current waypoint as one-point queue.
        if self.current_subgoal_local is not None:
            if time.time() - self.last_current_subgoal_wall_time <= self.args.waypoint_queue_timeout:
                return np.asarray([self.current_subgoal_local], dtype=np.float32)

        if self.current_waypoint_local is not None:
            if time.time() - self.last_waypoint_wall_time <= self.args.waypoint_timeout:
                return np.asarray([self.current_waypoint_local], dtype=np.float32)

        return None

    def prepare_reference_candidate(self, ref_xy: np.ndarray) -> Optional[np.ndarray]:
        pts = np.asarray(ref_xy, dtype=np.float32)
        if len(pts) < 2:
            return None

        # Ensure local trajectory starts at the vehicle origin.
        if float(np.linalg.norm(pts[0])) > self.args.ref_prepend_origin_distance:
            pts = np.vstack([np.asarray([[0.0, 0.0]], dtype=np.float32), pts])

        # Clip to trusted local range and length.
        clipped = [pts[0]]
        total = 0.0
        for i in range(1, len(pts)):
            a = clipped[-1]
            b = pts[i]
            if b[0] < self.args.ref_min_x:
                continue
            if b[0] > self.args.ref_max_x:
                break
            if abs(float(b[1])) > self.args.ref_y_limit:
                break

            seg = float(np.linalg.norm(b - a))
            if total + seg > self.args.ref_max_length:
                ratio = (self.args.ref_max_length - total) / max(seg, 1e-6)
                clipped.append(a + ratio * (b - a))
                break
            clipped.append(b)
            total += seg

        if len(clipped) < 2:
            return None

        xy = self.resample_polyline(np.asarray(clipped, dtype=np.float32), self.args.ref_publish_step)
        xy = self.smooth_polyline(xy, self.args.ref_smoothing_passes)
        yaws = self.polyline_yaws(xy)
        return np.column_stack([xy[:, 0], xy[:, 1], yaws]).astype(np.float32)

    def evaluate_candidate_for_arbiter(
        self,
        candidate: CandidatePath,
        grid: np.ndarray,
        info: GridInfo,
    ) -> ArbiterEval:
        safety = self.check_candidate_safety(candidate.points, grid, info)
        if not safety["safe"]:
            return ArbiterEval(
                candidate=candidate,
                safe=False,
                reason=safety["reason"],
                mode_hint="UNSAFE",
                total_cost=1e18,
                mean_ref_dist=1e6,
                max_ref_dist=1e6,
                endpoint_ref_dist=1e6,
                queue_cost=1e6,
                progress_s=-1e6,
                clearance_cost=safety["clearance_cost"],
                unknown_cost=safety["unknown_cost"],
                smooth_cost=1e6,
                curvature_cost=1e6,
                occupied_hits=safety["occupied_hits"],
                near_unknown_hits=safety["near_unknown_hits"],
                footprint_occupied_hits=safety["footprint_occupied_hits"],
                footprint_near_unknown_hits=safety["footprint_near_unknown_hits"],
            )

        ref_xy = self.get_active_reference_path()
        q_xy = self.get_active_waypoint_queue()

        mean_ref, max_ref, endpoint_ref, progress_s = self.reference_metrics(candidate.points[:, :2], ref_xy)
        queue_cost = self.queue_tracking_cost(candidate.points[:, :2], q_xy)
        smooth_cost = self.switch_smoothness_cost(candidate.points[:, :2])
        curvature_cost = self.curvature_cost(candidate.points)
        clearance_cost = safety["clearance_cost"]
        unknown_cost = safety["unknown_cost"]

        # Progress cost is negative reward, but bounded to avoid dominating safety/path matching.
        progress_cost = -self.args.progress_reward_weight * min(progress_s, self.args.progress_reward_cap)

        total = 0.0
        total += self.args.ref_mean_weight * mean_ref
        total += self.args.ref_max_weight * max_ref
        total += self.args.ref_endpoint_weight * endpoint_ref
        total += self.args.queue_weight * queue_cost
        total += progress_cost
        total += self.args.clearance_weight * clearance_cost
        total += self.args.unknown_cost_weight * unknown_cost
        total += self.args.switch_smooth_weight * smooth_cost
        total += self.args.curvature_weight * curvature_cost

        # Strong but not absolute preference for the reference candidate when it is safe.
        if candidate.source == "reference":
            total -= self.args.reference_candidate_bonus

        # Avoid mode should prefer candidates that end closer to the route.
        if mean_ref > self.args.route_tube_radius:
            total += self.args.rejoin_weight * endpoint_ref

        if candidate.rollout is not None:
            # Mildly discourage extreme lateral rollouts unless needed.
            total += self.args.rollout_shape_weight * abs(candidate.rollout.y_peak)

        mode_hint = "FOLLOW_REF" if (
            mean_ref <= self.args.route_tube_radius
            and max_ref <= self.args.route_tube_max_deviation
            and progress_s >= self.args.min_route_progress
        ) else "AVOID_LOCAL"

        return ArbiterEval(
            candidate=candidate,
            safe=True,
            reason="safe",
            mode_hint=mode_hint,
            total_cost=float(total),
            mean_ref_dist=float(mean_ref),
            max_ref_dist=float(max_ref),
            endpoint_ref_dist=float(endpoint_ref),
            queue_cost=float(queue_cost),
            progress_s=float(progress_s),
            clearance_cost=float(clearance_cost),
            unknown_cost=float(unknown_cost),
            smooth_cost=float(smooth_cost),
            curvature_cost=float(curvature_cost),
            occupied_hits=safety["occupied_hits"],
            near_unknown_hits=safety["near_unknown_hits"],
            footprint_occupied_hits=safety["footprint_occupied_hits"],
            footprint_near_unknown_hits=safety["footprint_near_unknown_hits"],
        )

    def reference_metrics(self, pts_xy: np.ndarray, ref_xy: Optional[np.ndarray]) -> Tuple[float, float, float, float]:
        if ref_xy is None or len(ref_xy) < 2:
            # Fall back to current waypoint if no reference path exists.
            if self.current_waypoint_local is None:
                return 2.0, 2.0, 2.0, 0.0
            target = np.asarray(self.current_waypoint_local, dtype=np.float32)
            d = np.linalg.norm(pts_xy - target[None, :], axis=1)
            return float(np.mean(d)), float(np.max(d)), float(np.linalg.norm(pts_xy[-1] - target)), float(pts_xy[-1, 0])

        query = pts_xy[::max(1, self.args.ref_metric_stride)]
        dists, progress = self.distance_and_progress_to_polyline(query, ref_xy)
        end_dist, end_prog = self.distance_and_progress_to_polyline(pts_xy[-1:, :], ref_xy)

        return (
            float(np.mean(dists)),
            float(np.max(dists)),
            float(end_dist[0]),
            float(end_prog[0]),
        )

    def queue_tracking_cost(self, pts_xy: np.ndarray, q_xy: Optional[np.ndarray]) -> float:
        if q_xy is None or len(q_xy) == 0:
            if self.current_waypoint_local is None:
                return 0.0
            q_xy = np.asarray([self.current_waypoint_local], dtype=np.float32)

        weights = list(self.args.queue_weights)
        if not weights:
            weights = [1.0]

        cost = 0.0
        w_sum = 0.0
        for i, q in enumerate(q_xy):
            w = weights[min(i, len(weights) - 1)]
            # Compare queue point qi with candidate point at similar arc-length.
            if len(self.args.queue_sample_distances) > i:
                s = float(self.args.queue_sample_distances[i])
            else:
                s = float(np.linalg.norm(q))
            p = self.point_on_polyline(pts_xy, s)
            d = float(np.linalg.norm(p - q))
            cost += w * d
            w_sum += w
        return cost / max(w_sum, 1e-6)

    def switch_smoothness_cost(self, pts_xy: np.ndarray) -> float:
        if self.last_selected_path_points is None or len(self.last_selected_path_points) < 2:
            return 0.0
        old = self.resample_polyline(self.last_selected_path_points[:, :2], self.args.switch_compare_step)
        new = self.resample_polyline(pts_xy, self.args.switch_compare_step)
        n = min(len(old), len(new))
        if n < 2:
            return 0.0
        return float(np.mean(np.linalg.norm(old[:n] - new[:n], axis=1)))

