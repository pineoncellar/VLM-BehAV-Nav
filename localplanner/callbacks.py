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

class LocalCallbacksMixin:
    def waypoint_callback(self, msg: PoseStamped) -> None:
        self.current_waypoint_local = (float(msg.pose.position.x), float(msg.pose.position.y))
        self.last_waypoint_wall_time = time.time()

    def far_path_callback(self, msg: Path) -> None:
        pts = []
        for pose_stamped in msg.poses:
            p = pose_stamped.pose.position
            pts.append((float(p.x), float(p.y)))

        if len(pts) < 2:
            self.far_path_xy = None
            return

        xy = np.asarray(pts, dtype=np.float32)
        keep = [0]
        for i in range(1, len(xy)):
            if np.linalg.norm(xy[i] - xy[keep[-1]]) > self.args.far_path_min_point_gap:
                keep.append(i)

        if len(keep) < 2:
            self.far_path_xy = None
            return

        self.far_path_xy = xy[keep]
        self.last_far_path_wall_time = time.time()

    def reference_path_callback(self, msg: Path) -> None:
        # Prefer the clean V6 topic. Internally it uses the same reference array.
        self.far_path_callback(msg)

    def waypoint_queue_callback(self, msg: PoseArray) -> None:
        pts = []
        for pose in msg.poses:
            pts.append((float(pose.position.x), float(pose.position.y)))
        if not pts:
            self.waypoint_queue_xy = None
            return
        self.waypoint_queue_xy = np.asarray(pts, dtype=np.float32)
        self.last_waypoint_queue_wall_time = time.time()

    def current_subgoal_callback(self, msg: PoseStamped) -> None:
        self.current_subgoal_local = (float(msg.pose.position.x), float(msg.pose.position.y))
        self.last_current_subgoal_wall_time = time.time()

    def route_status_callback(self, msg: String) -> None:
        self.far_route_status = str(msg.data)

    def goal_reached_callback(self, msg: Bool) -> None:
        self.goal_reached = bool(msg.data)
        if self.goal_reached:
            self.selected_group_id = None
            self.selected_rollout_id = None
            self.selected_candidate_id = None
            self.last_selected_rollout = None
            self.last_selected_path_points = None
            self.last_selected_eval = None
            self.local_mode = "RECOVERY_STOP"
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

        ev = self.select_by_arbiter(grid, info)
        if ev is None:
            self.get_logger().warn("No valid arbiter candidate selected. Publishing empty path.")
            self.publish_empty_path(msg.header.stamp, info.frame_id)
            self.publish_local_ok(False)
            self.publish_local_status("RECOVERY_STOP")
            return

        self.publish_candidate(ev.candidate, msg.header.stamp, info.frame_id)
        self.publish_local_ok(True)
        self.publish_local_status(self.local_mode)

        self.plan_count += 1
        if self.plan_count % max(1, self.args.print_every) == 0:
            wp_text = "none" if self.current_waypoint_local is None else f"({self.current_waypoint_local[0]:.2f},{self.current_waypoint_local[1]:.2f})"
            self.get_logger().info(
                f"arbiter mode={self.local_mode}, selected={ev.candidate.candidate_id}, "
                f"source={ev.candidate.source}, cost={ev.total_cost:.2f}, "
                f"mean_ref={ev.mean_ref_dist:.2f}, max_ref={ev.max_ref_dist:.2f}, "
                f"queue={ev.queue_cost:.2f}, progress={ev.progress_s:.2f}, waypoint={wp_text}"
            )

