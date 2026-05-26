#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import heapq
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
    from nav_msgs.msg import OccupancyGrid, Odometry, Path
    from visualization_msgs.msg import Marker, MarkerArray
    from std_msgs.msg import Bool, String
except Exception:
    # Allows syntax checking outside a ROS2 environment.
    pass

try:
    from .far_types import GridInfo, Point2, PolygonObstacle
except ImportError:
    from far_types import GridInfo, Point2, PolygonObstacle

class FarCallbacksMixin:
    def odom_callback(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.robot_x = float(p.x)
        self.robot_y = float(p.y)
        self.robot_yaw = self.quat_to_yaw(q.x, q.y, q.z, q.w)
        self.odom_frame = msg.header.frame_id if msg.header.frame_id else self.args.odom_frame_fallback

        if self.far_goal_odom is None:
            self.fix_far_goal_from_polar()

    def local_planner_ok_callback(self, msg: Bool) -> None:
        if bool(msg.data):
            self.local_fail_count = 0
        else:
            self.local_fail_count += 1

    def grid_callback(self, msg: OccupancyGrid) -> None:
        if self.robot_x is None or self.robot_y is None or self.robot_yaw is None:
            return
        if self.far_goal_odom is None:
            return

        now = time.time()
        if self.args.update_hz > 0.0:
            min_dt = 1.0 / self.args.update_hz
            if now - self.last_update_wall_time < min_dt:
                return
            self.last_update_wall_time = now

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

        free, unknown, occupied = self.split_grid(grid)

        # Safety layer:
        # occupied includes semantic cells if their value >= occupied_threshold.
        occupied_inflated = self.inflate_mask(occupied, info, self.args.occupied_inflation_radius)
        frontier = self.compute_frontier_mask(free, unknown, occupied_inflated)
        traversable = free & (~occupied_inflated)
        polygons = self.extract_obstacle_polygons(occupied_inflated, info)
        self.last_polygons = polygons

        self.update_waypoints(
            free=free,
            frontier=frontier,
            occupied_inflated=occupied_inflated,
            traversable=traversable,
            polygons=polygons,
            info=info,
        )
        self.publish_outputs(msg.header.stamp, info)

        self.update_count += 1
        if self.update_count % max(1, self.args.print_every) == 0:
            self.print_status(free, frontier, occupied_inflated)

