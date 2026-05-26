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

class FarGeometryMixin:
    @staticmethod
    def compact_points(points: List[Point2], min_gap: float) -> List[Point2]:
        if not points:
            return []
        out = [points[0]]
        for p in points[1:]:
            last = out[-1]
            if math.hypot(p.x - last.x, p.y - last.y) >= min_gap:
                out.append(p)
        return out

    @staticmethod
    def point_at_distance(points: List[Point2], lookahead: float) -> Point2:
        if not points:
            return Point2(0.0, 0.0)
        if len(points) == 1 or lookahead <= 0.0:
            return points[-1]

        accum = 0.0
        prev = points[0]
        for cur in points[1:]:
            seg = math.hypot(cur.x - prev.x, cur.y - prev.y)
            if accum + seg >= lookahead:
                ratio = (lookahead - accum) / max(seg, 1e-6)
                return Point2(
                    prev.x + ratio * (cur.x - prev.x),
                    prev.y + ratio * (cur.y - prev.y),
                )
            accum += seg
            prev = cur
        return points[-1]

    def local_to_odom(self, p: Point2) -> Point2:
        assert self.robot_x is not None
        assert self.robot_y is not None
        assert self.robot_yaw is not None
        c = math.cos(self.robot_yaw)
        s = math.sin(self.robot_yaw)
        return Point2(
            self.robot_x + c * p.x - s * p.y,
            self.robot_y + s * p.x + c * p.y,
        )

    def odom_to_local(self, p: Optional[Point2]) -> Optional[Point2]:
        if p is None:
            return None
        assert self.robot_x is not None
        assert self.robot_y is not None
        assert self.robot_yaw is not None
        dx = p.x - self.robot_x
        dy = p.y - self.robot_y
        c = math.cos(-self.robot_yaw)
        s = math.sin(-self.robot_yaw)
        return Point2(c * dx - s * dy, s * dx + c * dy)

    def distance_robot_to_odom_point(self, p: Point2) -> float:
        assert self.robot_x is not None
        assert self.robot_y is not None
        return math.hypot(p.x - self.robot_x, p.y - self.robot_y)

    @staticmethod
    def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

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

