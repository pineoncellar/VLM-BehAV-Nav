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

class LocalPublishersMixin:
    def publish_candidate(self, candidate: CandidatePath, stamp, frame_id: str) -> None:
        msg = Path()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        for x, y, yaw in candidate.points:
            msg.poses.append(self.make_pose(float(x), float(y), float(yaw), frame_id, stamp))
        self.selected_pub.publish(msg)

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

    def publish_local_status(self, status: str) -> None:
        msg = String()
        msg.data = str(status)
        self.local_status_pub.publish(msg)

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

