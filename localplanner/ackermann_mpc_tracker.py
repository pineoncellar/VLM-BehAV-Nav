#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ackermann_mpc_tracker.py - Modified for debugging, relaxed path length, throttled callback, and multithreaded executor.
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from std_msgs.msg import Bool


@dataclass
class Candidate:
    v: float
    steer_target: float
    cost: float
    yaw_rate: float


class AckermannMPCTracker(Node):
    def __init__(self, args):
        parameter_overrides = []
        if args.use_sim_time:
            parameter_overrides.append(Parameter("use_sim_time", Parameter.Type.BOOL, True))

        super().__init__("ackermann_mpc_tracker", parameter_overrides=parameter_overrides)
        self.args = args

        self.path_xy: Optional[np.ndarray] = None
        self.path_yaw: Optional[np.ndarray] = None
        self.last_path_wall_time: float = 0.0
        self.last_cmd_v: float = 0.0
        self.last_cmd_steer: float = 0.0
        self.goal_reached: bool = False

        # 限流：记录上次处理路径的时间
        self._last_path_process_time: float = 0.0

        self.path_sub = self.create_subscription(
            Path,
            args.path_topic,
            self.path_callback,
            10,
        )
        self.goal_reached_sub = self.create_subscription(
            Bool,
            args.far_goal_reached_topic,
            self.goal_reached_callback,
            10,
        )
        self.cmd_pub = self.create_publisher(Twist, args.cmd_vel_topic, 10)

        self.timer = self.create_timer(1.0 / max(args.control_hz, 1e-3), self.control_timer)

        self.get_logger().info("ackermann_mpc_tracker.py started")
        self.get_logger().info(f"path_topic        : {args.path_topic}")
        self.get_logger().info(f"cmd_vel_topic     : {args.cmd_vel_topic}")
        self.get_logger().info(f"control_hz        : {args.control_hz:.1f}")
        self.get_logger().info(f"wheelbase         : {args.wheelbase:.3f} m")
        self.get_logger().info(f"max_speed         : {args.max_speed:.3f} m/s")
        self.get_logger().info(f"max_steer         : {math.degrees(args.max_steer):.1f} deg")
        self.get_logger().info("output            : geometry_msgs/Twist, linear.x + angular.z")

    def path_callback(self, msg: Path) -> None:
        # 限流：每0.2秒最多处理一次，避免阻塞定时器
        now = time.time()
        if now - self._last_path_process_time < 0.2:
            return
        self._last_path_process_time = now

        if len(msg.poses) < 2:
            self.path_xy = None
            self.path_yaw = None
            self.get_logger().warn("Path has less than 2 poses, stopping")
            self.publish_stop()
            return

        pts = []
        yaws = []
        for pose_stamped in msg.poses:
            p = pose_stamped.pose.position
            q = pose_stamped.pose.orientation
            pts.append((float(p.x), float(p.y)))
            yaws.append(self.quat_to_yaw(q.x, q.y, q.z, q.w))

        xy = np.asarray(pts, dtype=np.float32)
        yaw = np.asarray(yaws, dtype=np.float32)

        # Remove duplicated or near-duplicated points
        keep = [0]
        for i in range(1, len(xy)):
            if np.linalg.norm(xy[i] - xy[keep[-1]]) > self.args.min_path_point_gap:
                keep.append(i)

        if len(keep) < 2:
            self.path_xy = None
            self.path_yaw = None
            self.get_logger().warn("Path collapsed to <2 points after dedup")
            return

        self.path_xy = xy[keep]
        self.path_yaw = yaw[keep]
        self.last_path_wall_time = time.time()
        self.get_logger().info(f"Received path with {len(keep)} points, length={self.path_length(self.path_xy):.2f}m")

    def goal_reached_callback(self, msg: Bool) -> None:
        self.goal_reached = bool(msg.data)
        if self.goal_reached:
            self.path_xy = None
            self.path_yaw = None
            self.publish_stop()
            self.get_logger().info("Goal reached, stopping")

    def control_timer(self) -> None:
        self.get_logger().info("control_timer called")  # 关键调试信息

        if self.goal_reached:
            self.get_logger().info("Goal reached, stop")
            self.publish_stop()
            return

        if self.path_xy is None or self.path_yaw is None:
            self.get_logger().info("No path received yet")
            self.publish_stop()
            return

        if time.time() - self.last_path_wall_time > self.args.path_timeout:
            self.get_logger().warn("Path timeout")
            self.publish_stop()
            return

        if self.path_xy.shape[0] < 2:
            self.get_logger().info("Path has <2 points")
            self.publish_stop()
            return

        path_len = self.path_length(self.path_xy)
        self.get_logger().info(f"Path length = {path_len:.3f} m, min_path_length = {self.args.min_path_length}")
        if path_len < self.args.min_path_length:
            self.get_logger().info("Path length too short, stopping")
            self.publish_stop()
            return

        best = self.solve_mpc(self.path_xy, self.path_yaw)
        if best is None:
            self.get_logger().warn("solve_mpc returned None")
            self.publish_stop()
            return

        self.get_logger().info(f"MPC -> v={best.v:.3f}, steer={best.steer_target:.3f}, cost={best.cost:.3f}")

        # Rate limit steering
        dt = 1.0 / max(self.args.control_hz, 1e-3)
        max_dsteer = self.args.max_steer_rate * dt
        steer_cmd = self.clamp(
            best.steer_target,
            self.last_cmd_steer - max_dsteer,
            self.last_cmd_steer + max_dsteer,
        )
        steer_cmd = self.clamp(steer_cmd, -self.args.max_steer, self.args.max_steer)

        # Rate limit speed
        max_dv = self.args.max_accel * dt
        v_cmd = self.clamp(best.v, self.last_cmd_v - max_dv, self.last_cmd_v + max_dv)
        v_cmd = self.clamp(v_cmd, 0.0, self.args.max_speed)

        yaw_rate = 0.0
        if abs(self.args.wheelbase) > 1e-6:
            yaw_rate = v_cmd / self.args.wheelbase * math.tan(steer_cmd)
        yaw_rate = self.clamp(yaw_rate, -self.args.max_yaw_rate, self.args.max_yaw_rate)

        self.last_cmd_v = v_cmd
        self.last_cmd_steer = steer_cmd

        msg = Twist()
        msg.linear.x = float(v_cmd)
        msg.angular.z = float(yaw_rate)
        self.cmd_pub.publish(msg)
        self.get_logger().info(f"Published cmd_vel: linear.x={v_cmd:.3f}, angular.z={yaw_rate:.3f}")

    def solve_mpc(self, path_xy: np.ndarray, path_yaw: np.ndarray) -> Optional[Candidate]:
        nominal_speed = self.compute_nominal_speed(path_xy)
        speed_samples = np.linspace(
            max(self.args.min_speed, nominal_speed - self.args.speed_sample_span),
            min(self.args.max_speed, nominal_speed + self.args.speed_sample_span),
            self.args.num_speed_samples,
            dtype=np.float32,
        )
        steer_samples = np.linspace(
            -self.args.max_steer,
            self.args.max_steer,
            self.args.num_steer_samples,
            dtype=np.float32,
        )
        best: Optional[Candidate] = None
        for v in speed_samples:
            for steer_target in steer_samples:
                cost, yaw_rate = self.evaluate_candidate(float(v), float(steer_target), path_xy, path_yaw)
                if best is None or cost < best.cost:
                    best = Candidate(v=float(v), steer_target=float(steer_target), cost=float(cost), yaw_rate=float(yaw_rate))
        return best

    def evaluate_candidate(self, v: float, steer_target: float, path_xy: np.ndarray, path_yaw: np.ndarray) -> Tuple[float, float]:
        x = 0.0
        y = 0.0
        yaw = 0.0
        steer = self.last_cmd_steer
        cost = 0.0
        dt = self.args.mpc_dt
        steps = self.args.horizon_steps
        max_dsteer_per_step = self.args.max_steer_rate * dt

        for k in range(steps):
            steer_error = steer_target - steer
            steer += self.clamp(steer_error, -max_dsteer_per_step, max_dsteer_per_step)
            steer = self.clamp(steer, -self.args.max_steer, self.args.max_steer)

            yaw_rate = 0.0
            if abs(self.args.wheelbase) > 1e-6:
                yaw_rate = v / self.args.wheelbase * math.tan(steer)
            yaw_rate = self.clamp(yaw_rate, -self.args.max_yaw_rate, self.args.max_yaw_rate)

            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            yaw = self.normalize_angle(yaw + yaw_rate * dt)

            nearest_idx, nearest_dist = self.nearest_path_point(x, y, path_xy)
            ref_yaw = float(path_yaw[min(nearest_idx, len(path_yaw) - 1)])
            yaw_err = abs(self.normalize_angle(yaw - ref_yaw))

            progress_cost = self.progress_cost(x, y, path_xy)
            step_weight = 1.0 + self.args.late_step_weight * (k / max(steps - 1, 1))
            cost += step_weight * self.args.cross_track_weight * nearest_dist * nearest_dist
            cost += step_weight * self.args.heading_weight * yaw_err * yaw_err
            cost += self.args.progress_weight * progress_cost
            cost += self.args.steer_weight * steer * steer
            cost += self.args.steer_rate_weight * (steer_target - self.last_cmd_steer) ** 2

        target = self.lookahead_target(path_xy, self.args.terminal_lookahead)
        terminal_dist = math.hypot(x - target[0], y - target[1])
        cost += self.args.terminal_weight * terminal_dist * terminal_dist

        cost += self.args.speed_weight * (v - self.compute_nominal_speed(path_xy)) ** 2

        final_yaw_rate = 0.0
        if abs(self.args.wheelbase) > 1e-6:
            final_yaw_rate = v / self.args.wheelbase * math.tan(steer_target)
        final_yaw_rate = self.clamp(final_yaw_rate, -self.args.max_yaw_rate, self.args.max_yaw_rate)

        return cost, final_yaw_rate

    def compute_nominal_speed(self, path_xy: np.ndarray) -> float:
        end = path_xy[-1]
        lateral = abs(float(end[1]))
        length = max(self.path_length(path_xy), 1e-3)
        lateral_ratio = min(1.0, lateral / max(length, 1e-3))
        speed = self.args.max_speed * (1.0 - self.args.lateral_slowdown * lateral_ratio)
        return self.clamp(speed, self.args.min_speed, self.args.max_speed)

    def nearest_path_point(self, x: float, y: float, path_xy: np.ndarray) -> Tuple[int, float]:
        d = path_xy - np.asarray([x, y], dtype=np.float32)
        dist2 = np.sum(d * d, axis=1)
        idx = int(np.argmin(dist2))
        return idx, math.sqrt(float(dist2[idx]))

    def progress_cost(self, x: float, y: float, path_xy: np.ndarray) -> float:
        target = self.lookahead_target(path_xy, self.args.progress_lookahead)
        dx = target[0] - x
        if dx <= 0.0:
            return 0.0
        return dx * dx

    def lookahead_target(self, path_xy: np.ndarray, lookahead: float) -> np.ndarray:
        if len(path_xy) == 0:
            return np.asarray([0.0, 0.0], dtype=np.float32)
        accum = 0.0
        prev = path_xy[0]
        for i in range(1, len(path_xy)):
            cur = path_xy[i]
            seg = float(np.linalg.norm(cur - prev))
            if accum + seg >= lookahead:
                ratio = (lookahead - accum) / max(seg, 1e-6)
                return prev + ratio * (cur - prev)
            accum += seg
            prev = cur
        return path_xy[-1]

    @staticmethod
    def path_length(path_xy: np.ndarray) -> float:
        if len(path_xy) < 2:
            return 0.0
        diffs = np.diff(path_xy, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def publish_stop(self) -> None:
        self.get_logger().info("publish_stop called")
        self.last_cmd_v = 0.0
        self.last_cmd_steer = 0.0
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)
        self.get_logger().info("Published stop (zero velocity)")

    # Math helpers
    @staticmethod
    def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

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
    parser = argparse.ArgumentParser(description="Ackermann MPC tracker")
    parser.add_argument("--path-topic", default="/local_selected_trajectory")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--far-goal-reached-topic", default="/far/goal_reached")
    parser.add_argument("--use-sim-time", dest="use_sim_time", action="store_true", default=True)
    parser.add_argument("--no-sim-time", dest="use_sim_time", action="store_false")

    parser.add_argument("--wheelbase", type=float, default=0.65)
    parser.add_argument("--max-steer", type=float, default=0.45)
    parser.add_argument("--max-steer-rate", type=float, default=0.8)
    parser.add_argument("--max-yaw-rate", type=float, default=1.2)

    parser.add_argument("--min-speed", type=float, default=0.0)
    parser.add_argument("--max-speed", type=float, default=0.45)
    parser.add_argument("--max-accel", type=float, default=0.4)

    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--mpc-dt", type=float, default=0.15)
    parser.add_argument("--horizon-steps", type=int, default=10)
    parser.add_argument("--num-speed-samples", type=int, default=4)
    parser.add_argument("--num-steer-samples", type=int, default=13)
    parser.add_argument("--speed-sample-span", type=float, default=0.20)

    parser.add_argument("--path-timeout", type=float, default=2.0)          # 放宽超时
    parser.add_argument("--min-path-length", type=float, default=0.0)       # 关键：设为0
    parser.add_argument("--min-path-point-gap", type=float, default=0.03)

    parser.add_argument("--cross-track-weight", type=float, default=8.0)
    parser.add_argument("--heading-weight", type=float, default=1.5)
    parser.add_argument("--terminal-weight", type=float, default=6.0)
    parser.add_argument("--progress-weight", type=float, default=0.4)
    parser.add_argument("--steer-weight", type=float, default=0.25)
    parser.add_argument("--steer-rate-weight", type=float, default=0.15)
    parser.add_argument("--speed-weight", type=float, default=0.2)
    parser.add_argument("--late-step-weight", type=float, default=1.0)
    parser.add_argument("--lateral-slowdown", type=float, default=0.7)

    parser.add_argument("--terminal-lookahead", type=float, default=1.8)
    parser.add_argument("--progress-lookahead", type=float, default=1.2)

    args, ros_args = parser.parse_known_args()
    return args, ros_args


def main():
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)
    node = AckermannMPCTracker(args)
    # 使用多线程执行器，避免回调阻塞定时器
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_stop()
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
