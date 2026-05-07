#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ackermann_mpc_tracker.py

Standalone ROS2 Python file. Put it in:
  ~/depthmap/localplanner/ackermann_mpc_tracker.py

Role:
  Track /local_selected_trajectory with a lightweight MPC-style rollout controller
  for an Ackermann vehicle, and publish /cmd_vel.

Inputs:
  /local_selected_trajectory      nav_msgs/msg/Path

Outputs:
  /cmd_vel                        geometry_msgs/msg/Twist

Important assumptions:
  1. /local_selected_trajectory is in the vehicle local frame, usually base_footprint.
  2. Vehicle state in that local frame is x=0, y=0, yaw=0 at every control cycle.
  3. This controller does not need /odom for the first version.
  4. Ackermann kinematics are enforced internally:
       yaw_rate = v / wheelbase * tan(steering_angle)
  5. It publishes Twist because the user requested /cmd_vel speed + angular velocity.

Safety:
  - If no path is received, it publishes zero velocity.
  - If path is stale, it publishes zero velocity.
  - Default speed is conservative.

No scipy, no cv2, no nav2 dependency.
Only rclpy + numpy + standard ROS2 messages.
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

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def path_callback(self, msg: Path) -> None:
        if len(msg.poses) < 2:
            self.path_xy = None
            self.path_yaw = None
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

        # Remove duplicated or near-duplicated points to avoid numerical issues.
        keep = [0]
        for i in range(1, len(xy)):
            if np.linalg.norm(xy[i] - xy[keep[-1]]) > self.args.min_path_point_gap:
                keep.append(i)

        if len(keep) < 2:
            self.path_xy = None
            self.path_yaw = None
            return

        self.path_xy = xy[keep]
        self.path_yaw = yaw[keep]
        self.last_path_wall_time = time.time()

    def goal_reached_callback(self, msg: Bool) -> None:
        self.goal_reached = bool(msg.data)
        if self.goal_reached:
            self.path_xy = None
            self.path_yaw = None
            self.publish_stop()

    def control_timer(self) -> None:
        if self.goal_reached:
            self.publish_stop()
            return

        if self.path_xy is None or self.path_yaw is None:
            self.publish_stop()
            return

        if time.time() - self.last_path_wall_time > self.args.path_timeout:
            self.publish_stop()
            return

        if self.path_xy.shape[0] < 2:
            self.publish_stop()
            return

        path_len = self.path_length(self.path_xy)
        if path_len < self.args.min_path_length:
            self.publish_stop()
            return

        best = self.solve_mpc(self.path_xy, self.path_yaw)
        if best is None:
            self.publish_stop()
            return

        if best.v == 0.0 and abs(best.steer_target) == 0.0:
             self.get_logger().info("MPC solver chose v=0.0")

        # Rate-limit steering target for Ackermann smoothness.
        dt = 1.0 / max(self.args.control_hz, 1e-3)
        max_dsteer = self.args.max_steer_rate * dt
        steer_cmd = self.clamp(
            best.steer_target,
            self.last_cmd_steer - max_dsteer,
            self.last_cmd_steer + max_dsteer,
        )
        steer_cmd = self.clamp(steer_cmd, -self.args.max_steer, self.args.max_steer)

        # Rate-limit speed.
        max_dv = self.args.max_accel * dt
        v_cmd = self.clamp(best.v, self.last_cmd_v - max_dv, self.last_cmd_v + max_dv)
        v_cmd = self.clamp(v_cmd, 0.0, self.args.max_speed)

        yaw_rate = 0.0
        if abs(self.args.wheelbase) > 1e-6:
            yaw_rate = v_cmd / self.args.wheelbase * math.tan(steer_cmd)
        yaw_rate = self.clamp(yaw_rate, -self.args.max_yaw_rate, self.args.max_yaw_rate)

        self.last_cmd_v = v_cmd
        self.last_cmd_steer = steer_cmd

        if not hasattr(self, 'plan_count'):
            self.plan_count = 0
        self.plan_count += 1

        msg = Twist()
        msg.linear.x = float(v_cmd)
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(yaw_rate)
        
        # DEBUG output to verify if tracker commands actual movement
        if self.plan_count % max(1, int(self.args.control_hz)) == 0:
            self.get_logger().info(f"Publishing cmd_vel: v={v_cmd:.2f}, steer={math.degrees(steer_cmd):.1f} deg, yaw_rate={yaw_rate:.2f}. Path dist: {path_len:.1f}")

        self.cmd_pub.publish(msg)

    # ------------------------------------------------------------------
    # MPC-style rollout solver
    # ------------------------------------------------------------------

    def solve_mpc(self, path_xy: np.ndarray, path_yaw: np.ndarray) -> Optional[Candidate]:
        # Dynamic target speed: slower for high curvature / large lateral target.
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

    def evaluate_candidate(
        self,
        v: float,
        steer_target: float,
        path_xy: np.ndarray,
        path_yaw: np.ndarray,
    ) -> Tuple[float, float]:
        x = 0.0
        y = 0.0
        yaw = 0.0
        steer = self.last_cmd_steer
        cost = 0.0

        dt = self.args.mpc_dt
        steps = self.args.horizon_steps

        max_dsteer_per_step = self.args.max_steer_rate * dt

        for k in range(steps):
            # Smoothly approach steering target inside prediction.
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

            # Prefer points farther along the path as the horizon evolves.
            progress_cost = self.progress_cost(x, y, path_xy)

            step_weight = 1.0 + self.args.late_step_weight * (k / max(steps - 1, 1))
            cost += step_weight * self.args.cross_track_weight * nearest_dist * nearest_dist
            cost += step_weight * self.args.heading_weight * yaw_err * yaw_err
            cost += self.args.progress_weight * progress_cost
            cost += self.args.steer_weight * steer * steer
            cost += self.args.steer_rate_weight * (steer_target - self.last_cmd_steer) ** 2

        # Terminal cost to a lookahead point on selected trajectory.
        target = self.lookahead_target(path_xy, self.args.terminal_lookahead)
        terminal_dist = math.hypot(x - target[0], y - target[1])
        cost += self.args.terminal_weight * terminal_dist * terminal_dist

        # Prefer reasonable forward velocity, but do not dominate tracking.
        cost += self.args.speed_weight * (v - self.compute_nominal_speed(path_xy)) ** 2

        final_yaw_rate = 0.0
        if abs(self.args.wheelbase) > 1e-6:
            final_yaw_rate = v / self.args.wheelbase * math.tan(steer_target)
        final_yaw_rate = self.clamp(final_yaw_rate, -self.args.max_yaw_rate, self.args.max_yaw_rate)

        return cost, final_yaw_rate

    def compute_nominal_speed(self, path_xy: np.ndarray) -> float:
        # Slow down if the selected local path bends strongly or target is very lateral.
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
        # Penalize being behind the lookahead target along x. Since the path is in base_footprint,
        # x should generally increase forward.
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

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def publish_stop(self) -> None:
        self.last_cmd_v = 0.0
        self.last_cmd_steer = 0.0
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------

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
    parser = argparse.ArgumentParser(description="Ackermann MPC-style tracker for /local_selected_trajectory.")

    parser.add_argument("--path-topic", default="/local_selected_trajectory")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--far-goal-reached-topic", default="/far_goal_reached")
    parser.add_argument("--use-sim-time", dest="use_sim_time", action="store_true", default=True)
    parser.add_argument("--no-sim-time", dest="use_sim_time", action="store_false")

    # Vehicle model.
    parser.add_argument("--wheelbase", type=float, default=0.65)
    parser.add_argument("--max-steer", type=float, default=0.45, help="Max steering angle in radians.")
    parser.add_argument("--max-steer-rate", type=float, default=0.8, help="Max steering angle rate in rad/s.")
    parser.add_argument("--max-yaw-rate", type=float, default=1.2)

    # Speed limits. Keep conservative at first.
    parser.add_argument("--min-speed", type=float, default=0.0)
    parser.add_argument("--max-speed", type=float, default=0.45)
    parser.add_argument("--max-accel", type=float, default=0.4)

    # Control and prediction.
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--mpc-dt", type=float, default=0.15)
    parser.add_argument("--horizon-steps", type=int, default=10)
    parser.add_argument("--num-speed-samples", type=int, default=4)
    parser.add_argument("--num-steer-samples", type=int, default=13)
    parser.add_argument("--speed-sample-span", type=float, default=0.20)

    # Path safety.
    parser.add_argument("--path-timeout", type=float, default=0.6)
    parser.add_argument("--min-path-length", type=float, default=0.8)
    parser.add_argument("--min-path-point-gap", type=float, default=0.03)

    # Cost weights.
    parser.add_argument("--cross-track-weight", type=float, default=8.0)
    parser.add_argument("--heading-weight", type=float, default=1.5)
    parser.add_argument("--terminal-weight", type=float, default=6.0)
    parser.add_argument("--progress-weight", type=float, default=0.4)
    parser.add_argument("--steer-weight", type=float, default=0.25)
    parser.add_argument("--steer-rate-weight", type=float, default=0.15)
    parser.add_argument("--speed-weight", type=float, default=0.2)
    parser.add_argument("--late-step-weight", type=float, default=1.0)
    parser.add_argument("--lateral-slowdown", type=float, default=0.7)

    # Lookahead terms in meters along the local selected trajectory.
    parser.add_argument("--terminal-lookahead", type=float, default=1.8)
    parser.add_argument("--progress-lookahead", type=float, default=1.2)

    args, ros_args = parser.parse_known_args()
    return args, ros_args


def main():
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)
    node = AckermannMPCTracker(args)

    try:
        rclpy.spin(node)
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
