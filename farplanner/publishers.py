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

class FarPublishersMixin:
    def publish_outputs(self, stamp, info: GridInfo) -> None:
        odom_frame = self.odom_frame

        # ----------------------------
        # Machine-readable data topics.
        # ----------------------------
        if self.far_goal_odom is not None:
            self.far_goal_pub.publish(
                self.make_pose(self.far_goal_odom.x, self.far_goal_odom.y, 0.0, odom_frame, stamp)
            )

        if self.current_wp_odom is not None:
            self.current_wp_pub.publish(
                self.make_pose(self.current_wp_odom.x, self.current_wp_odom.y, 0.0, odom_frame, stamp)
            )
            current_local = self.odom_to_local(self.current_wp_odom)
            if current_local is not None:
                yaw = math.atan2(current_local.y, current_local.x)
                self.current_wp_local_pub.publish(
                    self.make_pose(current_local.x, current_local.y, yaw, info.frame_id, stamp)
                )

        if self.future_wp_odom is not None:
            self.future_wp_pub.publish(
                self.make_pose(self.future_wp_odom.x, self.future_wp_odom.y, 0.0, odom_frame, stamp)
            )

        # Compatibility topic for existing local planner.
        self.publish_local_plan(stamp, info.frame_id)

        # V6 clean route topics for the upcoming arbiter.
        self.publish_reference_path(stamp, info.frame_id)
        self.publish_waypoint_queue(stamp, info.frame_id)
        self.publish_current_subgoal(stamp, info.frame_id)
        self.publish_route_status()

        # ----------------------------
        # RViz visualization topics.
        # ----------------------------
        self.publish_local_path_markers(stamp, info.frame_id)
        self.publish_waypoint_queue_markers(stamp, info.frame_id)
        self.publish_polygon_markers(stamp, info.frame_id)
        self.publish_odom_point_markers(stamp, odom_frame)

        reached_msg = Bool()
        reached_msg.data = bool(self.goal_reached)
        self.goal_reached_pub.publish(reached_msg)

    def publish_reference_path(self, stamp, frame_id: str) -> None:
        """Publish the ordered local reference path for the new local arbiter."""
        msg = Path()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        for i, p in enumerate(self.last_local_plan_points):
            if i + 1 < len(self.last_local_plan_points):
                nxt = self.last_local_plan_points[i + 1]
                yaw = math.atan2(nxt.y - p.y, nxt.x - p.x)
            elif i > 0:
                prev = self.last_local_plan_points[i - 1]
                yaw = math.atan2(p.y - prev.y, p.x - prev.x)
            else:
                yaw = 0.0
            msg.poses.append(self.make_pose(p.x, p.y, yaw, frame_id, stamp))
        self.reference_path_pub.publish(msg)

    def publish_waypoint_queue(self, stamp, frame_id: str) -> None:
        """Publish ordered future points as PoseArray in local frame."""
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        for i, p in enumerate(self.last_waypoint_queue_local):
            if i + 1 < len(self.last_waypoint_queue_local):
                nxt = self.last_waypoint_queue_local[i + 1]
                yaw = math.atan2(nxt.y - p.y, nxt.x - p.x)
            else:
                yaw = math.atan2(p.y, p.x) if abs(p.x) + abs(p.y) > 1e-6 else 0.0
            msg.poses.append(self.make_pose(p.x, p.y, yaw, frame_id, stamp).pose)
        self.waypoint_queue_pub.publish(msg)

    def publish_current_subgoal(self, stamp, frame_id: str) -> None:
        """Publish primary local subgoal. This mirrors the current waypoint local topic semantically."""
        p = self.last_current_subgoal_local
        if p is None:
            return
        yaw = math.atan2(p.y, p.x) if abs(p.x) + abs(p.y) > 1e-6 else 0.0
        self.current_subgoal_pub.publish(self.make_pose(p.x, p.y, yaw, frame_id, stamp))

    def publish_route_status(self) -> None:
        msg = String()
        if self.goal_reached:
            msg.data = "GOAL_REACHED"
        elif not self.last_local_plan_points:
            msg.data = "NO_ROUTE"
        else:
            msg.data = self.last_plan_source.upper()
        self.route_status_pub.publish(msg)

    def sample_waypoint_queue(self, points: List[Point2]) -> List[Point2]:
        """Sample a forward ordered waypoint queue along the reference path."""
        if not points or len(points) < 2:
            return []

        queue: List[Point2] = []
        distances = list(self.args.waypoint_queue_distances)
        if not distances:
            distances = [self.args.current_subgoal_distance]

        for d in distances:
            if d <= 0.0:
                continue
            p = self.point_at_distance(points, float(d))

            # Avoid repeating nearly identical points on short paths.
            if queue:
                last = queue[-1]
                if math.hypot(p.x - last.x, p.y - last.y) < self.args.min_waypoint_queue_spacing:
                    continue

            # Keep queue points inside the trusted local region.
            if p.x < self.args.queue_min_x or p.x > self.args.queue_max_x:
                continue
            if abs(p.y) > self.args.queue_y_limit:
                continue

            queue.append(p)
            if len(queue) >= self.args.max_waypoint_queue_size:
                break

        # Ensure the current subgoal distance is represented if possible.
        if not queue:
            queue.append(self.point_at_distance(points, self.args.current_subgoal_distance))
        return queue

    def publish_waypoint_queue_markers(self, stamp, frame_id: str) -> None:
        """Visualization-only: numbered large queue points and connecting line."""
        msg = MarkerArray()

        clear = Marker()
        clear.action = Marker.DELETEALL
        msg.markers.append(clear)

        if not self.last_waypoint_queue_local:
            self.waypoint_queue_viz_pub.publish(msg)
            return

        # Connecting cyan line.
        line = Marker()
        line.header.stamp = stamp
        line.header.frame_id = frame_id
        line.ns = "far_waypoint_queue_line"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = float(self.args.viz_queue_line_width)
        line.color.r = 0.0
        line.color.g = 0.95
        line.color.b = 1.0
        line.color.a = 0.95
        for p in self.last_waypoint_queue_local:
            from geometry_msgs.msg import Point
            qpt = Point()
            qpt.x = float(p.x)
            qpt.y = float(p.y)
            qpt.z = 0.16
            line.points.append(qpt)
        msg.markers.append(line)

        from geometry_msgs.msg import Point

        for i, p in enumerate(self.last_waypoint_queue_local):
            sphere = Marker()
            sphere.header.stamp = stamp
            sphere.header.frame_id = frame_id
            sphere.ns = "far_waypoint_queue_points"
            sphere.id = 100 + i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(p.x)
            sphere.pose.position.y = float(p.y)
            sphere.pose.position.z = 0.20
            sphere.pose.orientation.w = 1.0
            scale = self.args.viz_queue_point_scale * (1.15 if i == 0 else 1.0)
            sphere.scale.x = scale
            sphere.scale.y = scale
            sphere.scale.z = scale

            if i == 0:
                # Primary/near point: orange-red.
                sphere.color.r = 1.0
                sphere.color.g = 0.28
                sphere.color.b = 0.05
            else:
                # Future queue: cyan-blue.
                sphere.color.r = 0.0
                sphere.color.g = 0.85
                sphere.color.b = 1.0
            sphere.color.a = 0.95
            msg.markers.append(sphere)

            text = Marker()
            text.header.stamp = stamp
            text.header.frame_id = frame_id
            text.ns = "far_waypoint_queue_labels"
            text.id = 200 + i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(p.x)
            text.pose.position.y = float(p.y)
            text.pose.position.z = 0.55
            text.pose.orientation.w = 1.0
            text.scale.z = float(self.args.viz_queue_text_height)
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f"Q{i}"
            msg.markers.append(text)

        self.waypoint_queue_viz_pub.publish(msg)

    def publish_local_plan(self, stamp, frame_id: str) -> None:
        """Publish the guide path as nav_msgs/Path for other nodes."""
        msg = Path()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        for p in self.last_local_plan_points:
            yaw = math.atan2(p.y, p.x) if abs(p.x) + abs(p.y) > 1e-6 else 0.0
            msg.poses.append(self.make_pose(p.x, p.y, yaw, frame_id, stamp))
        self.local_plan_pub.publish(msg)

    def publish_polygon_markers(self, stamp, frame_id: str) -> None:
        """Orange outlines for obstacle polygons. Visualization-only."""
        msg = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        msg.markers.append(clear)

        for i, poly in enumerate(self.last_polygons[: self.args.max_polygon_markers]):
            if not poly.vertices:
                continue

            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = frame_id
            m.ns = "obstacle_polygon_outline"
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = float(self.args.viz_polygon_line_width)
            m.color.r = 1.0
            m.color.g = 0.48
            m.color.b = 0.0
            m.color.a = 0.95

            verts = poly.vertices + [poly.vertices[0]]
            for p in verts:
                pose = self.make_pose(p.x, p.y, 0.0, frame_id, stamp)
                m.points.append(pose.pose.position)
            msg.markers.append(m)

        self.polygon_pub.publish(msg)

    def publish_odom_point_markers(self, stamp, frame_id: str) -> None:
        """Big, clearly labeled odom-frame target markers.

        This topic is only for RViz. It deliberately uses spheres + labels,
        not Path lines, so target points do not look like route lines.
        """
        msg = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        msg.markers.append(clear)

        items = []
        if self.far_goal_odom is not None:
            items.append({
                "name": "FAR GOAL",
                "point": self.far_goal_odom,
                "id": 10,
                "scale": self.args.viz_far_goal_scale,
                "color": (1.0, 0.85, 0.05, 1.0),
                "z": 0.45,
            })
        if self.current_wp_odom is not None:
            items.append({
                "name": "CURRENT WP",
                "point": self.current_wp_odom,
                "id": 20,
                "scale": self.args.viz_current_wp_scale,
                "color": (1.0, 0.05, 0.05, 1.0),
                "z": 0.35,
            })
        if self.future_wp_odom is not None:
            items.append({
                "name": "FUTURE WP",
                "point": self.future_wp_odom,
                "id": 30,
                "scale": self.args.viz_future_wp_scale,
                "color": (0.1, 0.75, 1.0, 0.9),
                "z": 0.30,
            })

        for item in items:
            p = item["point"]
            r, g, b, a = item["color"]

            sphere = Marker()
            sphere.header.stamp = stamp
            sphere.header.frame_id = frame_id
            sphere.ns = "far_target_points"
            sphere.id = int(item["id"])
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(p.x)
            sphere.pose.position.y = float(p.y)
            sphere.pose.position.z = float(item["z"])
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = float(item["scale"])
            sphere.scale.y = float(item["scale"])
            sphere.scale.z = float(item["scale"])
            sphere.color.r = r
            sphere.color.g = g
            sphere.color.b = b
            sphere.color.a = a
            msg.markers.append(sphere)

            label = Marker()
            label.header.stamp = stamp
            label.header.frame_id = frame_id
            label.ns = "far_target_labels"
            label.id = int(item["id"]) + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(p.x)
            label.pose.position.y = float(p.y)
            label.pose.position.z = float(item["z"]) + float(item["scale"]) * 0.9
            label.pose.orientation.w = 1.0
            label.scale.z = float(self.args.viz_text_height)
            label.color.r = r
            label.color.g = g
            label.color.b = b
            label.color.a = 1.0
            label.text = item["name"]
            msg.markers.append(label)

        self.odom_points_viz_pub.publish(msg)

    def publish_local_path_markers(self, stamp, frame_id: str) -> None:
        """Thick local-frame guide path markers.

        This topic is only for RViz. The actual path data is published on
        /far/local_plan as nav_msgs/Path.
        """
        msg = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        msg.markers.append(clear)

        # Main guide path: thick green line.
        if len(self.last_local_plan_points) >= 2:
            guide = Marker()
            guide.header.stamp = stamp
            guide.header.frame_id = frame_id
            guide.ns = "far_local_guide_path"
            guide.id = 1
            guide.type = Marker.LINE_STRIP
            guide.action = Marker.ADD
            guide.scale.x = float(self.args.viz_local_plan_width)
            guide.color.r = 0.05
            guide.color.g = 1.0
            guide.color.b = 0.20
            guide.color.a = 1.0
            for p in self.last_local_plan_points:
                pose = self.make_pose(p.x, p.y, 0.0, frame_id, stamp)
                guide.points.append(pose.pose.position)
            msg.markers.append(guide)

            # Add a small arrow-like endpoint sphere, so it is obvious where the guide path goes.
            end = self.last_local_plan_points[-1]
            end_m = Marker()
            end_m.header.stamp = stamp
            end_m.header.frame_id = frame_id
            end_m.ns = "far_local_guide_endpoint"
            end_m.id = 2
            end_m.type = Marker.SPHERE
            end_m.action = Marker.ADD
            end_m.pose.position.x = float(end.x)
            end_m.pose.position.y = float(end.y)
            end_m.pose.position.z = 0.18
            end_m.pose.orientation.w = 1.0
            end_m.scale.x = 0.28
            end_m.scale.y = 0.28
            end_m.scale.z = 0.28
            end_m.color.r = 0.05
            end_m.color.g = 1.0
            end_m.color.b = 0.20
            end_m.color.a = 1.0
            msg.markers.append(end_m)

        # Current waypoint in local frame: big red sphere. This is the point local planner tracks.
        current_local = self.odom_to_local(self.current_wp_odom)
        if current_local is not None:
            wp = Marker()
            wp.header.stamp = stamp
            wp.header.frame_id = frame_id
            wp.ns = "far_current_waypoint_local"
            wp.id = 10
            wp.type = Marker.SPHERE
            wp.action = Marker.ADD
            wp.pose.position.x = float(current_local.x)
            wp.pose.position.y = float(current_local.y)
            wp.pose.position.z = 0.28
            wp.pose.orientation.w = 1.0
            wp.scale.x = float(self.args.viz_current_wp_local_scale)
            wp.scale.y = float(self.args.viz_current_wp_local_scale)
            wp.scale.z = float(self.args.viz_current_wp_local_scale)
            wp.color.r = 1.0
            wp.color.g = 0.05
            wp.color.b = 0.05
            wp.color.a = 1.0
            msg.markers.append(wp)

            vector = Marker()
            vector.header.stamp = stamp
            vector.header.frame_id = frame_id
            vector.ns = "far_current_waypoint_vector"
            vector.id = 11
            vector.type = Marker.LINE_STRIP
            vector.action = Marker.ADD
            vector.scale.x = float(self.args.viz_waypoint_vector_width)
            vector.color.r = 1.0
            vector.color.g = 0.05
            vector.color.b = 0.05
            vector.color.a = 0.9
            origin = self.make_pose(0.0, 0.0, 0.0, frame_id, stamp).pose.position
            target = self.make_pose(current_local.x, current_local.y, 0.0, frame_id, stamp).pose.position
            vector.points.append(origin)
            vector.points.append(target)
            msg.markers.append(vector)

        self.local_path_viz_pub.publish(msg)

    def make_pose(self, x: float, y: float, yaw: float, frame_id: str, stamp) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.05
        qx, qy, qz, qw = self.yaw_to_quat(yaw)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg

    def print_status(self, free: np.ndarray, frontier: np.ndarray, occupied_inflated: np.ndarray) -> None:
        current_dist = self.distance_robot_to_odom_point(self.current_wp_odom) if self.current_wp_odom is not None else float("nan")
        far_local = self.odom_to_local(self.far_goal_odom)
        far_text = "none" if far_local is None else f"({far_local.x:.2f},{far_local.y:.2f})"
        seed_text = "none" if self.last_seed_cell is None else f"({self.last_seed_cell[0]},{self.last_seed_cell[1]})"
        self.get_logger().info(
            f"current_dist={current_dist:.2f}m, far_local={far_text}, "
            f"free={int(np.count_nonzero(free))}, frontier={int(np.count_nonzero(frontier))}, "
            f"inflated_occ={int(np.count_nonzero(occupied_inflated))}, "
            f"reachable={self.last_reachable_count}, seed={seed_text}, "
            f"polygons={len(self.last_polygons)}, source={self.last_plan_source}, local_fail={self.local_fail_count}"
        )

