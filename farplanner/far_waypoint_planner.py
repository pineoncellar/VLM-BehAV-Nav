#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
import time
from typing import List, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool, String

try:
    from .far_types import GridInfo, Point2, PolygonObstacle
    from .callbacks import FarCallbacksMixin
    from .waypoint_logic import FarWaypointLogicMixin
    from .polygon_visibility import FarPolygonVisibilityMixin
    from .grid_search import FarGridSearchMixin
    from .grid_utils import FarGridUtilsMixin
    from .geometry import FarGeometryMixin
    from .publishers import FarPublishersMixin
except ImportError:
    from far_types import GridInfo, Point2, PolygonObstacle
    from callbacks import FarCallbacksMixin
    from waypoint_logic import FarWaypointLogicMixin
    from polygon_visibility import FarPolygonVisibilityMixin
    from grid_search import FarGridSearchMixin
    from grid_utils import FarGridUtilsMixin
    from geometry import FarGeometryMixin
    from publishers import FarPublishersMixin

class FarWaypointPlanner(
    Node,
    FarCallbacksMixin,
    FarWaypointLogicMixin,
    FarPolygonVisibilityMixin,
    FarGridSearchMixin,
    FarGridUtilsMixin,
    FarGeometryMixin,
    FarPublishersMixin,
):
    def __init__(self, args):
        parameter_overrides = []
        if args.use_sim_time:
            parameter_overrides.append(Parameter("use_sim_time", Parameter.Type.BOOL, True))

        super().__init__("far_waypoint_planner", parameter_overrides=parameter_overrides)
        self.args = args

        self.robot_x: Optional[float] = None
        self.robot_y: Optional[float] = None
        self.robot_yaw: Optional[float] = None
        self.odom_frame: str = args.odom_frame_fallback

        self.far_goal_odom: Optional[Point2] = None
        self.current_wp_odom: Optional[Point2] = None
        self.future_wp_odom: Optional[Point2] = None
        self.goal_reached: bool = False

        self.last_update_wall_time = 0.0
        self.update_count = 0

        self.local_fail_count = 0
        self.last_local_plan_points: List[Point2] = []
        self.last_reachable_count = 0
        self.last_seed_cell: Optional[Tuple[int, int]] = None
        self.last_polygons: List[PolygonObstacle] = []
        self.last_plan_source: str = "none"

        # V6 route contract:
        # - reference path: full local guide path for downstream modules
        # - waypoint queue: ordered future points sampled along that path
        # - current subgoal: primary point local planner should currently approach
        self.last_waypoint_queue_local: List[Point2] = []
        self.last_current_subgoal_local: Optional[Point2] = None

        self.grid_sub = self.create_subscription(
            OccupancyGrid,
            args.grid_topic,
            self.grid_callback,
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            args.odom_topic,
            self.odom_callback,
            20,
        )
        self.local_ok_sub = self.create_subscription(
            Bool,
            args.local_planner_ok_topic,
            self.local_planner_ok_callback,
            10,
        )

        # Data topics: these are meant for other nodes to consume.
        self.far_goal_pub = self.create_publisher(PoseStamped, args.goal_pose_topic, 10)
        self.current_wp_pub = self.create_publisher(PoseStamped, args.current_waypoint_topic, 10)
        self.current_wp_local_pub = self.create_publisher(PoseStamped, args.current_waypoint_local_topic, 10)
        self.future_wp_pub = self.create_publisher(PoseStamped, args.future_waypoint_topic, 10)
        self.local_plan_pub = self.create_publisher(Path, args.local_plan_topic, 10)

        # V6 data topics. These are the clean interface for the new arbiter.
        self.reference_path_pub = self.create_publisher(Path, args.reference_path_topic, 10)
        self.waypoint_queue_pub = self.create_publisher(PoseArray, args.waypoint_queue_topic, 10)
        self.current_subgoal_pub = self.create_publisher(PoseStamped, args.current_subgoal_topic, 10)
        self.route_status_pub = self.create_publisher(String, args.route_status_topic, 10)

        self.goal_reached_pub = self.create_publisher(Bool, args.far_goal_reached_topic, 10)

        # Visualization topics: these are only for RViz/debug.
        self.polygon_pub = self.create_publisher(MarkerArray, args.obstacle_polygons_topic, 10)
        self.odom_points_viz_pub = self.create_publisher(MarkerArray, args.odom_points_viz_topic, 10)
        self.local_path_viz_pub = self.create_publisher(MarkerArray, args.local_path_viz_topic, 10)
        self.waypoint_queue_viz_pub = self.create_publisher(MarkerArray, args.waypoint_queue_viz_topic, 10)

        self.get_logger().info("far_waypoint_planner.py Step3 polygon-vgraph started")
        self.get_logger().info(f"grid_topic       : {args.grid_topic}")
        self.get_logger().info(f"odom_topic       : {args.odom_topic}")
        self.get_logger().info(f"local_ok_topic   : {args.local_planner_ok_topic}")
        self.get_logger().info(f"far goal polar   : angle={args.goal_angle_deg:.2f} deg, distance={args.goal_distance:.2f} m")
        self.get_logger().info(f"arrival radius   : {args.arrival_radius:.2f} m")
        self.get_logger().info(f"final radius     : {args.final_goal_radius:.2f} m")
        self.get_logger().info(f"lookahead        : {args.lookahead_distance:.2f} m")
        self.get_logger().info("strategy         : forward seed + obstacle polygons + visibility graph + A* fallback")
        self.get_logger().info("vehicle control  : disabled")
        self.get_logger().info(f"reference_path   : {args.reference_path_topic}")
        self.get_logger().info(f"waypoint_queue   : {args.waypoint_queue_topic}")
        self.get_logger().info(f"current_subgoal  : {args.current_subgoal_topic}")
        self.get_logger().info(f"route_status     : {args.route_status_topic}")
        self.get_logger().info(f"goal_pose_topic  : {args.goal_pose_topic}")
        self.get_logger().info(f"current_local    : {args.current_waypoint_local_topic}")
        self.get_logger().info(f"local_plan_topic : {args.local_plan_topic}")
        self.get_logger().info(f"viz points       : {args.odom_points_viz_topic}")
        self.get_logger().info(f"viz local paths  : {args.local_path_viz_topic}")
        self.get_logger().info(f"viz polygons     : {args.obstacle_polygons_topic}")

def parse_args():
    parser = argparse.ArgumentParser(description="Depth-camera-friendly reachable waypoint planner. No vehicle control.")

    parser.add_argument("--grid-topic", default="/local_traversability_grid")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--local-planner-ok-topic", default="/local_planner_ok")

    # Data topics: stable API for other nodes.
    parser.add_argument("--goal-pose-topic", default="/far/goal_pose")
    parser.add_argument("--current-waypoint-topic", default="/far/current_waypoint_odom")
    parser.add_argument("--current-waypoint-local-topic", default="/far/current_waypoint_local")
    parser.add_argument("--future-waypoint-topic", default="/far/future_waypoint_odom")
    parser.add_argument("--local-plan-topic", default="/far/local_plan")
    parser.add_argument("--reference-path-topic", default="/far/reference_path_local")
    parser.add_argument("--waypoint-queue-topic", default="/far/waypoint_queue_local")
    parser.add_argument("--current-subgoal-topic", default="/far/current_subgoal_local")
    parser.add_argument("--route-status-topic", default="/far/route_status")
    parser.add_argument("--far-goal-reached-topic", default="/far/goal_reached")

    # Visualization topics: RViz only. Kept separate so points do not look like path lines.
    parser.add_argument("--odom-points-viz-topic", default="/far/viz/odom_points")
    parser.add_argument("--local-path-viz-topic", default="/far/viz/local_paths")
    parser.add_argument("--obstacle-polygons-topic", default="/far/viz/obstacle_polygons")
    parser.add_argument("--waypoint-queue-viz-topic", default="/far/viz/waypoint_queue")

    parser.add_argument("--use-sim-time", dest="use_sim_time", action="store_true", default=True)
    parser.add_argument("--no-sim-time", dest="use_sim_time", action="store_false")

    parser.add_argument("--goal-angle-deg", type=float, default=0.0)
    parser.add_argument("--goal-distance", type=float, default=25.0)
    parser.add_argument("--odom-frame-fallback", default="odom")
    parser.add_argument("--local-frame-fallback", default="base_footprint")

    # Waypoint distances.
    parser.add_argument("--arrival-radius", type=float, default=0.6)
    parser.add_argument("--final-goal-radius", type=float, default=1.0)
    parser.add_argument("--subgoal-min-distance", type=float, default=1.2)
    parser.add_argument("--subgoal-max-distance", type=float, default=4.0)
    parser.add_argument("--subgoal-min-x", type=float, default=0.8)
    parser.add_argument("--subgoal-y-limit", type=float, default=2.4)
    parser.add_argument("--lookahead-distance", type=float, default=2.5)
    parser.add_argument("--future-lookahead-distance", type=float, default=3.0)
    parser.add_argument("--future-min-separation", type=float, default=1.2)
    parser.add_argument("--waypoint-release-distance", type=float, default=6.0)
    parser.add_argument("--dynamic-current-waypoint", dest="dynamic_current_waypoint", action="store_true", default=True)
    parser.add_argument("--lock-current-waypoint", dest="dynamic_current_waypoint", action="store_false")
    parser.add_argument("--current-waypoint-smoothing-alpha", type=float, default=0.45)
    parser.add_argument("--current-waypoint-update-distance", type=float, default=0.8)

    # V6 ordered waypoint queue.
    parser.add_argument("--waypoint-queue-distances", type=float, nargs="+", default=[0.8, 1.5, 2.5, 3.3, 4.0])
    parser.add_argument("--current-subgoal-distance", type=float, default=2.5)
    parser.add_argument("--min-waypoint-queue-spacing", type=float, default=0.25)
    parser.add_argument("--max-waypoint-queue-size", type=int, default=5)
    parser.add_argument("--queue-min-x", type=float, default=-0.05)
    parser.add_argument("--queue-max-x", type=float, default=4.5)
    parser.add_argument("--queue-y-limit", type=float, default=2.8)

    # Forward seed search. This is the important Step2b part.
    parser.add_argument("--seed-min-x", type=float, default=0.4)
    parser.add_argument("--seed-max-x", type=float, default=2.6)
    parser.add_argument("--seed-y-limit", type=float, default=1.4)
    parser.add_argument("--seed-lateral-weight", type=float, default=2.0)

    # Occupancy semantics.
    parser.add_argument("--occupied-threshold", type=int, default=50)
    parser.add_argument("--occupied-inflation-radius", type=float, default=0.35)

    # Polygon / visibility graph structural layer.
    parser.add_argument("--use-polygon-visibility", dest="use_polygon_visibility", action="store_true", default=True)
    parser.add_argument("--no-polygon-visibility", dest="use_polygon_visibility", action="store_false")
    parser.add_argument("--polygon-min-cells", type=int, default=3)
    parser.add_argument("--polygon-bbox-cell-padding", type=int, default=0)
    parser.add_argument("--polygon-vertex-clearance", type=float, default=0.35)
    parser.add_argument("--max-polygons", type=int, default=20)
    parser.add_argument("--max-visibility-polygons", type=int, default=12)
    parser.add_argument("--max-visibility-nodes", type=int, default=48)
    parser.add_argument("--visibility-node-max-distance", type=float, default=5.5)
    parser.add_argument("--visibility-edge-max-length", type=float, default=6.0)
    parser.add_argument("--max-polygon-markers", type=int, default=20)

    # Search settings.
    parser.add_argument("--use-diagonal-connectivity", dest="use_diagonal_connectivity", action="store_true", default=True)
    parser.add_argument("--no-diagonal-connectivity", dest="use_diagonal_connectivity", action="store_false")
    parser.add_argument("--astar-heuristic-weight", type=float, default=1.2)
    parser.add_argument("--max-astar-expansions", type=int, default=3000)
    parser.add_argument("--max-astar-candidate-tries", type=int, default=20)
    parser.add_argument("--enable-old-style-fallback", dest="enable_old_style_fallback", action="store_true", default=True)
    parser.add_argument("--disable-old-style-fallback", dest="enable_old_style_fallback", action="store_false")

    # Main scoring weights. Less greedy than the original version.
    parser.add_argument("--remaining-distance-weight", type=float, default=3.0)
    parser.add_argument("--local-goal-distance-weight", type=float, default=4.0)
    parser.add_argument("--progress-weight", type=float, default=4.0)
    parser.add_argument("--angle-weight", type=float, default=1.5)
    parser.add_argument("--lateral-weight", type=float, default=0.8)
    parser.add_argument("--center-weight", type=float, default=0.1)
    parser.add_argument("--edge-weight", type=float, default=2.0)
    parser.add_argument("--edge-margin", type=float, default=0.5)
    parser.add_argument("--backward-weight", type=float, default=5.0)

    # Frontier is disabled by default for fixed-goal navigation.
    parser.add_argument("--frontier-bonus", type=float, default=0.0)
    parser.add_argument("--future-distance-weight", type=float, default=0.2)

    # Local planner failure feedback.
    parser.add_argument("--max-local-fail-count", type=int, default=8)

    # RViz style parameters.
    parser.add_argument("--viz-far-goal-scale", type=float, default=0.90)
    parser.add_argument("--viz-current-wp-scale", type=float, default=0.65)
    parser.add_argument("--viz-current-wp-local-scale", type=float, default=0.55)
    parser.add_argument("--viz-future-wp-scale", type=float, default=0.40)
    parser.add_argument("--viz-text-height", type=float, default=0.32)
    parser.add_argument("--viz-local-plan-width", type=float, default=0.12)
    parser.add_argument("--viz-waypoint-vector-width", type=float, default=0.08)
    parser.add_argument("--viz-polygon-line-width", type=float, default=0.055)
    parser.add_argument("--viz-queue-point-scale", type=float, default=0.34)
    parser.add_argument("--viz-queue-line-width", type=float, default=0.065)
    parser.add_argument("--viz-queue-text-height", type=float, default=0.26)

    parser.add_argument("--update-hz", type=float, default=5.0)
    parser.add_argument("--print-every", type=int, default=20)

    args, ros_args = parser.parse_known_args()
    return args, ros_args

def main():
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)
    node = FarWaypointPlanner(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        # ExternalShutdownException may not be importable in non-ROS lint environments.
        if exc.__class__.__name__ != "ExternalShutdownException":
            raise
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
