#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
far_waypoint_planner.py

Step 3 version: depth-camera-friendly polygon / visibility-graph waypoint manager.

This is NOT the full original CMU FAR Planner, because we only have a local
OccupancyGrid from one depth camera. It adds a CMU-like structure layer:
connected obstacle regions are represented as simple polygons, then a local
visibility graph is built around polygon vertices.

Core idea:
  - Keep the old planner's practical behavior: choose a waypoint that makes
    progress toward the fixed far goal.
  - Add a safer reachable-region layer, but do NOT require robot cell itself
    to be known free. With a depth camera, the area around x=0 is often unknown.
  - If robot cell is not traversable, search for the nearest forward free seed.
  - Flood-fill from that seed.
  - Score reachable candidate cells.
  - Extract local obstacle polygons from the inflated occupancy grid.
  - Try a polygon visibility graph path first; fall back to A* if needed.
  - Publish a lookahead point on that local path as /current_waypoint_local.

Inputs:
  /local_traversability_grid      nav_msgs/msg/OccupancyGrid
  /odom                           nav_msgs/msg/Odometry
  /local_planner_ok               std_msgs/msg/Bool, optional

Outputs:
  Data topics:
    /far/goal_pose                 geometry_msgs/msg/PoseStamped, odom frame
    /far/current_waypoint_odom     geometry_msgs/msg/PoseStamped, odom frame, locked
    /far/current_waypoint_local    geometry_msgs/msg/PoseStamped, local grid frame
    /far/future_waypoint_odom      geometry_msgs/msg/PoseStamped, odom frame, changeable
    /far/local_plan                nav_msgs/msg/Path, local frame, compatibility guide path
    /far/reference_path_local       nav_msgs/msg/Path, ordered local reference path
    /far/waypoint_queue_local       geometry_msgs/msg/PoseArray, ordered future waypoint queue
    /far/current_subgoal_local      geometry_msgs/msg/PoseStamped, primary local subgoal
    /far/route_status              std_msgs/msg/String, route source/status
    /far/goal_reached              std_msgs/msg/Bool

  Visualization topics:
    /far/viz/odom_points           visualization_msgs/msg/MarkerArray, big goal/waypoint markers
    /far/viz/local_paths           visualization_msgs/msg/MarkerArray, thick local guide path
    /far/viz/obstacle_polygons     visualization_msgs/msg/MarkerArray, local-frame polygon debug

Map values expected:
  -1 = unknown
   0 = free
  80 = semantic blocked, if CLIPSeg is enabled
 100 = occupied

Important:
  - If semantic_block_value is 80 and occupied_threshold is 50, semantic cells
    are treated as hard obstacles. For planner debugging, prefer disabling
    CLIPSeg or setting semantic_block_value below occupied_threshold.
"""

import argparse
import heapq
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from geometry_msgs.msg import PoseStamped, PoseArray, Point
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool, String


@dataclass
class GridInfo:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    frame_id: str


@dataclass
class Point2:
    x: float
    y: float


@dataclass
class PolygonObstacle:
    component_id: int
    vertices: List[Point2]
    cell_count: int


class FarWaypointPlanner(Node):
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
        self.goal_reached: bool = True

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
        self.polar_goal_sub = self.create_subscription(
            Point,
            '/behav/goal_polar',
            self.polar_goal_callback,
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

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def odom_callback(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.robot_x = float(p.x)
        self.robot_y = float(p.y)
        self.robot_yaw = self.quat_to_yaw(q.x, q.y, q.z, q.w)
        self.odom_frame = msg.header.frame_id if msg.header.frame_id else self.args.odom_frame_fallback

        # Do NOT auto-set a goal. Wait for the polar_goal_callback from VLM.
        # if self.far_goal_odom is None:
        #     self.fix_far_goal_from_polar()

    def polar_goal_callback(self, msg: Point) -> None:
        if msg.z == 1.0:
            self.far_goal_odom = Point2(msg.x, msg.y)
            self.goal_reached = False
            self.get_logger().info(f"Updated absolute goal from BeHav: x={msg.x:.2f}, y={msg.y:.2f}")
            return
            
        if self.robot_x is None or self.robot_y is None or self.robot_yaw is None:
            return
        distance = msg.x
        angle_deg = msg.y
        angle = math.radians(angle_deg)
        heading = self.robot_yaw + angle
        gx = self.robot_x + distance * math.cos(heading)
        gy = self.robot_y + distance * math.sin(heading)
        self.far_goal_odom = Point2(gx, gy)
        self.goal_reached = False
        self.get_logger().info(f"Updated goal from BeHav: dist={distance:.2f}m, angle={angle_deg:.1f}deg")

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

    # ------------------------------------------------------------------
    # Goal and waypoint logic
    # ------------------------------------------------------------------

    def fix_far_goal_from_polar(self) -> None:
        import traceback; traceback.print_stack()
        assert self.robot_x is not None
        assert self.robot_y is not None
        assert self.robot_yaw is not None

        angle = math.radians(self.args.goal_angle_deg)
        heading = self.robot_yaw + angle
        gx = self.robot_x + self.args.goal_distance * math.cos(heading)
        gy = self.robot_y + self.args.goal_distance * math.sin(heading)
        self.far_goal_odom = Point2(gx, gy)

        self.get_logger().info(
            f"Far goal fixed in {self.odom_frame}: x={gx:.3f}, y={gy:.3f}, "
            f"angle={self.args.goal_angle_deg:.1f}deg, distance={self.args.goal_distance:.1f}m"
        )

    def update_waypoints(
        self,
        free: np.ndarray,
        frontier: np.ndarray,
        occupied_inflated: np.ndarray,
        traversable: np.ndarray,
        polygons: List[PolygonObstacle],
        info: GridInfo,
    ) -> None:
        self.last_local_plan_points = []
        self.last_waypoint_queue_local = []
        self.last_current_subgoal_local = None
        self.last_reachable_count = 0
        self.last_seed_cell = None
        self.last_plan_source = "none"

        if self.far_goal_odom is not None:
            d_goal = self.distance_robot_to_odom_point(self.far_goal_odom)
            if d_goal <= self.args.final_goal_radius:
                if not self.goal_reached:
                    self.get_logger().info(f"Final far goal reached: distance={d_goal:.2f}m")
                self.goal_reached = True
                self.current_wp_odom = None
                self.future_wp_odom = None
                return

        if self.goal_reached:
            self.current_wp_odom = None
            self.future_wp_odom = None
            return

        # Always compute a fresh guide path. The old hard lock could make
        # /current_waypoint_local and /far_local_plan disagree, which then caused
        # the local rollout to diverge from the far path.
        guide_local = self.select_reachable_lookahead_waypoint(
            traversable=traversable,
            frontier=frontier,
            polygons=polygons,
            info=info,
            future_mode=False,
        )
        guide_plan_points = list(self.last_local_plan_points)

        if guide_local is None and self.args.enable_old_style_fallback:
            fallback_local = self.select_goal_closest_feasible_waypoint_old(
                free=free,
                frontier=frontier,
                occupied_inflated=occupied_inflated,
                info=info,
                future_mode=False,
            )
            if fallback_local is not None:
                guide_local = fallback_local
                guide_plan_points = [Point2(0.0, 0.0), fallback_local]
                self.last_plan_source = "old_fallback"
                self.get_logger().warn(
                    f"Polygon/VGraph selection failed. Using old-style fallback waypoint: "
                    f"local=({fallback_local.x:.2f},{fallback_local.y:.2f})"
                )

        if guide_local is None:
            self.current_wp_odom = None
            self.future_wp_odom = None
            self.last_local_plan_points = []
            self.last_waypoint_queue_local = []
            self.last_current_subgoal_local = None
            self.get_logger().warn("No current waypoint candidate found.")
            return

        old_local = self.odom_to_local(self.current_wp_odom) if self.current_wp_odom is not None else None
        force_update = False
        if self.current_wp_odom is None:
            force_update = True
        elif old_local is None:
            force_update = True
        elif math.hypot(old_local.x, old_local.y) < self.args.arrival_radius:
            force_update = True
        elif self.local_fail_count >= self.args.max_local_fail_count:
            force_update = True
        elif math.hypot(old_local.x - guide_local.x, old_local.y - guide_local.y) > self.args.current_waypoint_update_distance:
            force_update = True
        elif self.args.dynamic_current_waypoint:
            force_update = True

        if force_update:
            if old_local is not None and not (self.current_wp_odom is None or self.local_fail_count >= self.args.max_local_fail_count):
                a = self.args.current_waypoint_smoothing_alpha
                a = self.clamp(a, 0.0, 1.0)
                current_local = Point2(
                    (1.0 - a) * old_local.x + a * guide_local.x,
                    (1.0 - a) * old_local.y + a * guide_local.y,
                )
            else:
                current_local = guide_local

            self.current_wp_odom = self.local_to_odom(current_local)
            self.local_fail_count = 0
            if self.update_count % max(1, self.args.print_every) == 0:
                self.get_logger().info(
                    f"Current waypoint updated: local=({current_local.x:.2f},{current_local.y:.2f}), "
                    f"source={self.last_plan_source}"
                )

        # Keep guide path consistent with the current waypoint guidance.
        self.last_local_plan_points = guide_plan_points

        # V6 ordered route contract:
        # publish a sequence of future waypoints sampled by arc length.
        self.last_waypoint_queue_local = self.sample_waypoint_queue(self.last_local_plan_points)

        actual_current_local = self.odom_to_local(self.current_wp_odom) if self.current_wp_odom is not None else None
        if actual_current_local is not None:
            self.last_current_subgoal_local = actual_current_local
        elif self.last_local_plan_points:
            self.last_current_subgoal_local = self.point_at_distance(
                self.last_local_plan_points,
                self.args.current_subgoal_distance,
            )
        else:
            self.last_current_subgoal_local = None

        # Future waypoint is only a preview marker; do not let it overwrite /far_local_plan.
        saved_plan = list(self.last_local_plan_points)
        saved_source = self.last_plan_source
        future_local = self.select_reachable_lookahead_waypoint(
            traversable=traversable,
            frontier=frontier,
            polygons=polygons,
            info=info,
            future_mode=True,
        )
        if future_local is not None:
            self.future_wp_odom = self.local_to_odom(future_local)
        self.last_local_plan_points = saved_plan
        self.last_plan_source = saved_source

    def select_reachable_lookahead_waypoint(
        self,
        traversable: np.ndarray,
        frontier: np.ndarray,
        polygons: List[PolygonObstacle],
        info: GridInfo,
        future_mode: bool,
    ) -> Optional[Point2]:
        """Select a waypoint using forward free seed + reachable flood fill + A* path.

        This is the key Step2b behavior:
          - Do not require robot cell to be free.
          - Use robot cell only when it is traversable.
          - Otherwise search a forward free seed.
          - If no seed exists, fail or allow old-style fallback outside this method.
        """
        far_local = self.odom_to_local(self.far_goal_odom)
        if far_local is None:
            return None

        start_cell = self.choose_start_seed(traversable, info)
        if start_cell is None:
            if not future_mode:
                self.get_logger().warn("No reachable seed found in forward free area.")
            return None

        self.last_seed_cell = start_cell
        reachable = self.flood_fill_reachable(traversable, start_cell)
        self.last_reachable_count = int(np.count_nonzero(reachable))
        if self.last_reachable_count <= 0:
            if not future_mode:
                self.get_logger().warn("No reachable free region from selected seed.")
            return None

        candidates = self.collect_candidate_cells(
            reachable=reachable,
            traversable=traversable,
            frontier=frontier,
            info=info,
            far_local=far_local,
            future_mode=future_mode,
        )
        if not candidates:
            if not future_mode:
                self.get_logger().warn("No reachable candidate cells after filtering.")
            return None

        # Try top candidates until polygon visibility graph or A* succeeds.
        max_try = min(len(candidates), self.args.max_astar_candidate_tries)
        for _score, goal_cell in candidates[:max_try]:
            path_points: List[Point2] = []

            if self.args.use_polygon_visibility:
                path_points = self.visibility_graph_path(
                    traversable=traversable,
                    start_cell=start_cell,
                    goal_cell=goal_cell,
                    polygons=polygons,
                    info=info,
                )
                if path_points:
                    self.last_plan_source = "polygon_visibility"

            if not path_points:
                path_cells = self.astar(traversable, start_cell, goal_cell)
                if not path_cells:
                    continue
                path_points = [Point2(*self.grid_to_world_center(ix, iy, info)) for ix, iy in path_cells]
                self.last_plan_source = "astar"

            path_points = [Point2(0.0, 0.0)] + path_points
            path_points = self.compact_points(path_points, min_gap=0.05)
            if len(path_points) < 2:
                continue

            self.last_local_plan_points = path_points
            lookahead = self.args.future_lookahead_distance if future_mode else self.args.lookahead_distance
            return self.point_at_distance(path_points, lookahead)

        if not future_mode:
            self.get_logger().warn("Polygon visibility and A* failed for all reachable candidates.")
        return None

    def choose_start_seed(self, traversable: np.ndarray, info: GridInfo) -> Optional[Tuple[int, int]]:
        """Return a start cell for reachable flood fill.

        Prefer robot cell if it is free. Otherwise, find the nearest free cell
        in front of the robot. This is necessary for depth-camera maps where
        x=0 area is often unknown.
        """
        robot_cell = self.world_to_grid(0.0, 0.0, info)
        if robot_cell is not None:
            rx, ry = robot_cell
            if 0 <= rx < info.width and 0 <= ry < info.height and traversable[ry, rx]:
                return robot_cell

        ys, xs = np.nonzero(traversable)
        if xs.size == 0:
            return None

        best = None
        best_score = 1e18
        for ix, iy in zip(xs, ys):
            x, y = self.grid_to_world_center(int(ix), int(iy), info)
            if x < self.args.seed_min_x or x > self.args.seed_max_x:
                continue
            if abs(y) > self.args.seed_y_limit:
                continue

            # Prefer near-front and centerline cells.
            score = math.hypot(x, y) + self.args.seed_lateral_weight * abs(y)
            if score < best_score:
                best_score = score
                best = (int(ix), int(iy))

        return best

    def collect_candidate_cells(
        self,
        reachable: np.ndarray,
        traversable: np.ndarray,
        frontier: np.ndarray,
        info: GridInfo,
        far_local: Point2,
        future_mode: bool,
    ) -> List[Tuple[float, Tuple[int, int]]]:
        far_dist = math.hypot(far_local.x, far_local.y)
        theta_goal = math.atan2(far_local.y, far_local.x)
        ux = math.cos(theta_goal)
        uy = math.sin(theta_goal)

        active_local = self.odom_to_local(self.current_wp_odom) if self.current_wp_odom is not None else None

        ys, xs = np.nonzero(reachable & traversable)
        scored: List[Tuple[float, Tuple[int, int]]] = []

        for ix, iy in zip(xs, ys):
            x, y = self.grid_to_world_center(int(ix), int(iy), info)
            local_dist = math.hypot(x, y)

            if local_dist < self.args.subgoal_min_distance or local_dist > self.args.subgoal_max_distance:
                continue
            if x < self.args.subgoal_min_x:
                continue
            if abs(y) > self.args.subgoal_y_limit:
                continue

            if future_mode and active_local is not None:
                if math.hypot(x - active_local.x, y - active_local.y) < self.args.future_min_separation:
                    continue

            heading = math.atan2(y, x)
            angle_err = abs(self.normalize_angle(heading - theta_goal))
            progress = x * ux + y * uy
            lateral = abs(-x * uy + y * ux)
            center_abs = abs(y)
            edge = self.edge_penalty(x, y, info)
            remaining_to_far_goal = math.hypot(far_local.x - x, far_local.y - y)
            backward_penalty = abs(progress) if progress < 0.0 else 0.0
            frontier_bonus = self.args.frontier_bonus if frontier[iy, ix] else 0.0
            future_bonus = self.args.future_distance_weight * local_dist if future_mode else 0.0

            score = 0.0
            score -= self.args.remaining_distance_weight * remaining_to_far_goal
            score += self.args.progress_weight * progress
            score -= self.args.angle_weight * angle_err
            score -= self.args.lateral_weight * lateral
            score -= self.args.center_weight * center_abs
            score -= self.args.edge_weight * edge
            score -= self.args.backward_weight * backward_penalty
            score += frontier_bonus
            score += future_bonus

            if far_dist <= self.args.subgoal_max_distance:
                score -= self.args.local_goal_distance_weight * remaining_to_far_goal

            scored.append((float(score), (int(ix), int(iy))))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def select_goal_closest_feasible_waypoint_old(
        self,
        free: np.ndarray,
        frontier: np.ndarray,
        occupied_inflated: np.ndarray,
        info: GridInfo,
        future_mode: bool,
    ) -> Optional[Point2]:
        """Old-style fallback from the user's original planner.

        This intentionally does not require reachability. It is used only as a
        fallback so Step2b does not perform worse than the previous planner when
        depth grid has near-field holes.
        """
        far_local = self.odom_to_local(self.far_goal_odom)
        if far_local is None:
            return None

        far_dist = math.hypot(far_local.x, far_local.y)
        theta_goal = math.atan2(far_local.y, far_local.x)
        ux = math.cos(theta_goal)
        uy = math.sin(theta_goal)
        active_local = self.odom_to_local(self.current_wp_odom) if self.current_wp_odom is not None else None

        candidate_mask = free & (~occupied_inflated)
        ys, xs = np.nonzero(candidate_mask)
        best_score = -1e18
        best_point = None

        for ix, iy in zip(xs, ys):
            x, y = self.grid_to_world_center(int(ix), int(iy), info)
            local_dist = math.hypot(x, y)

            if local_dist < self.args.subgoal_min_distance or local_dist > self.args.subgoal_max_distance:
                continue
            if x < self.args.subgoal_min_x:
                continue
            if abs(y) > self.args.subgoal_y_limit:
                continue

            if future_mode and active_local is not None:
                if math.hypot(x - active_local.x, y - active_local.y) < self.args.future_min_separation:
                    continue

            heading = math.atan2(y, x)
            angle_err = abs(self.normalize_angle(heading - theta_goal))
            progress = x * ux + y * uy
            lateral = abs(-x * uy + y * ux)
            center_abs = abs(y)
            edge = self.edge_penalty(x, y, info)
            remaining_to_far_goal = math.hypot(far_local.x - x, far_local.y - y)
            backward_penalty = abs(progress) if progress < 0.0 else 0.0
            frontier_bonus = self.args.frontier_bonus if frontier[iy, ix] else 0.0
            future_bonus = self.args.future_distance_weight * local_dist if future_mode else 0.0

            score = 0.0
            score -= self.args.remaining_distance_weight * remaining_to_far_goal
            score += self.args.progress_weight * progress
            score -= self.args.angle_weight * angle_err
            score -= self.args.lateral_weight * lateral
            score -= self.args.center_weight * center_abs
            score -= self.args.edge_weight * edge
            score -= self.args.backward_weight * backward_penalty
            score += frontier_bonus
            score += future_bonus

            if far_dist <= self.args.subgoal_max_distance:
                score -= self.args.local_goal_distance_weight * remaining_to_far_goal

            if score > best_score:
                best_score = score
                best_point = Point2(x, y)

        return best_point

    # ------------------------------------------------------------------
    # Polygon structural map / visibility graph helpers
    # ------------------------------------------------------------------

    def extract_obstacle_polygons(self, mask: np.ndarray, info: GridInfo) -> List[PolygonObstacle]:
        """Extract simple rectangular polygons from connected obstacle regions.

        This is intentionally lightweight: no cv2/scipy. It is not a full CMU
        FAR polygon extractor, but it gives the far planner a structural layer
        instead of using raw grid cells only.
        """
        h, w = mask.shape
        visited = np.zeros((h, w), dtype=bool)
        polygons: List[PolygonObstacle] = []
        comp_id = 0

        for sy in range(h):
            for sx in range(w):
                if visited[sy, sx] or not mask[sy, sx]:
                    continue

                q = [(sx, sy)]
                visited[sy, sx] = True
                head = 0
                xs = []
                ys = []
                while head < len(q):
                    cx, cy = q[head]
                    head += 1
                    xs.append(cx)
                    ys.append(cy)
                    for nx, ny in self.grid_neighbors_8(cx, cy, w, h):
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        q.append((nx, ny))

                count = len(xs)
                if count < self.args.polygon_min_cells:
                    continue

                min_ix = max(0, min(xs) - self.args.polygon_bbox_cell_padding)
                max_ix = min(w - 1, max(xs) + self.args.polygon_bbox_cell_padding)
                min_iy = max(0, min(ys) - self.args.polygon_bbox_cell_padding)
                max_iy = min(h - 1, max(ys) + self.args.polygon_bbox_cell_padding)

                x0 = info.origin_x + min_ix * info.resolution
                x1 = info.origin_x + (max_ix + 1) * info.resolution
                y0 = info.origin_y + min_iy * info.resolution
                y1 = info.origin_y + (max_iy + 1) * info.resolution

                poly = PolygonObstacle(
                    component_id=comp_id,
                    vertices=[Point2(x0, y0), Point2(x1, y0), Point2(x1, y1), Point2(x0, y1)],
                    cell_count=count,
                )
                polygons.append(poly)
                comp_id += 1
                if len(polygons) >= self.args.max_polygons:
                    return polygons

        return polygons

    def polygon_navigation_nodes(
        self,
        polygons: List[PolygonObstacle],
        traversable: np.ndarray,
        info: GridInfo,
        start: Point2,
        goal: Point2,
    ) -> List[Point2]:
        nodes: List[Point2] = []
        clearance = self.args.polygon_vertex_clearance
        max_dist = self.args.visibility_node_max_distance

        for poly in polygons[: self.args.max_visibility_polygons]:
            cx = sum(v.x for v in poly.vertices) / max(len(poly.vertices), 1)
            cy = sum(v.y for v in poly.vertices) / max(len(poly.vertices), 1)
            for v in poly.vertices:
                dx = v.x - cx
                dy = v.y - cy
                n = math.hypot(dx, dy)
                if n < 1e-6:
                    continue
                p = Point2(v.x + clearance * dx / n, v.y + clearance * dy / n)

                if math.hypot(p.x - start.x, p.y - start.y) > max_dist and math.hypot(p.x - goal.x, p.y - goal.y) > max_dist:
                    continue
                cell = self.world_to_grid(p.x, p.y, info)
                if cell is None:
                    continue
                ix, iy = cell
                if not traversable[iy, ix]:
                    # Try a tiny pull toward the polygon's outside direction.
                    p2 = Point2(v.x + 2.0 * clearance * dx / n, v.y + 2.0 * clearance * dy / n)
                    cell2 = self.world_to_grid(p2.x, p2.y, info)
                    if cell2 is None:
                        continue
                    ix2, iy2 = cell2
                    if not traversable[iy2, ix2]:
                        continue
                    p = p2
                nodes.append(p)

        # Deduplicate by grid cell.
        unique: Dict[Tuple[int, int], Point2] = {}
        for p in nodes:
            cell = self.world_to_grid(p.x, p.y, info)
            if cell is not None:
                unique[cell] = p
        return list(unique.values())[: self.args.max_visibility_nodes]

    def visibility_graph_path(
        self,
        traversable: np.ndarray,
        start_cell: Tuple[int, int],
        goal_cell: Tuple[int, int],
        polygons: List[PolygonObstacle],
        info: GridInfo,
    ) -> List[Point2]:
        start = Point2(*self.grid_to_world_center(start_cell[0], start_cell[1], info))
        goal = Point2(*self.grid_to_world_center(goal_cell[0], goal_cell[1], info))
        nodes = [start, goal]
        nodes.extend(self.polygon_navigation_nodes(polygons, traversable, info, start, goal))

        n = len(nodes)
        if n < 2:
            return []

        adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = math.hypot(nodes[i].x - nodes[j].x, nodes[i].y - nodes[j].y)
                if d > self.args.visibility_edge_max_length:
                    continue
                if self.line_of_sight_free(nodes[i], nodes[j], traversable, info):
                    adj[i].append((j, d))
                    adj[j].append((i, d))

        return self.dijkstra_nodes(nodes, adj, 0, 1)

    def line_of_sight_free(self, a: Point2, b: Point2, traversable: np.ndarray, info: GridInfo) -> bool:
        ca = self.world_to_grid(a.x, a.y, info)
        cb = self.world_to_grid(b.x, b.y, info)
        if ca is None or cb is None:
            return False
        for ix, iy in self.bresenham(ca[0], ca[1], cb[0], cb[1]):
            if ix < 0 or ix >= info.width or iy < 0 or iy >= info.height:
                return False
            if not traversable[iy, ix]:
                return False
        return True

    @staticmethod
    def bresenham(x0: int, y0: int, x1: int, y1: int):
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            yield x, y
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    @staticmethod
    def dijkstra_nodes(nodes: List[Point2], adj: List[List[Tuple[int, float]]], start_idx: int, goal_idx: int) -> List[Point2]:
        heap: List[Tuple[float, int]] = [(0.0, start_idx)]
        dist: Dict[int, float] = {start_idx: 0.0}
        prev: Dict[int, int] = {}
        while heap:
            d, u = heapq.heappop(heap)
            if u == goal_idx:
                path_idx = [u]
                while u in prev:
                    u = prev[u]
                    path_idx.append(u)
                path_idx.reverse()
                return [nodes[i] for i in path_idx]
            if d > dist.get(u, 1e18):
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist.get(v, 1e18):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        return []

    # ------------------------------------------------------------------
    # Grid search helpers
    # ------------------------------------------------------------------

    def flood_fill_reachable(self, traversable: np.ndarray, start: Tuple[int, int]) -> np.ndarray:
        h, w = traversable.shape
        reachable = np.zeros((h, w), dtype=bool)
        sx, sy = start

        if sx < 0 or sx >= w or sy < 0 or sy >= h:
            return reachable
        if not traversable[sy, sx]:
            return reachable

        q: List[Tuple[int, int]] = [(sx, sy)]
        reachable[sy, sx] = True
        head = 0

        neighbors = self.grid_neighbors_8 if self.args.use_diagonal_connectivity else self.grid_neighbors_4

        while head < len(q):
            cx, cy = q[head]
            head += 1

            for nx, ny in neighbors(cx, cy, w, h):
                if reachable[ny, nx]:
                    continue
                if not traversable[ny, nx]:
                    continue
                reachable[ny, nx] = True
                q.append((nx, ny))

        return reachable

    def astar(
        self,
        traversable: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        h, w = traversable.shape
        sx, sy = start
        gx, gy = goal

        if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
            return []
        if not traversable[sy, sx] or not traversable[gy, gx]:
            return []

        if start == goal:
            return [start]

        neighbors = self.grid_neighbors_8 if self.args.use_diagonal_connectivity else self.grid_neighbors_4

        open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, 0.0, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_cost: Dict[Tuple[int, int], float] = {start: 0.0}

        expanded = 0
        while open_heap:
            _f, current_g, current = heapq.heappop(open_heap)
            if current == goal:
                return self.reconstruct_path(came_from, current)

            expanded += 1
            if expanded > self.args.max_astar_expansions:
                break

            cx, cy = current
            for nx, ny in neighbors(cx, cy, w, h):
                if not traversable[ny, nx]:
                    continue

                step = math.sqrt(2.0) if (nx != cx and ny != cy) else 1.0
                new_g = current_g + step

                nxt = (nx, ny)
                if nxt not in g_cost or new_g < g_cost[nxt]:
                    g_cost[nxt] = new_g
                    h_cost = math.hypot(gx - nx, gy - ny)
                    f = new_g + self.args.astar_heuristic_weight * h_cost
                    came_from[nxt] = current
                    heapq.heappush(open_heap, (f, new_g, nxt))

        return []

    @staticmethod
    def reconstruct_path(
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def grid_neighbors_4(cx: int, cy: int, w: int, h: int):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = cx + dx
            ny = cy + dy
            if 0 <= nx < w and 0 <= ny < h:
                yield nx, ny

    @staticmethod
    def grid_neighbors_8(cx: int, cy: int, w: int, h: int):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = cx + dx
                ny = cy + dy
                if 0 <= nx < w and 0 <= ny < h:
                    yield nx, ny

    # ------------------------------------------------------------------
    # Grid utilities
    # ------------------------------------------------------------------

    def split_grid(self, grid: np.ndarray):
        # Treat non-negative values below occupied_threshold as traversable.
        # This lets you set semantic_block_value below occupied_threshold during
        # debugging without turning those cells into neither-free-nor-occupied holes.
        unknown = grid < 0
        occupied = grid >= self.args.occupied_threshold
        free = (grid >= 0) & (~occupied)
        return free, unknown, occupied

    def compute_frontier_mask(self, free: np.ndarray, unknown: np.ndarray, occupied_inflated: np.ndarray) -> np.ndarray:
        near_unknown = self.neighbor_any(unknown)
        return free & near_unknown & (~occupied_inflated)

    def neighbor_any(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        padded = np.pad(mask, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        out = np.zeros((h, w), dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                out |= padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        return out

    def inflate_mask(self, mask: np.ndarray, info: GridInfo, radius_m: float) -> np.ndarray:
        if radius_m <= 0.0:
            return mask.astype(bool)
        cells = int(math.ceil(radius_m / info.resolution))
        if cells <= 0:
            return mask.astype(bool)

        inflated = mask.copy().astype(bool)
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return inflated

        r = cells
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue
                nx = xs + dx
                ny = ys + dy
                ok = (nx >= 0) & (nx < info.width) & (ny >= 0) & (ny < info.height)
                inflated[ny[ok], nx[ok]] = True
        return inflated

    def grid_to_world_center(self, ix: int, iy: int, info: GridInfo) -> Tuple[float, float]:
        x = info.origin_x + (float(ix) + 0.5) * info.resolution
        y = info.origin_y + (float(iy) + 0.5) * info.resolution
        return x, y

    def world_to_grid(self, x: float, y: float, info: GridInfo) -> Optional[Tuple[int, int]]:
        ix = int(math.floor((x - info.origin_x) / info.resolution))
        iy = int(math.floor((y - info.origin_y) / info.resolution))
        if ix < 0 or ix >= info.width or iy < 0 or iy >= info.height:
            return None
        return ix, iy

    def edge_penalty(self, x: float, y: float, info: GridInfo) -> float:
        margin = self.args.edge_margin
        if margin <= 0.0:
            return 0.0
        x_min = info.origin_x
        x_max = info.origin_x + info.width * info.resolution
        y_min = info.origin_y
        y_max = info.origin_y + info.height * info.resolution
        d = min(x - x_min, x_max - x, y - y_min, y_max - y)
        if d >= margin:
            return 0.0
        return (margin - d) / margin

    # ------------------------------------------------------------------
    # Point/path helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Odom/local transform using /odom pose
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------

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
    except Exception as e:
        try:
            node.get_logger().error(f"Exception: {e}")
        except Exception:
            pass
        raise
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
