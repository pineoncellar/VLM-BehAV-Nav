#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
far_waypoint_planner.py

Step 2b version: depth-camera-friendly FAR-like waypoint manager.

This is NOT the original FAR Planner visibility-graph algorithm.
It is designed for a single depth camera + local OccupancyGrid.

Core idea:
  - Keep the old planner's practical behavior: choose a waypoint that makes
    progress toward the fixed far goal.
  - Add a safer reachable-region layer, but do NOT require robot cell itself
    to be known free. With a depth camera, the area around x=0 is often unknown.
  - If robot cell is not traversable, search for the nearest forward free seed.
  - Flood-fill from that seed.
  - Score reachable candidate cells.
  - Optionally A* from seed to the chosen candidate.
  - Publish a lookahead point on that local path as /current_waypoint_local.

Inputs:
  /local_traversability_grid      nav_msgs/msg/OccupancyGrid
  /odom                           nav_msgs/msg/Odometry
  /local_planner_ok               std_msgs/msg/Bool, optional

Outputs:
  /far_goal                       geometry_msgs/msg/PoseStamped, odom frame
  /current_waypoint               geometry_msgs/msg/PoseStamped, odom frame, locked
  /current_waypoint_local         geometry_msgs/msg/PoseStamped, local grid frame
  /future_waypoint                geometry_msgs/msg/PoseStamped, odom frame, changeable
  /far_waypoints                  nav_msgs/msg/Path, odom frame, debug line
  /far_local_plan                 nav_msgs/msg/Path, local frame, debug A*/reachable path
  /far_waypoint_markers           visualization_msgs/msg/MarkerArray
  /far_goal_reached               std_msgs/msg/Bool

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

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool


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
        self.goal_reached: bool = False

        self.last_update_wall_time = 0.0
        self.update_count = 0

        self.local_fail_count = 0
        self.last_local_plan_points: List[Point2] = []
        self.last_reachable_count = 0
        self.last_seed_cell: Optional[Tuple[int, int]] = None

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

        self.far_goal_pub = self.create_publisher(PoseStamped, args.far_goal_topic, 10)
        self.current_wp_pub = self.create_publisher(PoseStamped, args.current_waypoint_topic, 10)
        self.current_wp_local_pub = self.create_publisher(PoseStamped, args.current_waypoint_local_topic, 10)
        self.future_wp_pub = self.create_publisher(PoseStamped, args.future_waypoint_topic, 10)
        self.waypoints_pub = self.create_publisher(Path, args.waypoints_topic, 10)
        self.local_plan_pub = self.create_publisher(Path, args.local_plan_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, args.waypoint_markers_topic, 10)
        self.goal_reached_pub = self.create_publisher(Bool, args.far_goal_reached_topic, 10)

        self.get_logger().info("far_waypoint_planner.py Step2b started")
        self.get_logger().info(f"grid_topic       : {args.grid_topic}")
        self.get_logger().info(f"odom_topic       : {args.odom_topic}")
        self.get_logger().info(f"local_ok_topic   : {args.local_planner_ok_topic}")
        self.get_logger().info(f"far goal polar   : angle={args.goal_angle_deg:.2f} deg, distance={args.goal_distance:.2f} m")
        self.get_logger().info(f"arrival radius   : {args.arrival_radius:.2f} m")
        self.get_logger().info(f"final radius     : {args.final_goal_radius:.2f} m")
        self.get_logger().info(f"lookahead        : {args.lookahead_distance:.2f} m")
        self.get_logger().info("strategy         : forward free seed + reachable flood fill + A* lookahead")
        self.get_logger().info("vehicle control  : disabled")

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

        self.update_waypoints(
            free=free,
            frontier=frontier,
            occupied_inflated=occupied_inflated,
            traversable=traversable,
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
        info: GridInfo,
    ) -> None:
        self.last_local_plan_points = []
        self.last_reachable_count = 0
        self.last_seed_cell = None

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

        # If local planner repeatedly fails, release current locked waypoint.
        if self.current_wp_odom is not None and self.local_fail_count >= self.args.max_local_fail_count:
            self.get_logger().warn(
                f"Local planner failed {self.local_fail_count} times. Releasing current waypoint."
            )
            self.current_wp_odom = None
            self.future_wp_odom = None
            self.local_fail_count = 0

        # Current waypoint is locked until arrival, but only if it still makes sense.
        if self.current_wp_odom is not None:
            d = self.distance_robot_to_odom_point(self.current_wp_odom)
            current_local = self.odom_to_local(self.current_wp_odom)

            should_release = False
            if d < self.args.arrival_radius:
                self.get_logger().info(f"Arrived current waypoint: distance={d:.2f}m")
                should_release = True
            elif current_local is None:
                should_release = True
            else:
                # If the locked waypoint drifted out of the trusted local window, reselect.
                local_dist = math.hypot(current_local.x, current_local.y)
                if local_dist > self.args.waypoint_release_distance:
                    should_release = True

            if should_release:
                self.current_wp_odom = None
                self.future_wp_odom = None

        # Select current waypoint only when no locked waypoint exists.
        if self.current_wp_odom is None:
            current_local = self.select_reachable_lookahead_waypoint(
                traversable=traversable,
                frontier=frontier,
                info=info,
                future_mode=False,
            )
            if current_local is not None:
                self.current_wp_odom = self.local_to_odom(current_local)
                self.get_logger().info(
                    f"New current waypoint LOCKED: local=({current_local.x:.2f},{current_local.y:.2f}), "
                    f"odom=({self.current_wp_odom.x:.2f},{self.current_wp_odom.y:.2f})"
                )
            else:
                # Important fallback:
                # If the reachable layer fails, optionally use the old goal-closest method
                # so the robot does not die immediately due to near-field unknown cells.
                if self.args.enable_old_style_fallback:
                    fallback_local = self.select_goal_closest_feasible_waypoint_old(
                        free=free,
                        frontier=frontier,
                        occupied_inflated=occupied_inflated,
                        info=info,
                        future_mode=False,
                    )
                    if fallback_local is not None:
                        self.current_wp_odom = self.local_to_odom(fallback_local)
                        self.last_local_plan_points = [Point2(0.0, 0.0), fallback_local]
                        self.get_logger().warn(
                            f"Reachable selection failed. Using old-style fallback waypoint: "
                            f"local=({fallback_local.x:.2f},{fallback_local.y:.2f})"
                        )
                    else:
                        self.get_logger().warn("No feasible current waypoint candidate found.")
                else:
                    self.get_logger().warn("No reachable current waypoint candidate found.")

        # Future waypoint is allowed to change every cycle.
        future_local = self.select_reachable_lookahead_waypoint(
            traversable=traversable,
            frontier=frontier,
            info=info,
            future_mode=True,
        )
        if future_local is not None:
            self.future_wp_odom = self.local_to_odom(future_local)

    def select_reachable_lookahead_waypoint(
        self,
        traversable: np.ndarray,
        frontier: np.ndarray,
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

        # Try top candidates until A* succeeds.
        max_try = min(len(candidates), self.args.max_astar_candidate_tries)
        for _score, goal_cell in candidates[:max_try]:
            path_cells = self.astar(traversable, start_cell, goal_cell)
            if not path_cells:
                continue

            path_points = [Point2(0.0, 0.0)]
            path_points.extend([Point2(*self.grid_to_world_center(ix, iy, info)) for ix, iy in path_cells])

            # Remove tiny duplicate between (0,0) and seed if any.
            path_points = self.compact_points(path_points, min_gap=0.05)
            if len(path_points) < 2:
                continue

            self.last_local_plan_points = path_points
            lookahead = self.args.future_lookahead_distance if future_mode else self.args.lookahead_distance
            return self.point_at_distance(path_points, lookahead)

        if not future_mode:
            self.get_logger().warn("A* failed for all reachable candidates.")
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
        free = grid == 0
        unknown = grid < 0
        occupied = grid >= self.args.occupied_threshold
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

        if self.far_goal_odom is not None:
            self.far_goal_pub.publish(self.make_pose(self.far_goal_odom.x, self.far_goal_odom.y, 0.0, odom_frame, stamp))

        if self.current_wp_odom is not None:
            self.current_wp_pub.publish(self.make_pose(self.current_wp_odom.x, self.current_wp_odom.y, 0.0, odom_frame, stamp))
            current_local = self.odom_to_local(self.current_wp_odom)
            if current_local is not None:
                yaw = math.atan2(current_local.y, current_local.x)
                self.current_wp_local_pub.publish(self.make_pose(current_local.x, current_local.y, yaw, info.frame_id, stamp))

        if self.future_wp_odom is not None:
            self.future_wp_pub.publish(self.make_pose(self.future_wp_odom.x, self.future_wp_odom.y, 0.0, odom_frame, stamp))

        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = odom_frame
        if self.current_wp_odom is not None:
            path.poses.append(self.make_pose(self.current_wp_odom.x, self.current_wp_odom.y, 0.0, odom_frame, stamp))
        if self.future_wp_odom is not None:
            path.poses.append(self.make_pose(self.future_wp_odom.x, self.future_wp_odom.y, 0.0, odom_frame, stamp))
        if self.far_goal_odom is not None:
            path.poses.append(self.make_pose(self.far_goal_odom.x, self.far_goal_odom.y, 0.0, odom_frame, stamp))
        self.waypoints_pub.publish(path)

        self.publish_local_plan(stamp, info.frame_id)
        self.publish_waypoint_markers(stamp, odom_frame)

        reached_msg = Bool()
        reached_msg.data = bool(self.goal_reached)
        self.goal_reached_pub.publish(reached_msg)

    def publish_local_plan(self, stamp, frame_id: str) -> None:
        msg = Path()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        for p in self.last_local_plan_points:
            yaw = math.atan2(p.y, p.x) if abs(p.x) + abs(p.y) > 1e-6 else 0.0
            msg.poses.append(self.make_pose(p.x, p.y, yaw, frame_id, stamp))
        self.local_plan_pub.publish(msg)

    def publish_waypoint_markers(self, stamp, frame_id: str) -> None:
        msg = MarkerArray()

        clear = Marker()
        clear.action = Marker.DELETEALL
        msg.markers.append(clear)

        items = []
        if self.far_goal_odom is not None:
            items.append(("far_goal", self.far_goal_odom, 0, 0.35))
        if self.current_wp_odom is not None:
            items.append(("current_waypoint", self.current_wp_odom, 1, 0.45))
        if self.future_wp_odom is not None:
            items.append(("future_waypoint", self.future_wp_odom, 2, 0.30))

        for name, p, marker_id, scale in items:
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = frame_id
            m.ns = "far_waypoints"
            m.id = marker_id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(p.x)
            m.pose.position.y = float(p.y)
            m.pose.position.z = 0.15
            m.pose.orientation.w = 1.0
            m.scale.x = scale
            m.scale.y = scale
            m.scale.z = scale

            if name == "current_waypoint":
                m.color.r = 1.0
                m.color.g = 0.2
                m.color.b = 0.2
            elif name == "future_waypoint":
                m.color.r = 0.2
                m.color.g = 0.8
                m.color.b = 1.0
            else:
                m.color.r = 1.0
                m.color.g = 1.0
                m.color.b = 0.1

            m.color.a = 1.0
            msg.markers.append(m)

        self.marker_pub.publish(msg)

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
            f"reachable={self.last_reachable_count}, seed={seed_text}, local_fail={self.local_fail_count}"
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


def parse_args():
    parser = argparse.ArgumentParser(description="Depth-camera-friendly reachable waypoint planner. No vehicle control.")

    parser.add_argument("--grid-topic", default="/local_traversability_grid")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--local-planner-ok-topic", default="/local_planner_ok")

    parser.add_argument("--far-goal-topic", default="/far_goal")
    parser.add_argument("--current-waypoint-topic", default="/current_waypoint")
    parser.add_argument("--current-waypoint-local-topic", default="/current_waypoint_local")
    parser.add_argument("--future-waypoint-topic", default="/future_waypoint")
    parser.add_argument("--waypoints-topic", default="/far_waypoints")
    parser.add_argument("--local-plan-topic", default="/far_local_plan")
    parser.add_argument("--waypoint-markers-topic", default="/far_waypoint_markers")
    parser.add_argument("--far-goal-reached-topic", default="/far_goal_reached")

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

    # Forward seed search. This is the important Step2b part.
    parser.add_argument("--seed-min-x", type=float, default=0.4)
    parser.add_argument("--seed-max-x", type=float, default=2.6)
    parser.add_argument("--seed-y-limit", type=float, default=1.4)
    parser.add_argument("--seed-lateral-weight", type=float, default=2.0)

    # Occupancy semantics.
    parser.add_argument("--occupied-threshold", type=int, default=50)
    parser.add_argument("--occupied-inflation-radius", type=float, default=0.35)

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
