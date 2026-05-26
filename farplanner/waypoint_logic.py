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

class FarWaypointLogicMixin:
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

