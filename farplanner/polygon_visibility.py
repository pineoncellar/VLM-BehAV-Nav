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

class FarPolygonVisibilityMixin:
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

