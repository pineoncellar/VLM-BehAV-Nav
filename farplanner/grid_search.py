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

class FarGridSearchMixin:
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

