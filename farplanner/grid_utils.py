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

class FarGridUtilsMixin:
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

