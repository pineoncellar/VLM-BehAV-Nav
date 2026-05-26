#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
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
    from nav_msgs.msg import OccupancyGrid, Path
    from std_msgs.msg import Bool, String
except Exception:
    # Allows syntax checking outside a ROS2 environment.
    pass

try:
    from .local_types import GridInfo, Rollout, PathEval, CandidatePath, ArbiterEval
except ImportError:
    from local_types import GridInfo, Rollout, PathEval, CandidatePath, ArbiterEval

class SafetyMixin:
    def check_candidate_safety(self, pts: np.ndarray, grid: np.ndarray, info: GridInfo) -> Dict[str, float]:
        occupied_hits = 0
        near_unknown_hits = 0
        far_unknown_hits = 0
        out_hits = 0
        footprint_occupied_hits = 0
        footprint_near_unknown_hits = 0
        footprint_far_unknown_hits = 0
        footprint_out_hits = 0

        sampled = pts[::max(1, self.args.centerline_stride)]
        for p in sampled:
            x = float(p[0])
            y = float(p[1])
            dist = math.hypot(x, y)
            cell = self.world_to_grid(x, y, info)
            if cell is None:
                out_hits += 1
                continue
            ix, iy = cell
            val = int(grid[iy, ix])
            if val >= self.args.occupied_threshold:
                occupied_hits += 1
            elif val < 0:
                if dist < self.args.near_unknown_distance:
                    near_unknown_hits += 1
                else:
                    far_unknown_hits += 1

        for p in pts[::max(1, self.args.footprint_check_stride)]:
            x = float(p[0])
            y = float(p[1])
            yaw = float(p[2])
            dist = math.hypot(x, y)
            c = math.cos(yaw)
            s = math.sin(yaw)

            for ox, oy in self.footprint_offsets:
                wx = x + c * float(ox) - s * float(oy)
                wy = y + s * float(ox) + c * float(oy)

                if self.args.allow_behind_origin and wx < info.origin_x:
                    continue

                cell = self.world_to_grid(wx, wy, info)
                if cell is None:
                    footprint_out_hits += 1
                    continue

                ix, iy = cell
                val = int(grid[iy, ix])
                if val >= self.args.occupied_threshold:
                    footprint_occupied_hits += 1
                elif val < 0:
                    if dist < self.args.near_unknown_distance:
                        footprint_near_unknown_hits += 1
                    else:
                        footprint_far_unknown_hits += 1

        safe = True
        reason = "safe"
        if occupied_hits > self.args.max_occupied_hits:
            safe = False
            reason = "occupied"
        elif near_unknown_hits > self.args.max_near_unknown_hits:
            safe = False
            reason = "near_unknown"
        elif out_hits > self.args.max_out_of_map_hits:
            safe = False
            reason = "out_of_map"
        elif footprint_occupied_hits > self.args.max_footprint_occupied_hits:
            safe = False
            reason = "footprint_occupied"
        elif footprint_near_unknown_hits > self.args.max_footprint_near_unknown_hits:
            safe = False
            reason = "footprint_near_unknown"

        clearance_cost = (
            self.args.occupied_weight * occupied_hits
            + self.args.occupied_footprint_penalty * footprint_occupied_hits
            + 0.3 * footprint_out_hits
        )
        unknown_cost = (
            self.args.unknown_near_weight * near_unknown_hits
            + self.args.unknown_far_weight * far_unknown_hits
            + self.args.unknown_near_footprint_penalty * footprint_near_unknown_hits
            + self.args.unknown_far_footprint_penalty * footprint_far_unknown_hits
        )

        return {
            "safe": safe,
            "reason": reason,
            "occupied_hits": int(occupied_hits),
            "near_unknown_hits": int(near_unknown_hits),
            "footprint_occupied_hits": int(footprint_occupied_hits),
            "footprint_near_unknown_hits": int(footprint_near_unknown_hits),
            "clearance_cost": float(clearance_cost),
            "unknown_cost": float(unknown_cost),
        }

    def score_centerline(
        self,
        rollout: Rollout,
        grid: np.ndarray,
        info: GridInfo,
        target_x: float,
        target_y: float,
    ) -> Tuple[bool, float, float, str]:
        occupied_hits = 0
        unknown_near = 0
        unknown_far = 0
        out_hits = 0
        free_hits = 0

        stride = max(1, self.args.centerline_stride)
        for x, y, _yaw in rollout.points[::stride]:
            x = float(x)
            y = float(y)
            cell = self.world_to_grid(x, y, info)
            if cell is None:
                out_hits += 1
                continue

            ix, iy = cell
            val = int(grid[iy, ix])
            dist = math.hypot(x, y)

            if val >= self.args.occupied_threshold:
                occupied_hits += 1
            elif val < 0:
                if dist < self.args.near_unknown_distance:
                    unknown_near += 1
                else:
                    unknown_far += 1
            else:
                free_hits += 1

        if occupied_hits > self.args.max_occupied_hits:
            return False, 1e18, 1e18, "occupied"
        if unknown_near > self.args.max_near_unknown_hits:
            return False, 1e18, 1e18, "near_unknown"
        if out_hits > self.args.max_out_of_map_hits:
            return False, 1e18, 1e18, "out_of_map"

        end_x = float(rollout.points[-1, 0])
        end_y = float(rollout.points[-1, 1])
        endpoint_dist = math.hypot(end_x - target_x, end_y - target_y)

        penalty = 0.0
        penalty += self.args.occupied_weight * occupied_hits
        penalty += self.args.unknown_near_weight * unknown_near
        penalty += self.args.unknown_far_weight * unknown_far
        penalty += self.args.out_of_map_weight * out_hits
        penalty -= self.args.free_weight * free_hits

        return True, penalty, endpoint_dist, "ok"

    def footprint_collision_penalty(self, rollout: Rollout, grid: np.ndarray, info: GridInfo) -> Tuple[bool, float]:
        penalty = 0.0
        occupied_hits = 0
        near_unknown_hits = 0
        stride = max(1, self.args.footprint_check_stride)

        for p in rollout.points[::stride]:
            x = float(p[0])
            y = float(p[1])
            yaw = float(p[2])
            c = math.cos(yaw)
            s = math.sin(yaw)
            dist = math.hypot(x, y)

            for ox, oy in self.footprint_offsets:
                wx = x + c * float(ox) - s * float(oy)
                wy = y + s * float(ox) + c * float(oy)

                if self.args.allow_behind_origin and wx < info.origin_x:
                    continue

                cell = self.world_to_grid(wx, wy, info)
                if cell is None:
                    penalty += self.args.out_of_map_footprint_penalty
                    continue

                ix, iy = cell
                val = int(grid[iy, ix])
                if val >= self.args.occupied_threshold:
                    occupied_hits += 1
                    penalty += self.args.occupied_footprint_penalty
                elif val < 0:
                    if dist < self.args.near_unknown_distance:
                        near_unknown_hits += 1
                        penalty += self.args.unknown_near_footprint_penalty
                    else:
                        penalty += self.args.unknown_far_footprint_penalty

        if occupied_hits > self.args.max_footprint_occupied_hits:
            return False, 1e18
        if near_unknown_hits > self.args.max_footprint_near_unknown_hits:
            return False, 1e18
        return True, penalty

    def world_to_grid(self, x: float, y: float, info: GridInfo) -> Optional[Tuple[int, int]]:
        ix = int(math.floor((x - info.origin_x) / info.resolution))
        iy = int(math.floor((y - info.origin_y) / info.resolution))
        if ix < 0 or ix >= info.width or iy < 0 or iy >= info.height:
            return None
        return ix, iy

