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

class RolloutLibraryMixin:
    def generate_rollout_library(self) -> List[Rollout]:
        """Generate candidate paths assigned into 7 fixed groups.

        Group IDs:
          0: hard right
          1: mid right
          2: slight right
          3: center
          4: slight left
          5: mid left
          6: hard left

        In base_footprint:
          +x forward
          +y left
          -y right
        """
        L = self.args.preview_distance

        # Representative lateral endpoint for each group.
        group_y = {
            0: -2.1,
            1: -1.4,
            2: -0.7,
            3:  0.0,
            4:  0.7,
            5:  1.4,
            6:  2.1,
        }

        rollouts: List[Rollout] = []
        traj_id = 0

        for gid in range(7):
            gy = group_y[gid]

            # 1) Representative direct path for the group.
            points = self.make_direct_rollout(L, gy)
            rollouts.append(Rollout(
                traj_id=traj_id,
                group_id=gid,
                variant_id=0,
                family="direct",
                is_representative=True,
                length=L,
                y_peak=gy,
                y_end=gy,
                points=points,
            ))
            traj_id += 1

            # 2) Direct variants around the group representative.
            # Center group gets fewer duplicates.
            direct_offsets = [] if gid == 3 else [-0.35, 0.35]
            for off in direct_offsets:
                y_end = self.clamp(gy + off, -self.args.max_lateral_offset, self.args.max_lateral_offset)
                points = self.make_direct_rollout(L, y_end)
                rollouts.append(Rollout(
                    traj_id=traj_id,
                    group_id=gid,
                    variant_id=traj_id,
                    family="direct",
                    is_representative=False,
                    length=L,
                    y_peak=y_end,
                    y_end=y_end,
                    points=points,
                ))
                traj_id += 1

            # 3) Recovery variants: avoid in the group's direction, then come back.
            # For center group, include mild left/right recoveries.
            if gid == 3:
                recover_peaks = [-1.0, 1.0]
            else:
                recover_peaks = [gy, self.clamp(gy * 1.15, -self.args.max_lateral_offset, self.args.max_lateral_offset)]

            for peak in recover_peaks:
                y_end = self.args.recovery_end_ratio * peak
                points = self.make_recovery_rollout(L, peak, y_end)
                rollouts.append(Rollout(
                    traj_id=traj_id,
                    group_id=gid,
                    variant_id=traj_id,
                    family="recovery",
                    is_representative=False,
                    length=L,
                    y_peak=peak,
                    y_end=y_end,
                    points=points,
                ))
                traj_id += 1

        return rollouts

    def make_direct_rollout(self, length: float, y_end: float) -> np.ndarray:
        step = max(0.10, self.args.trajectory_sample_step)
        n = max(2, int(math.ceil(length / step)) + 1)
        xs = np.linspace(0.0, length, n, dtype=np.float32)
        t = xs / max(length, 1e-6)

        s = 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3
        ds_dt = 30.0 * t**4 - 60.0 * t**3 + 30.0 * t**2
        ys = y_end * s
        dy_dx = y_end * ds_dt / max(length, 1e-6)
        yaws = np.arctan2(dy_dx, np.ones_like(dy_dx))

        return np.stack([xs, ys, yaws], axis=1).astype(np.float32)

    def make_recovery_rollout(self, length: float, y_peak: float, y_end: float) -> np.ndarray:
        step = max(0.10, self.args.trajectory_sample_step)
        n = max(2, int(math.ceil(length / step)) + 1)
        xs = np.linspace(0.0, length, n, dtype=np.float32)
        t = xs / max(length, 1e-6)

        smoother = 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3
        ds_dt = 30.0 * t**4 - 60.0 * t**3 + 30.0 * t**2

        bump = y_peak - 0.5 * y_end
        sin_term = np.sin(np.pi * t)
        bump_shape = sin_term * sin_term
        dbump_dt = np.pi * np.sin(2.0 * np.pi * t)

        ys = y_end * smoother + bump * bump_shape
        dy_dt = y_end * ds_dt + bump * dbump_dt
        dy_dx = dy_dt / max(length, 1e-6)
        yaws = np.arctan2(dy_dx, np.ones_like(dy_dx))

        return np.stack([xs, ys, yaws], axis=1).astype(np.float32)

    def make_footprint_offsets(self) -> np.ndarray:
        half = 0.5 * self.args.vehicle_size
        step = max(0.20, self.args.footprint_sample_step)
        coords = np.arange(-half, half + 1e-6, step, dtype=np.float32)
        offsets = []
        for x in coords:
            for y in coords:
                offsets.append((float(x), float(y)))

        offsets.extend([(-half, -half), (-half, half), (half, -half), (half, half), (0.0, 0.0)])
        return np.asarray(offsets, dtype=np.float32)

