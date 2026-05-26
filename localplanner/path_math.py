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

class PathMathMixin:
    @staticmethod
    def curvature_cost(pts: np.ndarray) -> float:
        if len(pts) < 3:
            return 0.0
        yaws = pts[:, 2]
        diffs = []
        for i in range(1, len(yaws)):
            d = yaws[i] - yaws[i - 1]
            while d > math.pi:
                d -= 2.0 * math.pi
            while d < -math.pi:
                d += 2.0 * math.pi
            diffs.append(abs(d))
        return float(np.mean(diffs)) if diffs else 0.0

    @staticmethod
    def distance_and_progress_to_polyline(query_xy: np.ndarray, ref_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        seg = ref_xy[1:] - ref_xy[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])

        d_out = np.zeros((len(query_xy),), dtype=np.float32)
        s_out = np.zeros((len(query_xy),), dtype=np.float32)

        for qi, q in enumerate(query_xy):
            best_d = 1e9
            best_s = 0.0
            for i in range(len(seg)):
                a = ref_xy[i]
                v = seg[i]
                l2 = float(np.dot(v, v))
                if l2 < 1e-8:
                    t = 0.0
                    proj = a
                else:
                    t = float(np.clip(np.dot(q - a, v) / l2, 0.0, 1.0))
                    proj = a + t * v
                d = float(np.linalg.norm(q - proj))
                if d < best_d:
                    best_d = d
                    best_s = float(cum[i] + t * seg_len[i])
            d_out[qi] = best_d
            s_out[qi] = best_s
        return d_out, s_out

    @staticmethod
    def resample_polyline(xy: np.ndarray, step: float) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float32)
        if len(xy) < 2:
            return xy
        step = max(0.05, float(step))
        seg_lens = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        total = float(np.sum(seg_lens))
        if total < 1e-6:
            return xy[:1]
        targets = np.arange(0.0, total + 1e-6, step, dtype=np.float32)
        if targets[-1] < total:
            targets = np.append(targets, total)

        cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
        out = []
        j = 0
        for t in targets:
            while j + 1 < len(cum) and cum[j + 1] < t:
                j += 1
            if j + 1 >= len(cum):
                out.append(xy[-1])
                continue
            denom = max(float(cum[j + 1] - cum[j]), 1e-6)
            ratio = float((t - cum[j]) / denom)
            out.append(xy[j] + ratio * (xy[j + 1] - xy[j]))
        return np.asarray(out, dtype=np.float32)

    @staticmethod
    def point_on_polyline(xy: np.ndarray, dist: float) -> np.ndarray:
        if len(xy) == 0:
            return np.asarray([0.0, 0.0], dtype=np.float32)
        if len(xy) == 1 or dist <= 0.0:
            return xy[0]
        seg_lens = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        accum = 0.0
        for i, seg in enumerate(seg_lens):
            if accum + seg >= dist:
                ratio = (dist - accum) / max(float(seg), 1e-6)
                return xy[i] + ratio * (xy[i + 1] - xy[i])
            accum += float(seg)
        return xy[-1]

    @staticmethod
    def smooth_polyline(xy: np.ndarray, passes: int) -> np.ndarray:
        if len(xy) < 4 or passes <= 0:
            return xy
        out = xy.copy()
        for _ in range(int(passes)):
            new = out.copy()
            new[1:-1] = 0.25 * out[:-2] + 0.50 * out[1:-1] + 0.25 * out[2:]
            out = new
        return out

    @staticmethod
    def polyline_yaws(xy: np.ndarray) -> np.ndarray:
        yaws = np.zeros((len(xy),), dtype=np.float32)
        if len(xy) < 2:
            return yaws
        for i in range(len(xy)):
            if i == 0:
                d = xy[1] - xy[0]
            elif i == len(xy) - 1:
                d = xy[-1] - xy[-2]
            else:
                d = xy[i + 1] - xy[i - 1]
            yaws[i] = math.atan2(float(d[1]), float(d[0]))
        return yaws

