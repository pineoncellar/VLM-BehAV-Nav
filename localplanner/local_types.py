#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

@dataclass
class GridInfo:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    frame_id: str

@dataclass
class Rollout:
    traj_id: int
    group_id: int
    variant_id: int
    family: str
    is_representative: bool
    length: float
    y_peak: float
    y_end: float
    points: np.ndarray

@dataclass
class PathEval:
    rollout: Rollout
    safe: bool
    path_score: float
    endpoint_dist: float
    footprint_penalty: float
    center_penalty: float
    reason: str

@dataclass
class CandidatePath:
    candidate_id: str
    source: str
    group_id: int
    rollout: Optional[Rollout]
    points: np.ndarray

@dataclass
class ArbiterEval:
    candidate: CandidatePath
    safe: bool
    reason: str
    mode_hint: str
    total_cost: float
    mean_ref_dist: float
    max_ref_dist: float
    endpoint_ref_dist: float
    queue_cost: float
    progress_s: float
    clearance_cost: float
    unknown_cost: float
    smooth_cost: float
    curvature_cost: float
    occupied_hits: int
    near_unknown_hits: int
    footprint_occupied_hits: int
    footprint_near_unknown_hits: int
