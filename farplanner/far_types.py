#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import List

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
