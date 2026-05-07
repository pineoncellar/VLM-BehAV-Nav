#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clipseg_semantic_fuser.py

普通 Python 工具模块，不创建 ROS2 Node，不发布话题。
由 depth_grid_standalone.py 调用。

功能：
  1. 调用本地 CLIPSeg 模型进行语义分割，推理使用 CUDA/GPU。
  2. 保存最新一帧 image-space semantic cost map。
  3. 使用 depth 像素反投影到 base_footprint / target_frame。
  4. 将语义 cost 聚合到 40x40 grid。
  5. 与 depth 几何 OccupancyGrid 融合。

输出仍然由 depth_grid_standalone.py 发布到：
  /local_traversability_grid

注意：
  - 不使用 cv_bridge / cv2。
  - RGB 与 depth 需要尽量对齐；如果尺寸不同，本模块只做简单 resize，不做相机外参配准。
  - 几何 occupied 永远优先级最高，语义不能把几何障碍改成 free。
"""

import os
import time
from typing import Dict, List, Optional, Tuple

# 强制本地模型，避免联网。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from PIL import Image as PILImage

import torch
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation


UNKNOWN = -1
FREE = 0
OCCUPIED = 100


class ClipsegSemanticFuser:
    def __init__(
        self,
        model_dir: str,
        prompts: List[str],
        behavior_rule: str = "avoid_grass",
        device: Optional[str] = None,
        input_width: int = 352,
        confidence_threshold: float = 0.25,
        use_amp: bool = True,
        semantic_cost_threshold: float = 35.0,
        semantic_block_value: int = 80,
    ):
        self.model_dir = model_dir
        self.prompts = list(prompts)
        self.behavior_rule = behavior_rule
        self.input_width = int(input_width)
        self.confidence_threshold = float(confidence_threshold)
        self.use_amp = bool(use_amp)
        self.semantic_cost_threshold = float(semantic_cost_threshold)
        self.semantic_block_value = int(semantic_block_value)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(f"Local CLIPSeg model directory not found: {self.model_dir}")

        self.processor = CLIPSegProcessor.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            self.model_dir,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()

        self.prompt_costs = self.build_prompt_costs(self.prompts, self.behavior_rule)

        self.latest_cost_map: Optional[np.ndarray] = None  # uint8, HxW, 0~100
        self.latest_debug_rgb: Optional[np.ndarray] = None  # uint8, HxWx3, RGB overlay for ROS debug image
        self.latest_stamp_sec: float = 0.0
        self.latest_wall_time: float = 0.0
        self.latest_shape: Optional[Tuple[int, int]] = None
        self.inference_count: int = 0
        self.last_inference_time: float = 0.0

    # ------------------------------------------------------------------
    # CLIPSeg inference
    # ------------------------------------------------------------------

    @staticmethod
    def build_prompt_costs(prompts: List[str], behavior_rule: str) -> Dict[str, float]:
        """Return occupancy-like semantic cost for each prompt.

        0   = allowed
        30  = allowed but not preferred
        80  = behavior forbidden
        100 = strong stop / blocked

        这里不是固定语义通行性，而是行为规则决定代价。
        """
        rule = behavior_rule.lower().strip()
        costs: Dict[str, float] = {}

        for p in prompts:
            name = p.lower().strip()

            is_grass = ("grass" in name) or ("vegetation" in name) or ("lawn" in name)
            is_pavement = ("pavement" in name) or ("sidewalk" in name) or ("road" in name) or ("concrete" in name)
            is_stop = ("stop" in name) or ("person" in name) or ("gesture" in name)

            if is_stop:
                costs[p] = 100.0
                continue

            if rule == "avoid_grass":
                if is_grass:
                    costs[p] = 80.0
                elif is_pavement:
                    costs[p] = 0.0
                else:
                    costs[p] = 30.0

            elif rule == "allow_grass":
                if is_grass:
                    costs[p] = 0.0
                elif is_pavement:
                    costs[p] = 0.0
                else:
                    costs[p] = 30.0

            elif rule == "prefer_pavement":
                if is_pavement:
                    costs[p] = 0.0
                elif is_grass:
                    costs[p] = 80.0
                else:
                    costs[p] = 40.0

            else:
                # 默认保守：草/植被禁行，其他未知语义弱惩罚。
                if is_grass:
                    costs[p] = 80.0
                elif is_pavement:
                    costs[p] = 0.0
                else:
                    costs[p] = 30.0

        return costs

    def update_rgb_msg(self, msg) -> bool:
        """Decode a ROS sensor_msgs/Image RGB message and run CLIPSeg once.

        Returns True if inference was successfully updated.
        """
        rgb = self.decode_rgb_image_msg(msg)
        self.update_rgb_array(rgb)

        # stamp is optional for non-ROS unit tests
        try:
            self.latest_stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        except Exception:
            self.latest_stamp_sec = time.time()

        return True

    def update_rgb_array(self, rgb: np.ndarray) -> None:
        """Run CLIPSeg on an RGB uint8 array, HxWx3."""
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"RGB image must be HxWx3, got shape={rgb.shape}")

        h, w = rgb.shape[:2]
        pil_image = PILImage.fromarray(rgb, mode="RGB")

        # 预缩放降低 CPU 预处理和 GPU 输入负载。
        # 后面仍然把概率图 resize 回原始 RGB/depth 尺寸用于像素投影。
        if self.input_width > 0 and w > self.input_width:
            new_w = int(self.input_width)
            new_h = max(1, int(round(h * (new_w / float(w)))))
            pil_for_model = pil_image.resize((new_w, new_h), resample=PILImage.BILINEAR)
        else:
            pil_for_model = pil_image

        inputs = self.processor(
            text=self.prompts,
            images=[pil_for_model] * len(self.prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

        start = time.time()
        with torch.inference_mode():
            if self.use_amp and self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

            preds = torch.sigmoid(outputs.logits)
            preds_resized = F.interpolate(
                preds.unsqueeze(1),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        probs = preds_resized.float().cpu().numpy()
        self.latest_cost_map = self.semantic_probs_to_cost_map(probs)
        self.latest_debug_rgb = self.make_debug_overlay(rgb, self.latest_cost_map)
        self.latest_shape = (h, w)
        self.latest_wall_time = time.time()
        self.last_inference_time = self.latest_wall_time - start
        self.inference_count += 1

    def semantic_probs_to_cost_map(self, probs: np.ndarray) -> np.ndarray:
        """Convert CxHxW CLIPSeg probabilities into an occupancy-like semantic cost map.

        不使用原 BehAV 的覆盖式 combined_cost_map。
        这里对每个 prompt 单独计算 cost，最后取 max，避免 prompt 顺序污染结果。
        """
        if probs.ndim != 3:
            raise ValueError(f"Expected probs CxHxW, got shape={probs.shape}")

        c, h, w = probs.shape
        cost_map = np.zeros((h, w), dtype=np.float32)

        for i, prompt in enumerate(self.prompts):
            if i >= c:
                break
            prompt_cost = float(self.prompt_costs.get(prompt, 0.0))
            if prompt_cost <= 0.0:
                continue

            p = probs[i]
            p = np.where(p >= self.confidence_threshold, p, 0.0)
            cost_map = np.maximum(cost_map, p * prompt_cost)

        return np.clip(cost_map, 0, 100).astype(np.uint8)

    def get_latest_debug_rgb(self) -> Optional[np.ndarray]:
        """Return latest CLIPSeg inference visualization as RGB uint8 HxWx3.

        This is only for debugging / RViz display. It does not affect grid fusion.
        """
        if self.latest_debug_rgb is None:
            return None
        return self.latest_debug_rgb.copy()

    @staticmethod
    def make_debug_overlay(rgb: np.ndarray, cost_map: np.ndarray, alpha: float = 0.55) -> np.ndarray:
        """Create a lightweight red/yellow overlay for semantic cost.

        Avoids cv2 to reduce extra dependency/CPU overhead.
        cost_map uses 0~100. Higher cost becomes stronger red/yellow.
        """
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return None
        h, w = rgb.shape[:2]
        if cost_map.shape != (h, w):
            cost_img = PILImage.fromarray(cost_map.astype(np.uint8), mode="L")
            cost_map = np.asarray(cost_img.resize((w, h), resample=PILImage.BILINEAR), dtype=np.uint8)

        rgb_f = rgb.astype(np.float32)
        c = np.clip(cost_map.astype(np.float32) / 100.0, 0.0, 1.0)

        heat = np.zeros_like(rgb_f)
        heat[..., 0] = 255.0 * c
        heat[..., 1] = 180.0 * c
        heat[..., 2] = 20.0 * c

        mask = c > 0.01
        out = rgb_f.copy()
        out[mask] = (1.0 - alpha) * rgb_f[mask] + alpha * heat[mask]
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def decode_rgb_image_msg(msg) -> np.ndarray:
        """Decode sensor_msgs/Image to RGB uint8 HxWx3 without cv_bridge/cv2."""
        encoding = msg.encoding.lower()
        width = int(msg.width)
        height = int(msg.height)

        if encoding in ("rgb8", "bgr8"):
            channels = 3
        elif encoding in ("rgba8", "bgra8"):
            channels = 4
        elif encoding in ("mono8", "8uc1"):
            channels = 1
        else:
            raise ValueError(f"Unsupported RGB image encoding: {msg.encoding}")

        expected_row_bytes = width * channels
        expected_total_bytes = height * int(msg.step)
        if len(msg.data) != expected_total_bytes:
            raise ValueError(
                f"Image data length mismatch: got {len(msg.data)}, expected {expected_total_bytes}"
            )

        raw = np.frombuffer(msg.data, dtype=np.uint8)
        if int(msg.step) == expected_row_bytes:
            arr = raw.reshape(height, width, channels)
        else:
            rows = raw.reshape(height, int(msg.step))
            useful = rows[:, :expected_row_bytes].copy()
            arr = useful.reshape(height, width, channels)

        if encoding == "rgb8":
            rgb = arr
        elif encoding == "bgr8":
            rgb = arr[:, :, ::-1]
        elif encoding == "rgba8":
            rgb = arr[:, :, :3]
        elif encoding == "bgra8":
            rgb = arr[:, :, [2, 1, 0]]
        else:
            rgb = np.repeat(arr[:, :, :1], 3, axis=2)

        return np.ascontiguousarray(rgb)

    # ------------------------------------------------------------------
    # 2D semantic image -> 40x40 local grid fusion
    # ------------------------------------------------------------------

    def is_ready(self, max_age: float) -> bool:
        if self.latest_cost_map is None:
            return False
        if max_age <= 0.0:
            return True
        return (time.time() - self.latest_wall_time) <= max_age

    def fuse_grid(
        self,
        geometry_grid: np.ndarray,
        depth: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        x_factor: np.ndarray,
        y_factor: np.ndarray,
        *,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        resolution: float,
        min_depth: float,
        max_depth: float,
        ground_z: float,
        grid_width: int,
        grid_height: int,
        depth_downsample: int,
        semantic_downsample: int,
        min_semantic_points_per_cell: int,
        semantic_ground_min: float,
        semantic_ground_max: float,
        semantic_max_age: float,
        override_unknown: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Project semantic image cost to local grid and fuse with geometry grid."""
        debug = {
            "semantic_ready": 0,
            "semantic_age": -1.0,
            "semantic_points": 0,
            "semantic_cells": 0,
            "semantic_blocked_cells": 0,
            "inference_count": float(self.inference_count),
            "last_inference_time": float(self.last_inference_time),
        }

        if not self.is_ready(semantic_max_age):
            return geometry_grid, debug

        assert self.latest_cost_map is not None

        cost_map = self.latest_cost_map
        if cost_map.shape != depth.shape:
            cost_map = self.resize_cost_map(cost_map, depth.shape)

        debug["semantic_ready"] = 1
        debug["semantic_age"] = float(time.time() - self.latest_wall_time)

        # 语义投影可比几何略低频/低密度，取二者较大 downsample。
        ds = max(1, int(depth_downsample), int(semantic_downsample))

        depth_ds = depth[::ds, ::ds]
        xf = x_factor[::ds, ::ds]
        yf = y_factor[::ds, ::ds]
        cost_ds = cost_map[::ds, ::ds].astype(np.float32)

        valid = (
            np.isfinite(depth_ds)
            & (depth_ds > min_depth)
            & (depth_ds < max_depth)
            & (cost_ds > 0)
        )

        if not np.any(valid):
            return geometry_grid, debug

        z = depth_ds[valid]
        Xc = xf[valid] * z
        Yc = yf[valid] * z
        Zc = z

        pts_cam = np.stack((Xc, Yc, Zc), axis=0).astype(np.float32)
        pts_base = R @ pts_cam + t

        xb = pts_base[0]
        yb = pts_base[1]
        zb = pts_base[2]
        costs = cost_ds[valid]

        # 只使用地面附近的语义点，避免树、人、墙上的语义污染地面通行网格。
        z_rel = zb - float(ground_z)

        in_area_ground = (
            (xb >= x_min)
            & (xb < x_max)
            & (yb >= y_min)
            & (yb < y_max)
            & (z_rel >= semantic_ground_min)
            & (z_rel <= semantic_ground_max)
        )

        if not np.any(in_area_ground):
            return geometry_grid, debug

        xb = xb[in_area_ground]
        yb = yb[in_area_ground]
        costs = costs[in_area_ground]

        ix = np.floor((xb - x_min) / resolution).astype(np.int32)
        iy = np.floor((yb - y_min) / resolution).astype(np.int32)
        ix = np.clip(ix, 0, grid_width - 1)
        iy = np.clip(iy, 0, grid_height - 1)
        cell_id = iy * grid_width + ix
        num_cells = grid_width * grid_height

        point_count = np.bincount(cell_id, minlength=num_cells)
        cost_sum = np.bincount(cell_id, weights=costs, minlength=num_cells)

        mean_cost = np.zeros(num_cells, dtype=np.float32)
        nonzero = point_count > 0
        mean_cost[nonzero] = cost_sum[nonzero] / point_count[nonzero]

        semantic_observed = point_count >= int(min_semantic_points_per_cell)
        semantic_blocked = semantic_observed & (mean_cost >= self.semantic_cost_threshold)

        fused_flat = geometry_grid.reshape(-1).copy()

        # 几何障碍优先，语义只能把 free 变成 forbidden；默认不填 unknown。
        if override_unknown:
            allowed_base = fused_flat != OCCUPIED
        else:
            allowed_base = fused_flat == FREE

        apply_mask = semantic_blocked & allowed_base
        fused_flat[apply_mask] = np.int8(self.semantic_block_value)

        debug["semantic_points"] = int(np.count_nonzero(in_area_ground))
        debug["semantic_cells"] = int(np.count_nonzero(semantic_observed))
        debug["semantic_blocked_cells"] = int(np.count_nonzero(apply_mask))

        return fused_flat.reshape(grid_height, grid_width), debug

    @staticmethod
    def resize_cost_map(cost_map: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        h, w = target_shape
        img = PILImage.fromarray(cost_map.astype(np.uint8), mode="L")
        img = img.resize((w, h), resample=PILImage.BILINEAR)
        return np.asarray(img, dtype=np.uint8)

