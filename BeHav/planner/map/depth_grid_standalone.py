#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
depth_grid_standalone.py

Standalone ROS2 Python script:
  Subscribe depth image + camera_info + TF
  Build a 4m x 4m local traversability grid, 40 x 40 = 1600 cells
  Publish nav_msgs/msg/OccupancyGrid

Important:
  The grid construction is implemented here directly with NumPy.
  It does NOT call ROS2 costmap, depthimage_to_laserscan, image_geometry, cv_bridge,
  or any ROS2-provided grid/conversion builder.

Recommended for your current setup:
  depth_topic       = /camera_sensor/depth/image_raw
  camera_info_topic = /camera_sensor/depth/camera_info
  target_frame      = base_footprint
  source_frame      = camera_link_optical from depth header
  depth encoding    = 32FC1
  image size        = 800 x 600
  camera intrinsics = fx=428.9488, fy=428.9488, cx=400.5, cy=300.5
  sim time          = enabled by default
"""

import argparse
import math
import time
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String
import json
from tf2_ros import Buffer, TransformListener

try:
    from clipseg_semantic_fuser import ClipsegSemanticFuser
except Exception:
    ClipsegSemanticFuser = None


UNKNOWN = -1
FREE = 0
OCCUPIED = 100


def quat_to_rot_matrix_xyzw(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Return 3x3 rotation matrix from quaternion x,y,z,w."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float32)

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def decode_depth_to_meters(msg: Image) -> np.ndarray:
    """Decode sensor_msgs/Image depth data into HxW float32 meters.

    No OpenCV/cv_bridge is used.

    Supports:
      32FC1: float32 meters
      16UC1/MONO16: uint16 millimeters -> meters

    Fast path:
      If msg.step == width * bytes_per_pixel, decode directly from the message
      buffer with np.frombuffer. This avoids image-library conversion overhead.
    """
    encoding = msg.encoding.upper()

    if encoding == "32FC1":
        dtype = np.dtype(">f4") if msg.is_bigendian else np.dtype("<f4")
        bytes_per_pixel = 4
        scale = 1.0
    elif encoding in ("16UC1", "MONO16"):
        dtype = np.dtype(">u2") if msg.is_bigendian else np.dtype("<u2")
        bytes_per_pixel = 2
        scale = 0.001
    else:
        raise ValueError(f"Unsupported depth encoding: {msg.encoding}")

    expected_row_bytes = msg.width * bytes_per_pixel
    expected_total_bytes = msg.height * msg.step

    if len(msg.data) != expected_total_bytes:
        raise ValueError(
            f"Image data length mismatch: got {len(msg.data)}, "
            f"expected height*step={expected_total_bytes}"
        )

    # Your current depth image is 32FC1, width=800, step=3200,
    # so it hits this direct path: no OpenCV and no row-padding copy.
    if msg.step == expected_row_bytes:
        arr = np.frombuffer(msg.data, dtype=dtype, count=msg.width * msg.height)
        depth = arr.reshape(msg.height, msg.width)
    else:
        # Safe fallback for padded rows. This path does one compacting copy.
        raw_u8 = np.frombuffer(msg.data, dtype=np.uint8)
        rows = raw_u8.reshape(msg.height, msg.step)
        useful_rows = rows[:, :expected_row_bytes]
        depth = np.frombuffer(useful_rows.tobytes(), dtype=dtype).reshape(msg.height, msg.width)

    if scale == 1.0 and depth.dtype == np.float32:
        return depth
    return depth.astype(np.float32) * scale


class DepthGridStandalone(Node):
    def __init__(self, args):
        parameter_overrides = []
        if args.use_sim_time:
            parameter_overrides.append(Parameter("use_sim_time", Parameter.Type.BOOL, True))

        super().__init__("depth_grid_standalone", parameter_overrides=parameter_overrides)

        self.args = args

        self.grid_width = int(round((args.x_max - args.x_min) / args.resolution))
        self.grid_height = int(round((args.y_max - args.y_min) / args.resolution))
        self.num_cells = self.grid_width * self.grid_height

        if self.grid_width <= 0 or self.grid_height <= 0:
            raise ValueError("Invalid grid size. Check x/y range and resolution.")

        self.has_camera_info = False
        self.image_width: Optional[int] = None
        self.image_height: Optional[int] = None
        self.camera_frame: Optional[str] = None
        self.x_factor: Optional[np.ndarray] = None
        self.y_factor: Optional[np.ndarray] = None

        self.frame_count = 0
        self.done = False
        self.last_pub_time = 0.0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.semantic_fuser = None
        self.latest_rgb_msg = None
        self.semantic_timer = None
        if args.enable_clipseg:
            self.rule_sub = self.create_subscription(
                String,
                '/semantic_behavior_rule',
                self.behavior_rule_callback,
                1
            )

            if ClipsegSemanticFuser is None:
                raise RuntimeError("clipseg_semantic_fuser.py could not be imported")
            self.semantic_fuser = ClipsegSemanticFuser(
                model_dir=args.clipseg_model_dir,
                prompts=args.clipseg_prompts,
                behavior_rule=args.behavior_rule,
                input_width=args.clipseg_input_width,
                confidence_threshold=args.clipseg_confidence_threshold,
                use_amp=args.clipseg_use_amp,
                semantic_cost_threshold=args.semantic_cost_threshold,
                semantic_block_value=args.semantic_block_value,
            )
            if args.clipseg_hz > 0.0:
                self.semantic_timer = self.create_timer(1.0 / args.clipseg_hz, self.clipseg_timer_callback)

        self.grid_pub = self.create_publisher(OccupancyGrid, args.grid_topic, 10)
        self.clipseg_debug_image_pub = None

        if self.semantic_fuser is not None and args.publish_clipseg_debug_image:
            self.clipseg_debug_image_pub = self.create_publisher(Image, args.clipseg_debug_image_topic, 10)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            args.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )

        self.color_sub = None
        if args.enable_clipseg:
            self.color_sub = self.create_subscription(
                Image,
                args.rgb_topic,
                self.color_callback,
                qos_profile_sensor_data,
            )

        self.depth_sub = self.create_subscription(
            Image,
            args.depth_topic,
            self.depth_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info("depth_grid_standalone.py started")
        self.get_logger().info(f"use_sim_time       : {args.use_sim_time}")
        self.get_logger().info(f"depth_topic        : {args.depth_topic}")
        self.get_logger().info(f"camera_info_topic  : {args.camera_info_topic}")
        self.get_logger().info(f"grid_topic         : {args.grid_topic}")
        self.get_logger().info(f"enable_clipseg     : {args.enable_clipseg}")
        if args.enable_clipseg:
            self.get_logger().info(f"rgb_topic          : {args.rgb_topic}")
            self.get_logger().info(f"clipseg_model_dir  : {args.clipseg_model_dir}")
            self.get_logger().info(f"behavior_rule      : {args.behavior_rule}")
            self.get_logger().info(f"clipseg_debug_img  : {args.clipseg_debug_image_topic}")
        self.get_logger().info(f"target_frame       : {args.target_frame}")
        self.get_logger().info(
            f"grid              : {self.grid_width} x {self.grid_height} = {self.num_cells} cells, "
            f"resolution={args.resolution:.3f} m"
        )
        self.get_logger().info(
            f"grid region       : x=[{args.x_min:.2f},{args.x_max:.2f}), "
            f"y=[{args.y_min:.2f},{args.y_max:.2f})"
        )
        self.get_logger().info(
            f"depth filter      : {args.min_depth:.2f} m < depth < {args.max_depth:.2f} m"
        )


    def color_callback(self, msg: Image) -> None:
        """Store latest RGB image only. CLIPSeg runs in a low-rate timer."""
        if self.semantic_fuser is None:
            return
        self.latest_rgb_msg = msg


    def behavior_rule_callback(self, msg: String) -> None:
        try:
            data_dict = json.loads(msg.data)
            prompts = data_dict.get("prompts", [])
            rule = data_dict.get("rule", "")
            if self.semantic_fuser:
                self.semantic_fuser.prompts = prompts
                self.semantic_fuser.behavior_rule = rule
                self.get_logger().info(f"Dynamically updated semantics: {prompts} -> {rule}")
        except Exception as e:
            self.get_logger().error(f"Failed to parse /semantic_behavior_rule: {e}")

    def clipseg_timer_callback(self) -> None:
        """Run CLIPSeg at a lower fixed rate to avoid blocking every depth frame."""
        if self.semantic_fuser is None:
            return
        if self.latest_rgb_msg is None:
            return
        try:
            updated = self.semantic_fuser.update_rgb_msg(self.latest_rgb_msg)
            if updated and self.clipseg_debug_image_pub is not None:
                self.publish_clipseg_debug_image(self.latest_rgb_msg)

            if updated and self.frame_count % max(1, self.args.print_every) == 0:
                self.get_logger().info(
                    f"CLIPSeg updated: count={self.semantic_fuser.inference_count}, "
                    f"time={self.semantic_fuser.last_inference_time:.3f}s"
                )
        except Exception as e:
            self.get_logger().warn(f"CLIPSeg inference failed: {e}")

    def publish_clipseg_debug_image(self, rgb_msg: Image) -> None:
        """Publish latest CLIPSeg inference visualization as sensor_msgs/Image/rgb8."""
        if self.semantic_fuser is None or self.clipseg_debug_image_pub is None:
            return

        debug_rgb = self.semantic_fuser.get_latest_debug_rgb()
        if debug_rgb is None:
            return

        h, w = debug_rgb.shape[:2]
        out = Image()
        out.header.stamp = rgb_msg.header.stamp
        out.header.frame_id = rgb_msg.header.frame_id
        out.height = int(h)
        out.width = int(w)
        out.encoding = "rgb8"
        out.is_bigendian = 0
        out.step = int(w * 3)
        out.data = debug_rgb.tobytes()
        self.clipseg_debug_image_pub.publish(out)

    def camera_info_callback(self, msg: CameraInfo) -> None:
        if self.has_camera_info:
            return

        fx = float(msg.k[0])
        fy = float(msg.k[4])
        cx = float(msg.k[2])
        cy = float(msg.k[5])

        if fx <= 0.0 or fy <= 0.0:
            self.get_logger().error(f"Invalid CameraInfo: fx={fx}, fy={fy}")
            return

        width = int(msg.width)
        height = int(msg.height)

        u = np.arange(width, dtype=np.float32)
        v = np.arange(height, dtype=np.float32)
        U, V = np.meshgrid(u, v)

        # Pixel -> camera optical normalized ray factors.
        # For depth Z:
        #   Xc = x_factor * Z
        #   Yc = y_factor * Z
        #   Zc = Z
        self.x_factor = (U - cx) / fx
        self.y_factor = (V - cy) / fy

        self.image_width = width
        self.image_height = height
        self.camera_frame = msg.header.frame_id
        self.has_camera_info = True

        self.get_logger().info(
            f"CameraInfo received: {width}x{height}, frame={msg.header.frame_id}, "
            f"fx={fx:.6f}, fy={fy:.6f}, cx={cx:.3f}, cy={cy:.3f}"
        )

    def lookup_transform_matrix(self, source_frame: str, stamp_msg) -> Tuple[np.ndarray, np.ndarray, object]:
        """Lookup transform target_frame <- source_frame.

        Returns:
          R: 3x3 rotation matrix
          t: 3x1 translation
          tf_msg: original transform message
        """
        image_time = Time.from_msg(stamp_msg)

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.args.target_frame,
                source_frame,
                image_time,
                timeout=Duration(seconds=self.args.tf_timeout),
            )
        except Exception as e_stamp:
            if not self.args.allow_latest_tf:
                raise e_stamp

            if self.frame_count % max(1, self.args.print_every) == 0:
                self.get_logger().warn(
                    "TF lookup at image timestamp failed; trying latest TF. "
                    f"Reason: {e_stamp}"
                )

            tf_msg = self.tf_buffer.lookup_transform(
                self.args.target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=self.args.tf_timeout),
            )

        tr = tf_msg.transform.translation
        qr = tf_msg.transform.rotation

        R = quat_to_rot_matrix_xyzw(qr.x, qr.y, qr.z, qr.w)
        t = np.array([[tr.x], [tr.y], [tr.z]], dtype=np.float32)
        return R, t, tf_msg

    def depth_callback(self, msg: Image) -> None:
        if not self.has_camera_info:
            return

        if msg.width != self.image_width or msg.height != self.image_height:
            self.get_logger().warn(
                f"Depth size {msg.width}x{msg.height} does not match CameraInfo "
                f"{self.image_width}x{self.image_height}"
            )
            return

        # Optional rate limit.
        if self.args.publish_hz > 0.0:
            now = time.time()
            min_dt = 1.0 / self.args.publish_hz
            if now - self.last_pub_time < min_dt:
                return
            self.last_pub_time = now

        try:
            depth = decode_depth_to_meters(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to decode depth image: {e}")
            return

        source_frame = msg.header.frame_id or self.camera_frame
        if source_frame is None or source_frame == "":
            self.get_logger().error("Depth frame_id is empty and CameraInfo frame_id is unavailable.")
            return

        try:
            R, t, tf_msg = self.lookup_transform_matrix(source_frame, msg.header.stamp)
        except Exception as e:
            if self.frame_count % max(1, self.args.print_every) == 0:
                self.get_logger().warn(
                    f"Cannot lookup TF {self.args.target_frame} <- {source_frame}: {e}"
                )
            return

        occupancy, debug = self.build_grid(depth, R, t)

        if self.semantic_fuser is not None:
            try:
                occupancy, semantic_debug = self.semantic_fuser.fuse_grid(
                    occupancy,
                    depth,
                    R,
                    t,
                    self.x_factor,
                    self.y_factor,
                    x_min=self.args.x_min,
                    x_max=self.args.x_max,
                    y_min=self.args.y_min,
                    y_max=self.args.y_max,
                    resolution=self.args.resolution,
                    min_depth=self.args.min_depth,
                    max_depth=self.args.max_depth,
                    ground_z=self.args.ground_z,
                    grid_width=self.grid_width,
                    grid_height=self.grid_height,
                    depth_downsample=self.args.downsample,
                    semantic_downsample=self.args.semantic_downsample,
                    min_semantic_points_per_cell=self.args.min_semantic_points_per_cell,
                    semantic_ground_min=self.args.semantic_ground_min,
                    semantic_ground_max=self.args.semantic_ground_max,
                    semantic_max_age=self.args.semantic_max_age,
                    override_unknown=self.args.semantic_override_unknown,
                )
                debug["semantic"] = semantic_debug
            except Exception as e:
                if self.frame_count % max(1, self.args.print_every) == 0:
                    self.get_logger().warn(f"Semantic fusion failed, using geometry only: {e}")

        self.publish_grid(occupancy, msg.header.stamp)

        if self.frame_count % max(1, self.args.print_every) == 0:
            self.print_debug_report(depth, occupancy, debug, tf_msg, source_frame)

        self.frame_count += 1

        if self.args.once:
            self.done = True

    def build_grid(self, depth: np.ndarray, R: np.ndarray, t: np.ndarray):
        """Build 40x40 local traversability grid using direct NumPy implementation.

        Grid value:
          -1 unknown
           0 free
         100 occupied

        Rule:
          - Valid depth points are projected to target_frame.
          - Points inside x/y grid region mark cells as observed.
          - A point with z > ground_z + obstacle_height and z < ground_z + max_obstacle_height
            votes as obstacle.
          - Observed cells become FREE unless obstacle votes exceed threshold.
        """
        assert self.x_factor is not None
        assert self.y_factor is not None

        ds = max(1, int(self.args.downsample))

        depth_ds = depth[::ds, ::ds]
        xf = self.x_factor[::ds, ::ds]
        yf = self.y_factor[::ds, ::ds]

        valid = (
            np.isfinite(depth_ds)
            & (depth_ds > self.args.min_depth)
            & (depth_ds < self.args.max_depth)
        )

        occupancy_flat = np.full(self.num_cells, UNKNOWN, dtype=np.int8)

        debug = {
            "valid_points": int(np.count_nonzero(valid)),
            "in_area_points": 0,
            "observed_cells": 0,
            "occupied_cells": 0,
            "free_cells": 0,
            "unknown_cells": self.num_cells,
            "center_grid": None,
        }

        if not np.any(valid):
            return occupancy_flat.reshape(self.grid_height, self.grid_width), debug

        z = depth_ds[valid]
        Xc = xf[valid] * z
        Yc = yf[valid] * z
        Zc = z

        pts_cam = np.stack((Xc, Yc, Zc), axis=0).astype(np.float32)
        pts_base = R @ pts_cam + t

        xb = pts_base[0]
        yb = pts_base[1]
        zb = pts_base[2]

        in_area = (
            (xb >= self.args.x_min)
            & (xb < self.args.x_max)
            & (yb >= self.args.y_min)
            & (yb < self.args.y_max)
        )

        debug["in_area_points"] = int(np.count_nonzero(in_area))

        if not np.any(in_area):
            return occupancy_flat.reshape(self.grid_height, self.grid_width), debug

        xb = xb[in_area]
        yb = yb[in_area]
        zb = zb[in_area]

        ix = np.floor((xb - self.args.x_min) / self.args.resolution).astype(np.int32)
        iy = np.floor((yb - self.args.y_min) / self.args.resolution).astype(np.int32)

        # Because in_area already filters bounds, clipping should be unnecessary,
        # but it protects against numeric edge cases.
        ix = np.clip(ix, 0, self.grid_width - 1)
        iy = np.clip(iy, 0, self.grid_height - 1)

        cell_id = iy * self.grid_width + ix

        point_count = np.bincount(cell_id, minlength=self.num_cells)

        height_above_ground = zb - self.args.ground_z
        obstacle_point_mask = (
            (height_above_ground > self.args.obstacle_height)
            & (height_above_ground < self.args.max_obstacle_height)
        )

        obstacle_count = np.bincount(cell_id[obstacle_point_mask], minlength=self.num_cells)

        observed = point_count >= self.args.min_points_per_cell
        occupied = obstacle_count >= self.args.min_obstacle_points_per_cell

        occupancy_flat[observed] = FREE
        occupancy_flat[occupied] = OCCUPIED

        debug["observed_cells"] = int(np.count_nonzero(observed))
        debug["occupied_cells"] = int(np.count_nonzero(occupancy_flat == OCCUPIED))
        debug["free_cells"] = int(np.count_nonzero(occupancy_flat == FREE))
        debug["unknown_cells"] = int(np.count_nonzero(occupancy_flat == UNKNOWN))

        # Record where the center pixel would fall, useful for TF sanity check.
        h, w = depth.shape
        cx_pix = w // 2
        cy_pix = h // 2
        center_depth = self.median_valid_depth(depth, cx_pix, cy_pix, half=2)
        if np.isfinite(center_depth):
            Xcc = self.x_factor[cy_pix, cx_pix] * center_depth
            Ycc = self.y_factor[cy_pix, cx_pix] * center_depth
            Zcc = center_depth
            pt = R @ np.array([[Xcc], [Ycc], [Zcc]], dtype=np.float32) + t
            x0 = float(pt[0, 0])
            y0 = float(pt[1, 0])
            z0 = float(pt[2, 0])
            if self.args.x_min <= x0 < self.args.x_max and self.args.y_min <= y0 < self.args.y_max:
                ix0 = int(math.floor((x0 - self.args.x_min) / self.args.resolution))
                iy0 = int(math.floor((y0 - self.args.y_min) / self.args.resolution))
                debug["center_grid"] = (cx_pix, cy_pix, center_depth, x0, y0, z0, ix0, iy0)
            else:
                debug["center_grid"] = (cx_pix, cy_pix, center_depth, x0, y0, z0, None, None)

        return occupancy_flat.reshape(self.grid_height, self.grid_width), debug

    def publish_grid(self, occupancy: np.ndarray, stamp_msg) -> None:
        msg = OccupancyGrid()

        msg.header.stamp = stamp_msg
        msg.header.frame_id = self.args.target_frame

        msg.info.map_load_time = self.get_clock().now().to_msg()
        msg.info.resolution = float(self.args.resolution)
        msg.info.width = int(self.grid_width)
        msg.info.height = int(self.grid_height)

        # OccupancyGrid origin is the pose of cell (0,0) lower-left corner
        # in target_frame. Here: x=[0,4), y=[-2,2).
        msg.info.origin.position.x = float(self.args.x_min)
        msg.info.origin.position.y = float(self.args.y_min)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.x = 0.0
        msg.info.origin.orientation.y = 0.0
        msg.info.origin.orientation.z = 0.0
        msg.info.origin.orientation.w = 1.0

        # Row-major: index = iy * width + ix.
        msg.data = occupancy.reshape(-1).astype(np.int8).tolist()

        self.grid_pub.publish(msg)

    def median_valid_depth(self, depth: np.ndarray, x: int, y: int, half: int = 2) -> float:
        h, w = depth.shape
        x0 = max(0, x - half)
        x1 = min(w, x + half + 1)
        y0 = max(0, y - half)
        y1 = min(h, y + half + 1)
        roi = depth[y0:y1, x0:x1]
        valid = np.isfinite(roi) & (roi > self.args.min_depth) & (roi < self.args.max_depth)
        if not np.any(valid):
            return float("nan")
        return float(np.median(roi[valid]))

    def project_pixel_for_report(self, depth: np.ndarray, R: np.ndarray, t: np.ndarray, x: int, y: int):
        assert self.x_factor is not None
        assert self.y_factor is not None

        z = self.median_valid_depth(depth, x, y, half=2)
        if not np.isfinite(z):
            return None

        Xc = float(self.x_factor[y, x] * z)
        Yc = float(self.y_factor[y, x] * z)
        Zc = float(z)

        pt = R @ np.array([[Xc], [Yc], [Zc]], dtype=np.float32) + t
        xb = float(pt[0, 0])
        yb = float(pt[1, 0])
        zb = float(pt[2, 0])

        if self.args.x_min <= xb < self.args.x_max and self.args.y_min <= yb < self.args.y_max:
            ix = int(math.floor((xb - self.args.x_min) / self.args.resolution))
            iy = int(math.floor((yb - self.args.y_min) / self.args.resolution))
        else:
            ix = None
            iy = None

        return z, Xc, Yc, Zc, xb, yb, zb, ix, iy

    def print_debug_report(self, depth: np.ndarray, occupancy: np.ndarray, debug: dict, tf_msg, source_frame: str) -> None:
        tr = tf_msg.transform.translation
        qr = tf_msg.transform.rotation

        print("\n========== TF ==========")
        print(f"target_frame: {self.args.target_frame}")
        print(f"source_frame: {source_frame}")
        print(f"translation : x={tr.x:.4f}, y={tr.y:.4f}, z={tr.z:.4f}")
        print(f"quaternion  : x={qr.x:.4f}, y={qr.y:.4f}, z={qr.z:.4f}, w={qr.w:.4f}")

        print("\n========== GRID ==========")
        print(f"topic       : {self.args.grid_topic}")
        print(f"frame_id    : {self.args.target_frame}")
        print(f"size        : {self.grid_width} x {self.grid_height}")
        print(f"data length : {self.num_cells}")
        print(f"valid pts   : {debug['valid_points']}")
        print(f"in-area pts : {debug['in_area_points']}")
        print(f"free        : {debug['free_cells']}")
        print(f"occupied    : {debug['occupied_cells']}")
        print(f"unknown     : {debug['unknown_cells']}")
        print("values      : -1 unknown, 0 free, 80 semantic-blocked, 100 occupied")
        sem = debug.get("semantic")
        if sem is not None:
            print("\n========== SEMANTIC ==========")
            print(f"ready       : {sem.get('semantic_ready', 0)}")
            print(f"age         : {sem.get('semantic_age', -1):.3f}s")
            print(f"points      : {sem.get('semantic_points', 0)}")
            print(f"cells       : {sem.get('semantic_cells', 0)}")
            print(f"blocked     : {sem.get('semantic_blocked_cells', 0)}")
            print(f"infer count : {sem.get('inference_count', 0)}")
            print(f"infer time  : {sem.get('last_inference_time', 0):.3f}s")

        h, w = depth.shape
        samples = [
            ("center", w // 2, h // 2),
            ("top_center", w // 2, h // 4),
            ("bottom_center", w // 2, int(h * 0.80)),
            ("left_center", w // 4, h // 2),
            ("right_center", int(w * 0.75), h // 2),
        ]

        # Rebuild R/t from tf_msg only for readable report.
        R = quat_to_rot_matrix_xyzw(qr.x, qr.y, qr.z, qr.w)
        t = np.array([[tr.x], [tr.y], [tr.z]], dtype=np.float32)

        print("\n========== PIXEL PROJECTION CHECK ==========")
        print("camera optical: X right, Y down, Z forward")
        print(f"{self.args.target_frame}: x forward, y left, z up")
        for name, x, y in samples:
            res = self.project_pixel_for_report(depth, R, t, x, y)
            if res is None:
                print(f"{name:14s} pixel=({x:3d},{y:3d}) depth=invalid")
                continue

            z, Xc, Yc, Zc, xb, yb, zb, ix, iy = res
            if ix is None:
                grid_text = "grid=OUT"
            else:
                grid_text = f"grid=({ix:02d},{iy:02d}), value={int(occupancy[iy, ix])}"

            print(
                f"{name:14s} pixel=({x:3d},{y:3d}) "
                f"depth={z:7.3f}m | "
                f"cam=({Xc:7.3f},{Yc:7.3f},{Zc:7.3f}) -> "
                f"{self.args.target_frame}=({xb:7.3f},{yb:7.3f},{zb:7.3f}) | "
                f"{grid_text}"
            )

        print("\nSanity check:")
        print("  center x should usually be positive, meaning in front of robot.")
        print("  left image should usually have larger y than right image in base_footprint.")
        print("  ground-like pixels should have z close to 0 in base_footprint.")
        print("  if many cells are occupied unexpectedly, tune --obstacle-height or --ground-z.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone ROS2 depth image to local 40x40 traversability OccupancyGrid."
    )

    parser.add_argument("--depth-topic", default="/camera_sensor/depth/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera_sensor/depth/camera_info")
    parser.add_argument("--grid-topic", default="/local_traversability_grid")

    # Optional CLIPSeg semantic fusion. The final output topic remains --grid-topic.
    parser.add_argument("--enable-clipseg", dest="enable_clipseg", action="store_true", default=True)
    parser.add_argument("--disable-clipseg", dest="enable_clipseg", action="store_false")
    parser.add_argument("--rgb-topic", default="/camera_sensor/color/image_raw")
    parser.add_argument("--clipseg-model-dir", default="/home/zyy/nvidia/models/clipseg-rd64-refined")
    parser.add_argument("--clipseg-prompts", nargs="+", default=["vegetation", "Pavement", "grass", "Stop gesture"])
    parser.add_argument("--behavior-rule", default="avoid_grass", choices=["avoid_grass", "allow_grass", "prefer_pavement"])
    parser.add_argument("--clipseg-hz", type=float, default=1.0)
    parser.add_argument("--clipseg-input-width", type=int, default=352)
    parser.add_argument("--clipseg-debug-image-topic", default="/clipseg_inference_image")
    parser.add_argument("--publish-clipseg-debug-image", dest="publish_clipseg_debug_image", action="store_true", default=True)
    parser.add_argument("--no-clipseg-debug-image", dest="publish_clipseg_debug_image", action="store_false")
    parser.add_argument("--clipseg-confidence-threshold", type=float, default=0.25)
    parser.add_argument("--clipseg-use-amp", dest="clipseg_use_amp", action="store_true", default=True)
    parser.add_argument("--no-clipseg-amp", dest="clipseg_use_amp", action="store_false")

    parser.add_argument("--target-frame", default="base_footprint")

    # Your system is Gazebo with /clock, so sim time is enabled by default.
    parser.add_argument("--use-sim-time", dest="use_sim_time", action="store_true", default=True)
    parser.add_argument("--no-sim-time", dest="use_sim_time", action="store_false")

    # 4m x 4m local grid: x forward 0~4m, y left/right -2~2m.
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=8.0)
    parser.add_argument("--y-min", type=float, default=-4.0)
    parser.add_argument("--y-max", type=float, default=4.0)
    parser.add_argument("--resolution", type=float, default=0.2)

    # Depth filtering.
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=10)

    # Geometry classification in target_frame.
    parser.add_argument("--ground-z", type=float, default=0.0)
    parser.add_argument("--obstacle-height", type=float, default=0.15)
    parser.add_argument("--max-obstacle-height", type=float, default=1.5)

    # Performance and noise control.
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--min-points-per-cell", type=int, default=3)
    parser.add_argument("--min-obstacle-points-per-cell", type=int, default=3)
    parser.add_argument("--publish-hz", type=float, default=10.0)

    # Semantic fusion parameters. CLIPSeg runs on GPU, projection/fusion is vectorized NumPy.
    parser.add_argument("--semantic-downsample", type=int, default=2)
    parser.add_argument("--semantic-max-age", type=float, default=2.0)
    parser.add_argument("--semantic-cost-threshold", type=float, default=35.0)
    parser.add_argument("--semantic-block-value", type=int, default=80)
    parser.add_argument("--min-semantic-points-per-cell", type=int, default=3)
    parser.add_argument("--semantic-ground-min", type=float, default=-0.20)
    parser.add_argument("--semantic-ground-max", type=float, default=0.30)
    parser.add_argument("--semantic-override-unknown", action="store_true", default=False)

    # TF.
    parser.add_argument("--tf-timeout", type=float, default=0.10)
    parser.add_argument("--allow-latest-tf", action="store_true", default=True)
    parser.add_argument("--no-latest-tf", dest="allow_latest_tf", action="store_false")

    # Runtime/debug.
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--wait-timeout", type=float, default=10.0)
    parser.add_argument("--print-every", type=int, default=30)

    # Keep ROS-specific arguments out of argparse trouble.
    args, ros_args = parser.parse_known_args()
    return args, ros_args


def main():
    args, ros_args = parse_args()

    rclpy.init(args=ros_args)
    node = DepthGridStandalone(args)

    try:
        if args.once:
            start = time.time()
            while rclpy.ok() and not node.done:
                rclpy.spin_once(node, timeout_sec=0.1)
                if time.time() - start > args.wait_timeout:
                    node.get_logger().error("Timeout waiting for camera_info/depth/tf.")
                    break
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
