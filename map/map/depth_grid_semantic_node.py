#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
depth_grid_semantic_node.py

Robust event-driven geometry + semantic fusion node with safe image-buffer handling.

Design:
  1. Depth callback only caches the latest depth frame.
  2. A worker thread builds geometry grid from the latest depth frame.
  3. Grid is published ONLY when a valid geometry grid was produced.
  4. No fake / empty / all-unknown keepalive grid is published.
  5. CLIPSeg is NOT run in this process.
  6. This node subscribes /clipseg_cost_map and projects it into the geometry grid.
  7. If depth callback becomes stale, the process exits for shell supervisor restart.
"""

import argparse
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time

from builtin_interfaces.msg import Time as TimeMsg
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformListener


UNKNOWN = -1
FREE = 0
SEMANTIC_BLOCKED = 80
OCCUPIED = 100


@dataclass(frozen=True)
class ImagePacket:
    data: bytes
    width: int
    height: int
    step: int
    encoding: str
    is_bigendian: int
    stamp: object
    frame_id: str
    seq: int
    recv_wall_time: float


class EwmaRate:
    def __init__(self, alpha: float = 0.15, min_period: float = 0.001, max_period: float = 30.0):
        self.alpha = float(np.clip(alpha, 0.001, 1.0))
        self.min_period = float(max(1e-6, min_period))
        self.max_period = float(max(self.min_period, max_period))
        self.last_wall_time = 0.0
        self.period = None

    def update(self, now: float) -> None:
        now = float(now)
        if self.last_wall_time > 0.0:
            dt = float(np.clip(now - self.last_wall_time, self.min_period, self.max_period))
            if self.period is None:
                self.period = dt
            else:
                self.period = (1.0 - self.alpha) * self.period + self.alpha * dt
        self.last_wall_time = now

    def hz(self) -> float:
        if self.period is None or self.period <= 0.0:
            return 0.0
        return 1.0 / self.period


class EwmaScalar:
    def __init__(self, alpha: float = 0.15):
        self.alpha = float(np.clip(alpha, 0.001, 1.0))
        self.value = None

    def update(self, x: float) -> None:
        x = float(max(0.0, x))
        if self.value is None:
            self.value = x
        else:
            self.value = (1.0 - self.alpha) * self.value + self.alpha * x

    def get(self, default: float = 0.0) -> float:
        if self.value is None:
            return float(default)
        return float(self.value)


def make_sensor_qos(depth: int) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=max(1, int(depth)),
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def make_reliable_qos(depth: int) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=max(1, int(depth)),
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def quat_to_rot_matrix_xyzw(x: float, y: float, z: float, w: float) -> np.ndarray:
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

    expected_row_bytes = int(msg.width) * bytes_per_pixel
    expected_total_bytes = int(msg.height) * int(msg.step)

    if len(msg.data) != expected_total_bytes:
        raise ValueError(
            f"Image data length mismatch: got {len(msg.data)}, expected {expected_total_bytes}"
        )

    if int(msg.step) == expected_row_bytes:
        arr = np.frombuffer(msg.data, dtype=dtype, count=int(msg.width) * int(msg.height))
        depth = arr.reshape(int(msg.height), int(msg.width))
    else:
        raw_u8 = np.frombuffer(msg.data, dtype=np.uint8)
        rows = raw_u8.reshape(int(msg.height), int(msg.step))
        useful_rows = rows[:, :expected_row_bytes]
        depth = np.frombuffer(useful_rows.tobytes(), dtype=dtype).reshape(
            int(msg.height), int(msg.width)
        )

    if scale == 1.0 and depth.dtype == np.float32:
        return depth

    return depth.astype(np.float32) * scale


def decode_mono8_image(msg: Image) -> np.ndarray:
    encoding = msg.encoding.lower()
    if encoding not in ("mono8", "8uc1"):
        raise ValueError(f"Expected mono8 semantic cost map, got encoding={msg.encoding}")

    width = int(msg.width)
    height = int(msg.height)
    expected_row_bytes = width
    expected_total_bytes = height * int(msg.step)

    if len(msg.data) != expected_total_bytes:
        raise ValueError(
            f"Image data length mismatch: got {len(msg.data)}, expected {expected_total_bytes}"
        )

    raw = np.frombuffer(msg.data, dtype=np.uint8)

    if int(msg.step) == expected_row_bytes:
        return raw.reshape(height, width)

    rows = raw.reshape(height, int(msg.step))
    useful = rows[:, :expected_row_bytes].copy()
    return useful.reshape(height, width)


def resize_u8_nearest(src: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    src_h, src_w = src.shape[:2]

    if src_h == target_h and src_w == target_w:
        return src

    y_idx = np.linspace(0, src_h - 1, target_h).astype(np.int32)
    x_idx = np.linspace(0, src_w - 1, target_w).astype(np.int32)
    return src[y_idx[:, None], x_idx[None, :]]


class DepthGridSemanticNode(Node):
    def __init__(self, args):
        parameter_overrides = []
        if args.use_sim_time:
            parameter_overrides.append(Parameter("use_sim_time", Parameter.Type.BOOL, True))

        super().__init__("depth_grid_semantic_node", parameter_overrides=parameter_overrides)

        self.args = args

        self.grid_width = int(round((args.x_max - args.x_min) / args.resolution))
        self.grid_height = int(round((args.y_max - args.y_min) / args.resolution))
        self.num_cells = self.grid_width * self.grid_height

        if self.grid_width <= 0 or self.grid_height <= 0:
            raise ValueError("Invalid grid size. Check x/y range and resolution.")

        self.lock = threading.Lock()
        self.depth_cv = threading.Condition(self.lock)
        self.running = True

        self.has_camera_info = False
        self.image_width: Optional[int] = None
        self.image_height: Optional[int] = None
        self.camera_frame: Optional[str] = None
        self.x_factor: Optional[np.ndarray] = None
        self.y_factor: Optional[np.ndarray] = None

        self.latest_depth_packet: Optional[ImagePacket] = None
        self.latest_depth_seq = 0
        self.processed_depth_seq = 0

        self.latest_semantic_cost: Optional[np.ndarray] = None
        self.latest_semantic_wall_time: float = 0.0
        self.latest_semantic_stamp_sec: float = 0.0
        self.semantic_msg_count = 0

        self.depth_count = 0
        self.grid_pub_count = 0
        self.skip_count = 0
        self.tf_fail_count = 0
        self.decode_fail_count = 0
        self.depth_copy_fail_count = 0
        self.last_depth_callback_copy_time = 0.0

        self.last_depth_wall_time = 0.0
        self.last_grid_wall_time = 0.0
        self.last_warn_wall_time = 0.0

        self.depth_rate = EwmaRate(args.rate_ewma_alpha, max_period=args.max_observed_period)
        self.semantic_rate = EwmaRate(args.rate_ewma_alpha, max_period=args.max_observed_period)
        self.grid_process_ewma = EwmaScalar(args.rate_ewma_alpha)
        self.effective_downsample = max(1, int(args.downsample))
        self.current_depth_stale_timeout = float(args.depth_exit_stale_sec)
        self.current_semantic_max_age = float(args.semantic_max_age)
        self.current_semantic_stamp_max_delta = float(args.semantic_stamp_max_delta)

        self.last_state = "INIT"
        self.last_debug: Dict[str, float] = {}

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.camera_info_group = ReentrantCallbackGroup()
        self.depth_group = ReentrantCallbackGroup()
        self.semantic_group = ReentrantCallbackGroup()
        self.heartbeat_group = ReentrantCallbackGroup()
        self.grid_pub_group = MutuallyExclusiveCallbackGroup()

        self.grid_pub = self.create_publisher(
            OccupancyGrid,
            args.grid_topic,
            make_reliable_qos(args.pub_qos_depth),
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            args.camera_info_topic,
            self.camera_info_callback,
            make_sensor_qos(args.image_qos_depth),
            callback_group=self.camera_info_group,
        )

        self.depth_sub = self.create_subscription(
            Image,
            args.depth_topic,
            self.depth_callback,
            make_sensor_qos(args.image_qos_depth),
            callback_group=self.depth_group,
        )

        self.semantic_sub = None
        if args.enable_semantic_fusion:
            self.semantic_sub = self.create_subscription(
                Image,
                args.semantic_cost_topic,
                self.semantic_callback,
                make_sensor_qos(args.semantic_qos_depth),
                callback_group=self.semantic_group,
            )

        self.heartbeat_timer = self.create_timer(args.heartbeat_sec, self.heartbeat_callback, callback_group=self.heartbeat_group)

        self.worker_thread = threading.Thread(
            target=self.grid_worker_loop,
            name="grid_worker",
            daemon=True,
        )
        self.worker_thread.start()

        self.get_logger().info("depth_grid_semantic_node.py safe-buffer worker version started")
        self.get_logger().info(f"depth_topic          : {args.depth_topic}")
        self.get_logger().info(f"camera_info_topic    : {args.camera_info_topic}")
        self.get_logger().info(f"grid_topic           : {args.grid_topic}")
        self.get_logger().info(f"semantic_fusion      : {args.enable_semantic_fusion}")
        self.get_logger().info(f"semantic_cost_topic  : {args.semantic_cost_topic}")
        self.get_logger().info(f"target_frame         : {args.target_frame}")
        self.get_logger().info(
            f"grid                 : {self.grid_width} x {self.grid_height}, "
            f"resolution={args.resolution:.3f} m"
        )
        self.get_logger().info(
            f"region               : x=[{args.x_min:.2f},{args.x_max:.2f}), "
            f"y=[{args.y_min:.2f},{args.y_max:.2f})"
        )
        self.get_logger().info(
            f"depth stale exit     : {args.depth_exit_stale_sec:.2f}s"
        )
        self.get_logger().info(f"image_qos_depth     : {args.image_qos_depth}")
        self.get_logger().info(f"semantic_qos_depth  : {args.semantic_qos_depth}")
        self.get_logger().info(f"pub_qos_depth       : {args.pub_qos_depth}")

    def destroy_node(self):
        self.running = False
        with self.depth_cv:
            self.depth_cv.notify_all()
        return super().destroy_node()

    def warn_throttled(self, text: str, period: float = 1.0) -> None:
        now = time.time()
        if now - self.last_warn_wall_time >= period:
            self.last_warn_wall_time = now
            self.get_logger().warn(text)

    @staticmethod
    def stamp_to_sec(stamp_msg) -> float:
        return float(stamp_msg.sec) + float(stamp_msg.nanosec) * 1e-9

    def compute_depth_stale_timeout_locked(self) -> float:
        base = float(self.args.depth_exit_stale_sec)
        if base <= 0.0:
            return 0.0
        if not self.args.adaptive_stale:
            return base
        p = self.depth_rate.period
        if p is None or p <= 0.0:
            return base
        dynamic = max(base, float(self.args.stale_period_mult) * p)
        if self.args.max_stale_sec > 0.0:
            dynamic = min(dynamic, float(self.args.max_stale_sec))
        return dynamic

    def compute_semantic_limits_locked(self) -> Tuple[float, float]:
        max_age = float(self.args.semantic_max_age)
        stamp_delta = float(self.args.semantic_stamp_max_delta)
        if self.args.adaptive_semantic_age:
            p = self.semantic_rate.period
            if p is not None and p > 0.0:
                max_age = max(max_age, float(self.args.semantic_age_period_mult) * p)
                stamp_delta = max(stamp_delta, float(self.args.semantic_stamp_period_mult) * p)
        if self.args.semantic_max_age_cap > 0.0:
            max_age = min(max_age, float(self.args.semantic_max_age_cap))
        if self.args.semantic_stamp_delta_cap > 0.0:
            stamp_delta = min(stamp_delta, float(self.args.semantic_stamp_delta_cap))
        return max_age, stamp_delta

    def get_effective_downsample_locked(self) -> int:
        if not self.args.auto_downsample:
            return max(1, int(self.args.downsample))
        return max(1, int(self.effective_downsample))

    def update_auto_downsample_locked(self, process_time: float) -> None:
        self.grid_process_ewma.update(process_time)
        if not self.args.auto_downsample:
            self.effective_downsample = max(1, int(self.args.downsample))
            return

        depth_period = self.depth_rate.period
        if depth_period is None or depth_period <= 0.0:
            return

        proc_avg = self.grid_process_ewma.get(default=process_time)
        budget = max(0.005, float(self.args.grid_target_load) * depth_period)
        min_ds = max(1, int(self.args.min_downsample))
        max_ds = max(min_ds, int(self.args.max_downsample))
        ds = max(min_ds, min(max_ds, int(self.effective_downsample)))

        if proc_avg > budget * 1.20 and ds < max_ds:
            ds += 1
        elif proc_avg < budget * 0.45 and ds > min_ds:
            ds -= 1

        self.effective_downsample = ds

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

        with self.lock:
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

    def depth_callback(self, msg: Image) -> None:
        start = time.time()
        now = start

        try:
            # Safety change: copy the payload in the callback and never store the
            # ROS Image message object outside this function. This keeps DDS/rclpy
            # image-buffer lifetime short even if the grid worker is busy.
            data_copy = bytes(msg.data)
            stamp_copy = TimeMsg(
                sec=int(msg.header.stamp.sec),
                nanosec=int(msg.header.stamp.nanosec),
            )
            packet = ImagePacket(
                data=data_copy,
                width=int(msg.width),
                height=int(msg.height),
                step=int(msg.step),
                encoding=str(msg.encoding),
                is_bigendian=int(msg.is_bigendian),
                stamp=stamp_copy,
                frame_id=str(msg.header.frame_id),
                seq=0,
                recv_wall_time=now,
            )
        except Exception as e:
            with self.lock:
                self.depth_copy_fail_count += 1
            self.warn_throttled(f"Failed to copy depth image in callback: {e}", period=2.0)
            return

        with self.depth_cv:
            self.depth_rate.update(now)
            self.latest_depth_seq += 1
            packet = ImagePacket(
                data=packet.data,
                width=packet.width,
                height=packet.height,
                step=packet.step,
                encoding=packet.encoding,
                is_bigendian=packet.is_bigendian,
                stamp=packet.stamp,
                frame_id=packet.frame_id,
                seq=self.latest_depth_seq,
                recv_wall_time=now,
            )
            self.latest_depth_packet = packet
            self.last_depth_wall_time = now
            self.last_depth_callback_copy_time = time.time() - start
            self.depth_count += 1
            self.depth_cv.notify()

    def semantic_callback(self, msg: Image) -> None:
        try:
            cost = decode_mono8_image(msg)
        except Exception as e:
            self.warn_throttled(f"Failed to decode semantic cost map: {e}", period=2.0)
            return

        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

        now = time.time()
        with self.lock:
            self.semantic_rate.update(now)
            self.latest_semantic_cost = cost.copy()
            self.latest_semantic_wall_time = now
            self.latest_semantic_stamp_sec = stamp_sec
            self.semantic_msg_count += 1

    def heartbeat_callback(self) -> None:
        now = time.time()

        with self.lock:
            depth_age = now - self.last_depth_wall_time if self.last_depth_wall_time > 0 else -1.0
            grid_age = now - self.last_grid_wall_time if self.last_grid_wall_time > 0 else -1.0
            sem_age = now - self.latest_semantic_wall_time if self.latest_semantic_wall_time > 0 else -1.0
            state = self.last_state
            debug = dict(self.last_debug)
            depth_count = self.depth_count
            grid_pub = self.grid_pub_count
            skip_count = self.skip_count
            tf_fail = self.tf_fail_count
            decode_fail = self.decode_fail_count
            depth_copy_fail = self.depth_copy_fail_count
            cb_copy = self.last_depth_callback_copy_time
            sem_count = self.semantic_msg_count
            depth_hz = self.depth_rate.hz()
            sem_hz = self.semantic_rate.hz()
            proc_avg = self.grid_process_ewma.get(default=0.0)
            eff_ds = self.get_effective_downsample_locked()
            depth_timeout = self.compute_depth_stale_timeout_locked()
            sem_max_age, sem_stamp_delta = self.compute_semantic_limits_locked()
            self.current_depth_stale_timeout = depth_timeout
            self.current_semantic_max_age = sem_max_age
            self.current_semantic_stamp_max_delta = sem_stamp_delta

        self.get_logger().info(
            "HEARTBEAT grid "
            f"state={state} "
            f"depth_count={depth_count} "
            f"grid_pub={grid_pub} "
            f"skip={skip_count} "
            f"tf_fail={tf_fail} "
            f"decode_fail={decode_fail} "
            f"depth_copy_fail={depth_copy_fail} "
            f"cb_copy={cb_copy:.4f}s "
            f"depth_hz={depth_hz:.2f} "
            f"grid_proc={proc_avg:.4f}s "
            f"ds={eff_ds} "
            f"sem_count={sem_count} "
            f"sem_hz={sem_hz:.2f} "
            f"depth_age={depth_age:.2f}s "
            f"grid_age={grid_age:.2f}s "
            f"sem_age={sem_age:.2f}s "
            f"depth_timeout={depth_timeout:.2f}s "
            f"sem_max_age={sem_max_age:.2f}s "
            f"sem_stamp_max={sem_stamp_delta:.2f}s "
            f"free={int(debug.get('free_cells', -1))} "
            f"occ={int(debug.get('occupied_cells', -1))} "
            f"sem_blocked={int(debug.get('semantic_blocked_cells', -1))}"
        )

        if depth_timeout > 0.0 and depth_count > 0 and depth_age > depth_timeout:
            self.get_logger().error(
                f"Depth input stale for {depth_age:.2f}s, timeout={depth_timeout:.2f}s. "
                "Exiting grid process for supervisor restart."
            )
            os._exit(20)

    def grid_worker_loop(self) -> None:
        while self.running and rclpy.ok():
            with self.depth_cv:
                while (
                    self.running
                    and self.latest_depth_seq == self.processed_depth_seq
                    and rclpy.ok()
                ):
                    self.depth_cv.wait(timeout=0.2)

                if not self.running or not rclpy.ok():
                    return

                packet = self.latest_depth_packet
                seq = self.latest_depth_seq

            if packet is None:
                continue

            self.process_depth_msg(packet, seq)

    def process_depth_msg(self, msg: ImagePacket, seq: int) -> None:
        process_start = time.time()
        with self.lock:
            has_camera_info = self.has_camera_info
            image_width = self.image_width
            image_height = self.image_height
            camera_frame = self.camera_frame

        if not has_camera_info:
            self.mark_processed(seq, "NO_CAMERA_INFO")
            return

        if msg.width != image_width or msg.height != image_height:
            self.warn_throttled(
                f"Depth size {msg.width}x{msg.height} does not match CameraInfo "
                f"{image_width}x{image_height}",
                period=2.0,
            )
            self.mark_processed(seq, "DEPTH_SIZE_MISMATCH")
            return

        try:
            depth = decode_depth_to_meters(msg)
        except Exception as e:
            with self.lock:
                self.decode_fail_count += 1
            self.warn_throttled(f"Failed to decode depth image: {e}", period=2.0)
            self.mark_processed(seq, "DEPTH_DECODE_FAIL")
            return

        source_frame = msg.frame_id or camera_frame
        if source_frame is None or source_frame == "":
            self.mark_processed(seq, "EMPTY_DEPTH_FRAME")
            return

        try:
            R, t, tf_msg = self.lookup_transform_matrix(source_frame, msg.stamp)
        except Exception as e:
            with self.lock:
                self.tf_fail_count += 1
            self.warn_throttled(
                f"Cannot lookup TF {self.args.target_frame} <- {source_frame}: {e}",
                period=1.0,
            )
            self.mark_processed(seq, "TF_FAIL")
            return

        try:
            occupancy, debug = self.build_geometry_grid(depth, R, t)
        except Exception as e:
            self.warn_throttled(f"build_geometry_grid failed: {e}", period=2.0)
            self.mark_processed(seq, "BUILD_GRID_FAIL")
            return

        valid_points = int(debug.get("valid_points", 0))
        in_area_points = int(debug.get("in_area_points", 0))

        if valid_points < self.args.min_valid_points or in_area_points < self.args.min_in_area_points:
            self.warn_throttled(
                "Depth frame has insufficient usable points. "
                f"valid_points={valid_points}, in_area_points={in_area_points}. "
                "Not publishing grid.",
                period=1.0,
            )
            self.mark_processed(seq, "INSUFFICIENT_DEPTH")
            return

        if self.args.enable_semantic_fusion:
            occupancy, sem_debug = self.fuse_semantic_cost_map(occupancy, depth, R, t, msg.stamp)
            debug.update(sem_debug)

        self.publish_grid(occupancy, msg.stamp)

        process_time = time.time() - process_start
        with self.lock:
            self.update_auto_downsample_locked(process_time)
            debug["grid_process_time"] = float(process_time)
            debug["grid_process_avg"] = float(self.grid_process_ewma.get(default=process_time))
            debug["effective_downsample"] = int(self.get_effective_downsample_locked())
            debug["depth_input_hz"] = float(self.depth_rate.hz())
            self.processed_depth_seq = seq
            self.grid_pub_count += 1
            self.last_grid_wall_time = time.time()
            self.last_state = "ACTIVE"
            self.last_debug = dict(debug)
            grid_pub_count = self.grid_pub_count

        if self.args.print_every > 0 and grid_pub_count % self.args.print_every == 0:
            self.print_debug_report(depth, occupancy, debug, tf_msg, source_frame)

    def mark_processed(self, seq: int, state: str) -> None:
        with self.lock:
            self.processed_depth_seq = seq
            self.skip_count += 1
            self.last_state = state

    def lookup_transform_matrix(self, source_frame: str, stamp_msg) -> Tuple[np.ndarray, np.ndarray, object]:
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

            self.warn_throttled(
                "TF lookup at image timestamp failed; trying latest TF. "
                f"Reason: {e_stamp}",
                period=2.0,
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

    def build_geometry_grid(self, depth: np.ndarray, R: np.ndarray, t: np.ndarray):
        with self.lock:
            x_factor = self.x_factor
            y_factor = self.y_factor

        assert x_factor is not None
        assert y_factor is not None

        with self.lock:
            ds = self.get_effective_downsample_locked()

        depth_ds = depth[::ds, ::ds]
        xf = x_factor[::ds, ::ds]
        yf = y_factor[::ds, ::ds]

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
            "semantic_ready": 0,
            "semantic_blocked_cells": 0,
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

        return occupancy_flat.reshape(self.grid_height, self.grid_width), debug

    def fuse_semantic_cost_map(
        self,
        occupancy: np.ndarray,
        depth: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        depth_stamp,
    ):
        now = time.time()

        with self.lock:
            sem = None if self.latest_semantic_cost is None else self.latest_semantic_cost.copy()
            sem_wall_time = self.latest_semantic_wall_time
            sem_stamp_sec = self.latest_semantic_stamp_sec
            sem_max_age, sem_stamp_max_delta = self.compute_semantic_limits_locked()
            self.current_semantic_max_age = sem_max_age
            self.current_semantic_stamp_max_delta = sem_stamp_max_delta
            x_factor = self.x_factor
            y_factor = self.y_factor

        debug = {
            "semantic_ready": 0,
            "semantic_age": -1.0,
            "semantic_max_age": float(sem_max_age),
            "semantic_stamp_delta": -1.0,
            "semantic_stamp_max_delta": float(sem_stamp_max_delta),
            "semantic_points": 0,
            "semantic_cells": 0,
            "semantic_blocked_cells": 0,
        }

        if sem is None or sem_wall_time <= 0.0:
            return occupancy, debug

        sem_age = now - sem_wall_time
        debug["semantic_age"] = float(sem_age)

        if sem_age > sem_max_age:
            return occupancy, debug

        if self.args.semantic_stamp_check and sem_stamp_sec > 0.0:
            try:
                depth_stamp_sec = self.stamp_to_sec(depth_stamp)
            except Exception:
                depth_stamp_sec = 0.0
            if depth_stamp_sec > 0.0:
                stamp_delta = abs(depth_stamp_sec - sem_stamp_sec)
                debug["semantic_stamp_delta"] = float(stamp_delta)
                if stamp_delta > sem_stamp_max_delta:
                    return occupancy, debug

        if sem.shape != depth.shape:
            sem = resize_u8_nearest(sem, depth.shape)

        assert x_factor is not None
        assert y_factor is not None

        with self.lock:
            ds = max(self.get_effective_downsample_locked(), int(self.args.semantic_downsample))

        depth_ds = depth[::ds, ::ds]
        cost_ds = sem[::ds, ::ds].astype(np.float32)
        xf = x_factor[::ds, ::ds]
        yf = y_factor[::ds, ::ds]

        valid = (
            np.isfinite(depth_ds)
            & (depth_ds > self.args.min_depth)
            & (depth_ds < self.args.max_depth)
            & (cost_ds >= self.args.semantic_min_pixel_cost)
        )

        if not np.any(valid):
            return occupancy, debug

        z = depth_ds[valid]
        Xc = xf[valid] * z
        Yc = yf[valid] * z
        Zc = z
        costs = cost_ds[valid]

        pts_cam = np.stack((Xc, Yc, Zc), axis=0).astype(np.float32)
        pts_base = R @ pts_cam + t

        xb = pts_base[0]
        yb = pts_base[1]
        zb = pts_base[2]

        z_rel = zb - self.args.ground_z

        in_area_ground = (
            (xb >= self.args.x_min)
            & (xb < self.args.x_max)
            & (yb >= self.args.y_min)
            & (yb < self.args.y_max)
            & (z_rel >= self.args.semantic_ground_min)
            & (z_rel <= self.args.semantic_ground_max)
        )

        if not np.any(in_area_ground):
            return occupancy, debug

        xb = xb[in_area_ground]
        yb = yb[in_area_ground]
        costs = costs[in_area_ground]

        ix = np.floor((xb - self.args.x_min) / self.args.resolution).astype(np.int32)
        iy = np.floor((yb - self.args.y_min) / self.args.resolution).astype(np.int32)

        ix = np.clip(ix, 0, self.grid_width - 1)
        iy = np.clip(iy, 0, self.grid_height - 1)

        cell_id = iy * self.grid_width + ix

        max_cost = np.zeros(self.num_cells, dtype=np.float32)
        point_count = np.zeros(self.num_cells, dtype=np.int32)

        np.maximum.at(max_cost, cell_id, costs)
        np.add.at(point_count, cell_id, 1)

        semantic_observed = point_count >= self.args.min_semantic_points_per_cell
        semantic_blocked = semantic_observed & (max_cost >= self.args.semantic_cost_threshold)

        fused_flat = occupancy.reshape(-1).copy()
        apply_mask = semantic_blocked & (fused_flat == FREE)
        fused_flat[apply_mask] = np.int8(self.args.semantic_block_value)

        debug["semantic_ready"] = 1
        debug["semantic_points"] = int(len(costs))
        debug["semantic_cells"] = int(np.count_nonzero(semantic_observed))
        debug["semantic_blocked_cells"] = int(np.count_nonzero(apply_mask))

        return fused_flat.reshape(self.grid_height, self.grid_width), debug

    def publish_grid(self, occupancy: np.ndarray, stamp_msg) -> None:
        msg = OccupancyGrid()

        msg.header.stamp = stamp_msg
        msg.header.frame_id = self.args.target_frame

        msg.info.map_load_time = self.get_clock().now().to_msg()
        msg.info.resolution = float(self.args.resolution)
        msg.info.width = int(self.grid_width)
        msg.info.height = int(self.grid_height)

        msg.info.origin.position.x = float(self.args.x_min)
        msg.info.origin.position.y = float(self.args.y_min)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.x = 0.0
        msg.info.origin.orientation.y = 0.0
        msg.info.origin.orientation.z = 0.0
        msg.info.origin.orientation.w = 1.0

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
        with self.lock:
            x_factor = self.x_factor
            y_factor = self.y_factor

        assert x_factor is not None
        assert y_factor is not None

        z = self.median_valid_depth(depth, x, y, half=2)
        if not np.isfinite(z):
            return None

        Xc = float(x_factor[y, x] * z)
        Yc = float(y_factor[y, x] * z)
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

        print("\n========== GRID REPORT ==========")
        print(f"state       : {self.last_state}")
        print(f"target_frame: {self.args.target_frame}")
        print(f"source_frame: {source_frame}")
        print(f"translation : x={tr.x:.4f}, y={tr.y:.4f}, z={tr.z:.4f}")
        print(f"quaternion  : x={qr.x:.4f}, y={qr.y:.4f}, z={qr.z:.4f}, w={qr.w:.4f}")
        print(f"topic       : {self.args.grid_topic}")
        print(f"size        : {self.grid_width} x {self.grid_height}")
        print(f"valid pts   : {debug.get('valid_points', 0)}")
        print(f"in-area pts : {debug.get('in_area_points', 0)}")
        print(f"free        : {debug.get('free_cells', 0)}")
        print(f"occupied    : {debug.get('occupied_cells', 0)}")
        print(f"unknown     : {debug.get('unknown_cells', 0)}")
        print(f"semantic ok : {debug.get('semantic_ready', 0)}")
        print(f"sem blocked : {debug.get('semantic_blocked_cells', 0)}")

        h, w = depth.shape
        samples = [
            ("bottom_center", w // 2, int(h * 0.80)),
            ("left_center", w // 4, h // 2),
            ("right_center", int(w * 0.75), h // 2),
        ]

        R = quat_to_rot_matrix_xyzw(qr.x, qr.y, qr.z, qr.w)
        t = np.array([[tr.x], [tr.y], [tr.z]], dtype=np.float32)

        print("\n========== PIXEL CHECK ==========")
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
                f"depth={z:7.3f}m -> "
                f"{self.args.target_frame}=({xb:7.3f},{yb:7.3f},{zb:7.3f}) | "
                f"{grid_text}"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Worker-based geometry grid with external CLIPSeg semantic fusion."
    )

    parser.add_argument("--depth-topic", default="/gemini330/depth/image_raw")
    parser.add_argument("--camera-info-topic", default="/gemini330/depth/camera_info")
    parser.add_argument("--grid-topic", default="/local_traversability_grid")
    parser.add_argument("--target-frame", default="tita4264886/base_link")

    parser.add_argument("--use-sim-time", dest="use_sim_time", action="store_true", default=False)
    parser.add_argument("--no-sim-time", dest="use_sim_time", action="store_false")

    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=5.0)
    parser.add_argument("--y-min", type=float, default=-2.5)
    parser.add_argument("--y-max", type=float, default=2.5)
    parser.add_argument("--resolution", type=float, default=0.2)

    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=6.0)

    parser.add_argument("--ground-z", type=float, default=0.0)
    parser.add_argument("--obstacle-height", type=float, default=0.25)
    parser.add_argument("--max-obstacle-height", type=float, default=1.5)

    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--auto-downsample", dest="auto_downsample", action="store_true", default=True)
    parser.add_argument("--no-auto-downsample", dest="auto_downsample", action="store_false")
    parser.add_argument("--min-downsample", type=int, default=2)
    parser.add_argument("--max-downsample", type=int, default=5)
    parser.add_argument("--grid-target-load", type=float, default=0.65)
    parser.add_argument("--min-points-per-cell", type=int, default=3)
    parser.add_argument("--min-obstacle-points-per-cell", type=int, default=3)

    parser.add_argument("--min-valid-points", type=int, default=1000)
    parser.add_argument("--min-in-area-points", type=int, default=200)

    parser.add_argument("--tf-timeout", type=float, default=0.10)
    parser.add_argument("--allow-latest-tf", action="store_true", default=True)
    parser.add_argument("--no-latest-tf", dest="allow_latest_tf", action="store_false")

    parser.add_argument("--enable-semantic-fusion", action="store_true", default=True)
    parser.add_argument("--disable-semantic-fusion", dest="enable_semantic_fusion", action="store_false")
    parser.add_argument("--semantic-cost-topic", default="/clipseg_cost_map")
    parser.add_argument("--semantic-max-age", type=float, default=10.0)
    parser.add_argument("--adaptive-semantic-age", dest="adaptive_semantic_age", action="store_true", default=True)
    parser.add_argument("--no-adaptive-semantic-age", dest="adaptive_semantic_age", action="store_false")
    parser.add_argument("--semantic-age-period-mult", type=float, default=3.0)
    parser.add_argument("--semantic-max-age-cap", type=float, default=30.0)
    parser.add_argument("--semantic-stamp-check", dest="semantic_stamp_check", action="store_true", default=True)
    parser.add_argument("--no-semantic-stamp-check", dest="semantic_stamp_check", action="store_false")
    parser.add_argument("--semantic-stamp-max-delta", type=float, default=4.0)
    parser.add_argument("--semantic-stamp-period-mult", type=float, default=2.5)
    parser.add_argument("--semantic-stamp-delta-cap", type=float, default=30.0)
    parser.add_argument("--semantic-downsample", type=int, default=2)
    parser.add_argument("--semantic-min-pixel-cost", type=float, default=1.0)
    parser.add_argument("--semantic-cost-threshold", type=float, default=35.0)
    parser.add_argument("--semantic-block-value", type=int, default=80)
    parser.add_argument("--min-semantic-points-per-cell", type=int, default=2)
    parser.add_argument("--semantic-ground-min", type=float, default=-0.20)
    parser.add_argument("--semantic-ground-max", type=float, default=0.35)

    parser.add_argument("--image-qos-depth", type=int, default=1)
    parser.add_argument("--semantic-qos-depth", type=int, default=1)
    parser.add_argument("--pub-qos-depth", type=int, default=1)

    parser.add_argument("--depth-exit-stale-sec", type=float, default=6.0)
    parser.add_argument("--adaptive-stale", dest="adaptive_stale", action="store_true", default=True)
    parser.add_argument("--no-adaptive-stale", dest="adaptive_stale", action="store_false")
    parser.add_argument("--stale-period-mult", type=float, default=4.0)
    parser.add_argument("--max-stale-sec", type=float, default=30.0)
    parser.add_argument("--rate-ewma-alpha", type=float, default=0.15)
    parser.add_argument("--max-observed-period", type=float, default=30.0)

    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--heartbeat-sec", type=float, default=2.0)
    parser.add_argument("--executor-threads", type=int, default=3)

    args, ros_args = parser.parse_known_args()
    return args, ros_args


def main():
    args, ros_args = parse_args()

    rclpy.init(args=ros_args)
    node = DepthGridSemanticNode(args)

    try:
        executor = MultiThreadedExecutor(num_threads=max(2, int(args.executor_threads)))
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
