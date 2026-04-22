#!/usr/bin/env python3

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import math
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.time import Time

import tf2_ros

from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from sensor_msgs.msg import Image, CameraInfo

import torch
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation


# =========================
# Math helpers
# =========================

def euler_from_quaternion(quaternion: List[float]) -> Tuple[float, float, float]:
    x, y, z, w = quaternion

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = max(min(t2, 1.0), -1.0)
    pitch = math.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return x, y, z, w


def quaternion_msg_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    x, y, z, w = quaternion_from_euler(0.0, 0.0, yaw)
    q.x = x
    q.y = y
    q.z = z
    q.w = w
    return q


def rotation_matrix_from_quaternion_msg(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


# =========================
# RRT* on cost_map
# =========================

class RRTStarNode2D:
    def __init__(self, x: float, y: float, parent: Optional['RRTStarNode2D'] = None, cost: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.parent = parent
        self.cost = float(cost)


class LocalRRTStar:
    def __init__(
        self,
        owner: 'FtFootStyleRGBDRRTStar',
        step_size: float = 0.25,
        goal_radius: float = 0.30,
        max_iter: int = 500,
        search_radius: float = 0.60,
        goal_sample_rate: float = 0.20,
    ):
        self.owner = owner
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.max_iter = max_iter
        self.search_radius = search_radius
        self.goal_sample_rate = goal_sample_rate

    @staticmethod
    def distance(n1: RRTStarNode2D, n2: RRTStarNode2D) -> float:
        return math.hypot(n1.x - n2.x, n1.y - n2.y)

    def sample(self, goal: RRTStarNode2D) -> RRTStarNode2D:
        if random.random() < self.goal_sample_rate:
            return RRTStarNode2D(goal.x, goal.y)

        x = random.uniform(self.owner.local_x_min, self.owner.local_x_max)
        y = random.uniform(self.owner.local_y_min, self.owner.local_y_max)
        return RRTStarNode2D(x, y)

    def nearest(self, nodes: List[RRTStarNode2D], q_rand: RRTStarNode2D) -> RRTStarNode2D:
        dists = [self.distance(n, q_rand) for n in nodes]
        return nodes[int(np.argmin(dists))]

    def steer(self, q_near: RRTStarNode2D, q_rand: RRTStarNode2D) -> Optional[RRTStarNode2D]:
        dx = q_rand.x - q_near.x
        dy = q_rand.y - q_near.y
        d = math.hypot(dx, dy)
        if d < 1e-9:
            return None
        step = min(self.step_size, d)
        yaw = math.atan2(dy, dx)
        return RRTStarNode2D(
            q_near.x + step * math.cos(yaw),
            q_near.y + step * math.sin(yaw),
        )

    def near_nodes(self, nodes: List[RRTStarNode2D], q_new: RRTStarNode2D) -> List[RRTStarNode2D]:
        return [n for n in nodes if self.distance(n, q_new) <= self.search_radius]

    def point_valid(self, x: float, y: float) -> bool:
        valid, cost = self.owner.lookup_cost_map(x, y)
        return valid and (cost <= self.owner.block_cost_threshold)

    def edge_valid(self, n1: RRTStarNode2D, n2: RRTStarNode2D, samples: int = 12) -> bool:
        for t in np.linspace(0.0, 1.0, samples):
            x = n1.x + t * (n2.x - n1.x)
            y = n1.y + t * (n2.y - n1.y)
            if not self.point_valid(x, y):
                return False
        return True

    def edge_semantic_cost(self, n1: RRTStarNode2D, n2: RRTStarNode2D, samples: int = 12) -> float:
        vals = []
        for t in np.linspace(0.0, 1.0, samples):
            x = n1.x + t * (n2.x - n1.x)
            y = n1.y + t * (n2.y - n1.y)
            valid, cost = self.owner.lookup_cost_map(x, y)
            if not valid:
                return 1.0
            vals.append(cost)
        return float(np.mean(vals)) if vals else 1.0

    def edge_cost(self, n1: RRTStarNode2D, n2: RRTStarNode2D) -> float:
        dist_cost = self.distance(n1, n2)
        semantic_cost = self.edge_semantic_cost(n1, n2)
        return dist_cost + self.owner.semantic_weight * semantic_cost

    def choose_parent(
        self,
        near_nodes: List[RRTStarNode2D],
        q_near: RRTStarNode2D,
        q_new: RRTStarNode2D,
    ) -> RRTStarNode2D:
        best_parent = q_near
        best_cost = q_near.cost + self.edge_cost(q_near, q_new)

        for n in near_nodes:
            if self.edge_valid(n, q_new):
                c = n.cost + self.edge_cost(n, q_new)
                if c < best_cost:
                    best_parent = n
                    best_cost = c

        q_new.parent = best_parent
        q_new.cost = best_cost
        return q_new

    def rewire(self, near_nodes: List[RRTStarNode2D], q_new: RRTStarNode2D) -> None:
        for n in near_nodes:
            if n is q_new.parent:
                continue
            if not self.edge_valid(q_new, n):
                continue
            candidate_cost = q_new.cost + self.edge_cost(q_new, n)
            if candidate_cost < n.cost:
                n.parent = q_new
                n.cost = candidate_cost

    @staticmethod
    def backtrack_path(goal_node: RRTStarNode2D) -> List[Tuple[float, float]]:
        path = []
        cur = goal_node
        while cur is not None:
            path.append((cur.x, cur.y))
            cur = cur.parent
        return path[::-1]

    def plan(self, start_xy: Tuple[float, float], goal_xy: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        start = RRTStarNode2D(start_xy[0], start_xy[1], None, 0.0)
        goal = RRTStarNode2D(goal_xy[0], goal_xy[1], None, 0.0)

        if not self.point_valid(start.x, start.y):
            self.owner.get_logger().warn("RRT*: start invalid in cost_map.")
            return None
        if not self.point_valid(goal.x, goal.y):
            self.owner.get_logger().warn("RRT*: goal invalid in cost_map.")
            return None

        nodes = [start]
        best_goal = None
        best_goal_cost = float('inf')

        for _ in range(self.max_iter):
            q_rand = self.sample(goal)
            q_near = self.nearest(nodes, q_rand)
            q_new = self.steer(q_near, q_rand)
            if q_new is None:
                continue
            if not self.edge_valid(q_near, q_new):
                continue

            near = self.near_nodes(nodes, q_new)
            q_new = self.choose_parent(near, q_near, q_new)
            nodes.append(q_new)
            self.rewire(near, q_new)

            if math.hypot(q_new.x - goal.x, q_new.y - goal.y) <= self.goal_radius:
                if self.edge_valid(q_new, goal):
                    goal_candidate = RRTStarNode2D(
                        goal.x,
                        goal.y,
                        q_new,
                        q_new.cost + self.edge_cost(q_new, goal),
                    )
                    if goal_candidate.cost < best_goal_cost:
                        best_goal = goal_candidate
                        best_goal_cost = goal_candidate.cost

        if best_goal is None:
            return None

        return self.backtrack_path(best_goal)


# =========================
# Main node
# =========================

class FtFootStyleRGBDRRTStar(Node):
    def __init__(self):
        super().__init__('ftfoot_style_rgbd_rrtstar')

        qos_best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ---------- Topics ----------
        self.declare_parameter('odom_topic', '/lio_sam/mapping/odometry')
        self.declare_parameter('rgb_topic', '/color/image_raw')
        self.declare_parameter('depth_topic', '/depth/image_raw')
        self.declare_parameter('color_info_topic', '/color/camera_info')
        self.declare_parameter('depth_info_topic', '/depth/camera_info')

        self.declare_parameter('path_topic', '/planned_path')
        self.declare_parameter('cost_image_topic', '/clipseg_cost_image')
        self.declare_parameter('cost_image_overlay_topic', '/clipseg_cost_overlay')
        self.declare_parameter('cost_map_topic', '/local_cost_map')

        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('path_frame_id', 'odom')
        self.declare_parameter('cost_map_frame_id', 'odom')

        # ---------- Fixed goal ----------
        self.declare_parameter('goal_radius', 3.0)
        self.declare_parameter('goal_theta_deg', 0.0)

        # ---------- CLIPSeg ----------
        self.declare_parameter('clipseg_dir', '/home/zyy/nvidia/models/clipseg-rd64-refined')
        self.declare_parameter('prompts', ['vegetation', 'pavement', 'grass', 'person'])
        self.declare_parameter('cost_values', [0.90, 0.05, 0.20, 1.00])
        self.declare_parameter('prompt_threshold', 0.10)
        self.declare_parameter('unknown_cost', 0.60)

        # ---------- Depth params ----------
        self.declare_parameter('depth_scale', 0.001)   # uint16 mm -> m
        self.declare_parameter('min_depth', 0.2)
        self.declare_parameter('max_depth', 6.0)
        self.declare_parameter('pixel_stride', 4)      # 保留参数，逆投影版本中不再使用
        self.declare_parameter('depth_consistency_tol', 0.20)
        self.declare_parameter('depth_occlusion_tol', 0.10)

        # ---------- Local cost_map ----------
        self.declare_parameter('local_x_min', -0.20)
        self.declare_parameter('local_x_max', 4.00)
        self.declare_parameter('local_y_min', -2.50)
        self.declare_parameter('local_y_max', 2.50)
        self.declare_parameter('cost_map_resolution', 0.05)

        self.declare_parameter('origin_clear_radius', 0.25)
        self.declare_parameter('goal_search_radius', 0.50)

        # ---------- Planner ----------
        self.declare_parameter('semantic_weight', 2.0)
        self.declare_parameter('block_cost_threshold', 0.85)
        self.declare_parameter('replan_period_sec', 0.5)
        self.declare_parameter('rrt_step_size', 0.25)
        self.declare_parameter('rrt_goal_radius', 0.30)
        self.declare_parameter('rrt_max_iter', 500)
        self.declare_parameter('rrt_search_radius', 0.60)

        # ---------- Params load ----------
        self.odom_topic = self.get_parameter('odom_topic').value
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.color_info_topic = self.get_parameter('color_info_topic').value
        self.depth_info_topic = self.get_parameter('depth_info_topic').value

        self.path_topic = self.get_parameter('path_topic').value
        self.cost_image_topic = self.get_parameter('cost_image_topic').value
        self.cost_image_overlay_topic = self.get_parameter('cost_image_overlay_topic').value
        self.cost_map_topic = self.get_parameter('cost_map_topic').value

        self.base_frame = self.get_parameter('base_frame').value
        self.path_frame_id = self.get_parameter('path_frame_id').value
        self.cost_map_frame_id = self.get_parameter('cost_map_frame_id').value

        self.goal_radius = float(self.get_parameter('goal_radius').value)
        self.goal_theta_deg = float(self.get_parameter('goal_theta_deg').value)

        self.clipseg_dir = self.get_parameter('clipseg_dir').value
        self.prompts = list(self.get_parameter('prompts').value)
        self.cost_values = [float(x) for x in self.get_parameter('cost_values').value]
        if len(self.prompts) != len(self.cost_values):
            raise ValueError("prompts and cost_values must have the same length")

        self.prompt_threshold = float(self.get_parameter('prompt_threshold').value)
        self.unknown_cost = float(self.get_parameter('unknown_cost').value)

        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.min_depth = float(self.get_parameter('min_depth').value)
        self.max_depth = float(self.get_parameter('max_depth').value)
        self.pixel_stride = int(self.get_parameter('pixel_stride').value)
        self.depth_consistency_tol = float(self.get_parameter('depth_consistency_tol').value)
        self.depth_occlusion_tol = float(self.get_parameter('depth_occlusion_tol').value)

        self.local_x_min = float(self.get_parameter('local_x_min').value)
        self.local_x_max = float(self.get_parameter('local_x_max').value)
        self.local_y_min = float(self.get_parameter('local_y_min').value)
        self.local_y_max = float(self.get_parameter('local_y_max').value)
        self.cost_map_resolution = float(self.get_parameter('cost_map_resolution').value)

        self.origin_clear_radius = float(self.get_parameter('origin_clear_radius').value)
        self.goal_search_radius = float(self.get_parameter('goal_search_radius').value)

        self.semantic_weight = float(self.get_parameter('semantic_weight').value)
        self.block_cost_threshold = float(self.get_parameter('block_cost_threshold').value)
        self.replan_period_sec = float(self.get_parameter('replan_period_sec').value)

        self.map_w = int(math.ceil((self.local_x_max - self.local_x_min) / self.cost_map_resolution))
        self.map_h = int(math.ceil((self.local_y_max - self.local_y_min) / self.cost_map_resolution))

        # ---------- State ----------
        self.odom_received = False
        self.rgb_received = False
        self.depth_received = False
        self.goal_initialized = False
        self.color_info_received = False
        self.depth_info_received = False

        self.x = None
        self.y = None
        self.yaw = None
        self.fixed_goal_odom = None

        self.latest_rgb = None
        self.latest_depth = None

        self.color_fx = None
        self.color_fy = None
        self.color_cx = None
        self.color_cy = None

        self.depth_fx = None
        self.depth_fy = None
        self.depth_cx = None
        self.depth_cy = None

        self.color_frame = None
        self.depth_frame = None

        self.cost_image = None          # float32 [0,1], image coords
        self.cost_image_u8 = None       # uint8 [0,255]
        self.cost_map = None            # float32 [0,1], local ground grid
        self.cost_map_valid = None      # bool, whether the cell is inside projected view

        # ---------- CLIPSeg ----------
        if not os.path.isdir(self.clipseg_dir):
            raise FileNotFoundError(f"Local CLIPSeg directory not found: {self.clipseg_dir}")

        self.processor = CLIPSegProcessor.from_pretrained(
            self.clipseg_dir,
            local_files_only=True,
        )
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            self.clipseg_dir,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        self.get_logger().info(f"CLIPSeg loaded from: {self.clipseg_dir}")

        # ---------- Planner ----------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.rrt_star = LocalRRTStar(
            owner=self,
            step_size=float(self.get_parameter('rrt_step_size').value),
            goal_radius=float(self.get_parameter('rrt_goal_radius').value),
            max_iter=int(self.get_parameter('rrt_max_iter').value),
            search_radius=float(self.get_parameter('rrt_search_radius').value),
        )

        # ---------- ROS I/O ----------
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, qos_best_effort)
        self.sub_rgb = self.create_subscription(Image, self.rgb_topic, self.rgb_callback, qos_best_effort)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.depth_callback, qos_best_effort)

        self.sub_color_info = self.create_subscription(
            CameraInfo, self.color_info_topic, self.color_info_callback, qos_best_effort
        )
        self.sub_depth_info = self.create_subscription(
            CameraInfo, self.depth_info_topic, self.depth_info_callback, qos_best_effort
        )

        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        self.cost_image_pub = self.create_publisher(Image, self.cost_image_topic, 10)
        self.cost_image_overlay_pub = self.create_publisher(Image, self.cost_image_overlay_topic, 10)
        self.cost_map_pub = self.create_publisher(OccupancyGrid, self.cost_map_topic, 10)

        self.timer = self.create_timer(self.replan_period_sec, self.plan_cycle)

    # =========================
    # Callbacks
    # =========================

    def odom_callback(self, msg: Odometry) -> None:
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.odom_received = True

        if not self.goal_initialized:
            gx_r = self.goal_radius * math.cos(math.radians(self.goal_theta_deg))
            gy_r = self.goal_radius * math.sin(math.radians(self.goal_theta_deg))
            gx_o, gy_o = self.robot_to_odom_xy(gx_r, gy_r)
            self.fixed_goal_odom = (gx_o, gy_o)
            self.goal_initialized = True
            self.get_logger().info(f"Fixed goal initialized in odom: ({gx_o:.2f}, {gy_o:.2f})")

    def rgb_callback(self, msg: Image) -> None:
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.rgb_received = True
        except Exception as e:
            self.get_logger().error(f"RGB callback failed: {e}")

    def depth_callback(self, msg: Image) -> None:
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.depth_received = True
        except Exception as e:
            self.get_logger().error(f"Depth callback failed: {e}")

    def color_info_callback(self, msg: CameraInfo) -> None:
        self.color_fx = float(msg.k[0])
        self.color_fy = float(msg.k[4])
        self.color_cx = float(msg.k[2])
        self.color_cy = float(msg.k[5])
        self.color_frame = msg.header.frame_id
        self.color_info_received = True

    def depth_info_callback(self, msg: CameraInfo) -> None:
        self.depth_fx = float(msg.k[0])
        self.depth_fy = float(msg.k[4])
        self.depth_cx = float(msg.k[2])
        self.depth_cy = float(msg.k[5])
        self.depth_frame = msg.header.frame_id
        self.depth_info_received = True

    # =========================
    # Basic transforms
    # =========================

    def robot_to_odom_xy(self, x_r: float, y_r: float) -> Tuple[float, float]:
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        x_o = self.x + c * x_r - s * y_r
        y_o = self.y + s * x_r + c * y_r
        return x_o, y_o

    def odom_to_robot_xy(self, x_o: float, y_o: float) -> Tuple[float, float]:
        dx = x_o - self.x
        dy = y_o - self.y
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        x_r = c * dx + s * dy
        y_r = -s * dx + c * dy
        return x_r, y_r

    def path_robot_to_odom(self, path_robot: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        return [self.robot_to_odom_xy(px, py) for px, py in path_robot]

    def lookup_rt(self, target_frame: str, source_frame: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            tf_msg = self.tf_buffer.lookup_transform(target_frame, source_frame, Time())
        except Exception as e:
            self.get_logger().warn(f"Waiting for TF {source_frame} -> {target_frame}: {e}")
            return None

        R = rotation_matrix_from_quaternion_msg(tf_msg.transform.rotation)
        t = np.array([
            tf_msg.transform.translation.x,
            tf_msg.transform.translation.y,
            tf_msg.transform.translation.z,
        ], dtype=np.float64)
        return R, t

    @staticmethod
    def transform_points(R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
        return (R @ pts.T).T + t[None, :]

    @staticmethod
    def bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]

        inside = (u >= 0.0) & (u <= (w - 1)) & (v >= 0.0) & (v <= (h - 1))
        values = np.zeros(u.shape, dtype=np.float32)
        if not np.any(inside):
            return values, inside

        u_in = u[inside]
        v_in = v[inside]

        x0 = np.floor(u_in).astype(np.int32)
        y0 = np.floor(v_in).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)

        du = (u_in - x0).astype(np.float32)
        dv = (v_in - y0).astype(np.float32)

        w00 = (1.0 - du) * (1.0 - dv)
        w10 = du * (1.0 - dv)
        w01 = (1.0 - du) * dv
        w11 = du * dv

        i00 = image[y0, x0].astype(np.float32)
        i10 = image[y0, x1].astype(np.float32)
        i01 = image[y1, x0].astype(np.float32)
        i11 = image[y1, x1].astype(np.float32)

        if valid_mask is None:
            sampled = w00 * i00 + w10 * i10 + w01 * i01 + w11 * i11
            ok_inside = np.ones(sampled.shape, dtype=bool)
        else:
            m00 = valid_mask[y0, x0].astype(np.float32)
            m10 = valid_mask[y0, x1].astype(np.float32)
            m01 = valid_mask[y1, x0].astype(np.float32)
            m11 = valid_mask[y1, x1].astype(np.float32)

            denom = w00 * m00 + w10 * m10 + w01 * m01 + w11 * m11
            sampled = np.zeros(u_in.shape, dtype=np.float32)
            ok_inside = denom > 1e-6
            sampled[ok_inside] = (
                w00[ok_inside] * i00[ok_inside] * m00[ok_inside] +
                w10[ok_inside] * i10[ok_inside] * m10[ok_inside] +
                w01[ok_inside] * i01[ok_inside] * m01[ok_inside] +
                w11[ok_inside] * i11[ok_inside] * m11[ok_inside]
            ) / denom[ok_inside]

        values[inside] = sampled
        ok = np.zeros(u.shape, dtype=bool)
        ok[inside] = ok_inside
        return values, ok

    # =========================
    # cost_image: image coordinates
    # =========================

    def build_cost_image(self, rgb_image: np.ndarray) -> None:
        h, w, _ = rgb_image.shape
        pil_image = PILImage.fromarray(rgb_image)

        inputs = self.processor(
            text=self.prompts,
            images=[pil_image] * len(self.prompts),
            return_tensors='pt',
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        preds = torch.sigmoid(outputs.logits)
        preds_resized = F.interpolate(
            preds.unsqueeze(1),
            size=(h, w),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1).cpu().numpy()

        cost_image = np.full((h, w), self.unknown_cost, dtype=np.float32)
        best_score = np.zeros((h, w), dtype=np.float32)

        for i, pred in enumerate(preds_resized):
            mask = pred > self.prompt_threshold
            stronger = pred > best_score
            use = mask & stronger
            cost_image[use] = self.cost_values[i]
            best_score[use] = pred[use]

        self.cost_image = cost_image
        self.cost_image_u8 = np.clip(cost_image * 255.0, 0, 255).astype(np.uint8)

    def publish_cost_image(self, rgb_image: np.ndarray) -> None:
        if self.cost_image_u8 is None:
            return

        mono_msg = self.bridge.cv2_to_imgmsg(self.cost_image_u8, encoding='mono8')
        self.cost_image_pub.publish(mono_msg)

        colored = cv2.applyColorMap(self.cost_image_u8, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(rgb_image, 0.45, colored, 0.55, 0.0)
        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='rgb8')
        self.cost_image_overlay_pub.publish(overlay_msg)

    # =========================
    # cost_map: continuous inverse projection
    # =========================

    def depth_to_meters(self, depth: np.ndarray) -> np.ndarray:
        if depth.dtype == np.uint16:
            return depth.astype(np.float32) * self.depth_scale
        return depth.astype(np.float32)

    def build_cost_map_from_rgbd(self, depth_image: np.ndarray) -> None:
        if self.cost_image is None:
            return

        if not (self.color_info_received and self.depth_info_received):
            self.get_logger().warn("Waiting for color/depth camera_info.")
            return

        rt_color = self.lookup_rt(self.color_frame, self.base_frame)
        if rt_color is None:
            return
        R_cb, t_cb = rt_color

        rt_depth = self.lookup_rt(self.depth_frame, self.base_frame)
        if rt_depth is None:
            return
        R_db, t_db = rt_depth

        depth_m = self.depth_to_meters(depth_image)
        depth_valid_pixels = np.isfinite(depth_m) & (depth_m > self.min_depth) & (depth_m < self.max_depth)

        # 1) 生成局部地面栅格中心点，坐标系在 base_frame/base_footprint，z=0
        xs = self.local_x_min + (np.arange(self.map_w) + 0.5) * self.cost_map_resolution
        ys = self.local_y_min + (np.arange(self.map_h) + 0.5) * self.cost_map_resolution
        xx, yy = np.meshgrid(xs, ys)
        zz = np.zeros_like(xx, dtype=np.float32)

        pts_base = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float64)

        # 2) 栅格中心 -> color optical
        pts_color = self.transform_points(R_cb, t_cb, pts_base)
        zc = pts_color[:, 2]
        front_color = zc > 1e-6

        u_color = np.full(zc.shape, -1.0, dtype=np.float32)
        v_color = np.full(zc.shape, -1.0, dtype=np.float32)
        u_color[front_color] = self.color_fx * (pts_color[front_color, 0] / pts_color[front_color, 2]) + self.color_cx
        v_color[front_color] = self.color_fy * (pts_color[front_color, 1] / pts_color[front_color, 2]) + self.color_cy

        color_h, color_w = self.cost_image.shape[:2]
        in_color = (
            (u_color >= 0.0) & (u_color <= (color_w - 1)) &
            (v_color >= 0.0) & (v_color <= (color_h - 1))
        )

        # 3) 栅格中心 -> depth optical
        pts_depth = self.transform_points(R_db, t_db, pts_base)
        zd = pts_depth[:, 2]
        front_depth = zd > 1e-6

        u_depth = np.full(zd.shape, -1.0, dtype=np.float32)
        v_depth = np.full(zd.shape, -1.0, dtype=np.float32)
        u_depth[front_depth] = self.depth_fx * (pts_depth[front_depth, 0] / pts_depth[front_depth, 2]) + self.depth_cx
        v_depth[front_depth] = self.depth_fy * (pts_depth[front_depth, 1] / pts_depth[front_depth, 2]) + self.depth_cy

        depth_h, depth_w = depth_m.shape[:2]
        in_depth = (
            (u_depth >= 0.0) & (u_depth <= (depth_w - 1)) &
            (v_depth >= 0.0) & (v_depth <= (depth_h - 1))
        )

        # 4) 几何上可投影到 color + depth 的地面 cell
        geo_mask = front_color & front_depth & in_color & in_depth

        cost_flat = np.full((pts_base.shape[0],), self.unknown_cost, dtype=np.float32)
        valid_flat = np.zeros((pts_base.shape[0],), dtype=bool)

        if np.any(geo_mask):
            valid_flat[geo_mask] = True

            # 从 cost_image 稠密采样语义代价
            semantic_vals, semantic_ok = self.bilinear_sample(
                self.cost_image,
                u_color[geo_mask],
                v_color[geo_mask],
                valid_mask=None,
            )

            # 从 depth 图采样观测深度，做遮挡一致性检查
            depth_obs, depth_ok = self.bilinear_sample(
                depth_m,
                u_depth[geo_mask],
                v_depth[geo_mask],
                valid_mask=depth_valid_pixels,
            )

            z_pred = pts_depth[geo_mask, 2].astype(np.float32)

            geo_cost = np.full(z_pred.shape, self.unknown_cost, dtype=np.float32)

            # 一致：真实深度和该地面cell预测深度接近
            consistent = depth_ok & semantic_ok & (np.abs(depth_obs - z_pred) <= self.depth_consistency_tol)

            # 遮挡：前面有更近的物体挡住该地面cell
            occluded = depth_ok & ((depth_obs + self.depth_occlusion_tol) < z_pred)

            geo_cost[consistent] = semantic_vals[consistent]
            geo_cost[occluded] = 1.0

            cost_flat[geo_mask] = geo_cost

        self.cost_map = cost_flat.reshape(self.map_h, self.map_w)
        self.cost_map_valid = valid_flat.reshape(self.map_h, self.map_w)
        self._mark_robot_origin_clear()

    def _mark_robot_origin_clear(self) -> None:
        if self.cost_map is None or self.cost_map_valid is None:
            return

        xs = self.local_x_min + (np.arange(self.map_w) + 0.5) * self.cost_map_resolution
        ys = self.local_y_min + (np.arange(self.map_h) + 0.5) * self.cost_map_resolution
        xx, yy = np.meshgrid(xs, ys)
        bubble = (xx ** 2 + yy ** 2) <= (self.origin_clear_radius ** 2)

        self.cost_map[bubble] = np.minimum(self.cost_map[bubble], 0.05)
        self.cost_map_valid[bubble] = True

    def publish_cost_map(self) -> None:
        if self.cost_map is None or self.cost_map_valid is None or not self.odom_received:
            return

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.cost_map_frame_id

        msg.info.resolution = float(self.cost_map_resolution)
        msg.info.width = self.map_w
        msg.info.height = self.map_h

        ox, oy = self.robot_to_odom_xy(self.local_x_min, self.local_y_min)
        msg.info.origin.position.x = ox
        msg.info.origin.position.y = oy
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation = quaternion_msg_from_yaw(self.yaw)

        occ = np.full((self.map_h, self.map_w), -1, dtype=np.int8)
        occ[self.cost_map_valid] = np.clip(
            np.round(self.cost_map[self.cost_map_valid] * 100.0), 0, 100
        ).astype(np.int8)

        msg.data = occ.flatten().tolist()
        self.cost_map_pub.publish(msg)

    def lookup_cost_map(self, x_r: float, y_r: float) -> Tuple[bool, float]:
        if self.cost_map is None or self.cost_map_valid is None:
            return False, 1.0

        ix = int(math.floor((x_r - self.local_x_min) / self.cost_map_resolution))
        iy = int(math.floor((y_r - self.local_y_min) / self.cost_map_resolution))

        if ix < 0 or ix >= self.map_w or iy < 0 or iy >= self.map_h:
            return False, 1.0

        if not self.cost_map_valid[iy, ix]:
            return False, 1.0

        return True, float(self.cost_map[iy, ix])

    # =========================
    # Goal handling
    # =========================

    def choose_planning_goal_robot(self) -> Optional[Tuple[float, float]]:
        if self.fixed_goal_odom is None:
            return None

        gx_r, gy_r = self.odom_to_robot_xy(self.fixed_goal_odom[0], self.fixed_goal_odom[1])

        desired_x, desired_y = gx_r, gy_r

        inside = (
            self.local_x_min <= gx_r <= self.local_x_max and
            self.local_y_min <= gy_r <= self.local_y_max
        )

        if not inside:
            if gx_r <= 1e-6:
                self.get_logger().warn("Fixed goal is behind robot / outside local front map.")
                return None

            scales = []
            if gx_r > 0.0:
                scales.append((self.local_x_max - 0.10) / gx_r)
            if gy_r > 0.0:
                scales.append((self.local_y_max - 0.10) / gy_r)
            elif gy_r < 0.0:
                scales.append((self.local_y_min + 0.10) / gy_r)

            scale = min([s for s in scales if s > 0.0], default=None)
            if scale is None:
                return None

            desired_x = gx_r * scale
            desired_y = gy_r * scale

        return self.find_valid_goal_near(desired_x, desired_y)

    def find_valid_goal_near(self, x_r: float, y_r: float) -> Optional[Tuple[float, float]]:
        valid, cost = self.lookup_cost_map(x_r, y_r)
        if valid and cost <= self.block_cost_threshold:
            return x_r, y_r

        max_cells = int(math.ceil(self.goal_search_radius / self.cost_map_resolution))
        best = None
        best_dist = float('inf')

        center_ix = int(math.floor((x_r - self.local_x_min) / self.cost_map_resolution))
        center_iy = int(math.floor((y_r - self.local_y_min) / self.cost_map_resolution))

        for dy in range(-max_cells, max_cells + 1):
            for dx in range(-max_cells, max_cells + 1):
                ix = center_ix + dx
                iy = center_iy + dy
                if ix < 0 or ix >= self.map_w or iy < 0 or iy >= self.map_h:
                    continue

                cx = self.local_x_min + (ix + 0.5) * self.cost_map_resolution
                cy = self.local_y_min + (iy + 0.5) * self.cost_map_resolution

                valid, c = self.lookup_cost_map(cx, cy)
                if not valid:
                    continue
                if c > self.block_cost_threshold:
                    continue

                d = math.hypot(cx - x_r, cy - y_r)
                if d < best_dist:
                    best = (cx, cy)
                    best_dist = d

        return best

    # =========================
    # Path publishing
    # =========================

    def publish_path(self, path_odom: List[Tuple[float, float]]) -> None:
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.path_frame_id

        for i, (x, y) in enumerate(path_odom):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0

            if i < len(path_odom) - 1:
                nx, ny = path_odom[i + 1]
                yaw = math.atan2(ny - y, nx - x)
            else:
                yaw = self.yaw if self.yaw is not None else 0.0

            ps.pose.orientation = quaternion_msg_from_yaw(yaw)
            msg.poses.append(ps)

        self.path_pub.publish(msg)

    # =========================
    # Main planning cycle
    # =========================

    def plan_cycle(self) -> None:
        if not (
            self.odom_received and
            self.rgb_received and
            self.depth_received and
            self.goal_initialized and
            self.color_info_received and
            self.depth_info_received
        ):
            return

        rgb_image = self.latest_rgb
        depth_image = self.latest_depth

        if rgb_image is None or depth_image is None:
            return

        if depth_image.ndim == 3:
            depth_image = depth_image[:, :, 0]

        # 1) image-space semantic cost
        self.build_cost_image(rgb_image)
        self.publish_cost_image(rgb_image)

        # 2) continuous ground cost_map
        self.build_cost_map_from_rgbd(depth_image)
        self.publish_cost_map()

        # 3) local planning goal
        goal_robot = self.choose_planning_goal_robot()
        if goal_robot is None:
            self.get_logger().warn("No valid local planning goal found.")
            return

        # 4) RRT*
        start_robot = (0.0, 0.0)
        path_robot = self.rrt_star.plan(start_robot, goal_robot)
        if path_robot is None or len(path_robot) < 2:
            self.get_logger().warn("RRT* failed on current continuous cost_map.")
            return

        # 5) publish path in odom
        path_odom = self.path_robot_to_odom(path_robot)
        self.publish_path(path_odom)


def main(args=None):
    rclpy.init(args=args)
    node = FtFootStyleRGBDRRTStar()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()