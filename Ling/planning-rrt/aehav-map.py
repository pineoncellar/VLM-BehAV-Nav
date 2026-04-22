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
# RRT* on global cost map
# =========================

class RRTStarNode2D:
    def __init__(self, x: float, y: float, parent: Optional['RRTStarNode2D'] = None, cost: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.parent = parent
        self.cost = float(cost)


class GlobalRRTStar:
    def __init__(
        self,
        owner: 'FtFootStyleRGBDRRTStar',
        step_size: float = 0.25,
        goal_radius: float = 0.30,
        max_iter: int = 1200,
        search_radius: float = 0.60,
        goal_sample_rate: float = 0.25,
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

        x = random.uniform(self.owner.rrt_sample_x_min, self.owner.rrt_sample_x_max)
        y = random.uniform(self.owner.rrt_sample_y_min, self.owner.rrt_sample_y_max)
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
        valid, cost = self.owner.lookup_global_cost_map(x, y)
        return valid and (cost <= self.owner.block_cost_threshold)

    def edge_valid(self, n1: RRTStarNode2D, n2: RRTStarNode2D, samples: int = 16) -> bool:
        for t in np.linspace(0.0, 1.0, samples):
            x = n1.x + t * (n2.x - n1.x)
            y = n1.y + t * (n2.y - n1.y)
            if not self.point_valid(x, y):
                return False
        return True

    def edge_semantic_cost(self, n1: RRTStarNode2D, n2: RRTStarNode2D, samples: int = 16) -> float:
        vals = []
        for t in np.linspace(0.0, 1.0, samples):
            x = n1.x + t * (n2.x - n1.x)
            y = n1.y + t * (n2.y - n1.y)
            valid, cost = self.owner.lookup_global_cost_map(x, y)
            if not valid:
                return 1.0
            vals.append(cost)
        return float(np.mean(vals)) if vals else 1.0

    def edge_cost(self, n1: RRTStarNode2D, n2: RRTStarNode2D) -> float:
        return self.distance(n1, n2) + self.owner.semantic_weight * self.edge_semantic_cost(n1, n2)

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
            self.owner.get_logger().warn("RRT*: start invalid in global cost_map.")
            return None
        if not self.point_valid(goal.x, goal.y):
            self.owner.get_logger().warn("RRT*: goal invalid in global cost_map.")
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
        self.declare_parameter('global_cost_map_topic', '/global_cost_map')

        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('path_frame_id', 'odom')
        self.declare_parameter('cost_map_frame_id', 'odom')

        # ---------- Goal ----------
        # 保持原来的终点给定方式：第一次 odom 到来时，按半径和角度在 odom 中固定终点
        self.declare_parameter('goal_radius', 3.0)
        self.declare_parameter('goal_theta_deg', 0.0)
        self.declare_parameter('goal_search_radius', 0.80)
        self.declare_parameter('goal_reached_dist', 0.25)

        # ---------- CLIPSeg ----------
        self.declare_parameter('clipseg_dir', '/home/zyy/nvidia/models/clipseg-rd64-refined')
        self.declare_parameter('prompts', ['vegetation', 'pavement', 'grass', 'person'])
        self.declare_parameter('cost_values', [0.90, 0.05, 0.20, 1.00])
        self.declare_parameter('prompt_threshold', 0.10)
        self.declare_parameter('unknown_cost', 0.60)

        # ---------- Depth params ----------
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('min_depth', 0.2)
        self.declare_parameter('max_depth', 6.0)
        self.declare_parameter('depth_consistency_tol', 0.25)
        self.declare_parameter('depth_occlusion_tol', 0.12)

        # ---------- Local cost_map ----------
        self.declare_parameter('local_x_min', -0.20)
        self.declare_parameter('local_x_max', 4.00)
        self.declare_parameter('local_y_min', -2.50)
        self.declare_parameter('local_y_max', 2.50)
        self.declare_parameter('cost_map_resolution', 0.05)
        self.declare_parameter('origin_clear_radius', 0.25)
        self.declare_parameter('smooth_kernel', 5)

        # ---------- Global cost_map ----------
        self.declare_parameter('global_x_min', -30.0)
        self.declare_parameter('global_x_max', 30.0)
        self.declare_parameter('global_y_min', -30.0)
        self.declare_parameter('global_y_max', 30.0)
        self.declare_parameter('global_cost_map_resolution', 0.10)
        self.declare_parameter('global_fusion_mode', 'max')  # max / mean / overwrite
        self.declare_parameter('global_unknown_is_obstacle', True)

        # ---------- Planner ----------
        self.declare_parameter('semantic_weight', 2.0)
        self.declare_parameter('block_cost_threshold', 0.85)
        self.declare_parameter('replan_period_sec', 0.5)
        self.declare_parameter('rrt_step_size', 0.25)
        self.declare_parameter('rrt_goal_radius', 0.30)
        self.declare_parameter('rrt_max_iter', 1200)
        self.declare_parameter('rrt_search_radius', 0.60)
        self.declare_parameter('rrt_goal_sample_rate', 0.25)
        self.declare_parameter('rrt_sampling_margin', 2.0)

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
        self.global_cost_map_topic = self.get_parameter('global_cost_map_topic').value

        self.base_frame = self.get_parameter('base_frame').value
        self.path_frame_id = self.get_parameter('path_frame_id').value
        self.cost_map_frame_id = self.get_parameter('cost_map_frame_id').value

        self.goal_radius = float(self.get_parameter('goal_radius').value)
        self.goal_theta_deg = float(self.get_parameter('goal_theta_deg').value)
        self.goal_search_radius = float(self.get_parameter('goal_search_radius').value)
        self.goal_reached_dist = float(self.get_parameter('goal_reached_dist').value)

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
        self.depth_consistency_tol = float(self.get_parameter('depth_consistency_tol').value)
        self.depth_occlusion_tol = float(self.get_parameter('depth_occlusion_tol').value)

        self.local_x_min = float(self.get_parameter('local_x_min').value)
        self.local_x_max = float(self.get_parameter('local_x_max').value)
        self.local_y_min = float(self.get_parameter('local_y_min').value)
        self.local_y_max = float(self.get_parameter('local_y_max').value)
        self.cost_map_resolution = float(self.get_parameter('cost_map_resolution').value)
        self.origin_clear_radius = float(self.get_parameter('origin_clear_radius').value)
        self.smooth_kernel = int(self.get_parameter('smooth_kernel').value)
        if self.smooth_kernel > 1 and (self.smooth_kernel % 2 == 0):
            self.smooth_kernel += 1

        self.global_x_min = float(self.get_parameter('global_x_min').value)
        self.global_x_max = float(self.get_parameter('global_x_max').value)
        self.global_y_min = float(self.get_parameter('global_y_min').value)
        self.global_y_max = float(self.get_parameter('global_y_max').value)
        self.global_cost_map_resolution = float(self.get_parameter('global_cost_map_resolution').value)
        self.global_fusion_mode = self.get_parameter('global_fusion_mode').value
        self.global_unknown_is_obstacle = bool(self.get_parameter('global_unknown_is_obstacle').value)

        self.semantic_weight = float(self.get_parameter('semantic_weight').value)
        self.block_cost_threshold = float(self.get_parameter('block_cost_threshold').value)
        self.replan_period_sec = float(self.get_parameter('replan_period_sec').value)
        self.rrt_sampling_margin = float(self.get_parameter('rrt_sampling_margin').value)

        self.local_map_w = int(math.ceil((self.local_x_max - self.local_x_min) / self.cost_map_resolution))
        self.local_map_h = int(math.ceil((self.local_y_max - self.local_y_min) / self.cost_map_resolution))

        self.global_map_w = int(math.ceil((self.global_x_max - self.global_x_min) / self.global_cost_map_resolution))
        self.global_map_h = int(math.ceil((self.global_y_max - self.global_y_min) / self.global_cost_map_resolution))

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
        self.pose_frame_id = None

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

        self.cost_image = None
        self.cost_image_u8 = None
        self.local_cost_map = None
        self.local_cost_map_valid = None

        self.global_cost_map = np.full(
            (self.global_map_h, self.global_map_w),
            self.unknown_cost,
            dtype=np.float32
        )
        self.global_cost_map_valid = np.zeros((self.global_map_h, self.global_map_w), dtype=bool)
        self.global_cost_map_count = np.zeros((self.global_map_h, self.global_map_w), dtype=np.int32)

        self.rrt_sample_x_min = self.global_x_min
        self.rrt_sample_x_max = self.global_x_max
        self.rrt_sample_y_min = self.global_y_min
        self.rrt_sample_y_max = self.global_y_max

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

        # ---------- TF / Planner ----------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.rrt_star = GlobalRRTStar(
            owner=self,
            step_size=float(self.get_parameter('rrt_step_size').value),
            goal_radius=float(self.get_parameter('rrt_goal_radius').value),
            max_iter=int(self.get_parameter('rrt_max_iter').value),
            search_radius=float(self.get_parameter('rrt_search_radius').value),
            goal_sample_rate=float(self.get_parameter('rrt_goal_sample_rate').value),
        )

        # ---------- ROS I/O ----------
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, qos_best_effort)
        self.sub_rgb = self.create_subscription(Image, self.rgb_topic, self.rgb_callback, qos_best_effort)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.depth_callback, qos_best_effort)

        self.sub_color_info = self.create_subscription(CameraInfo, self.color_info_topic, self.color_info_callback, qos_best_effort)
        self.sub_depth_info = self.create_subscription(CameraInfo, self.depth_info_topic, self.depth_info_callback, qos_best_effort)

        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        self.cost_image_pub = self.create_publisher(Image, self.cost_image_topic, 10)
        self.cost_image_overlay_pub = self.create_publisher(Image, self.cost_image_overlay_topic, 10)
        self.cost_map_pub = self.create_publisher(OccupancyGrid, self.cost_map_topic, 10)
        self.global_cost_map_pub = self.create_publisher(OccupancyGrid, self.global_cost_map_topic, 10)

        self.timer = self.create_timer(self.replan_period_sec, self.plan_cycle)

        self.get_logger().info("Node started. Goal generation mode kept unchanged: fixed goal from initial odom pose.")

    # =========================
    # Callbacks
    # =========================

    def odom_callback(self, msg: Odometry) -> None:
        self.x = float(msg.pose.pose.position.x)
        self.y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.pose_frame_id = msg.header.frame_id if msg.header.frame_id else self.path_frame_id
        self.odom_received = True

        if self.pose_frame_id != self.path_frame_id:
            self.get_logger().warn(
                f"Odometry frame is '{self.pose_frame_id}', but path_frame_id is '{self.path_frame_id}'. "
                f"This code assumes odometry is already in the global planning frame."
            )

        # 保持原逻辑：第一次收到 odom 时固定终点
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
    def bilinear_sample(
        image: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        valid_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        inside = (u >= 0.0) & (u <= (w - 1)) & (v >= 0.0) & (v <= (h - 1))

        values = np.zeros(u.shape, dtype=np.float32)
        ok = np.zeros(u.shape, dtype=bool)
        if not np.any(inside):
            return values, ok

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
            values[inside] = sampled
            ok[inside] = True
            return values, ok

        m00 = valid_mask[y0, x0].astype(np.float32)
        m10 = valid_mask[y0, x1].astype(np.float32)
        m01 = valid_mask[y1, x0].astype(np.float32)
        m11 = valid_mask[y1, x1].astype(np.float32)

        denom = w00 * m00 + w10 * m10 + w01 * m01 + w11 * m11
        good = denom > 1e-6

        sampled = np.zeros(u_in.shape, dtype=np.float32)
        sampled[good] = (
            w00[good] * i00[good] * m00[good] +
            w10[good] * i10[good] * m10[good] +
            w01[good] * i01[good] * m01[good] +
            w11[good] * i11[good] * m11[good]
        ) / denom[good]

        values[inside] = sampled
        ok_inside = np.zeros(u_in.shape, dtype=bool)
        ok_inside[good] = True
        ok[inside] = ok_inside
        return values, ok

    # =========================
    # cost_image
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
        mono_msg.header.stamp = self.get_clock().now().to_msg()
        mono_msg.header.frame_id = self.color_frame if self.color_frame is not None else ''
        self.cost_image_pub.publish(mono_msg)

        colored = cv2.applyColorMap(self.cost_image_u8, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(rgb_image, 0.45, colored, 0.55, 0.0)
        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='rgb8')
        overlay_msg.header.stamp = mono_msg.header.stamp
        overlay_msg.header.frame_id = mono_msg.header.frame_id
        self.cost_image_overlay_pub.publish(overlay_msg)

    # =========================
    # local cost_map
    # =========================

    def depth_to_meters(self, depth: np.ndarray) -> np.ndarray:
        if depth.dtype == np.uint16:
            return depth.astype(np.float32) * self.depth_scale
        return depth.astype(np.float32)

    def smooth_local_cost_map(self) -> None:
        if self.local_cost_map is None or self.local_cost_map_valid is None:
            return
        if self.smooth_kernel <= 1:
            return

        k = self.smooth_kernel
        src = self.local_cost_map.astype(np.float32)
        mask = self.local_cost_map_valid.astype(np.float32)

        num = cv2.GaussianBlur(src * mask, (k, k), 0)
        den = cv2.GaussianBlur(mask, (k, k), 0)

        good = den > 1e-6
        out = src.copy()
        out[good] = num[good] / den[good]
        self.local_cost_map = out

    def build_local_cost_map_from_rgbd(self, depth_image: np.ndarray) -> None:
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

        xs = self.local_x_min + (np.arange(self.local_map_w) + 0.5) * self.cost_map_resolution
        ys = self.local_y_min + (np.arange(self.local_map_h) + 0.5) * self.cost_map_resolution
        xx, yy = np.meshgrid(xs, ys)
        zz = np.zeros_like(xx, dtype=np.float32)

        pts_base = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float64)

        # base -> color optical
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

        # base -> depth optical
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

        valid_flat = (front_color & in_color).copy()
        cost_flat = np.full((pts_base.shape[0],), self.unknown_cost, dtype=np.float32)

        geo_mask = front_color & in_color & front_depth & in_depth
        if np.any(geo_mask):
            semantic_vals, semantic_ok = self.bilinear_sample(
                self.cost_image,
                u_color[geo_mask],
                v_color[geo_mask],
                valid_mask=None,
            )

            depth_obs, depth_ok = self.bilinear_sample(
                depth_m,
                u_depth[geo_mask],
                v_depth[geo_mask],
                valid_mask=depth_valid_pixels,
            )

            z_pred = pts_depth[geo_mask, 2].astype(np.float32)

            local_cost = np.full(z_pred.shape, self.unknown_cost, dtype=np.float32)

            consistent = depth_ok & semantic_ok & (np.abs(depth_obs - z_pred) <= self.depth_consistency_tol)
            occluded = depth_ok & ((depth_obs + self.depth_occlusion_tol) < z_pred)

            local_cost[consistent] = semantic_vals[consistent]
            local_cost[occluded] = 1.0

            cost_flat[geo_mask] = local_cost

        self.local_cost_map = cost_flat.reshape(self.local_map_h, self.local_map_w)
        self.local_cost_map_valid = valid_flat.reshape(self.local_map_h, self.local_map_w)

        self._mark_robot_origin_clear()
        self.smooth_local_cost_map()

    def _mark_robot_origin_clear(self) -> None:
        if self.local_cost_map is None or self.local_cost_map_valid is None:
            return

        xs = self.local_x_min + (np.arange(self.local_map_w) + 0.5) * self.cost_map_resolution
        ys = self.local_y_min + (np.arange(self.local_map_h) + 0.5) * self.cost_map_resolution
        xx, yy = np.meshgrid(xs, ys)
        bubble = (xx ** 2 + yy ** 2) <= (self.origin_clear_radius ** 2)

        self.local_cost_map[bubble] = np.minimum(self.local_cost_map[bubble], 0.05)
        self.local_cost_map_valid[bubble] = True

    def publish_local_cost_map(self) -> None:
        if self.local_cost_map is None or self.local_cost_map_valid is None or not self.odom_received:
            return

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.cost_map_frame_id

        msg.info.resolution = float(self.cost_map_resolution)
        msg.info.width = self.local_map_w
        msg.info.height = self.local_map_h

        ox, oy = self.robot_to_odom_xy(self.local_x_min, self.local_y_min)
        msg.info.origin.position.x = ox
        msg.info.origin.position.y = oy
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation = quaternion_msg_from_yaw(self.yaw)

        occ = np.full((self.local_map_h, self.local_map_w), -1, dtype=np.int8)
        occ[self.local_cost_map_valid] = np.clip(
            np.round(self.local_cost_map[self.local_cost_map_valid] * 100.0), 0, 100
        ).astype(np.int8)

        msg.data = occ.flatten().tolist()
        self.cost_map_pub.publish(msg)

    # =========================
    # global cost_map
    # =========================

    def odom_to_global_map_index(self, x_o: float, y_o: float) -> Tuple[bool, int, int]:
        ix = int(math.floor((x_o - self.global_x_min) / self.global_cost_map_resolution))
        iy = int(math.floor((y_o - self.global_y_min) / self.global_cost_map_resolution))
        inside = (0 <= ix < self.global_map_w) and (0 <= iy < self.global_map_h)
        return inside, ix, iy

    def lookup_global_cost_map(self, x_o: float, y_o: float) -> Tuple[bool, float]:
        inside, ix, iy = self.odom_to_global_map_index(x_o, y_o)
        if not inside:
            return False, 1.0

        if not self.global_cost_map_valid[iy, ix]:
            if self.global_unknown_is_obstacle:
                return False, 1.0
            return True, self.unknown_cost

        return True, float(self.global_cost_map[iy, ix])

    def fuse_local_cost_map_into_global(self) -> None:
        if self.local_cost_map is None or self.local_cost_map_valid is None:
            return
        if not self.odom_received:
            return

        ys_idx, xs_idx = np.where(self.local_cost_map_valid)
        if xs_idx.size == 0:
            return

        x_r = self.local_x_min + (xs_idx + 0.5) * self.cost_map_resolution
        y_r = self.local_y_min + (ys_idx + 0.5) * self.cost_map_resolution
        local_costs = self.local_cost_map[ys_idx, xs_idx]

        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        x_o = self.x + c * x_r - s * y_r
        y_o = self.y + s * x_r + c * y_r

        for xo, yo, lc in zip(x_o, y_o, local_costs):
            inside, ix, iy = self.odom_to_global_map_index(float(xo), float(yo))
            if not inside:
                continue

            if not self.global_cost_map_valid[iy, ix]:
                self.global_cost_map[iy, ix] = float(lc)
                self.global_cost_map_valid[iy, ix] = True
                self.global_cost_map_count[iy, ix] = 1
                continue

            if self.global_fusion_mode == 'overwrite':
                self.global_cost_map[iy, ix] = float(lc)
                self.global_cost_map_count[iy, ix] += 1

            elif self.global_fusion_mode == 'mean':
                n = int(self.global_cost_map_count[iy, ix])
                old = float(self.global_cost_map[iy, ix])
                new = (old * n + float(lc)) / float(n + 1)
                self.global_cost_map[iy, ix] = new
                self.global_cost_map_count[iy, ix] = n + 1

            else:  # max
                self.global_cost_map[iy, ix] = max(float(self.global_cost_map[iy, ix]), float(lc))
                self.global_cost_map_count[iy, ix] += 1

    def publish_global_cost_map(self) -> None:
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.path_frame_id

        msg.info.resolution = float(self.global_cost_map_resolution)
        msg.info.width = self.global_map_w
        msg.info.height = self.global_map_h

        msg.info.origin.position.x = self.global_x_min
        msg.info.origin.position.y = self.global_y_min
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        occ = np.full((self.global_map_h, self.global_map_w), -1, dtype=np.int8)
        occ[self.global_cost_map_valid] = np.clip(
            np.round(self.global_cost_map[self.global_cost_map_valid] * 100.0), 0, 100
        ).astype(np.int8)

        msg.data = occ.flatten().tolist()
        self.global_cost_map_pub.publish(msg)

    # =========================
    # Goal handling
    # =========================

    def find_valid_global_goal_near(self, x_o: float, y_o: float) -> Optional[Tuple[float, float]]:
        half = 0.5 * self.global_cost_map_resolution
        x_o = min(max(x_o, self.global_x_min + half), self.global_x_max - half)
        y_o = min(max(y_o, self.global_y_min + half), self.global_y_max - half)

        valid, cost = self.lookup_global_cost_map(x_o, y_o)
        if valid and cost <= self.block_cost_threshold:
            return x_o, y_o

        max_cells = int(math.ceil(self.goal_search_radius / self.global_cost_map_resolution))
        inside, center_ix, center_iy = self.odom_to_global_map_index(x_o, y_o)
        if not inside:
            return None

        best = None
        best_dist = float('inf')

        for dy in range(-max_cells, max_cells + 1):
            for dx in range(-max_cells, max_cells + 1):
                ix = center_ix + dx
                iy = center_iy + dy

                if ix < 0 or ix >= self.global_map_w or iy < 0 or iy >= self.global_map_h:
                    continue

                if not self.global_cost_map_valid[iy, ix]:
                    continue

                c = float(self.global_cost_map[iy, ix])
                if c > self.block_cost_threshold:
                    continue

                cx = self.global_x_min + (ix + 0.5) * self.global_cost_map_resolution
                cy = self.global_y_min + (iy + 0.5) * self.global_cost_map_resolution

                d = math.hypot(cx - x_o, cy - y_o)
                if d < best_dist:
                    best = (cx, cy)
                    best_dist = d

        return best

    def set_rrt_sampling_bounds(self, start_xy: Tuple[float, float], goal_xy: Tuple[float, float]) -> None:
        self.rrt_sample_x_min = max(self.global_x_min, min(start_xy[0], goal_xy[0]) - self.rrt_sampling_margin)
        self.rrt_sample_x_max = min(self.global_x_max, max(start_xy[0], goal_xy[0]) + self.rrt_sampling_margin)
        self.rrt_sample_y_min = max(self.global_y_min, min(start_xy[1], goal_xy[1]) - self.rrt_sampling_margin)
        self.rrt_sample_y_max = min(self.global_y_max, max(start_xy[1], goal_xy[1]) + self.rrt_sampling_margin)

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
                if len(path_odom) >= 2:
                    px, py = path_odom[-2]
                    yaw = math.atan2(y - py, x - px)
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

        # 1) 每帧语义图像代价
        self.build_cost_image(rgb_image)
        self.publish_cost_image(rgb_image)

        # 2) 当前帧 local cost map
        self.build_local_cost_map_from_rgbd(depth_image)
        self.publish_local_cost_map()

        # 3) 累积成 global cost map
        self.fuse_local_cost_map_into_global()
        self.publish_global_cost_map()

        # 4) 起点 = 当前机器人位置
        start_odom = (self.x, self.y)

        # 5) 终点 = 原方式生成的 fixed_goal_odom
        if self.fixed_goal_odom is None:
            self.get_logger().warn("Fixed goal is not initialized.")
            return

        goal_odom = self.find_valid_global_goal_near(
            self.fixed_goal_odom[0],
            self.fixed_goal_odom[1]
        )
        if goal_odom is None:
            self.get_logger().warn("No valid goal found near fixed endpoint in global cost_map.")
            return

        dist_to_goal = math.hypot(start_odom[0] - goal_odom[0], start_odom[1] - goal_odom[1])
        if dist_to_goal <= self.goal_reached_dist:
            self.publish_path([start_odom, goal_odom])
            self.get_logger().info("Goal already reached or very close.")
            return

        # 6) 在起点到终点附近采样
        self.set_rrt_sampling_bounds(start_odom, goal_odom)

        # 7) 在 global cost map 上做 RRT*
        path_odom = self.rrt_star.plan(start_odom, goal_odom)
        if path_odom is None or len(path_odom) < 2:
            self.get_logger().warn("RRT* failed on accumulated global cost_map.")
            return

        # 8) 发布路径
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