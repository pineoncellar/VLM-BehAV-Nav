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

import tf2_ros
from rclpy.time import Time

from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
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
    # standard formula
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
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=np.float64)


def rotz(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr,  cr]], dtype=np.float64)
    ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]], dtype=np.float64)
    return rz @ ry @ rx


# Standard ROS optical frame -> robot base frame
# optical: x right, y down, z forward
# robot:   x forward, y left, z up
R_ROBOT_FROM_OPTICAL = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
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
        self.declare_parameter('path_overlay_topic', '/path_overlay_image')


        self.declare_parameter('path_frame_id', 'odom')
        self.declare_parameter('cost_map_frame_id', 'odom')

        # ---------- Fixed goal ----------
        # Fixed once from initial robot pose + relative command
        self.declare_parameter('goal_radius', 10.0)
        self.declare_parameter('goal_theta_deg', 0.0)

        # ---------- CLIPSeg ----------
        self.declare_parameter('clipseg_dir', '/home/zyy/nvidia/models/clipseg-rd64-refined')
        self.declare_parameter('prompts', ['vegetation', 'pavement', 'grass', 'person'])
        self.declare_parameter('cost_values', [0.90, 0.05, 0.20, 1.00])
        self.declare_parameter('prompt_threshold', 0.10)
        self.declare_parameter('unknown_cost', 0.60)

        # ---------- RGBD camera intrinsics ----------
        self.declare_parameter('fx', 607.175048828125)
        self.declare_parameter('fy', 607.222900390625)
        self.declare_parameter('cx', 322.55340576171875)
        self.declare_parameter('cy', 248.86021423339844)

        # camera pose in robot frame
        self.declare_parameter('camera_x', 0.0)
        self.declare_parameter('camera_y', 0.0)
        self.declare_parameter('camera_z', 0.59)
        self.declare_parameter('camera_roll_deg', 0.0)
        self.declare_parameter('camera_pitch_deg', 0.0)
        self.declare_parameter('camera_yaw_deg', 0.0)

        # ---------- Depth params ----------
        self.declare_parameter('depth_scale', 0.001)   # uint16 mm -> m
        self.declare_parameter('min_depth', 0.2)
        self.declare_parameter('max_depth', 6.0)
        self.declare_parameter('pixel_stride', 4)

        # ---------- Local cost_map ----------
        self.declare_parameter('local_x_min', -0.20)
        self.declare_parameter('local_x_max', 4.00)
        self.declare_parameter('local_y_min', -2.50)
        self.declare_parameter('local_y_max',  2.50)
        self.declare_parameter('cost_map_resolution', 0.05)

        self.declare_parameter('min_z_for_map', -0.30)
        self.declare_parameter('max_z_for_map',  1.50)
        self.declare_parameter('origin_clear_radius', 0.25)
        self.declare_parameter('high_cost_lock_threshold', 0.90)
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
        self.path_overlay_topic = self.get_parameter('path_overlay_topic').value

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

        self.fx = float(self.get_parameter('fx').value)
        self.fy = float(self.get_parameter('fy').value)
        self.cx = float(self.get_parameter('cx').value)
        self.cy = float(self.get_parameter('cy').value)

        self.camera_x = float(self.get_parameter('camera_x').value)
        self.camera_y = float(self.get_parameter('camera_y').value)
        self.camera_z = float(self.get_parameter('camera_z').value)
        self.camera_roll = math.radians(float(self.get_parameter('camera_roll_deg').value))
        self.camera_pitch = math.radians(float(self.get_parameter('camera_pitch_deg').value))
        self.camera_yaw = math.radians(float(self.get_parameter('camera_yaw_deg').value))

        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.min_depth = float(self.get_parameter('min_depth').value)
        self.max_depth = float(self.get_parameter('max_depth').value)
        self.pixel_stride = int(self.get_parameter('pixel_stride').value)

        self.local_x_min = float(self.get_parameter('local_x_min').value)
        self.local_x_max = float(self.get_parameter('local_x_max').value)
        self.local_y_min = float(self.get_parameter('local_y_min').value)
        self.local_y_max = float(self.get_parameter('local_y_max').value)
        self.cost_map_resolution = float(self.get_parameter('cost_map_resolution').value)

        self.min_z_for_map = float(self.get_parameter('min_z_for_map').value)
        self.max_z_for_map = float(self.get_parameter('max_z_for_map').value)
        self.origin_clear_radius = float(self.get_parameter('origin_clear_radius').value)
        self.high_cost_lock_threshold = float(self.get_parameter('high_cost_lock_threshold').value)
        self.goal_search_radius = float(self.get_parameter('goal_search_radius').value)

        self.semantic_weight = float(self.get_parameter('semantic_weight').value)
        self.block_cost_threshold = float(self.get_parameter('block_cost_threshold').value)
        self.replan_period_sec = float(self.get_parameter('replan_period_sec').value)

        self.map_w = int(math.ceil((self.local_x_max - self.local_x_min) / self.cost_map_resolution))
        self.map_h = int(math.ceil((self.local_y_max - self.local_y_min) / self.cost_map_resolution))

        # ---------- Camera extrinsics ----------
        r_mount = rotation_matrix_from_rpy(self.camera_roll, self.camera_pitch, self.camera_yaw)
        self.R_robot_from_cam = r_mount @ R_ROBOT_FROM_OPTICAL
        self.t_robot_from_cam = np.array([self.camera_x, self.camera_y, self.camera_z], dtype=np.float64)

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

        self.fixed_goal_odom = None  # (x, y) fixed once

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

        # Strict distinction:
        self.cost_image = None          # float32 [0,1], image coordinates
        self.cost_image_u8 = None       # uint8 [0,255], for display
        self.cost_map = None            # float32 [0,1], robot/map coordinates
        self.cost_map_valid = None      # bool valid observed mask

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
        self.path_overlay_pub = self.create_publisher(Image, self.path_overlay_topic, 10)

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
            self.get_logger().info(
                f"Fixed goal initialized in odom: ({gx_o:.2f}, {gy_o:.2f})"
            )

    def rgb_callback(self, msg: Image) -> None:
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.rgb_received = True
        except Exception as e:
            self.get_logger().error(f"RGB callback failed: {e}")

    def depth_callback(self, msg: Image) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth = depth
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
    # Coordinate transforms
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
        overlay = cv2.addWeighted(rgb_image, 0.45, colored, 0.55, 0.0)
        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='rgb8')
        self.cost_image_overlay_pub.publish(overlay_msg)

    # =========================
    # cost_map: map coordinates
    # =========================

    def depth_to_meters(self, depth: np.ndarray) -> np.ndarray:
        if depth.dtype == np.uint16:
            return depth.astype(np.float32) * self.depth_scale
        return depth.astype(np.float32)

    def build_cost_map_from_rgbd(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> None:
        if self.cost_image is None:
            return

        if not (self.color_info_received and self.depth_info_received):
            self.get_logger().warn("Waiting for color/depth camera_info.")
            return

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.color_frame,   # target frame
                self.depth_frame,   # source frame
                Time()
            )
        except Exception as e:
            self.get_logger().warn(f"Waiting for TF depth->color: {e}")
            return

        try:
            tf_msg_base = self.tf_buffer.lookup_transform(
                'base_footprint',        # target frame
                self.depth_frame,   # source frame
                Time()
            )
        except Exception as e:
            self.get_logger().warn(f"Waiting for TF depth->base_footprint: {e}")
            return

        R_cd = rotation_matrix_from_quaternion_msg(tf_msg.transform.rotation)
        t_cd = np.array([
            tf_msg.transform.translation.x,
            tf_msg.transform.translation.y,
            tf_msg.transform.translation.z
        ], dtype=np.float64)

        R_db = rotation_matrix_from_quaternion_msg(tf_msg_base.transform.rotation)
        t_db = np.array([
            tf_msg_base.transform.translation.x,
            tf_msg_base.transform.translation.y,
            tf_msg_base.transform.translation.z
        ], dtype=np.float64)

        h, w = depth_image.shape[:2]
        depth_m = self.depth_to_meters(depth_image)

        us = np.arange(0, w, self.pixel_stride, dtype=np.int32)
        vs = np.arange(0, h, self.pixel_stride, dtype=np.int32)
        uu, vv = np.meshgrid(us, vs)

        d = depth_m[vv, uu]

        valid = np.isfinite(d)
        valid &= d > self.min_depth
        valid &= d < self.max_depth

        if not np.any(valid):
            self.cost_map = np.full((self.map_h, self.map_w), self.unknown_cost, dtype=np.float32)
            self.cost_map_valid = np.zeros((self.map_h, self.map_w), dtype=bool)
            self._mark_robot_origin_clear()
            return

        u_d = uu[valid].astype(np.float32)
        v_d = vv[valid].astype(np.float32)
        z_d = d[valid].astype(np.float32)

        # 1) depth 像素 -> depth optical 3D
        x_d = (u_d - self.depth_cx) * z_d / self.depth_fx
        y_d = (v_d - self.depth_cy) * z_d / self.depth_fy
        pts_d = np.stack([x_d, y_d, z_d], axis=1)

        # 2) depth optical -> color optical
        pts_c = (R_cd @ pts_d.T).T + t_cd[None, :]

        z_c = pts_c[:, 2]
        keep = np.isfinite(z_c) & (z_c > 1e-6)
        pts_d = pts_d[keep]
        pts_c = pts_c[keep]

        if pts_c.shape[0] == 0:
            self.cost_map = np.full((self.map_h, self.map_w), self.unknown_cost, dtype=np.float32)
            self.cost_map_valid = np.zeros((self.map_h, self.map_w), dtype=bool)
            self._mark_robot_origin_clear()
            return

        # 3) color optical -> color image pixel
        u_c = self.color_fx * (pts_c[:, 0] / pts_c[:, 2]) + self.color_cx
        v_c = self.color_fy * (pts_c[:, 1] / pts_c[:, 2]) + self.color_cy

        uc_i = np.round(u_c).astype(np.int32)
        vc_i = np.round(v_c).astype(np.int32)

        ch, cw = self.cost_image.shape[:2]
        keep = (
            (uc_i >= 0) & (uc_i < cw) &
            (vc_i >= 0) & (vc_i < ch)
        )

        pts_d = pts_d[keep]
        uc_i = uc_i[keep]
        vc_i = vc_i[keep]

        if pts_d.shape[0] == 0:
            self.cost_map = np.full((self.map_h, self.map_w), self.unknown_cost, dtype=np.float32)
            self.cost_map_valid = np.zeros((self.map_h, self.map_w), dtype=bool)
            self._mark_robot_origin_clear()
            return

        # 4) 在 color 图像上取 CLIPSeg 代价
        cost_vals = self.cost_image[vc_i, uc_i].astype(np.float32)

        # 5) depth optical -> robot
        # 5) depth optical -> base_link
        pts_base = (R_db @ pts_d.T).T + t_db[None, :]

        xr = pts_base[:, 0]
        yr = pts_base[:, 1]
        zr = pts_base[:, 2]

        keep = np.isfinite(xr) & np.isfinite(yr) & np.isfinite(zr)
        keep &= xr >= self.local_x_min
        keep &= xr < self.local_x_max
        keep &= yr >= self.local_y_min
        keep &= yr < self.local_y_max
        keep &= zr >= self.min_z_for_map
        keep &= zr <= self.max_z_for_map

        xr = xr[keep]
        yr = yr[keep]
        cost_vals = cost_vals[keep]

        sum_grid = np.zeros((self.map_h, self.map_w), dtype=np.float32)
        cnt_grid = np.zeros((self.map_h, self.map_w), dtype=np.int32)
        max_grid = np.zeros((self.map_h, self.map_w), dtype=np.float32)

        if xr.size > 0:
            ix = np.floor((xr - self.local_x_min) / self.cost_map_resolution).astype(np.int32)
            iy = np.floor((yr - self.local_y_min) / self.cost_map_resolution).astype(np.int32)

            inside = (ix >= 0) & (ix < self.map_w) & (iy >= 0) & (iy < self.map_h)
            ix = ix[inside]
            iy = iy[inside]
            cost_vals = cost_vals[inside]

            np.add.at(sum_grid, (iy, ix), cost_vals)
            np.add.at(cnt_grid, (iy, ix), 1)
            np.maximum.at(max_grid, (iy, ix), cost_vals)

        cost_map = np.full((self.map_h, self.map_w), self.unknown_cost, dtype=np.float32)
        valid_map = cnt_grid > 0

        if np.any(valid_map):
            avg = np.zeros_like(cost_map)
            avg[valid_map] = sum_grid[valid_map] / cnt_grid[valid_map]
            cost_map[valid_map] = avg[valid_map]

            lock_mask = max_grid >= self.high_cost_lock_threshold
            cost_map[lock_mask] = np.maximum(cost_map[lock_mask], max_grid[lock_mask])

        self.cost_map = cost_map
        self.cost_map_valid = valid_map
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

        # OccupancyGrid origin is the lower-left corner of local grid in odom frame.
        ox, oy = self.robot_to_odom_xy(self.local_x_min, self.local_y_min)
        msg.info.origin.position.x = ox
        msg.info.origin.position.y = oy
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation = quaternion_msg_from_yaw(self.yaw)

        occ = np.full((self.map_h, self.map_w), -1, dtype=np.int8)
        valid = self.cost_map_valid
        occ[valid] = np.clip(np.round(self.cost_map[valid] * 100.0), 0, 100).astype(np.int8)

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
            return True, self.unknown_cost

        return True, float(self.cost_map[iy, ix])

    # =========================
    # Goal handling
    # =========================

    def choose_planning_goal_robot(self) -> Optional[Tuple[float, float]]:
        if self.fixed_goal_odom is None:
            return None

        gx_r, gy_r = self.odom_to_robot_xy(self.fixed_goal_odom[0], self.fixed_goal_odom[1])

        # If the fixed goal lies inside local map, try to use it directly.
        desired_x, desired_y = gx_r, gy_r

        # If it is outside local map, clip along the ray from robot origin.
        inside = (
            self.local_x_min <= gx_r <= self.local_x_max and
            self.local_y_min <= gy_r <= self.local_y_max
        )

        if not inside:
            if gx_r <= 1e-6:
                self.get_logger().warn("Fixed goal is behind robot / outside front local map.")
                return None

            scales = []
            if gx_r > 0:
                scales.append((self.local_x_max - 0.10) / gx_r)
            if gy_r > 0:
                scales.append((self.local_y_max - 0.10) / gy_r)
            elif gy_r < 0:
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
    # Visualization
    # =========================

    def publish_path_overlay(self, rgb_image: np.ndarray, path_robot: List[Tuple[float, float]]) -> None:
        if self.cost_image_u8 is None:
            return

        overlay = rgb_image.copy()
        colored = cv2.applyColorMap(self.cost_image_u8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(overlay, 0.45, colored, 0.55, 0.0)

        pts_robot = np.array(path_robot, dtype=np.float32)
        if pts_robot.size == 0:
            return

        # Convert robot-frame path points to camera frame, then project to image
        # inverse of pts_robot = R_robot_from_cam @ pts_cam + t
        pts_cam = (np.linalg.inv(self.R_robot_from_cam) @ (pts_robot_to_3d(pts_robot).T - self.t_robot_from_cam[:, None])).T

        z = pts_cam[:, 2]
        front = z > 1e-6
        pts_cam = pts_cam[front]
        if pts_cam.shape[0] == 0:
            return

        u = self.fx * (pts_cam[:, 0] / pts_cam[:, 2]) + self.cx
        v = self.fy * (pts_cam[:, 1] / pts_cam[:, 2]) + self.cy

        xpix = np.round(u).astype(np.int32)
        ypix = np.round(v).astype(np.int32)

        valid = (xpix >= 0) & (xpix < overlay.shape[1]) & (ypix >= 0) & (ypix < overlay.shape[0])
        pts = np.stack([xpix[valid], ypix[valid]], axis=1) if np.any(valid) else np.empty((0, 2), dtype=np.int32)

        if len(pts) >= 2:
            cv2.polylines(overlay, [pts], isClosed=False, color=(255, 255, 255), thickness=3)
        for p in pts:
            cv2.circle(overlay, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)

        msg = self.bridge.cv2_to_imgmsg(overlay, encoding='rgb8')
        self.path_overlay_pub.publish(msg)

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

        # 1) cost_image in image coordinates
        self.build_cost_image(rgb_image)
        self.publish_cost_image(rgb_image)

        # 2) cost_map in coordinate/map frame, using depth
        if depth_image.ndim == 3:
            depth_image = depth_image[:, :, 0]
        self.build_cost_map_from_rgbd(rgb_image, depth_image)
        self.publish_cost_map()

        # 3) choose planning goal in robot frame from fixed odom goal
        goal_robot = self.choose_planning_goal_robot()
        if goal_robot is None:
            self.get_logger().warn("No valid local planning goal found.")
            return

        # 4) RRT* on cost_map
        start_robot = (0.0, 0.0)
        path_robot = self.rrt_star.plan(start_robot, goal_robot)
        if path_robot is None or len(path_robot) < 2:
            self.get_logger().warn("RRT* failed on current cost_map.")
            return

        # 5) publish path in odom
        path_odom = self.path_robot_to_odom(path_robot)
        self.publish_path(path_odom)

        # Optional image visualization
        # self.publish_path_overlay(rgb_image, path_robot)


def pts_robot_to_3d(pts_robot_xy: np.ndarray) -> np.ndarray:
    """Lift robot-frame 2D path points to z=0 for image projection."""
    return np.column_stack([pts_robot_xy[:, 0], pts_robot_xy[:, 1], np.zeros(pts_robot_xy.shape[0], dtype=np.float32)])


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


