#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from scipy.spatial import KDTree
from sensor_msgs.msg import Image, PointCloud2
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

import sensor_msgs_py.point_cloud2 as pc2
import casadi as ca

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


@dataclass
class MPCConfig:
    """
    全局配置。
    这个类先把后面所有模块会用到的参数集中起来，避免参数散落在 Node 里。
    """

    # ---------- robot / planner ----------
    goal_reach_threshold: float = 0.7
    sensing_range: float = 3.5
    safe_dist_threshold: float = 0.5
    robot_radius: float = 0.35

    min_obstacle_height: float = -0.3
    max_obstacle_height: float = 1.2

    # ---------- camera / projection ----------
    projection_matrix: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [607.175048828125, 0.0, 322.55340576171875, 0.0],
                [0.0, 607.222900390625, 248.86021423339844, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    camera_height: float = 0.59
    camera_tilt_angle_deg: float = 0.0
    camera_offset_x: float = 0.0
    camera_offset_y: float = 0.0

    # ---------- clipseg ----------
    enable_clipseg: bool = True
    clipseg_dir: str = "/home/zyy/nvidia/models/clipseg-rd64-refined"
    prompts: List[str] = field(
        default_factory=lambda: ["vegetation", "Pavement", "grass", "Stop gesture"]
    )

    # 这里先保留你原来的语义代价风格
    semantic_class_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "vegetation": 0.05,
            "Pavement": 0.05,
            "grass": 0.90,
            "Stop gesture": 0.0,
        }   
    )

    # ---------- acados / mpc ----------
    horizon_steps: int = 10
    dt: float = 0.2

    # state = [x, y, yaw, v, w]
    # control = [a_v, a_w]
    v_min: float = 0.0
    v_max: float = 0.8
    w_min: float = -0.7
    w_max: float = 0.7

    a_v_min: float = -1.0
    a_v_max: float = 1.0
    a_w_min: float = -2.0
    a_w_max: float = 2.0

    # ---------- anchor settings ----------
    obstacle_anchor_count: int = 12
    semantic_anchor_count: int = 16
    sigma_obs: float = 0.35
    sigma_sem: float = 0.45

    semantic_x_samples: List[float] = field(
        default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    )
    semantic_y_samples: List[float] = field(
        default_factory=lambda: [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    )

    # ---------- cost weights ----------
    q_goal_stage: float = 0.15
    q_goal_terminal: float = 12.0
    q_goal_terminal_yaw: float = 1.5
    q_sem: float = 40.0

    r_v: float = 0.02
    r_w: float = 0.03
    r_a_v: float = 0.05
    r_a_w: float = 0.08

    # ---------- topics ----------
    odom_topic: str = "/lio_sam/mapping/odometry"
    pointcloud_topic: str = "/lio_sam/mapping/cloud_registered"
    image_topic: str = "/color/image_raw"
    cmd_vel_topic: str = "/cmd_vel"
    behav_costmap_topic: str = "/behav_costmap"

    # ---------- qos ----------
    @property
    def best_effort_qos(self) -> QoSProfile:
        return QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

    @property
    def reliable_qos(self) -> QoSProfile:
        return QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

class GeometryProjectionUtils:
    """
    几何、坐标变换、相机投影工具类。

    这个类不依赖 ROS Node，本质上只是把后面 MPC / 感知 / 可视化都会反复用到的
    数学操作集中起来，避免主类里全是几何细节。
    """

    def __init__(self, config: MPCConfig):
        self.cfg = config

    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def euler_from_quaternion(quaternion: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
        x, y, z, w = quaternion

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = max(min(t2, 1.0), -1.0)
        pitch = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        return roll, pitch, yaw

    @staticmethod
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

    def yaw_from_pose(self, pose: Pose) -> float:
        return self.euler_from_quaternion(
            (
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            )
        )[2]

    def robot_to_odom_points(
        self,
        pts_robot: np.ndarray,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> np.ndarray:
        """
        pts_robot: shape (N, 2), robot frame
        return:    shape (N, 2), odom frame
        """
        pts_robot = np.asarray(pts_robot, dtype=np.float64)
        if pts_robot.size == 0:
            return np.zeros((0, 2), dtype=np.float64)

        c = math.cos(robot_yaw)
        s = math.sin(robot_yaw)

        x_odom = robot_x + pts_robot[:, 0] * c - pts_robot[:, 1] * s
        y_odom = robot_y + pts_robot[:, 0] * s + pts_robot[:, 1] * c
        return np.column_stack((x_odom, y_odom))

    def odom_to_robot_points(
        self,
        pts_odom: np.ndarray,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> np.ndarray:
        """
        pts_odom: shape (N, 2), odom frame
        return:   shape (N, 2), robot frame
        """
        pts_odom = np.asarray(pts_odom, dtype=np.float64)
        if pts_odom.size == 0:
            return np.zeros((0, 2), dtype=np.float64)

        dx = pts_odom[:, 0] - robot_x
        dy = pts_odom[:, 1] - robot_y

        c = math.cos(robot_yaw)
        s = math.sin(robot_yaw)

        x_robot = dx * c + dy * s
        y_robot = -dx * s + dy * c
        return np.column_stack((x_robot, y_robot))

    def motion_step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """
        state   = [x, y, yaw, v, w]
        control = [a_v, a_w]
        """
        x, y, yaw, v, w = state
        a_v, a_w = control

        v_new = v + a_v * dt
        w_new = w + a_w * dt

        v_new = float(np.clip(v_new, self.cfg.v_min, self.cfg.v_max))
        w_new = float(np.clip(w_new, self.cfg.w_min, self.cfg.w_max))

        yaw_new = self.wrap_to_pi(yaw + w_new * dt)
        x_new = x + v_new * math.cos(yaw_new) * dt
        y_new = y + v_new * math.sin(yaw_new) * dt

        return np.array([x_new, y_new, yaw_new, v_new, w_new], dtype=np.float64)

    def project_robot_points_to_image(
        self,
        pts_robot: np.ndarray,
        img_h: int,
        img_w: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        把机器人坐标系下的 2D 点投影到当前相机图像平面。

        pts_robot: shape (N, 2), each row [x_robot, y_robot]

        return:
            valid_indices: 原始点中成功投影且落在图像内的索引
            x_pix:         对应像素 x
            y_pix:         对应像素 y
        """
        pts_robot = np.asarray(pts_robot, dtype=np.float64)
        if pts_robot.size == 0:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
            )

        x_rob = pts_robot[:, 0]
        y_rob = pts_robot[:, 1]

        # robot frame -> camera-like 3D coordinates
        pts_xyz = np.column_stack(
            (
                -y_rob + self.cfg.camera_offset_y,
                np.full(x_rob.shape, self.cfg.camera_height, dtype=np.float64),
                x_rob - self.cfg.camera_offset_x,
            )
        )

        alpha = np.deg2rad(-self.cfg.camera_tilt_angle_deg)
        rotation_matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(alpha), -math.sin(alpha)],
                [0.0, math.sin(alpha),  math.cos(alpha)],
            ],
            dtype=np.float64,
        )

        pts_rot = pts_xyz @ rotation_matrix.T
        pts_h = np.hstack((pts_rot, np.ones((pts_rot.shape[0], 1), dtype=np.float64)))

        uvw = self.cfg.projection_matrix @ pts_h.T
        u_vec, v_vec, w_vec = uvw[0], uvw[1], uvw[2]

        front_mask = w_vec > 1e-6
        idx_front = np.where(front_mask)[0]
        if len(idx_front) == 0:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
            )

        x_pix = np.round(u_vec[front_mask] / w_vec[front_mask]).astype(np.int32)
        y_pix = np.round(v_vec[front_mask] / w_vec[front_mask]).astype(np.int32)

        valid_mask = (
            (x_pix >= 0) & (x_pix < img_w) &
            (y_pix >= 0) & (y_pix < img_h)
        )

        valid_indices = idx_front[valid_mask]
        return valid_indices, x_pix[valid_mask], y_pix[valid_mask]

class AcadosBehaviorMPC:
    """
    acados + CasADi 的 MPC 求解器封装。

    设计目标：
    1. 状态:   [x, y, yaw, v, w]
    2. 控制:   [a_v, a_w]
    3. 参数 p: [goal_x, goal_y, goal_yaw,
                obs_x..., obs_y..., obs_w...,
                sem_x..., sem_y..., sem_w...]

    其中：
    - obstacle anchors 表示“局部障碍代价场”
    - semantic anchors 表示“局部语义代价场”
    - MPC 在 solver 内部直接优化未来一段控制
    """

    def __init__(self, cfg: MPCConfig):
        self.cfg = cfg

        self.nx = 5   # [x, y, yaw, v, w]
        self.nu = 2   # [a_v, a_w]
        self.n_obs = cfg.obstacle_anchor_count
        self.n_sem = cfg.semantic_anchor_count

        self.np_dim = 3 + 3 * self.n_obs + 3 * self.n_sem

        self.model = self._build_model()
        self.solver = self._build_solver()

        # warm start buffers
        self.x_init_guess = np.zeros((self.cfg.horizon_steps + 1, self.nx), dtype=np.float64)
        self.u_init_guess = np.zeros((self.cfg.horizon_steps, self.nu), dtype=np.float64)

    def _build_model(self) -> AcadosModel:
        model = AcadosModel()
        model.name = "behavior_mpc"

        # ---------- states ----------
        x = ca.SX.sym("x", self.nx)
        px = x[0]
        py = x[1]
        yaw = x[2]
        v = x[3]
        w = x[4]

        # ---------- controls ----------
        u = ca.SX.sym("u", self.nu)
        a_v = u[0]
        a_w = u[1]

        # ---------- parameters ----------
        # p = [goal_x, goal_y, goal_yaw,
        #      obs_x(n_obs), obs_y(n_obs), obs_w(n_obs),
        #      sem_x(n_sem), sem_y(n_sem), sem_w(n_sem)]
        p = ca.SX.sym("p", self.np_dim)

        goal_x = p[0]
        goal_y = p[1]
        goal_yaw = p[2]

        obs_start = 3
        obs_x = p[obs_start : obs_start + self.n_obs]
        obs_y = p[obs_start + self.n_obs : obs_start + 2 * self.n_obs]
        obs_w = p[obs_start + 2 * self.n_obs : obs_start + 3 * self.n_obs]

        sem_start = obs_start + 3 * self.n_obs
        sem_x = p[sem_start : sem_start + self.n_sem]
        sem_y = p[sem_start + self.n_sem : sem_start + 2 * self.n_sem]
        sem_w = p[sem_start + 2 * self.n_sem : sem_start + 3 * self.n_sem]

        # ---------- continuous dynamics ----------
        xdot_expr = ca.vertcat(
            v * ca.cos(yaw),
            v * ca.sin(yaw),
            w,
            a_v,
            a_w,
        )

        xdot = ca.SX.sym("xdot", self.nx)

        # ---------- stage cost ----------
        dx = px - goal_x
        dy = py - goal_y
        goal_stage_cost = self.cfg.q_goal_stage * (dx * dx + dy * dy)

        # 状态/控制平滑与幅值惩罚
        # 这里惩罚 v,w,a_v,a_w，避免控制过于激进
        regularization_cost = (
            self.cfg.r_v * v * v +
            self.cfg.r_w * w * w +
            self.cfg.r_a_v * a_v * a_v +
            self.cfg.r_a_w * a_w * a_w
        )

        # ---------- obstacle anchor cost ----------
        # 用 RBF/高斯形式构造平滑障碍代价场
        obs_cost = 0
        for i in range(self.n_obs):
            d2 = (px - obs_x[i]) ** 2 + (py - obs_y[i]) ** 2
            obs_cost += obs_w[i] * ca.exp(-d2 / (2.0 * self.cfg.sigma_obs ** 2))

        # ---------- semantic anchor cost ----------
        sem_cost = 0
        for i in range(self.n_sem):
            d2 = (px - sem_x[i]) ** 2 + (py - sem_y[i]) ** 2
            sem_cost += sem_w[i] * ca.exp(-d2 / (2.0 * self.cfg.sigma_sem ** 2))

        
        stage_cost = goal_stage_cost + regularization_cost + obs_cost + self.cfg.q_sem * sem_cost


        # ---------- terminal cost ----------
        # 终端位置强收敛
        terminal_goal_cost = self.cfg.q_goal_terminal * (dx * dx + dy * dy)

        # 终端姿态误差
        # 用 1-cos(yaw error) 比 angle^2 更平滑、周期性更自然
        terminal_yaw_cost = self.cfg.q_goal_terminal_yaw * (1.0 - ca.cos(yaw - goal_yaw))

        terminal_cost = terminal_goal_cost + terminal_yaw_cost

        # ---------- bind model ----------
        model.x = x
        model.u = u
        model.p = p
        model.xdot = xdot
        model.f_expl_expr = xdot_expr
        model.f_impl_expr = xdot - xdot_expr

        # acados external cost
        model.cost_expr_ext_cost = stage_cost
        model.cost_expr_ext_cost_e = terminal_cost

        return model

    def _build_solver(self) -> AcadosOcpSolver:
        ocp = AcadosOcp()
        ocp.model = self.model

        N = self.cfg.horizon_steps
        ocp.dims.N = N
        ocp.solver_options.tf = N * self.cfg.dt

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        ocp.parameter_values = np.zeros(self.np_dim, dtype=np.float64)

        # stage state bounds: only v, w
        ocp.constraints.idxbx = np.array([3, 4], dtype=np.int64)
        ocp.constraints.lbx = np.array([self.cfg.v_min, self.cfg.w_min], dtype=np.float64)
        ocp.constraints.ubx = np.array([self.cfg.v_max, self.cfg.w_max], dtype=np.float64)

        # initial state bounds: full x0
        ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        ocp.constraints.lbx_0 = np.zeros(self.nx, dtype=np.float64)
        ocp.constraints.ubx_0 = np.zeros(self.nx, dtype=np.float64)

        # control bounds
        ocp.constraints.idxbu = np.array([0, 1], dtype=np.int64)
        ocp.constraints.lbu = np.array([self.cfg.a_v_min, self.cfg.a_w_min], dtype=np.float64)
        ocp.constraints.ubu = np.array([self.cfg.a_v_max, self.cfg.a_w_max], dtype=np.float64)

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"

        ocp.solver_options.nlp_solver_max_iter = 50
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 1

        solver = AcadosOcpSolver(ocp, json_file="acados_behavior_mpc.json")
        return solver

    def reset_warm_start(self, x0: np.ndarray) -> None:
        """
        当目标变化较大或系统重置时调用。
        """
        self.x_init_guess[:] = 0.0
        self.u_init_guess[:] = 0.0

        if x0 is not None and len(x0) == self.nx:
            self.x_init_guess[0] = x0
            for k in range(1, self.cfg.horizon_steps + 1):
                self.x_init_guess[k] = self.x_init_guess[k - 1]

    def solve(self, x0: np.ndarray, p_vec: np.ndarray) -> Tuple[float, float, int]:
        """
        x0:    当前状态 [x, y, yaw, v, w]
        p_vec: 当前周期参数向量

        return:
            v_cmd, w_cmd, status
        """
        x0 = np.asarray(x0, dtype=np.float64).reshape(self.nx)
        p_vec = np.asarray(p_vec, dtype=np.float64).reshape(self.np_dim)

        # 1. 初始状态约束
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

        # 2. 更新所有 stage 的参数
        for k in range(self.cfg.horizon_steps):
            self.solver.set(k, "p", p_vec)
        self.solver.set(self.cfg.horizon_steps, "p", p_vec)

        # 3. warm start
        for k in range(self.cfg.horizon_steps):
            self.solver.set(k, "u", self.u_init_guess[k])

        for k in range(self.cfg.horizon_steps + 1):
            self.solver.set(k, "x", self.x_init_guess[k])

        # 4. 求解
        status = self.solver.solve()

        if status != 0:
            return 0.0, 0.0, int(status)

        # 5. 取第一步预测后的状态作为实际命令
        #    因为状态里包含 v,w，控制里是加速度
        x1 = self.solver.get(1, "x")
        v_cmd = float(x1[3])
        w_cmd = float(x1[4])

        # 再做一次保险裁剪
        v_cmd = float(np.clip(v_cmd, self.cfg.v_min, self.cfg.v_max))
        w_cmd = float(np.clip(w_cmd, self.cfg.w_min, self.cfg.w_max))

        # 6. 更新 warm start
        # u 向前移一格
        for k in range(self.cfg.horizon_steps - 1):
            self.u_init_guess[k] = self.solver.get(k + 1, "u")
        self.u_init_guess[-1] = self.u_init_guess[-2]

        # x 向前移一格
        for k in range(self.cfg.horizon_steps):
            self.x_init_guess[k] = self.solver.get(k + 1, "x")
        self.x_init_guess[-1] = self.x_init_guess[-2]

        return v_cmd, w_cmd, int(status)

class PerceptionAnchorBuilder:
    """
    把原始感知结果转换成 MPC 可直接使用的 anchors / 参数向量。

    输入来源：
    1. 点云预处理后的 obstacles_odom: List[(x,y)] 或 ndarray(N,2)
    2. CLIPSeg 的四张独立语义图 semantic_maps
    3. 当前机器人位姿 / 当前目标

    输出：
    1. obstacle_anchors: shape (n_obs, 3)   -> [x, y, weight]
    2. semantic_anchors: shape (n_sem, 3)   -> [x, y, weight]
    3. acados 参数向量 p_vec
    """

    def __init__(self, cfg: MPCConfig, geom: GeometryProjectionUtils):
        self.cfg = cfg
        self.geom = geom

        self.obstacle_anchors = np.zeros(
            (self.cfg.obstacle_anchor_count, 3), dtype=np.float64
        )
        self.semantic_anchors = np.zeros(
            (self.cfg.semantic_anchor_count, 3), dtype=np.float64
        )

        self._semantic_robot_grid = self._build_semantic_robot_grid()

    def reset(self) -> None:
        self.obstacle_anchors[:] = 0.0
        self.semantic_anchors[:] = 0.0

    def _build_semantic_robot_grid(self) -> np.ndarray:
        """
        在机器人前方构建固定采样格点。
        这些点会被投影到当前图像中，从而把图像语义代价映射回 odom 平面。
        """
        pts = []
        for xr in self.cfg.semantic_x_samples:
            for yr in self.cfg.semantic_y_samples:
                pts.append([float(xr), float(yr)])
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray(pts, dtype=np.float64)

    def update_obstacle_anchors(
        self,
        obstacles_odom: Optional[np.ndarray],
        robot_x: float,
        robot_y: float,
    ) -> np.ndarray:
        """
        根据当前障碍点，选最近的若干个作为 obstacle anchors。

        anchors[:, 0] = obs_x
        anchors[:, 1] = obs_y
        anchors[:, 2] = weight
        """
        self.obstacle_anchors[:] = 0.0

        if obstacles_odom is None:
            return self.obstacle_anchors

        pts = np.asarray(obstacles_odom, dtype=np.float64)
        if pts.size == 0:
            return self.obstacle_anchors

        if pts.ndim != 2 or pts.shape[1] != 2:
            pts = pts.reshape(-1, 2)

        robot_xy = np.array([robot_x, robot_y], dtype=np.float64)
        d = np.linalg.norm(pts - robot_xy, axis=1)

        if len(d) == 0:
            return self.obstacle_anchors

        idx = np.argsort(d)[: self.cfg.obstacle_anchor_count]
        chosen = pts[idx]
        chosen_d = d[idx]

        # 用“净空距离”构造权重：越近权重越大
        clearance = np.clip(chosen_d - self.cfg.robot_radius, 0.05, None)

        # 这里的形式不是唯一的，但比较稳
        weights = 1.0 / (clearance ** 2)

        # 在安全阈值以内进一步增强
        near_mask = clearance < self.cfg.safe_dist_threshold
        weights[near_mask] *= 2.0

        # 裁剪到一个合理范围，避免数值过大
        weights = np.clip(weights, 0.0, 200.0)

        n = len(chosen)
        self.obstacle_anchors[:n, 0:2] = chosen
        self.obstacle_anchors[:n, 2] = weights

        return self.obstacle_anchors

    def _compute_semantic_score_from_maps(
        self,
        semantic_maps: Dict[str, np.ndarray],
        x_pix: np.ndarray,
        y_pix: np.ndarray,
    ) -> np.ndarray:
        """
        根据四张语义图，在给定像素位置上计算语义综合代价值。
        """
        if len(x_pix) == 0:
            return np.zeros((0,), dtype=np.float64)

        required_keys = ["vegetation", "Pavement", "grass", "Stop gesture"]
        for key in required_keys:
            if key not in semantic_maps:
                return np.zeros((len(x_pix),), dtype=np.float64)

        veg = semantic_maps["vegetation"][y_pix, x_pix].astype(np.float64)
        pave = semantic_maps["Pavement"][y_pix, x_pix].astype(np.float64)
        grass = semantic_maps["grass"][y_pix, x_pix].astype(np.float64)
        stop = semantic_maps["Stop gesture"][y_pix, x_pix].astype(np.float64)

        score = (
            self.cfg.semantic_class_weights["vegetation"] * veg
            + self.cfg.semantic_class_weights["Pavement"] * pave
            + self.cfg.semantic_class_weights["grass"] * grass
            + self.cfg.semantic_class_weights["Stop gesture"] * stop
        )
        return score

    def update_semantic_anchors(
        self,
        semantic_maps: Dict[str, np.ndarray],
        img_h: Optional[int],
        img_w: Optional[int],
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> np.ndarray:
        """
        从当前图像语义图中抽取 semantic anchors。

        方法：
        1. 先在机器人前方定义一批固定 robot-frame 采样点
        2. 投影到图像
        3. 从四张独立语义图取值，形成综合 semantic score
        4. 选 score 最高的 top-k
        5. 把这些点转回 odom 平面，作为 semantic anchors
        """
        self.semantic_anchors[:] = 0.0

        if semantic_maps is None or len(semantic_maps) == 0:
            return self.semantic_anchors
        if img_h is None or img_w is None:
            return self.semantic_anchors
        if self._semantic_robot_grid.size == 0:
            return self.semantic_anchors

        valid_indices, x_pix, y_pix = self.geom.project_robot_points_to_image(
            self._semantic_robot_grid, img_h, img_w
        )

        if len(valid_indices) == 0:
            return self.semantic_anchors

        pts_robot_valid = self._semantic_robot_grid[valid_indices]
        pts_odom_valid = self.geom.robot_to_odom_points(
            pts_robot_valid, robot_x, robot_y, robot_yaw
        )

        sem_score = self._compute_semantic_score_from_maps(
            semantic_maps, x_pix, y_pix
        )

        if len(sem_score) == 0:
            return self.semantic_anchors

        # 选代价值最高的 top-k
        idx = np.argsort(-sem_score)[: self.cfg.semantic_anchor_count]
        chosen_xy = pts_odom_valid[idx]
        chosen_w = sem_score[idx]

        # 裁剪权重，避免数值过大
        chosen_w = np.clip(chosen_w, 0.0, 5.0)

        n = len(chosen_xy)
        self.semantic_anchors[:n, 0:2] = chosen_xy
        self.semantic_anchors[:n, 2] = chosen_w

        return self.semantic_anchors

    def build_parameter_vector(
        self,
        goal_x: float,
        goal_y: float,
        goal_yaw: float,
    ) -> np.ndarray:
        """
        按 AcadosBehaviorMPC 里定义的顺序组装参数向量：

        [goal_x, goal_y, goal_yaw,
         obs_x..., obs_y..., obs_w...,
         sem_x..., sem_y..., sem_w...]
        """
        n_obs = self.cfg.obstacle_anchor_count
        n_sem = self.cfg.semantic_anchor_count

        p_dim = 3 + 3 * n_obs + 3 * n_sem
        p = np.zeros((p_dim,), dtype=np.float64)

        p[0] = goal_x
        p[1] = goal_y
        p[2] = goal_yaw

        base = 3
        p[base : base + n_obs] = self.obstacle_anchors[:, 0]
        p[base + n_obs : base + 2 * n_obs] = self.obstacle_anchors[:, 1]
        p[base + 2 * n_obs : base + 3 * n_obs] = self.obstacle_anchors[:, 2]

        sem_base = base + 3 * n_obs
        p[sem_base : sem_base + n_sem] = self.semantic_anchors[:, 0]
        p[sem_base + n_sem : sem_base + 2 * n_sem] = self.semantic_anchors[:, 1]
        p[sem_base + 2 * n_sem : sem_base + 3 * n_sem] = self.semantic_anchors[:, 2]

        return p

class BehaviorMPCPlannerNode(Node):
    """
    主 ROS2 节点：
    1. 订阅 odom / pointcloud / image
    2. 维护当前状态、目标、障碍点、语义图
    3. 用 PerceptionAnchorBuilder 构建 acados 参数
    4. 调用 AcadosBehaviorMPC 求解并发布 cmd_vel
    """

    def __init__(self, config: Optional[MPCConfig] = None):
        super().__init__("behavior_mpc_planner")

        self.cfg = config if config is not None else MPCConfig()
        self.geom = GeometryProjectionUtils(self.cfg)
        self.anchor_builder = PerceptionAnchorBuilder(self.cfg, self.geom)
        self.mpc_solver = AcadosBehaviorMPC(self.cfg)

        # ---------- runtime flags ----------
        self.publish_outputs = True
        self.publish_to_robot = self._ask_publish_to_robot()

        # ---------- ROS I/O ----------
        self.bridge = CvBridge()

        self.sub_odom = self.create_subscription(
            Odometry,
            self.cfg.odom_topic,
            self.assign_odom_coords,
            self.cfg.best_effort_qos,
        )

        self.sub_pointcloud = self.create_subscription(
            PointCloud2,
            self.cfg.pointcloud_topic,
            self.pointcloud_callback,
            self.cfg.best_effort_qos,
        )

        if self.cfg.enable_clipseg:
            self.sub_image = self.create_subscription(
                Image,
                self.cfg.image_topic,
                self.image_callback,
                10,
            )
        else:
            self.sub_image = None

        self.pub_cmd = self.create_publisher(
            Twist,
            self.cfg.cmd_vel_topic if self.publish_to_robot else "/dont_publish",
            10 if self.publish_to_robot else 1,
        )

        self.pub_behav_costmap = self.create_publisher(
            Image,
            self.cfg.behav_costmap_topic,
            10,
        )

        # ---------- robot state ----------
        self.x: Optional[float] = None
        self.y: Optional[float] = None
        self.th: Optional[float] = None
        self.v_meas: float = 0.0
        self.w_meas: float = 0.0

        self.current_pose: Optional[Odometry] = None

        self.received_odom_once = False
        self.received_img_once = not self.cfg.enable_clipseg

        # ---------- goal ----------
        self.goal_radius, self.goal_theta_deg, self.goal_delta_deg = self._ask_goal_from_user()

        self.goalX: Optional[float] = None
        self.goalY: Optional[float] = None
        self.final_goal_pose = Pose()

        self.received_final_goal_odom = False
        self.current_to_goal_dist = self.cfg.goal_reach_threshold + 1.0

        # ---------- point cloud / obstacle ----------
        self.obstacles_odom = np.zeros((0, 2), dtype=np.float64)
        self.obs_tree: Optional[KDTree] = None
        self.b_has_cost_map = False

        # ---------- semantic ----------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_h: Optional[int] = None
        self.img_w: Optional[int] = None
        self.semantic_maps: Dict[str, np.ndarray] = {}
        self.behav_costmap: Optional[np.ndarray] = None

        self.processor: Optional[CLIPSegProcessor] = None
        self.model: Optional[CLIPSegForImageSegmentation] = None
        if self.cfg.enable_clipseg:
            self._load_clipseg_local()

        # ---------- output command ----------
        self.speed = Twist()

        # ---------- debug ----------
        self.get_logger().info(f"publish_to_robot = {self.publish_to_robot}")
        self.get_logger().info(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
        self.get_logger().info(
            f"goal in robot frame: r={self.goal_radius:.3f} m, "
            f"theta={self.goal_theta_deg:.3f} deg, delta={self.goal_delta_deg:.3f} deg"
        )

    # ------------------------------------------------------------------
    # user input helpers
    # ------------------------------------------------------------------
    def _ask_publish_to_robot(self) -> bool:
        try:
            choice = input("Publish to Robot Motors ? 1 or 0: ").strip()
            return int(choice) == 1
        except Exception:
            return False

    def _ask_goal_from_user(self) -> Tuple[float, float, float]:
        try:
            r = float(input("Enter the goal distance r (meters) : ").strip())
            theta = float(input("Enter the goal heading angle theta (degrees, left +ve) : ").strip())
            delta = float(input("Enter the goal pose angle (degrees) : ").strip())
            return r, theta, delta
        except Exception:
            self.get_logger().warn("Goal input failed, fallback to r=2.0, theta=0.0, delta=0.0")
            return 2.0, 0.0, 0.0

    # ------------------------------------------------------------------
    # model / perception loading
    # ------------------------------------------------------------------
    def _load_clipseg_local(self) -> None:
        if not os.path.isdir(self.cfg.clipseg_dir):
            raise FileNotFoundError(
                f"Local CLIPSeg directory not found: {self.cfg.clipseg_dir}"
            )

        self.processor = CLIPSegProcessor.from_pretrained(
            self.cfg.clipseg_dir,
            local_files_only=True,
        )
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            self.cfg.clipseg_dir,
            local_files_only=True,
        ).to(self.device)

        self.get_logger().info(
            f"CLIPSeg loaded locally from: {self.cfg.clipseg_dir}"
        )

    # ------------------------------------------------------------------
    # odom / goal
    # ------------------------------------------------------------------
    def assign_odom_coords(self, msg: Odometry) -> None:
        self.current_pose = msg

        self.x = float(msg.pose.pose.position.x)
        self.y = float(msg.pose.pose.position.y)

        q = msg.pose.pose.orientation
        _, _, yaw = self.geom.euler_from_quaternion(
            (q.x, q.y, q.z, q.w)
        )
        self.th = float(yaw)

        self.v_meas = float(msg.twist.twist.linear.x)
        self.w_meas = float(msg.twist.twist.angular.z)

        if self.received_final_goal_odom and self.goalX is not None and self.goalY is not None:
            self.current_to_goal_dist = math.sqrt(
                (self.goalX - self.x) ** 2 + (self.goalY - self.y) ** 2
            )

        self.received_odom_once = True

    def goal_to_odom_pose(self) -> None:
        """
        把用户输入的机器人坐标系目标 (r, theta, delta) 转到 odom 下。
        """
        if self.x is None or self.y is None or self.th is None:
            return

        goal_x_robot = self.goal_radius * math.cos(math.radians(self.goal_theta_deg))
        goal_y_robot = self.goal_radius * math.sin(math.radians(self.goal_theta_deg))

        self.goalX = self.x + goal_x_robot * math.cos(self.th) - goal_y_robot * math.sin(self.th)
        self.goalY = self.y + goal_x_robot * math.sin(self.th) + goal_y_robot * math.cos(self.th)

        goal_yaw_odom = self.th + math.radians(self.goal_delta_deg)
        qx, qy, qz, qw = self.geom.quaternion_from_euler(0.0, 0.0, goal_yaw_odom)

        pose = Pose()
        pose.position.x = float(self.goalX)
        pose.position.y = float(self.goalY)
        pose.position.z = 0.0
        pose.orientation.x = float(qx)
        pose.orientation.y = float(qy)
        pose.orientation.z = float(qz)
        pose.orientation.w = float(qw)

        self.final_goal_pose = pose
        self.received_final_goal_odom = True

        self.get_logger().info(
            f"Goal odom pose: x={self.goalX:.3f}, y={self.goalY:.3f}, yaw={goal_yaw_odom:.3f}"
        )

    # ------------------------------------------------------------------
    # point cloud -> obstacles -> obstacle anchors
    # ------------------------------------------------------------------
    def pointcloud_callback(self, msg: PointCloud2) -> None:
        if self.x is None or self.y is None or self.current_pose is None:
            return

        points_2d = []

        try:
            robot_z = float(self.current_pose.pose.pose.position.z)

            for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                px, py, pz = p

                z_rel = float(pz) - robot_z
                if z_rel < self.cfg.min_obstacle_height or z_rel > self.cfg.max_obstacle_height:
                    continue

                dx = float(px) - self.x
                dy = float(py) - self.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > self.cfg.sensing_range:
                    continue

                points_2d.append((float(px), float(py)))

            if len(points_2d) > 0:
                self.obstacles_odom = np.asarray(points_2d, dtype=np.float64)
                self.obs_tree = KDTree(self.obstacles_odom)
                self.b_has_cost_map = True
            else:
                self.obstacles_odom = np.zeros((0, 2), dtype=np.float64)
                self.obs_tree = None
                self.b_has_cost_map = False

            self.anchor_builder.update_obstacle_anchors(
                self.obstacles_odom,
                self.x,
                self.y,
            )

        except Exception as e:
            self.get_logger().error(f"pointcloud_callback error: {str(e)}")
            self.obstacles_odom = np.zeros((0, 2), dtype=np.float64)
            self.obs_tree = None
            self.b_has_cost_map = False
            self.anchor_builder.obstacle_anchors[:] = 0.0

    # ------------------------------------------------------------------
    # image -> clipseg -> semantic maps -> semantic anchors
    # ------------------------------------------------------------------
    def image_callback(self, msg: Image) -> None:
        if not self.cfg.enable_clipseg or self.processor is None or self.model is None:
            self.received_img_once = True
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self.img_h, self.img_w, _ = cv_image.shape
            pil_image = PILImage.fromarray(cv_image)

            inputs = self.processor(
                text=self.cfg.prompts,
                images=[pil_image] * len(self.cfg.prompts),
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(**inputs)

            preds = torch.sigmoid(outputs.logits)
            preds_resized = F.interpolate(
                preds.unsqueeze(1),
                size=(self.img_h, self.img_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1).cpu().numpy()

            # 保存四张独立语义图
            self.semantic_maps = {}
            for i, prompt in enumerate(self.cfg.prompts):
                self.semantic_maps[prompt] = preds_resized[i].astype(np.float32)

            # 仅用于可视化的 combined cost map
            combined_cost_map = np.zeros((self.img_h, self.img_w), dtype=np.float32)
            for prompt in self.cfg.prompts:
                combined_cost_map += (
                    self.semantic_maps[prompt] * 255.0 * self.cfg.semantic_class_weights[prompt]
                )

            combined_cost_map = np.clip(combined_cost_map, 0, 255).astype(np.uint8)
            self.behav_costmap = combined_cost_map

            if self.publish_outputs:
                combined_cost_map_colored = cv2.applyColorMap(
                    combined_cost_map, cv2.COLORMAP_JET
                )
                overlaid_image = cv2.addWeighted(
                    cv_image, 0.4, combined_cost_map_colored, 0.6, 0
                )
                ros_img = self.bridge.cv2_to_imgmsg(overlaid_image, encoding="rgb8")
                self.pub_behav_costmap.publish(ros_img)

            # 有位姿时再更新 semantic anchors
            if self.x is not None and self.y is not None and self.th is not None:
                self.anchor_builder.update_semantic_anchors(
                    semantic_maps=self.semantic_maps,
                    img_h=self.img_h,
                    img_w=self.img_w,
                    robot_x=self.x,
                    robot_y=self.y,
                    robot_yaw=self.th,
                )

            self.received_img_once = True

            end_time = time.time()
            hz = 1.0 / max(end_time - start_time, 1e-6)
            self.get_logger().info(f"CLIPSeg inference rate: {hz:.3f} Hz")

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

    # ------------------------------------------------------------------
    # mpc parameter build / solve
    # ------------------------------------------------------------------
    def build_mpc_parameter_vector(self) -> np.ndarray:
        goal_yaw = self.geom.yaw_from_pose(self.final_goal_pose)
        return self.anchor_builder.build_parameter_vector(
            goal_x=float(self.goalX),
            goal_y=float(self.goalY),
            goal_yaw=float(goal_yaw),
        )

    def solve_current_mpc(self) -> Tuple[float, float, int]:
        x0 = np.array(
            [
                float(self.x),
                float(self.y),
                float(self.th),
                float(self.v_meas),
                float(self.w_meas),
            ],
            dtype=np.float64,
        )

        p_vec = self.build_mpc_parameter_vector()
        v_cmd, w_cmd, status = self.mpc_solver.solve(x0, p_vec)
        return v_cmd, w_cmd, status

    # ------------------------------------------------------------------
    # run loop
    # ------------------------------------------------------------------
    def wait_for_odom(self) -> None:
        while not self.received_odom_once and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

    def wait_for_img(self) -> None:
        if not self.cfg.enable_clipseg:
            return
        while not self.received_img_once and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

    def main_loop(self) -> None:
        loop_start_time = time.time()

        if self.received_odom_once and not self.received_final_goal_odom:
            self.goal_to_odom_pose()

        ready = (
            self.received_odom_once
            and self.received_final_goal_odom
            and self.received_img_once
        )

        if not ready:
            print(" -- Waiting for odom / image / goal initialization -- ")
            return

        if self.current_to_goal_dist < self.cfg.goal_reach_threshold:
            self.speed.linear.x = 0.0
            self.speed.angular.z = 0.0
            self.pub_cmd.publish(self.speed)
            print("--- Goal Reached !! ---")
            return

        v_cmd, w_cmd, status = self.solve_current_mpc()

        if status != 0:
            self.get_logger().warn(f"MPC solver failed, status={status}, publish zero cmd")
            v_cmd = 0.0
            w_cmd = 0.0

        self.speed.linear.x = float(v_cmd)
        self.speed.angular.z = float(w_cmd)

        self.pub_cmd.publish(self.speed)

        loop_end_time = time.time()
        dt = max(loop_end_time - loop_start_time, 1e-6)
        print("--- Total inference rate per cycle ---", 1.0 / dt)

    def run(self) -> None:
        self.wait_for_odom()
        self.wait_for_img()
        self.get_logger().info("Planner ready, entering main loop.")

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            self.main_loop()


def main(args=None):
    rclpy.init(args=args)

    node = BehaviorMPCPlannerNode()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()