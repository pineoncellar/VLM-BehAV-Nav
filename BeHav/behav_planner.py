import math
import numpy as np
import os
import time
import cv2
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import nlopt
from scipy.spatial import KDTree
from threading import Condition, Lock

import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import Twist, PoseStamped, Pose
from cv_bridge import CvBridge

def euler_from_quaternion(quaternion):
    x, y, z, w = quaternion
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw 

def quaternion_from_euler(roll, pitch, yaw):
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
    return (x, y, z, w)

def motion(planner, state, u, dt):
    x, y, theta = state
    v, omega = u
    theta_new = theta + omega * dt
    x_new = x + v * np.cos(theta_new) * dt
    y_new = y + v * np.sin(theta_new) * dt
    return np.array([x_new, y_new, theta_new])

class ControlLawSettings:
    def __init__(self, K1=1, K2=3, BETA=1, LAMBDA=1, V_MAX=1.0, V_MIN=0.0, R_THRESH=0.05):
        self.m_K1 = K1
        self.m_K2 = K2
        self.m_BETA = BETA
        self.m_LAMBDA = LAMBDA
        self.m_V_MAX = V_MAX
        self.m_V_MIN = V_MIN
        self.m_R_THRESH = R_THRESH

class EgoPolar:
    def __init__(self, r=0.0, delta=0.0, theta=0.0):
        self.r = r
        self.delta = delta
        self.theta = theta

class ControlLaw:
    def __init__(self, settings):
        self.settings = settings

    @staticmethod
    def mod(x, y):
        m = x - y * math.floor(x / y)
        if y > 0:
            if m >= y: return 0
            if m < 0: return 0 if y + m == y else y + m
        else:
            if m <= y: return 0
            if m > 0: return 0 if y + m == y else y + m
        return m

    def wrap_pos_neg_pi(self, angle):
        return self.mod(angle + math.pi, 2 * math.pi) - math.pi

    def convert_to_egopolar(self, current_state, current_goal_pose):
        coords = EgoPolar()
        dx = float(current_goal_pose.position.x) - float(current_state[0])
        dy = float(current_goal_pose.position.y) - float(current_state[1])
        obs_heading = math.atan2(dy, dx)
        current_yaw = float(current_state[2])
        goal_yaw = euler_from_quaternion([
            float(current_goal_pose.orientation.x), float(current_goal_pose.orientation.y),
            float(current_goal_pose.orientation.z), float(current_goal_pose.orientation.w)
        ])[2]
        coords.r = math.sqrt(dx**2 + dy**2)
        coords.delta = self.wrap_pos_neg_pi(current_yaw - obs_heading)
        coords.theta = self.wrap_pos_neg_pi(goal_yaw - obs_heading)
        return coords
    
    def convert_from_egopolar(self, current_state, current_goal_coords):
        current_yaw = float(current_state[2])
        current_goal_pose = Pose()
        current_goal_pose.position.x = float(current_state[0]) + float(current_goal_coords.r) * math.cos(current_yaw - float(current_goal_coords.delta))
        current_goal_pose.position.y = float(current_state[1]) + float(current_goal_coords.r) * math.sin(current_yaw - float(current_goal_coords.delta))
        current_goal_pose.position.z = 0.0  
        quaternion = quaternion_from_euler(0, 0, current_yaw - float(current_goal_coords.delta) + float(current_goal_coords.theta))
        current_goal_pose.orientation.x = float(quaternion[0])
        current_goal_pose.orientation.y = float(quaternion[1])
        current_goal_pose.orientation.z = float(quaternion[2])
        current_goal_pose.orientation.w = float(quaternion[3])
        return current_goal_pose
    
    def get_kappa(self, current_ego_goal, k1, k2):
        kappa = (-1 / (current_ego_goal.r+1e-5)) * (
            k2 * (current_ego_goal.delta - math.atan(-1 * k1 * current_ego_goal.theta)) +
            (1 + k1 / (1 + k1**2 * current_ego_goal.theta**2)) * math.sin(current_ego_goal.delta)
        )
        return kappa

    def get_linear_vel(self, kappa, current_ego_goal, vMax):
        lin_vel = min(self.settings.m_V_MAX / self.settings.m_R_THRESH * current_ego_goal.r,
                      self.settings.m_V_MAX / (1 + self.settings.m_BETA * abs(kappa)**self.settings.m_LAMBDA))
        if self.settings.m_V_MIN < lin_vel < 0.00:
            lin_vel = self.settings.m_V_MIN
        return lin_vel

    def _get_velocity_command(self, goal_coords, k1, k2, vMax):
        cmd_vel = Twist()
        kappa = self.get_kappa(goal_coords, k1, k2)
        cmd_vel.linear.x = self.get_linear_vel(kappa, goal_coords, vMax)
        cmd_vel.angular.z = kappa * cmd_vel.linear.x
        R_SPEED_LIMIT = self.settings.m_V_MAX - 0.1  
        if abs(cmd_vel.angular.z) > R_SPEED_LIMIT:
            cmd_vel.angular.z = math.copysign(R_SPEED_LIMIT, cmd_vel.angular.z)
            cmd_vel.linear.x = cmd_vel.angular.z / (kappa + 1e-5)
        return cmd_vel

class BehavPlannerCore:
    def __init__(self, logger=None, goal_radius=2.5, goal_theta=0.0, goal_delta=0.0):
        self.logger = logger
        self.settings = ControlLawSettings(K1=1.2, K2=1, BETA=0.4, LAMBDA=2, R_THRESH=0.05, V_MAX=0.8, V_MIN=0.0)
        self.control_law = ControlLaw(self.settings)
        
        self.odom_msg = None
        self.sensing_range = 3.5
        self.obstacles_odom = None
        self.safe_dist_threshold = 0.5

        self.obs_tree = None
        self.b_has_cost_map = False   
        self.b_has_odom = False       

        self.min_obstacle_height = -0.3
        self.max_obstacle_height = 1.2
        self.robot_radius = 0.35
        self.behav_weight = 500.0

        self.enable_clipseg = True   
        self.max_speed = 0.8  

        self.x = None
        self.y = None
        self.goalX = None 
        self.goalY = None
        self.th = None

        self.speed = Twist()
        self.goal_reach_thrshold = 0.7

        self.init_x = None
        self.init_y = None
        self.received_init_odom = False
        self.received_odom_once = False
        self.received_final_goal_odom = False

        self.to_global_goal_from_init = 0
        self.current_to_goal_dist = self.goal_reach_thrshold + 1 
        self.current_pose = None
        self.final_goal_pose = Pose()

        self.goal_radius = goal_radius
        self.goal_theta = goal_theta
        self.goal_delta = goal_delta

        self.velocityGain = 1.0
        self.V_MAX = 1.0
        self.V_MIN = 0.0
        self.trajectory_count = 0
        self.TIME_HORIZON = 4 
        self.DELTA_SIM_TIME = 0.5 
        self.SAFETY_ZONE = 0.225
        self.WAYPOINT_THRESH = 1.75
        self.goal_factor = 1
        self.goal_angle_factor = 3 
        self.C1 = 0.05
        self.C2 = 2.5
        self.C3 = 0.05
        self.C4 = 0.05
        self.PHI_COL = 1.0
        self.SIGMA = 0.2

        self.pose_mutex = Lock()
        self.cost_map_mutex = Lock()

        self.received_img_once = False
        self.bridge = CvBridge()
        self.img_h, self.img_w = None, None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.local_clipseg_dir = "/home/zyy/nvidia/models/clipseg-rd64-refined"
        self.prompts = ["vegetation", "Pavement", "grass", "Stop gesture"]
        self.cost_values = [0.05, 0.95, 0.08, 0]
        self.behav_costmap = None
        self.publish_outputs = True

        if self.enable_clipseg:
            if not os.path.exists(self.local_clipseg_dir):
                if self.logger: self.logger.error(f"Local CLIPSeg directory not found: {self.local_clipseg_dir}")
                self.enable_clipseg = False
            else:
                self.processor = CLIPSegProcessor.from_pretrained(
                    self.local_clipseg_dir, local_files_only=True
                )
                self.model = CLIPSegForImageSegmentation.from_pretrained(
                    self.local_clipseg_dir, local_files_only=True
                ).to(self.device)
                if self.logger: self.logger.info(f"CLIPSeg loaded locally from: {self.local_clipseg_dir}")
        else:
            self.processor = None
            self.model = None
            self.received_img_once = True
            if self.logger: self.logger.info("CLIPSeg disabled.")

        self.Projection_Matrix = np.array([
            [607.175048828125, 0.0, 322.55340576171875, 0.0],
            [0.0, 607.222900390625, 248.86021423339844, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=np.float64)

        self.camera_height = 0.59 
        self.camera_tilt_angle = 0 
        self.camera_offset_x = 0 
        self.camera_offset_y = 0 
        self.on_behav_costmap = None
        self.on_traj_image = None
        
    def process_odom(self, msg):
        self.current_pose = msg
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        rot_q = msg.pose.pose.orientation
        (roll, pitch, theta) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])
        self.th = theta

        if self.received_final_goal_odom:
            self.current_to_goal_dist = np.sqrt((self.goalX - self.x) ** 2 + (self.goalY - self.y) ** 2)

        if not self.received_init_odom and self.received_final_goal_odom:
            self.init_x = msg.pose.pose.position.x
            self.init_y = msg.pose.pose.position.y
            self.to_global_goal_from_init = np.sqrt((self.goalX - self.init_x) ** 2 + (self.goalY - self.init_y) ** 2)
            self.received_init_odom = True

        self.received_odom_once = True
        self.b_has_odom = True

    def process_pointcloud(self, msg):
        if self.x is None or self.y is None or self.current_pose is None:
            return
        points_2d = []
        try:
            robot_z = self.current_pose.pose.pose.position.z
            for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                x, y, z = p
                z_rel = z - robot_z
                if z_rel < self.min_obstacle_height or z_rel > self.max_obstacle_height:
                    continue
                dx = x - self.x
                dy = y - self.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > self.sensing_range:
                    continue
                points_2d.append((x, y))

            with self.cost_map_mutex:
                if len(points_2d) > 0:
                    self.obstacles_odom = points_2d
                    self.obs_tree = KDTree(points_2d)
                    self.b_has_cost_map = True
                else:
                    self.obstacles_odom = []
                    self.obs_tree = None
                    self.b_has_cost_map = False
        except Exception as e:
            if self.logger: self.logger.error(f"pointcloud_callback error: {str(e)}")
            with self.cost_map_mutex:
                self.obstacles_odom = []
                self.obs_tree = None
                self.b_has_cost_map = False

    def process_image(self, msg):
        if not self.enable_clipseg:
            return
        if not self.prompts:
            return
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.img_h, self.img_w, _ = cv_image.shape
            pil_image = PILImage.fromarray(cv_image)

            inputs = self.processor(text=self.prompts, images=[pil_image] * len(self.prompts), return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(**inputs)

            preds = torch.sigmoid(outputs.logits)
            combined_cost_map = np.full((self.img_h, self.img_w), 128, dtype=np.float32)

            preds_resized = F.interpolate(preds.unsqueeze(1), size=(self.img_h, self.img_w), mode="bilinear", align_corners=False).squeeze(1).cpu().numpy()

            for i, pred_resized in enumerate(preds_resized):
                mask = pred_resized > 0.1
                combined_cost_map[mask] = pred_resized[mask] * 255 * self.cost_values[i]

            combined_cost_map = np.clip(combined_cost_map, 0, 255).astype(np.uint8)
            self.behav_costmap = combined_cost_map

            if self.publish_outputs and self.on_behav_costmap:
                combined_cost_map_colored = cv2.applyColorMap(combined_cost_map, cv2.COLORMAP_JET)
                overlaid_image = cv2.addWeighted(cv_image, 0.4, combined_cost_map_colored, 0.6, 0)
                ros_overlaid_image = self.bridge.cv2_to_imgmsg(overlaid_image, encoding='rgb8')
                self.on_behav_costmap(ros_overlaid_image)

            self.received_img_once = True
        except Exception as e:
            if self.logger: self.logger.error(f"Error processing image: {str(e)}")

    def find_intermediate_goal_params(self):
        self.trajectory_count = 0
        opt = nlopt.opt(nlopt.GN_ESCH, 4)
        opt.set_min_objective(self.score_trajectory)
        opt.set_xtol_rel(0.001)
        lb = [0.1, -0.8, -0.8, self.V_MIN]
        ub = [5.0, 0.8, 0.8, self.V_MAX]
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)
        opt.set_maxeval(100)
        x = [2.5, 0.0, 0.0, (self.V_MAX + self.V_MIN) / 2.0]
        minf = opt.optimize(x)
        
        opt2 = nlopt.opt(nlopt.LN_BOBYQA, 4)
        opt2.set_min_objective(self.score_trajectory)
        opt2.set_xtol_rel(0.002)
        opt2.set_lower_bounds(lb)
        opt2.set_upper_bounds(ub)
        opt2.set_maxeval(30)
        minf2 = opt2.optimize(minf)
        
        coords = EgoPolar()
        coords.r = minf2[0]
        coords.delta = minf2[1]
        coords.theta = minf2[2]
        return coords, minf2[3]

    def score_trajectory(self, x, grad=None):
        if grad is not None: grad[:] = 0.0  
        return self.sim_trajectory(x[0], x[1], x[2], x[3], self.TIME_HORIZON)

    def sim_trajectory(self, r, delta, theta, vMax, time_horizon):
        with self.pose_mutex:
            state = np.array([self.x, self.y, self.th])
            num_steps = int(time_horizon / self.DELTA_SIM_TIME)
            trajectory = np.zeros((num_steps + 1, 3))  
            trajectory[0, :] = state  

        expected_collision = 0.0
        expected_behav = 0.0

        sim_goal = EgoPolar(r=r, delta=delta, theta=theta)
        current_goal = self.control_law.convert_from_egopolar(state, sim_goal)
        control_inputs = np.zeros((num_steps, 2))  

        for i in range(num_steps):
            sim_cmd_vel = self.control_law._get_velocity_command(self.control_law.convert_to_egopolar(state, current_goal), self.settings.m_K1, self.settings.m_K2, vMax)
            control_inputs[i, :] = [sim_cmd_vel.linear.x, sim_cmd_vel.angular.z]
            state = motion(self, state, control_inputs[i, :], self.DELTA_SIM_TIME)
            trajectory[i + 1, :] = state  

        trajectory_arr = np.array(trajectory)
        x_coords = trajectory_arr[:, 0]
        y_coords = trajectory_arr[:, 1]
        yaw_angles = trajectory_arr[:, 2]
        goal_x = self.final_goal_pose.position.x
        goal_y = self.final_goal_pose.position.y
        distances = np.sqrt((x_coords - goal_x) ** 2 + (y_coords - goal_y) ** 2)
        goal_directions = np.arctan2(goal_y - y_coords, goal_x - x_coords)
        heading_errors = np.abs(np.arctan2(np.sin(goal_directions - yaw_angles), np.cos(goal_directions - yaw_angles)))
        expected_progress = 2 * np.sum(distances) + 1 * np.sum(heading_errors)

        if self.b_has_cost_map and self.obs_tree is not None:
            dist_to_obs = self.get_distances_to_obstacles(trajectory)
            clearance = dist_to_obs - self.robot_radius
            if np.any(clearance <= 0.0):
                expected_collision += 1e6
            else:
                penalty_zone = np.clip(self.safe_dist_threshold - clearance, 0.0, None)
                expected_collision += 200.0 * np.sum(penalty_zone ** 2)
                expected_collision += 2.0 * np.sum(1.0 / (clearance + 1e-3))

        if self.enable_clipseg and self.behav_costmap is not None:
            x_rob = (x_coords - self.x) * np.cos(self.th) + (y_coords - self.y) * np.sin(self.th)
            y_rob = -(x_coords - self.x) * np.sin(self.th) + (y_coords - self.y) * np.cos(self.th)
            robot_frame_trajectory = np.column_stack((x_rob, y_rob, yaw_angles))
            _, max_behav_cost = self.get_traj_behav_cost(robot_frame_trajectory)
            expected_behav += (max_behav_cost / 255)  

        return expected_collision + expected_progress + self.behav_weight * expected_behav

    def get_distances_to_obstacles(self, trajectory):
        with self.cost_map_mutex:
            if (not self.b_has_cost_map) or (self.obs_tree is None):
                return np.full(len(trajectory), self.sensing_range)
            xy_points = trajectory[:, :2]
            distances, _ = self.obs_tree.query(xy_points)
        return distances

    def get_traj_behav_cost(self, robot_frame_trajectory):
        marked_img = self.behav_costmap.copy()
        robot_frame_trajectory = np.array(robot_frame_trajectory)
        x_rob = robot_frame_trajectory[:, 0]
        y_rob = robot_frame_trajectory[:, 1]

        traj_coords_xyz = np.column_stack((-y_rob + self.camera_offset_y, np.full(x_rob.shape, self.camera_height), x_rob - self.camera_offset_x))
        if traj_coords_xyz.shape[0] == 0: return marked_img, 0.0

        alpha = np.deg2rad(-self.camera_tilt_angle)
        rotation_matrix = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
        points_rotated = np.dot(traj_coords_xyz, rotation_matrix.T)
        points_homogeneous = np.hstack((points_rotated, np.ones((points_rotated.shape[0], 1))))
        uvw = np.dot(self.Projection_Matrix, points_homogeneous.T)
        u_vec, v_vec, w_vec = uvw[0], uvw[1], uvw[2]

        front_mask = w_vec > 1e-6
        u_vec = u_vec[front_mask]
        v_vec = v_vec[front_mask]
        w_vec = w_vec[front_mask]
        x_vec = np.round(u_vec / w_vec).astype(np.int32)
        y_vec = np.round(v_vec / w_vec).astype(np.int32)

        valid_indices = ((y_vec >= 0) & (y_vec < self.img_h) & (x_vec >= 0) & (x_vec < self.img_w))
        valid_x = x_vec[valid_indices]
        valid_y = y_vec[valid_indices]

        if len(valid_x) == 0: return marked_img, 0.0
        costs = self.behav_costmap[valid_y, valid_x]
        max_cost = np.max(costs) if costs.size > 0 else 0.0

        if self.publish_outputs and self.on_traj_image:
            points = np.vstack((valid_x, valid_y)).T
            if len(points) > 1: cv2.polylines(marked_img, [points], isClosed=False, color=255, thickness=8)
            marked_image_msg = self.bridge.cv2_to_imgmsg(marked_img, encoding="mono8")
            self.on_traj_image(marked_image_msg)

        return marked_img, max_cost 

    def goal_to_odom_pose(self):
        goalX_rob = self.goal_radius * math.cos(math.radians(self.goal_theta))
        goalY_rob = self.goal_radius * math.sin(math.radians(self.goal_theta))
        self.goalX = self.x + goalX_rob * math.cos(self.th) - goalY_rob * math.sin(self.th)
        self.goalY = self.y + goalX_rob * math.sin(self.th) + goalY_rob * math.cos(self.th)
        goal_yaw_rob = math.radians(self.goal_delta)
        goal_yaw_odom = self.th + goal_yaw_rob

        pose = Pose()
        pose.position.x = self.goalX
        pose.position.y = self.goalY
        pose.position.z = 0.0  
        quaternion = quaternion_from_euler(0, 0, goal_yaw_odom)
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]
        self.final_goal_pose = pose
        self.current_to_goal_dist = math.sqrt((self.goalX - self.x) ** 2 + (self.goalY - self.y) ** 2)
        if self.logger: self.logger.info(f"Goal x,y w.r.t. robot and odom : {(goalX_rob,goalY_rob)} {(self.goalX,self.goalY)}")

    def compute_velocity(self):
        cmd = Twist()
        if self.received_odom_once and not self.received_final_goal_odom:
            self.goal_to_odom_pose()
            self.received_final_goal_odom = True

        if self.received_odom_once and self.received_final_goal_odom and self.received_init_odom and self.received_img_once:
            if self.current_to_goal_dist < self.goal_reach_thrshold:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
            else:
                new_coords, new_vMax = self.find_intermediate_goal_params()
                cmd_vel = self.control_law._get_velocity_command(new_coords, self.settings.m_K1, self.settings.m_K2, new_vMax)
                cmd.linear.x = cmd_vel.linear.x
                cmd.angular.z = cmd_vel.angular.z
        return cmd
