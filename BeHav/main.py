import logging
import numpy as np
from instruction_processor import get_instruction_breakdown, get_similarity_scores, calculate_input_action_costs, get_ith_key_list
from landmark_vision import LandmarkDetectorCore
from behav_planner import BehavPlannerCore

class BehavMainPipeline:
    """
    主要业务逻辑管线：整合大语言模型指令分解、视觉地标检测与避障路径规划。
    将核心逻辑与 ROS 接口解耦。
    """
    def __init__(self, logger=None):
        self.logger = logger
        
        # 实例化各个核心算法模块
        self.detector_core = LandmarkDetectorCore(logger=self.logger)
        self.behav_planner = BehavPlannerCore(logger=self.logger, goal_radius=2.5, goal_theta=0.0, goal_delta=0.0)
        
        # 用于输出可视化的回调函数（供 ROS 接口绑定）
        self.on_behav_costmap = None
        self.on_traj_image = None
        
        # 绑定 BehavPlannerCore 的可视化输出到本类的分发口
        self.behav_planner.on_behav_costmap = self._distribute_costmap
        self.behav_planner.on_traj_image = self._distribute_traj

    def _distribute_costmap(self, msg):
        if self.on_behav_costmap:
            self.on_behav_costmap(msg)

    def _distribute_traj(self, msg):
        if self.on_traj_image:
            self.on_traj_image(msg)

    def run_instruction_reasoning(self, language_instruction='Walk to the red car and stop in front of it', skip_nlp=False):
        """
        NLP 顶层指令分解模块
        """
        if self.logger:
            if skip_nlp:
                self.logger.info(f"Running instruction reasoning for: {language_instruction} (Skipped LLM, using hardcoded)")
            else:
                self.logger.info(f"Running instruction reasoning for: {language_instruction}")
            
        reference_list = ['Stay on', 'Avoid', 'Yield', 'Stop']
        reference_costs = [0, 0.5, 0.7, 1]

        if skip_nlp:
            import json
            import os
            instruction_breakdown = {}
            landmark_list = json.loads(os.environ.get('PRESET_LANDMARKS', '["red car"]'))
            navigation_action_list = json.loads(os.environ.get('PRESET_NAV_ACTIONS', '["walk to", "stop"]'))
            behavioral_action_list = json.loads(os.environ.get('PRESET_BEHAV_ACTIONS', '["stay on", "avoid"]'))
            behavioral_target_list = json.loads(os.environ.get('PRESET_BEHAV_TARGETS', '["pavement"]'))
        else:
            instruction_breakdown = get_instruction_breakdown(language_instruction)
            landmark_list = get_ith_key_list(instruction_breakdown, key_idx=1)
            navigation_action_list = get_ith_key_list(instruction_breakdown, key_idx=2)
            behavioral_action_list = get_ith_key_list(instruction_breakdown, key_idx=3)
            behavioral_target_list = get_ith_key_list(instruction_breakdown, key_idx=4)

        if self.logger:
            self.logger.info(f"Landmarks: {landmark_list}")
            self.logger.info(f"Navigation Actions: {navigation_action_list}")
            self.logger.info(f"Behavioral Actions: {behavioral_action_list}")
            self.logger.info(f"Behavioral Targets: {behavioral_target_list}")

        similarity_scores = get_similarity_scores(behavioral_action_list, reference_list)
        input_action_costs = calculate_input_action_costs(similarity_scores, reference_costs)

        # Convert numpy arrays back to regular Python lists to avoid HuggingFace strict type checking errors
        if isinstance(behavioral_target_list, np.ndarray):
            behavioral_target_list = behavioral_target_list.tolist()
        if isinstance(landmark_list, np.ndarray):
            landmark_list = landmark_list.tolist()
        if isinstance(navigation_action_list, np.ndarray):
            navigation_action_list = navigation_action_list.tolist()

        self.behav_planner.prompts = behavioral_target_list or []
        self.behav_planner.cost_values = input_action_costs or []

        self.detector_core.navigation_landmarks = landmark_list or []
        self.detector_core.navigation_actions = navigation_action_list or []

    def process_vision_cv2(self, cv_image, depth_image=None):
        """
        传递给视觉追踪模块的 OpenCV 图像和深度图像
        """
        self.detector_core.process_image(cv_image, depth_image)

    def update_sensor_data(self, image_msg=None, pointcloud_msg=None, odom_msg=None):
        """
        传递 ROS 数据至 BehavPlannerCore
        """
        if image_msg is not None:
            self.behav_planner.process_image(image_msg)
        if pointcloud_msg is not None:
            self.behav_planner.process_pointcloud(pointcloud_msg)
        if odom_msg is not None:
            self.behav_planner.process_odom(odom_msg)

    def compute_control_command(self):
        """
        在 MainPipeline 中决策最终的控制下发。
        结合 behav_planner 的运动学求解以及 detector_core 定位的当前目标方位进行加权或逻辑覆盖
        """
        # 1. 默认：根据代价地图及障碍物获取规划路径的速度命令
        cmd_msg = self.behav_planner.compute_velocity()

        # 2. 参考地标的直接定位情况
        meas = self.detector_core.latest_measurement
        if meas is not None:
            distance_m, bearing_deg = meas
            # 如果地标在视野内，将其作为局部目标传递给 planner
            self.behav_planner.goal_radius = distance_m
            self.behav_planner.goal_theta = bearing_deg
            self.behav_planner.received_final_goal_odom = False # 强制重新计算以车辆中心为基准的新目标点
            if distance_m < 1.0 and cmd_msg is not None: # 如果距离目标极近，可直接发送停止指令或大幅降 低速度
                cmd_msg.linear.x *= 0.5
                cmd_msg.angular.z *= 0.5

        return cmd_msg
