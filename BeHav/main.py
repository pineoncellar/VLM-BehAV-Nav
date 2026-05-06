import logging
import numpy as np
from instruction_processor import get_instruction_breakdown, get_similarity_scores, calculate_input_action_costs, get_ith_key_list
from landmark_vision import LandmarkDetectorCore

class BehavMainPipeline:
    """
    主要业务逻辑管线：整合大语言模型指令分解、视觉地标检测与避障路径规划。
    将核心逻辑与 ROS 接口解耦。
    """
    def __init__(self, logger=None):
        self.logger = logger
        
        # 实例化各个核心算法模块
        self.detector_core = LandmarkDetectorCore(logger=self.logger)
        
        # 用于输出可视化的回调函数（供 ROS 接口绑定）
        self.on_behav_costmap = None
        self.on_traj_image = None

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

        
        # 将最新的 LLM 提取属性缓存，供 control loop 频率下发
        self.current_prompts = behavioral_target_list or []
        self.current_behavior_rule = "avoid_" + "_".join(self.current_prompts) if self.current_prompts else ""

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
        pass


    def compute_control_command(self):
        """
        分布式 ROS 节点通信版本：
        不再直接计算运动学速度，而是将 LLM 解析出的语义和目标转化为 Waypoint 和 Semantic Topic。
        只有在距离极度接近时，才发布直接覆盖的 Twsit 让车停下。
        """
        from geometry_msgs.msg import Twist, PoseStamped
        from std_msgs.msg import String
        import json
        import math

        cmd_override = None
        wp_msg = None
        semantic_msg = None

        # 1. 向下层建图节点发送行为准则 (Behavior Rule)
        if hasattr(self, 'current_prompts') and self.current_prompts:
            rule_dict = {
                "prompts": self.current_prompts,
                "rule": self.current_behavior_rule
            }
            s = String()
            s.data = json.dumps(rule_dict)
            semantic_msg = s

        # 2. 从 VLM 获取地标位置，下发给 Rollout 节点
        meas = self.detector_core.latest_measurement
        if meas is not None:
            distance_m, bearing_deg = meas
            if distance_m < 1.0:
                # 距离过近，发送直接停止指令 (急停保护)
                cmd_override = Twist()
                cmd_override.linear.x = 0.0
                cmd_override.angular.z = 0.0
            else:
                # 还有距离，发送局部轨迹坐标给底层 Rollout，要求底端自行避障行驶
                wp = PoseStamped()
                wp.header.frame_id = "base_footprint"
                # bearing_deg 转换为弧度
                theta = math.radians(bearing_deg)
                wp.pose.position.x = distance_m * math.cos(theta)
                wp.pose.position.y = distance_m * math.sin(theta)
                wp.pose.position.z = 0.0
                wp.pose.orientation.w = 1.0  # 简化的四元数
                wp_msg = wp
        else:
            # 视野内没有地标，让底层走默认路点或停下
            # 这里先不做任何处理，等待底层自主停下
            pass

        return cmd_override, wp_msg, semantic_msg

