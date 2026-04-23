import cv2
import math
import os
import datetime
import logging
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError

# Import algorithms
from instruction_processor import get_instruction_breakdown, extract_lists_from_dict, get_similarity_scores, calculate_input_action_costs, get_ith_key_list
from landmark_vision import LandmarkDetectorCore
from behav_planner import BehavPlannerCore


def setup_file_logger():
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"run_{timestamp}.log")
    
    file_logger = logging.getLogger("behav_file_logger")
    file_logger.setLevel(logging.INFO)
    if not file_logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s]: %(message)s'))
        file_logger.addHandler(fh)
    return file_logger

class DualLogger:
    def __init__(self, ros_logger, file_logger):
        self.ros_logger = ros_logger
        self.file_logger = file_logger
        
    def info(self, msg):
        self.ros_logger.info(msg)
        self.file_logger.info(msg)
        
    def error(self, msg):
        self.ros_logger.error(msg)
        self.file_logger.error(msg)
        
    def warn(self, msg):
        self.ros_logger.warn(msg)
        self.file_logger.warning(msg)

class LandmarkDetectorNode(Node):
    def __init__(self):
        super().__init__('landmark_detector_node')
        
        self.file_logger = setup_file_logger()
        self.dual_logger = DualLogger(self.get_logger(), self.file_logger)
        
        self.dual_logger.info('Started landmark_detector_node')

        # Initialize the algorithm logic instance and inject dual logger
        self.detector_core = LandmarkDetectorCore(logger=self.dual_logger)
        self.behav_planner = BehavPlannerCore(
            logger=self.dual_logger, 
            goal_radius=2.5, 
            goal_theta=0.0, 
            goal_delta=0.0
        )
        
        # Override some properties if needed
        self.image_topic = "/camera_sensor/image_raw"
        self.lidar_topic = "/velodyne_points"
        self.odom_topic = "/odom"
        self.cmd_topic = "/cmd_vel"   # Or /ackermann_steering_controller/reference based on multiplexer
        self.period_sec = 10.0
        self.control_period_sec = 0.1 # 10 Hz for control loop

        # ========= ROS =========
        self.bridge = CvBridge()
        
        # 1. Perception Interfaces
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            1
        )
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            self.lidar_topic,
            self.lidar_callback,
            1
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            1
        )
        
        # 2. Control Interface (Ackermann Steering Output)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.behav_costmap_pub = self.create_publisher(Image, '/behav_costmap', 10)
        self.traj_image_pub = self.create_publisher(Image, '/traj_marked_image', 10)

        # Connect publishers conceptually to BehavPlannerCore callbacks
        self.behav_planner.on_behav_costmap = lambda m: self.behav_costmap_pub.publish(m)
        self.behav_planner.on_traj_image = lambda m: self.traj_image_pub.publish(m)

        # 3. Timers
        self.timer = self.create_timer(self.period_sec, self.timer_callback)
        self.control_timer = self.create_timer(self.control_period_sec, self.control_loop)
        
        self.latest_image = None
        self.latest_pointcloud = None
        self.is_processing = False

    def image_callback(self, msg: Image):
        self.dual_logger.info('Received an image message')
        
        # pass raw ros message to behav_planner
        self.behav_planner.process_image(msg)
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.dual_logger.info(f"Image converted successfully, shape: {cv_image.shape}")
            if msg.encoding == 'rgb8':
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                pass
            elif len(cv_image.shape) == 2:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            self.latest_image = cv_image
        except CvBridgeError as e:
            self.dual_logger.error(f'cv_bridge error: {str(e)}')

    def lidar_callback(self, msg: PointCloud2):
        # Store for obstacle avoidance algorithms
        self.latest_pointcloud = msg
        self.behav_planner.process_pointcloud(msg)

    def odom_callback(self, msg: Odometry):
        self.behav_planner.process_odom(msg)

    def timer_callback(self):
        if self.latest_image is None or self.is_processing:
            self.dual_logger.info("No image or already processing...")
            return

        self.dual_logger.info("Starting image processing...")
        self.is_processing = True
        try:
            # Delegate to the core logic layer
            self.detector_core.process_image(self.latest_image.copy())
        except Exception as e:
            self.dual_logger.error(f'process_image failed: {str(e)}')
        finally:
            self.is_processing = False

    def control_loop(self):
        """
        闭环控制接口：读取 detector_core 和 behav_planner
        """
        # 可以直接采用 behav_planner 生成的最优命令
        msg = self.behav_planner.compute_velocity()
        
        # （原有基于 Vision 的 P 控制已作为备用参考，或者组合使用）
        meas = self.detector_core.latest_measurement
        if meas is not None:
            distance_m, bearing_deg = meas
            # 例如：当有目标时可覆盖或引入 behav 逻辑
            
        self.cmd_pub.publish(msg)

def run_instruction_pipeline():
    print("Running initial instruction reasoning...")
    language_instruction = 'Walk to the red car and stop in front of it'
    reference_list = ['Stay on', 'Avoid', 'Yield', 'Stop']
    reference_costs = [0, 0.5, 0.7, 1]

    # Use the separated instruction processor logic
    instruction_breakdown = get_instruction_breakdown(language_instruction)
    extracted_lists = extract_lists_from_dict(instruction_breakdown)

    landmark_list = get_ith_key_list(instruction_breakdown, key_idx=1)
    navigation_action_list = get_ith_key_list(instruction_breakdown, key_idx=2)
    behavioral_action_list = get_ith_key_list(instruction_breakdown, key_idx=3)
    behavioral_target_list = get_ith_key_list(instruction_breakdown, key_idx=4)

    print("Landmarks List:", landmark_list)
    print("Navigation Actions List:", navigation_action_list)
    print("Behavioral Actions List:", behavioral_action_list)
    print("Behavioral Targets List:", behavioral_target_list)

    similarity_scores = get_similarity_scores(behavioral_action_list, reference_list)
    input_action_costs = calculate_input_action_costs(similarity_scores, reference_costs)
    print("Input Action Costs:\n", input_action_costs)

def main(args=None):
    # Optional: run the instruction text parsing first to generate `landmark_data.json` 
    # In full system, this could happen remotely, but we provide it here as part of main interface.
    # run_instruction_pipeline()
    
    rclpy.init(args=args)
    node = LandmarkDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
