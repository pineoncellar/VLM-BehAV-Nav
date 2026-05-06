import cv2
import math
import os
import datetime
import logging
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError

# Import algorithms orchestration pipeline
from main import BehavMainPipeline

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
    """日志模块，既输出到 ROS 控制台，也写入文件"""
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
    
    def debug(self, msg):
        self.ros_logger.debug(msg)
        self.file_logger.debug(msg)

class LandmarkDetectorNode(Node):
    """ros2 节点：整合视觉地标检测、指令分解与行为规划，提供 ROS 接口"""
    def __init__(self):
        super().__init__('landmark_detector_node')
        
        self.file_logger = setup_file_logger()
        self.dual_logger = DualLogger(self.get_logger(), self.file_logger)
        
        self.dual_logger.info('Started landmark_detector_node')

        # Initialize the algorithm pipeline logic
        self.pipeline = BehavMainPipeline(logger=self.dual_logger)
        
        self.declare_parameter('instruction', 'Walk on the road, avoid stepping on the grass, go straight to the fire hydrant, turn right and keep walking until you reach the post office, then stop.')
        instruction = self.get_parameter('instruction').value
        
        print("\n" + "="*50)
        print("    欢迎使用 VLM-BehAV-Nav 导航系统")
        print("="*50)
        
        skip_nlp = os.environ.get('SKIP_NLP') == '1'
        
        if skip_nlp:
            print("[系统] SKIP_NLP=1，自动跳过用户输入和底层的LLM大语言模型推理流程。")
            instruction = 'Walk on the road, avoid stepping on the grass, go straight to the fire hydrant, turn right and keep walking until you reach the post office, then stop.'
        else:
            user_input = input(">>> 请输入您的自然语言导航指令\n(直接回车将使用默认指令): ").strip()
            if user_input:
                instruction = user_input
                
        print(f"\n[系统] 正在处理指令: {instruction}\n")
        
        self.dual_logger.info(f"Using instruction: {instruction}")
        
        # 将指令与 skip_nlp 标志传递给 pipeline
        self.pipeline.run_instruction_reasoning(instruction, skip_nlp=skip_nlp)
        
        # ROS节点定义
        self.image_topic = "/camera_sensor/image_raw"
        self.depth_topic = "/camera_sensor/depth/image_raw"
        self.lidar_topic = "/velodyne_points"
        self.odom_topic = "/odom"
        self.cmd_topic = "/cmd_vel"
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
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
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
        
        # 2. Control Interface / External Outputs

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.waypoint_pub = self.create_publisher(PoseStamped, '/current_waypoint_local', 10)
        self.rule_pub = self.create_publisher(String, '/semantic_behavior_rule', 10)

        self.behav_costmap_pub = self.create_publisher(Image, '/behav_costmap', 10)
        self.traj_image_pub = self.create_publisher(Image, '/traj_marked_image', 10)

        # 把 pipeline 的图像结果绑定到 ROS 原生发布对象上
        self.pipeline.on_behav_costmap = lambda m: self.behav_costmap_pub.publish(m)
        self.pipeline.on_traj_image = lambda m: self.traj_image_pub.publish(m)

        # 3. Timers
        self.timer = self.create_timer(self.period_sec, self.timer_callback)
        self.control_timer = self.create_timer(self.control_period_sec, self.control_loop)
        
        self.latest_image = None
        self.latest_depth_image = None
        self.is_processing = False

    def image_callback(self, msg: Image):
        self.dual_logger.debug('Received an image message')
        
        # 将原始 ros topic 数据发给 pipeline
        self.pipeline.update_sensor_data(image_msg=msg)
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.dual_logger.debug(f"Image converted successfully, shape: {cv_image.shape}")
            if msg.encoding == 'rgb8':
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                pass
            elif len(cv_image.shape) == 2:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            self.latest_image = cv_image
        except CvBridgeError as e:
            self.dual_logger.error(f'cv_bridge error: {str(e)}')

    def depth_callback(self, msg: Image):
        try:
            # depth images from gazebo are often 32FC1
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.latest_depth_image = cv_image
        except CvBridgeError as e:
            self.dual_logger.error(f'cv_bridge depth error: {str(e)}')

    def lidar_callback(self, msg: PointCloud2):
        self.pipeline.update_sensor_data(pointcloud_msg=msg)

    def odom_callback(self, msg: Odometry):
        self.pipeline.update_sensor_data(odom_msg=msg)

    def timer_callback(self):
        if self.latest_image is None or self.is_processing:
            self.dual_logger.info("No image or already processing...")
            return

        self.dual_logger.info("Starting image processing...")
        self.is_processing = True
        try:
            # Send latest image to vision landmark detector in pipeline
            depth_img = self.latest_depth_image.copy() if self.latest_depth_image is not None else None
            self.pipeline.process_vision_cv2(self.latest_image.copy(), depth_image=depth_img)
        except Exception as e:
            self.dual_logger.error(f'process_image failed: {str(e)}')
        finally:
            self.is_processing = False

    def control_loop(self):
        """
        核心控制闭环（通过 Pipeline 下发调用获取最终决策层给出的速度包）
        """
        cmd_override, wp_msg, semantic_msg = self.pipeline.compute_control_command()
        
        if cmd_override is not None:
            self.cmd_pub.publish(cmd_override)
            self.dual_logger.debug(f"==> [ros_interface] Publishing CMD override: {cmd_override.linear.x}, {cmd_override.angular.z}")
            
        if wp_msg is not None:
            self.waypoint_pub.publish(wp_msg)
            
        if semantic_msg is not None:
            self.rule_pub.publish(semantic_msg)

def run_instruction_pipeline():
    """测试用，可直接调用 Pipeline 单次测试 NLP 并获取行为 costs"""
    print("Running initial instruction reasoning...")
    pipeline = BehavMainPipeline()
    pipeline.run_instruction_reasoning()

def main(args=None):
    # Optional: run the instruction text parsing first
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
