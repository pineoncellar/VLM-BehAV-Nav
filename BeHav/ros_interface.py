import cv2
import math
import os
import time
import datetime
import logging
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CompressedImage
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

# Import algorithms orchestration pipeline
from main import BehavMainPipeline

def setup_file_logger():
    # 改为存入根目录下的 logs/BeHav 文件夹中
    log_dir = "../logs/BeHav"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{timestamp}.log")
    
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
        llm_start = time.time()
        self.pipeline.run_instruction_reasoning(instruction, skip_nlp=skip_nlp)
        llm_end = time.time()
        
        if os.environ.get('ENABLE_EXP1_LATENCY_LOG') == '1':
            self.dual_logger.info(f"[EXP1_LOG] 大语言模型指令推理端到端耗时: {llm_end - llm_start:.4f} 秒, timestamp: {llm_end}")
        
        # ROS节点定义
        self.declare_parameter('rgb_topic', '/gemini330/color/image_raw')
        self.declare_parameter('depth_topic', '/gemini330/depth/image_raw')
        self.image_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.lidar_topic = "/tower/mapping/cloud_colored"
        self.odom_topic = "/tita4264886/chassis/odometry"
        self.cmd_topic = "/cmd_vel"
        self.period_sec = 10.0
        self.control_period_sec = 0.1 # 10 Hz for control loop

        # ========= ROS =========
        self.bridge = CvBridge()
        
        # 1. Perception Interfaces
        rgb_msg_type = CompressedImage if 'compressed' in self.image_topic else Image
        self.image_sub = self.create_subscription(
            rgb_msg_type,
            self.image_topic,
            self.image_callback,
            1
        )
        depth_msg_type = CompressedImage if 'compressed' in self.depth_topic else Image
        self.depth_sub = self.create_subscription(
            depth_msg_type,
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
        self.behav_costmap_pub = self.create_publisher(Image, '/behav_costmap', 10)
        self.traj_image_pub = self.create_publisher(Image, '/traj_marked_image', 10)
        self.vision_image_pub = self.create_publisher(Image, '/vision_marked_image', 10)

        # 把 pipeline 的图像结果绑定到 ROS 原生发布对象上
        self.latest_vision_image_bgr = None
        self.pipeline.on_behav_costmap = lambda m: self.behav_costmap_pub.publish(m)
        self.pipeline.on_traj_image = lambda m: self.traj_image_pub.publish(m)
        self.pipeline.on_vision_image = lambda cv_bgr: setattr(self, 'latest_vision_image_bgr', cv_bgr)

        # 3. Timers
        self.last_process_time = 0.0
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.control_timer = self.create_timer(self.control_period_sec, self.control_loop)
        self.vision_pub_timer = self.create_timer(1.0 / 5.0, self.vision_pub_loop) # 5Hz publish
        
        self.latest_image = None
        self.latest_depth_image = None
        self.is_processing = False

    def image_callback(self, msg):
        self.dual_logger.debug('Received an image message')
        
        # 将原始 ros topic 数据发给 pipeline
        self.pipeline.update_sensor_data(image_msg=msg)
        
        try:
            if hasattr(msg, 'format'):  # CompressedImage
                cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
                if len(cv_image.shape) == 2:
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
                else:
                    # compressed_imgmsg_to_cv2 默认通常转出 bgr8
                    pass
            else:  # Raw Image
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

    def depth_callback(self, msg):
        try:
            if hasattr(msg, 'format'):  # CompressedImage
                # 对于 Depth 通常是 16UC1压缩，ROS 中部分相机的 compressedDepth 包含 12 个字节的自定义头部
                format_str = msg.format.lower()
                
                # 尝试用标准 cv_bridge 解码
                cv_image = None
                try:
                    cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
                except Exception as e:
                    pass

                if cv_image is None:
                    # 如果标准解码失败，很可能是 RViz / Realsense 的带头部 compressedDepth
                    # 跳过前 12 字节头部，直接用 cv2.imdecode 解码后面真实的 png/tiff 数据
                    raw_data = np.frombuffer(msg.data, np.uint8)[12:]
                    cv_image = cv2.imdecode(raw_data, cv2.IMREAD_UNCHANGED)

                if cv_image is not None:
                    # 统一转换为米制的 32FC1 以适配现有管道
                    if cv_image.dtype == np.uint16:
                        cv_image = cv_image.astype(np.float32) / 1000.0
                    elif cv_image.dtype == np.float64:
                        cv_image = cv_image.astype(np.float32)
                    self.latest_depth_image = cv_image
                else:
                    self.dual_logger.error("Failed to decode compressed depth image.")
            else:
                # depth images from gazebo are often 32FC1
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
                self.latest_depth_image = cv_image
        except CvBridgeError as e:
            self.dual_logger.error(f'cv_bridge depth error: {str(e)}')

    def lidar_callback(self, msg: PointCloud2):
        self.pipeline.update_sensor_data(pointcloud_msg=msg)

    def odom_callback(self, msg: Odometry):
        self.pipeline.update_sensor_data(odom_msg=msg)

    def _process_thread(self, cv_image, depth_image, img_odom=None):
        vision_start = time.time()
        try:
            self.pipeline.process_vision_cv2(cv_image, depth_image=depth_image, img_odom=img_odom)
        except Exception as e:
            self.dual_logger.error(f'process_image failed: {str(e)}')
        finally:
            vision_end = time.time()
            if os.environ.get('ENABLE_EXP1_LATENCY_LOG') == '1':
                self.dual_logger.info(f"[EXP1_LOG] 视觉管线处理端到端耗时: {vision_end - vision_start:.4f} 秒, timestamp: {vision_end}")
            self.is_processing = False

    def timer_callback(self):
        if self.latest_image is None or self.is_processing:
            return
            
        if time.time() - self.last_process_time < self.period_sec:
            return

        self.last_process_time = time.time()
        self.dual_logger.info("Starting image processing...")
        self.is_processing = True
        import threading
        depth_img = self.latest_depth_image.copy() if self.latest_depth_image is not None else None
        cv_img = self.latest_image.copy()
        img_odom = getattr(self.pipeline, 'latest_odom', None)
        t = threading.Thread(target=self._process_thread, args=(cv_img, depth_img, img_odom))
        t.daemon = True
        t.start()

    def control_loop(self):
        """
        核心控制闭环（通过 Pipeline 下发调用获取最终决策层给出的速度包）
        """
        msg = self.pipeline.compute_control_command()
        if msg is not None:
            self.cmd_pub.publish(msg)
            self.dual_logger.debug(f"==> [ros_interface] Publishing CMD: {msg.linear.x}, {msg.angular.z}")

    def vision_pub_loop(self):
        """
        以 5Hz 的频率持续向外发布最新保存好的带状态标注帧，防止 RViz 断流或画面跳变闪烁
        """
        if getattr(self, 'latest_vision_image_bgr', None) is not None:
            try:
                img_msg = self.bridge.cv2_to_imgmsg(self.latest_vision_image_bgr, encoding="bgr8")
                self.vision_image_pub.publish(img_msg)
            except CvBridgeError as e:
                self.dual_logger.error(f'cv_bridge publish error: {str(e)}')

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
