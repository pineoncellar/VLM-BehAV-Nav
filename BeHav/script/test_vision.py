import sys
import os
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instruction_processor import get_instruction_breakdown, get_ith_key_list
from landmark_vision import LandmarkDetectorCore

class TestVisionNode(Node):
    def __init__(self, instruction):
        super().__init__('test_vision_node')

        self.get_logger().info(f"=== [Module 1] Instruction Processing ===")
        # 只启动NLP处理模块提取视觉所需信息
        breakdown = get_instruction_breakdown(instruction)
        self.landmarks = get_ith_key_list(breakdown, 1)
        self.nav_actions = get_ith_key_list(breakdown, 2)
        
        self.get_logger().info(f"Extracted Landmarks: {self.landmarks}")
        self.get_logger().info(f"Extracted Navigation Acts: {self.nav_actions}")

        self.get_logger().info(f"=== [Module 2] Landmark Vision ===")
        # 启动视觉处理检测模块
        self.detector = LandmarkDetectorCore(logger=self.get_logger())
        self.detector.navigation_landmarks = self.landmarks
        self.detector.navigation_actions = self.nav_actions

        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, '/gemini330/color/image_raw', self.image_callback, 1)
        
        self.get_logger().info("Subscribed to /gemini330/color/image_raw. Waiting for simulation images...")
        self.is_processing = False

    def image_callback(self, msg: Image):
        if self.is_processing:
            return
        
        self.is_processing = True
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if msg.encoding == 'rgb8':
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            elif len(cv_image.shape) == 2:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

            # 接收仿真图像并预测
            self.detector.process_image(cv_image)
            meas = self.detector.latest_measurement
            if meas:
                distance, bearing = meas
                self.get_logger().info(f"==> Detected Target '{self.landmarks}'! Distance: {distance:.2f}m, Bearing: {bearing:.2f}deg")
            else:
                self.get_logger().info(f"Target '{self.landmarks}' not in view. Searching...")
        except Exception as e:
            self.get_logger().error(f"Image processing error: {e}")
        finally:
            self.is_processing = False

def main(args=None):
    print("\n" + "="*50)
    print("    [测试模块] 视觉检测 (Instruction -> Vision)")
    print("="*50)
    
    instruction = input(">>> 请输入您的自然语言导航指令: ").strip()
    if not instruction:
        instruction = "Walk to the red car"
        print(f"使用默认指令: {instruction}")

    rclpy.init(args=args)
    node = TestVisionNode(instruction)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n测试结束")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
