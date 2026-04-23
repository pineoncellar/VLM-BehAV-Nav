import sys
import os
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import BehavMainPipeline

class TestPlanningNode(Node):
    def __init__(self, instruction):
        super().__init__('test_planning_node')

        self.get_logger().info(f"=== Initializing Full Pipeline for Planning Test ===")
        # 直接使用项目中包装好的完整Pipeline
        self.pipeline = BehavMainPipeline(logger=self.get_logger())

        # Module 1: Instructions (启动自然语言处理解析)
        self.get_logger().info(f"=== [Module 1] Instruction Processing ===")
        self.pipeline.run_instruction_reasoning(instruction)

        # Module 2 & 3: Vision + Planning (订阅所有需要的 ROS Node 消息)
        self.get_logger().info(f"=== [Module 2+3] Vision & Planning ===")
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/camera_sensor/image_raw', self.image_callback, 1)
        self.lidar_sub = self.create_subscription(PointCloud2, '/velodyne_points', self.lidar_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 1)

        # 控制指令发布计时器 (10Hz 轮询打印输出，代替直接发布)
        self.control_timer = self.create_timer(0.1, self.control_loop)
        # 视觉追踪计时器 (1Hz 处理一张画面)
        self.vision_timer = self.create_timer(1.0, self.vision_loop)

        self.latest_cv_image = None
        self.vision_is_processing = False

        self.get_logger().info("Subscribed to /camera_sensor/image_raw, /velodyne_points, /odom.")
        self.get_logger().info("Waiting for simulation data flow...")

    def image_callback(self, msg: Image):
        # Image 发给 Planning 用作行为 costmap 叠选
        self.pipeline.update_sensor_data(image_msg=msg)

        # 保存为 cv2 留给 Vision 模型单独以较低频率推理
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if msg.encoding == 'rgb8':
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            elif len(cv_image.shape) == 2:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            self.latest_cv_image = cv_image
        except Exception as e:
            self.get_logger().error(f"Image convert error: {e}")

    def vision_loop(self):
        if self.latest_cv_image is not None and not self.vision_is_processing:
            self.vision_is_processing = True
            try:
                # 传入最新图像送给视觉地标追踪模型
                self.pipeline.process_vision_cv2(self.latest_cv_image.copy())
            except Exception as e:
                self.get_logger().error(f"Vision processing error: {e}")
            finally:
                self.vision_is_processing = False

    def lidar_callback(self, msg: PointCloud2):
        self.pipeline.update_sensor_data(pointcloud_msg=msg)

    def odom_callback(self, msg: Odometry):
        self.pipeline.update_sensor_data(odom_msg=msg)

    def control_loop(self):
        try:
            # 融合传感器计算出避障+寻迹最终速度
            cmd = self.pipeline.compute_control_command()
            self.get_logger().info(f"==> [Planning CMD] linear.x: {cmd.linear.x:.2f}, angular.z: {cmd.angular.z:.2f}")
        except Exception as e:
            pass

def main(args=None):
    print("\n" + "="*50)
    print("    [测试模块] 总体规划输出 (Instruction -> Vision -> Planning)")
    print("="*50)
    
    instruction = input(">>> 请输入您的自然语言导航指令: ").strip()
    if not instruction:
        instruction = "Walk to the red car and avoid the person"
        print(f"使用默认指令: {instruction}")

    rclpy.init(args=args)
    node = TestPlanningNode(instruction)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n测试结束")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
