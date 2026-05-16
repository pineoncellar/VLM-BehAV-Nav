import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from datetime import datetime

class PhotoTaker(Node):
    def __init__(self):
        super().__init__('photo_taker')
        self.bridge = CvBridge()
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(script_dir, "input")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.rgb_saved = False
        self.depth_saved = False
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.image_sub = self.create_subscription(
            Image,
            '/camera_sensor/image_raw',
            self.image_callback,
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            '/camera_sensor/depth/image_raw',
            self.depth_callback,
            10
        )
        
        self.get_logger().info('正在等待图像数据 (RGB 和 深度图)...')
        
    def image_callback(self, msg):
        if not self.rgb_saved:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                path = os.path.join(self.output_dir, f"{self.timestamp}_rgb.jpg")
                cv2.imwrite(path, cv_image)
                self.get_logger().info(f'RGB图保存到: {path}')
                self.rgb_saved = True
                self.check_done()
            except Exception as e:
                self.get_logger().error(f'RGB保存失败: {e}')
                
    def depth_callback(self, msg):
        if not self.depth_saved:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                # Save as npy to preserve float32 values accurately
                path_npy = os.path.join(self.output_dir, f"{self.timestamp}_depth.npy")
                np.save(path_npy, cv_image)
                # Also save a pseudo-color image for visualization if desired, but npy is enough
                self.get_logger().info(f'深度图(npy格式)保存到: {path_npy}')
                self.depth_saved = True
                self.check_done()
            except Exception as e:
                self.get_logger().error(f'深度图保存失败: {e}')
                
    def check_done(self):
        if self.rgb_saved and self.depth_saved:
            self.get_logger().info('所有图像已保存，退出中...')
            raise SystemExit

def main(args=None):
    rclpy.init(args=args)
    node = PhotoTaker()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
