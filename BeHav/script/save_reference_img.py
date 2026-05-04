import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import sys
import os

class ImageSaver(Node):
    def __init__(self, topic_name, file_name):
        super().__init__('image_saver')
        self.bridge = CvBridge()
        self.file_name = file_name
        
        # 订阅相机话题
        self.subscription = self.create_subscription(
            Image,
            topic_name,
            self.image_callback,
            10)
        self.get_logger().info(f"等待接收话题 '{topic_name}' 的图像以保存至 '{file_name}'...")

    def image_callback(self, msg):
        try:
            # 转换为 OpenCV 格式 (BGR编码)
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 确保保存目录存在
            save_dir = os.path.dirname(os.path.abspath(self.file_name))
            if not os.path.exists(save_dir) and save_dir != '':
                os.makedirs(save_dir)
                
            # 保存图片
            cv2.imwrite(self.file_name, cv_image)
            self.get_logger().info(f"✅ 成功截取并保存图片: {self.file_name}")
            
            # 保存完一张图后自动退出节点
            sys.exit(0)
        except Exception as e:
            self.get_logger().error(f"保存图像失败: {e}")
            sys.exit(1)

def main(args=None):
    rclpy.init(args=args)
    
    # 默认值
    topic = '/camera_sensor/image_raw'  # 请根据你的小车相机真实话题修改
    filename = 'script/reference_images/ref_5m.jpg'
    
    # 支持命令行传参
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if len(sys.argv) > 2:
        topic = sys.argv[2]

    image_saver = ImageSaver(topic, filename)
    
    try:
        rclpy.spin(image_saver)
    except SystemExit:
        rclpy.logging.get_logger("image_saver").info("脚本运行结束。")
    finally:
        image_saver.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()