#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import os
import glob

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        
        # Declare parameters (paths to your image directories)
        self.declare_parameter('rgb_path', '/home/tita-remote/ZyyPlanner/word-vedio/004-1/rgb')
        self.declare_parameter('depth_path', '/home/tita-remote/ZyyPlanner/word-vedio/004-1/depth')
        self.declare_parameter('fps', 10.0)
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/depth/camera_info')
        
        # Get parameters
        self.rgb_path = self.get_parameter('rgb_path').value
        self.depth_path = self.get_parameter('depth_path').value
        self.fps = self.get_parameter('fps').value
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        
        # Publishers
        self.rgb_pub = self.create_publisher(Image, self.rgb_topic, 10)
        self.depth_pub = self.create_publisher(Image, self.depth_topic, 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, self.camera_info_topic, 10)
        self.camera_info_msg = self._build_camera_info()
        
        self.cv_bridge = CvBridge()
        
        # Collect image files, sorted by name to ensure sequential 'xxxxxx' order
        self.rgb_files = sorted(glob.glob(os.path.join(self.rgb_path, 'rgb_*.png')))
        self.depth_files = sorted(glob.glob(os.path.join(self.depth_path, 'depth_*.png')))
        
        if not self.rgb_files:
            self.get_logger().warn(f"No RGB images found in {self.rgb_path}. Looking for 'rgb_*.png'")
        if not self.depth_files:
            self.get_logger().warn(f"No Depth images found in {self.depth_path}. Looking for 'depth_*.png'")
            
        self.frame_idx = 0
        
        # Timer to read and publish frames at specified FPS
        period = 1.0 / self.fps if self.fps > 0 else 0.1
        self.timer = self.create_timer(period, self.timer_callback)
        self.get_logger().info(f"Image publisher started reading from:\nRGB: {self.rgb_path}\nDepth: {self.depth_path}\nat {self.fps} FPS")

    def _build_camera_info(self):
        msg = CameraInfo()
        msg.height = 480
        msg.width = 640
        msg.distortion_model = "plumb_bob"
        msg.d = [-0.03134607896208763, 0.034163735806941986, -0.0002452715125400573, 7.945882316562347e-06, -0.011619596742093563, 0.0, 0.0, 0.0]
        msg.k = [367.2445373535156, 0.0, 319.23297119140625, 0.0, 367.30047607421875, 236.9733428955078, 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [367.2445373535156, 0.0, 319.23297119140625, 0.0, 0.0, 367.30047607421875, 236.9733428955078, 0.0, 0.0, 0.0, 1.0, 0.0]
        msg.binning_x = 0
        msg.binning_y = 0
        msg.roi.x_offset = 0
        msg.roi.y_offset = 0
        msg.roi.height = 0
        msg.roi.width = 0
        msg.roi.do_rectify = False
        return msg

    def timer_callback(self):
        if not self.rgb_files and not self.depth_files:
            return
            
        # Loop back to begining if we reach the end of the video sequence
        if self.frame_idx >= len(self.rgb_files) and self.frame_idx >= len(self.depth_files):
            self.get_logger().info("Finished all frames, looping back to the beginning.")
            self.frame_idx = 0

        now = self.get_clock().now().to_msg()
        
        # Publish Camera Info
        self.camera_info_msg.header.stamp = now
        self.camera_info_msg.header.frame_id = "camera_depth_optical_frame"
        self.camera_info_pub.publish(self.camera_info_msg)

        # Read and publish RGB
        if self.frame_idx < len(self.rgb_files):
            rgb_file = self.rgb_files[self.frame_idx]
            cv_img = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
            if cv_img is not None:
                msg = self.cv_bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
                msg.header.stamp = now
                msg.header.frame_id = "camera_color_optical_frame"
                self.rgb_pub.publish(msg)

        # Read and publish Depth
        if self.frame_idx < len(self.depth_files):
            depth_file = self.depth_files[self.frame_idx]
            # IMREAD_UNCHANGED to read potential 16-bit depth values unmodified
            cv_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
            if cv_img is not None:
                # determine encoding by dtype
                encoding = "passthrough"
                if cv_img.dtype.name == 'uint16':
                    encoding = "16UC1"
                elif cv_img.dtype.name == 'uint8':
                    encoding = "mono8"
                    
                msg = self.cv_bridge.cv2_to_imgmsg(cv_img, encoding=encoding)
                msg.header.stamp = now
                msg.header.frame_id = "camera_depth_optical_frame"
                self.depth_pub.publish(msg)

        self.frame_idx += 1

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
