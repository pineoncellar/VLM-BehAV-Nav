#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose
import sys
import threading
import cv2
import numpy as np
import math
import os

from rclpy.parameter import Parameter

class ImageMapPublisher(Node):
    def __init__(self, image_path):
        super().__init__('image_map_publisher')
        
        # VERY IMPORTANT for Gazebo: use sim time!
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        
        # Like BeHav/planner standalone scripts, simple QoS depth 10 handles standard default matching
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        
        self.image_path = image_path
        self.resolution = 0.05  # 5cm per pixel default
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_theta = 0.0 # radians
        
        self.img_width = 0
        self.img_height = 0
        self.grid_data = self.process_image(self.image_path)
        
        # Publish periodically to ensure late subscribers (like RViz2) receive it
        self.timer = self.create_timer(1.0, self.publish_map_timer_callback)
        self.publish_map()

    def publish_map_timer_callback(self):
        self.publish_map(log=False)


    def process_image(self, img_path):
        if not os.path.exists(img_path):
            self.get_logger().error(f"Image not found at {img_path}. Please check the path.")
            sys.exit(1)
            
        img = cv2.imread(img_path)
        if img is None:
            self.get_logger().error(f"Failed to load image at {img_path}")
            sys.exit(1)
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.img_height, self.img_width = gray.shape
        
        # Map pixels to OccupancyGrid values (0-100)
        # 0: Free space, 100: Obstacle, -1: Unknown
        map_data = np.zeros_like(gray, dtype=np.int8)
        
        # Gazebo maps tend to be gray and black. Let's make anything black (<=50) an obstacle
        # and everything else (>=51) free space to avoid invisible transparent gray!
        map_data[gray > 120] = 0        # Free space
        map_data[gray <= 120] = 100     # Obstacles
        # map_data[(gray >= 50) & (gray <= 240)] = -1  # (Removed to prevent transparent gray)
        
        # ROS map origin is bottom-left, OpenCV origin is top-left
        # So we must flip the image vertically
        flipped = cv2.flip(map_data, 0)
        return flipped.flatten().tolist()

    def publish_map(self, log=True):
        msg = OccupancyGrid()
        now = self.get_clock().now().to_msg()
        msg.header.stamp = now
        msg.header.frame_id = 'map'
        
        msg.info = MapMetaData()
        msg.info.map_load_time = now
        msg.info.resolution = self.resolution
        msg.info.width = self.img_width
        msg.info.height = self.img_height
        
        # Setup coordinates (origin of the map)
        msg.info.origin = Pose()
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        
        # Convert yaw (theta) to quaternion
        msg.info.origin.orientation.z = math.sin(self.origin_theta / 2.0)
        msg.info.origin.orientation.w = math.cos(self.origin_theta / 2.0)
        
        msg.data = self.grid_data
        self.map_pub.publish(msg)
        if log:
            self.get_logger().info(f"Map updated | X: {self.origin_x:.3f}, Y: {self.origin_y:.3f}, Theta: {math.degrees(self.origin_theta):.1f}°, Res: {self.resolution:.4f}")

def input_loop(node):
    print("\n" + "="*40)
    print("Map Alignment Tool Started")
    print("="*40)
    print("Commands:")
    print("  x <value>     - Set X origin in meters (e.g., x -5.0)")
    print("  y <value>     - Set Y origin in meters (e.g., y 2.5)")
    print("  theta <value> - Set Yaw (rotation) in degrees (e.g., theta 90)")
    print("  res <value>   - Set resolution in m/px (e.g., res 0.05)")
    print("  q or quit     - Exit program")
    print("="*40 + "\n")
    
    while rclpy.ok():
        try:
            cmd = input("> ").strip().lower()
            if not cmd:
                continue
            if cmd in ['q', 'quit', 'exit']:
                rclpy.shutdown()
                break
                
            parts = cmd.split()
            if len(parts) != 2:
                print("Invalid format. Use a command followed by a number (e.g. 'x 2.5').")
                continue
                
            command, val_str = parts[0], parts[1]
            val = float(val_str)
            
            if command == 'x':
                node.origin_x = val
            elif command == 'y':
                node.origin_y = val
            elif command == 'theta':
                node.origin_theta = math.radians(val)  # degrees to radians
            elif command == 'res':
                if val <= 0:
                    print("Resolution must be greater than 0.")
                    continue
                node.resolution = val
            else:
                print(f"Unknown command: {command}")
                continue
                
            # Republish map with new settings
            node.publish_map()
                
        except ValueError:
            print("Invalid value. Please enter a valid number.")
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    # Define image path
    current_dir = os.path.dirname(os.path.realpath(__file__))
    default_image_path = os.path.join(current_dir, "input", "map.jpg")
    
    image_path = default_image_path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
    print(f"Target map image: {image_path}")
    
    # Initialize Node
    node = ImageMapPublisher(image_path)
    
    # Run ROS spin in a separate daemon thread so it can process ROS callbacks (if any) while we block on input()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    
    # Run interactive command prompt loop in main thread
    input_loop(node)
    
    # Cleanup on exit
    if rclpy.ok():
        rclpy.shutdown()
    print("Exiting Map Publisher...")

if __name__ == '__main__':
    main()
