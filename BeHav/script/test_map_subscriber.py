#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from nav_msgs.msg import OccupancyGrid
import numpy as np

class MapVerificationSubscriber(Node):
    def __init__(self):
        super().__init__('map_verification_subscriber')
        # 同样开启 sim_time 保持时钟体系一致
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        
        # 订阅 /map 话题，QoS 使用默认的 10 即可匹配
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        self.get_logger().info("Map verification subscriber initialized. Waiting for /map data...")
        self.received_count = 0

    def map_callback(self, msg: OccupancyGrid):
        self.received_count += 1
        self.get_logger().info(f"==== Received Map Data #{self.received_count} ====")
        
        # 1. 检查基础网格信息
        width = msg.info.width
        height = msg.info.height
        res = msg.info.resolution
        self.get_logger().info(f"[Metadata] Frame ID: {msg.header.frame_id}")
        self.get_logger().info(f"[Metadata] Stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
        self.get_logger().info(f"[Grid Info] Width: {width}, Height: {height}, Resolution: {res:.4f} m/px")
        
        # 2. 检查 Origin
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y
        self.get_logger().info(f"[Origin] x: {ox:.3f}, y: {oy:.3f}, z: {msg.info.origin.position.z:.3f}")
        
        # 3. 检查数据长度是否合法
        data_len = len(msg.data)
        expected_len = width * height
        if data_len == expected_len:
            self.get_logger().info(f"[Data Check] ✅ Array length ({data_len}) strictly matched width * height.")
        else:
            self.get_logger().warn(f"[Data Check] ❌ Array length ({data_len}) DOES NOT match expected {expected_len}!")
            
        # 4. 分析具体包含的值
        data_np = np.array(msg.data)
        unique_vals, counts = np.unique(data_np, return_counts=True)
        val_distribution = dict(zip(unique_vals, counts))
        self.get_logger().info(f"[Value Distribution] {val_distribution}")
        
        if 0 in val_distribution:
            self.get_logger().info(f" -> Found {val_distribution[0]} free space cells (white)")
        if 100 in val_distribution:
            self.get_logger().info(f" -> Found {val_distribution[100]} obstacle cells (black)")
        if -1 in val_distribution:
            self.get_logger().info(f" -> Found {val_distribution[-1]} unknown cells (gray)")
            
        self.get_logger().info("=========================================\n")


def main(args=None):
    rclpy.init(args=args)
    node = MapVerificationSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
