#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
import os

class CmdVelMonitor(Node):
    def __init__(self):
        super().__init__('cmd_vel_monitor_node')
        self.declare_parameter('topic', '/cmd_vel')
        self.topic = self.get_parameter('topic').value
        
        self.subscription = self.create_subscription(
            Twist,
            self.topic,
            self.cmd_callback,
            10)
            
        self.last_time = None
        self.interval_list = []
        self.get_logger().info(f"Started CmdVelMonitor for experimental evaluation on {self.topic}")
        
        self.log_file = os.path.join(os.path.dirname(__file__), "exp1_hz_log.txt")
        if os.environ.get('ENABLE_EXP1_LATENCY_LOG') == '1':
            with open(self.log_file, 'w') as f:
                f.write("timestamp,interval_sec,hz\n")
            self.get_logger().info(f"Logging intervals to {self.log_file}")

    def cmd_callback(self, msg):
        current_time = time.time()
        
        if self.last_time is not None:
            interval = current_time - self.last_time
            self.interval_list.append(interval)
            hz = 1.0 / interval if interval > 0 else 0
            
            if os.environ.get('ENABLE_EXP1_LATENCY_LOG') == '1':
                with open(self.log_file, 'a') as f:
                    f.write(f"{current_time},{interval:.5f},{hz:.2f}\n")
                    
                # To prevent spamming the console, only print every 20 messages (~2 secs at 10Hz)
                if len(self.interval_list) % 20 == 0:
                    avg_hz = 1.0 / (sum(self.interval_list[-20:]) / 20)
                    self.get_logger().info(f"[EXP1_LOG] Average control frequency over last 20 msgs: {avg_hz:.2f} Hz")
        
        self.last_time = current_time

def main(args=None):
    rclpy.init(args=args)
    
    if os.environ.get('ENABLE_EXP1_LATENCY_LOG') != '1':
        print("Warning: ENABLE_EXP1_LATENCY_LOG is not 1. Experimental logging is disabled.", flush=True)
        print("To enable, prefix your command with 'ENABLE_EXP1_LATENCY_LOG=1'", flush=True)
    else:
        print("CmdVel frequencies will be recorded for Experiment 1.", flush=True)

    monitor = CmdVelMonitor()
    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        monitor.get_logger().info("Keyboard Interrupt. Stopping monitor.")
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
