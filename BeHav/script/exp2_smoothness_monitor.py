#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import numpy as np
import matplotlib.pyplot as plt
import os
import signal
import sys

class SmoothnessMonitor(Node):
    def __init__(self):
        super().__init__('smoothness_monitor')
        self.cmd_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_callback,
            10
        )
        self.angular_z_data = []
        self.time_data = []
        self.start_time = None
        self.get_logger().info('Smoothness monitor started, waiting for /cmd_vel...')

    def cmd_callback(self, msg):
        current_time = self.get_clock().now().nanoseconds / 1e9
        if self.start_time is None:
            self.start_time = current_time
            
        rel_time = current_time - self.start_time
        
        # Limit data to two minutes
        if rel_time > 120.0:
            return

        self.time_data.append(rel_time)
        self.angular_z_data.append(msg.angular.z)

    def save_and_plot(self):
        if not self.angular_z_data:
            self.get_logger().warn('No /cmd_vel data received to plot.')
            return

        out_dir = '/home/zyy/VLM-BehAV-Nav/BeHav/Image_plots'
        os.makedirs(out_dir, exist_ok=True)
        
        is_disabled = os.environ.get('DISABLE_ANTI_JITTER', '0') == '1'
        prefix = 'Group_A_NoAntiJitter' if is_disabled else 'Group_B_Ours'
        
        plt.figure(figsize=(10, 5))
        color = 'red' if is_disabled else 'blue'
        linestyle = '--' if is_disabled else '-'
        label = 'A - No Anti-Jitter' if is_disabled else 'B - Ours'
        
        plt.plot(self.time_data, self.angular_z_data, color=color, linestyle=linestyle, label=label)
        plt.title('Angular Velocity vs Time over Narrow Pass')
        plt.xlabel('Time (s)')
        plt.ylabel('Angular Velocity $\\omega_z$ (rad/s)')
        plt.grid(True)
        plt.legend()
        
        plot_path = os.path.join(out_dir, f'{prefix}_angular_z.png')
        plt.savefig(plot_path)
        plt.close()
        
        var_z = np.var(self.angular_z_data)
        
        # Save raw data for combined plotting later
        np.savez(os.path.join(out_dir, f'{prefix}_data.npz'), time=self.time_data, angular=self.angular_z_data, var=var_z)
        
        print(f"\n==========================")
        print(f"Experiment completed for {prefix}")
        print(f"Data points collected: {len(self.angular_z_data)}")
        print(f"Angular Velocity Variance: {var_z:.4f} rad^2/s^2")
        print(f"Plot saved to: {plot_path}")
        print(f"==========================\n")

def main(args=None):
    rclpy.init(args=args)
    monitor = SmoothnessMonitor()
    
    def signal_handler(sig, frame):
        print('Interrupted, saving plot...')
        monitor.save_and_plot()
        rclpy.shutdown()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()