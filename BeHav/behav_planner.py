import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
import threading

class BehavPlannerBridgeNode(Node):
    """
    一个简单的 ROS2 桥接节点，专门用于将高层的目标距离与角度下发给 Depthmap 模块。
    """
    def __init__(self):
        super().__init__('behav_planner_bridge_node')
        # 构建发布器，向 far_waypoint_planner 发送目标
        self.goal_pub = self.create_publisher(Point, '/behav/goal_polar', 10)

class BehavPlannerCore:
    def __init__(self, logger=None, goal_radius=2.5, goal_theta=0.0, goal_delta=0.0):
        self.logger = logger
        self.goal_radius = goal_radius
        self.goal_theta = goal_theta
        self.goal_delta = goal_delta
        
        self.prompts = []
        self.cost_values = []
        
        self.on_behav_costmap = None
        self.on_traj_image = None
        
        self.received_odom_once = True
        self.received_final_goal_odom = False

        self.bridge_node = BehavPlannerBridgeNode()
        # 必须使用独立的 Executor，以防和外部主线程的 rclpy.spin 发生全局 Executor 冲突
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.bridge_node)
        # 启动后台线程维持 ROS Spin 以处理发布
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

    def process_image(self, msg):
        pass

    def process_pointcloud(self, msg):
        pass

    def process_odom(self, msg):
        self.latest_odom = msg

    def compute_velocity(self):
        """
        不再在此处计算运动学轨迹，而是将 VLM 识别出的最新距离和角度下发给 depthmap 的 far_planner 去生成轨迹。
        返回 None 以通知 ros_interface 取消原生的 cmd_vel 下发，因为 depthmap 中的 local_planner 将接管控制。
        """
        if not getattr(self, 'has_new_target', False):
            return None
        self.has_new_target = False

        # 增加目标距离截断：提前停留在目标物前面一段距离，避免局部避障将其当成障碍物绕开
        # 实际停止距离 = stop_distance_margin (1.5) + far_waypoint_planner 的 final_goal_radius (1.0) = 2.5m
        stop_distance_margin = 1.5
        adjusted_radius = max(0.0, float(self.goal_radius) - stop_distance_margin)
        
        # Determine the base odom for calculating target
        base_odom = getattr(self, 'goal_odom', None)
        if base_odom is None:
            base_odom = getattr(self, 'latest_odom', None)

        polar_msg = Point()
        
        if base_odom is not None:
            # Anchor to the absolute map location
            px = base_odom.pose.pose.position.x
            py = base_odom.pose.pose.position.y
            q = base_odom.pose.pose.orientation
            
            # euler from quaternion
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            heading = yaw + math.radians(self.goal_theta)
            gx = px + adjusted_radius * math.cos(heading)
            gy = py + adjusted_radius * math.sin(heading)
            
            polar_msg.x = float(gx)
            polar_msg.y = float(gy)
            polar_msg.z = 1.0 # Signal to far_waypoint_planner that this is absolute
            mapped_type = "ABSOLUTE"
        else:
            # Fallback to local polar
            polar_msg.x = float(adjusted_radius)
            polar_msg.y = float(self.goal_theta)
            polar_msg.z = 0.0
            mapped_type = "POLAR"
        
        self.bridge_node.goal_pub.publish(polar_msg)
        
        if self.logger:
            self.logger.debug(f"[BehavPlannerCore] Redirecting Target to depthmap ({mapped_type}): coord=({polar_msg.x:.2f}, {polar_msg.y:.2f})")
            
        # 返回 Dummy_twist 让 ROS 接盘或者直接使用 None 处理
        return None
