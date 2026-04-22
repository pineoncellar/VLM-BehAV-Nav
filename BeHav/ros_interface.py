import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Import algorithms
from instruction_processor import get_instruction_breakdown, extract_lists_from_dict, get_similarity_scores, calculate_input_action_costs, get_ith_key_list
from landmark_vision import LandmarkDetectorCore

class LandmarkDetectorNode(Node):
    def __init__(self):
        super().__init__('landmark_detector_node')
        self.get_logger().info('Started landmark_detector_node')

        # Initialize the algorithm logic instance and inject ROS logger
        self.detector_core = LandmarkDetectorCore(logger=self.get_logger())
        
        # Override some properties if needed
        self.image_topic = "/camera_sensor/image_raw"
        self.period_sec = 10.0

        # ========= ROS =========
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            1
        )
        self.timer = self.create_timer(self.period_sec, self.timer_callback)
        
        self.latest_image = None
        self.is_processing = False

    def image_callback(self, msg: Image):
        self.get_logger().info('Received an image message')
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.get_logger().info(f"Image converted successfully, shape: {cv_image.shape}")
            if msg.encoding == 'rgb8':
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                pass
            elif len(cv_image.shape) == 2:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            self.latest_image = cv_image
        except CvBridgeError as e:
            self.get_logger().error(f'cv_bridge error: {str(e)}')

    def timer_callback(self):
        if self.latest_image is None or self.is_processing:
            self.get_logger().info("No image or already processing...")
            return

        self.get_logger().info("Starting image processing...")
        self.is_processing = True
        try:
            # Delegate to the core logic layer
            self.detector_core.process_image(self.latest_image.copy())
        except Exception as e:
            self.get_logger().error(f'process_image failed: {str(e)}')
        finally:
            self.is_processing = False

def run_instruction_pipeline():
    print("Running initial instruction reasoning...")
    language_instruction = 'Walk to the red car and stop in front of it'
    reference_list = ['Stay on', 'Avoid', 'Yield', 'Stop']
    reference_costs = [0, 0.5, 0.7, 1]

    # Use the separated instruction processor logic
    instruction_breakdown = get_instruction_breakdown(language_instruction)
    extracted_lists = extract_lists_from_dict(instruction_breakdown)

    landmark_list = get_ith_key_list(instruction_breakdown, key_idx=1)
    navigation_action_list = get_ith_key_list(instruction_breakdown, key_idx=2)
    behavioral_action_list = get_ith_key_list(instruction_breakdown, key_idx=3)
    behavioral_target_list = get_ith_key_list(instruction_breakdown, key_idx=4)

    print("Landmarks List:", landmark_list)
    print("Navigation Actions List:", navigation_action_list)
    print("Behavioral Actions List:", behavioral_action_list)
    print("Behavioral Targets List:", behavioral_target_list)

    similarity_scores = get_similarity_scores(behavioral_action_list, reference_list)
    input_action_costs = calculate_input_action_costs(similarity_scores, reference_costs)
    print("Input Action Costs:\n", input_action_costs)

def main(args=None):
    # Optional: run the instruction text parsing first to generate `landmark_data.json` 
    # In full system, this could happen remotely, but we provide it here as part of main interface.
    # run_instruction_pipeline()
    
    rclpy.init(args=args)
    node = LandmarkDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
