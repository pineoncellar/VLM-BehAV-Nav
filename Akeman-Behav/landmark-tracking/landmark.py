import os
import re
import io
import time
import math
import json
import base64
from textwrap import dedent

import requests
from requests.exceptions import RequestException
import cv2
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from PIL import Image as PILImage


class LandmarkDetectorNode(Node):
    def __init__(self):
        super().__init__('landmark_detector_node')
        self.get_logger().info('Started landmark_detector_node')

        # ========= 固定配置：不从外部接收任务信息 =========
        self.api_key = "sk-04c50afd503a4a108b0112f3184ea921"
        self.vlm_model = "qwen3-vl-plus"

        # 固定导航目标列表：按顺序执行
        self.navigation_landmarks = ["traffic barrier", "library"]
        self.navigation_actions = ["go to", "then go to"]  # 保留但不参与计算

        # 只订阅图像
        self.image_topic = "/color/image_raw"

        # 每10秒执行一次
        self.period_sec = 10.0

        # 固定相机水平视场角，用于由像素位置估算角度
        self.horizontal_fov_deg = 69.0

        # 如果模型没给出距离，就用默认值
        self.default_distance_m = 5.0
        self.use_vlm_distance = True

        # 距离小于这个值时，切换到下一个 landmark
        self.goal_reached_threshold_m = 3.0

        # 调试图
        self.save_debug_plot = True
        self.save_debug_dir = "./Image_plots/"

        # ========= 运行时状态 =========
        self.current_landmark_index = 0
        self.latest_image = None
        self.latest_measurement = None   # [distance_m, bearing_rad]
        self.is_processing = False

        self.max_retries = 3
        self.delay = 5

        # ========= ROS =========
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            1
        )
        self.timer = self.create_timer(self.period_sec, self.timer_callback)

    # ============================================================
    # 基础函数
    # ============================================================
    def current_target_text(self) -> str:
        return self.navigation_landmarks[self.current_landmark_index]

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
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
            return

        self.is_processing = True
        try:
            self.process_image(self.latest_image.copy())
        except Exception as e:
            self.get_logger().error(f'process_image failed: {str(e)}')
        finally:
            self.is_processing = False

    def load_image(self, image_np_rgb):
        pil_image = PILImage.fromarray(image_np_rgb)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

    def make_api_request(self, headers, data):
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60
                )
                response.raise_for_status()
                return response.json()
            except RequestException as e:
                if attempt + 1 < self.max_retries:
                    time.sleep(self.delay)
                else:
                    self.get_logger().error(f'API request failed: {e}')
                    return None

    # ============================================================
    # 几何辅助
    # ============================================================
    def compute_bearing(self, pixel_x, image_width):
        """
        返回角度制（degree）
        假设使用 D435 RGB，相机水平视场角约 69°
        像素原点在左上角，x 向右增加
        """
        if image_width <= 1:
            return 0.0

        cx = (image_width - 1) / 2.0
        half_fov_rad = math.radians(self.horizontal_fov_deg / 2.0)

        # 用已知 FOV 近似恢复 fx（像素单位）
        fx = cx / math.tan(half_fov_rad)

        # 针孔模型：theta = atan((x - cx) / fx)
        angle_rad = math.atan((pixel_x - cx) / fx)

        # 直接返回角度制
        return math.degrees(angle_rad)

    def maybe_advance_to_next_landmark(self, distance_m):
        if distance_m is None:
            return
        if distance_m <= self.goal_reached_threshold_m:
            if self.current_landmark_index + 1 < len(self.navigation_landmarks):
                prev_target = self.current_target_text()
                self.current_landmark_index += 1
                print(f'[LandmarkDetector] reached "{prev_target}", switch to "{self.current_target_text()}"')
            else:
                print('[LandmarkDetector] final landmark reached')

    # ============================================================
    # 调试可视化
    # ============================================================
    def get_next_file_number(self):
        os.makedirs(self.save_debug_dir, exist_ok=True)
        files = os.listdir(self.save_debug_dir)
        existing_numbers = []
        for f in files:
            if f.startswith('Image_plots_') and f.endswith('.jpg'):
                match = re.search(r'Image_plots_(\d+)\.jpg', f)
                if match:
                    existing_numbers.append(int(match.group(1)))
        return max(existing_numbers, default=0) + 1

    def draw_detection_overlay(self, image_rgb, x, y, target_text, bearing, distance):
        vis = image_rgb.copy()
        cv2.circle(vis, (int(x), int(y)), 18, (255, 0, 0), -1)
        label = f"{target_text} | bearing={bearing:.2f} rad | dist={distance:.1f} m"
        cv2.putText(
            vis, label, (max(10, int(x) - 80), max(30, int(y) - 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
        )
        return vis

    def save_images(self, original_img_rgb, circled_img_rgb, x, y, distance, bearing, target_text):
        os.makedirs(self.save_debug_dir, exist_ok=True)
        plt.figure(figsize=(12, 5))

        plt.subplot(121)
        plt.imshow(original_img_rgb)
        plt.title('Original image')
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(circled_img_rgb)
        plt.title('Detected target')
        plt.axis('off')

        plt.tight_layout()
        next_number = self.get_next_file_number()
        save_path = os.path.join(self.save_debug_dir, f"Image_plots_{next_number}.jpg")

        text = (
            f"Target: {target_text}\n"
            f"x={x}, y={y}\n"
            f"Distance(m): {distance}\n"
            f"Bearing(rad): {bearing:.3f}"
        )
        plt.figtext(
            0.06, 0.95, text,
            ha="center", va="top", fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.7, "pad": 5}
        )
        plt.savefig(save_path)
        plt.close()

    # ============================================================
    # VLM 查询
    # ============================================================
    def query_target_position_and_distance(self, image_rgb, target_text):
        h, w = image_rgb.shape[:2]

        prompt = dedent(f"""
            You are given ONE outdoor scene image.

            Target landmark description:
            "{target_text}"

            Image size:
            width = {w} pixels
            height = {h} pixels

            Your task:
            1. Decide whether the target landmark is visible in the image.
            2. If visible, estimate the CENTER pixel coordinates (x, y) of the target landmark.
            3. If visible, estimate the approximate camera-to-target distance in meters.

            Important rules:
            - x must be an integer in [0, {w - 1}]
            - y must be an integer in [0, {h - 1}]
            - (x, y) must be the visual center of the target landmark in the image
            - If the landmark is only partially visible, estimate the center of the visible landmark region
            - If the target is not visible, set x, y, and distance_m to null
            - Return JSON only
            - Do not add markdown
            - Do not add explanations

            Return exactly this JSON schema:
            {{
              "visible": true or false,
              "x": integer or null,
              "y": integer or null,
              "distance_m": number or null
            }}
        """).strip()

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            "model": self.vlm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self.load_image(image_rgb),
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0,
            "max_tokens": 300
        }

        response_json = self.make_api_request(headers, data)
        if not response_json:
            return ""

        return response_json["choices"][0]["message"]["content"].strip()

    def parse_vlm_response(self, response_text):
        result = {
            "visible": False,
            "x": None,
            "y": None,
            "distance_m": None,
            "raw": response_text
        }

        if not response_text:
            return result

        cleaned = response_text.strip()

        # 去掉 ```json ... ``` 包裹
        cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # 优先按 JSON 解析
        try:
            data = json.loads(cleaned)
            result["visible"] = bool(data.get("visible", False))
            result["x"] = data.get("x", None)
            result["y"] = data.get("y", None)
            result["distance_m"] = data.get("distance_m", None)
            return result
        except Exception:
            pass

        # 兜底：正则解析
        visible_match = re.search(r'"visible"\s*:\s*(true|false)', cleaned, re.IGNORECASE)
        if visible_match:
            result["visible"] = (visible_match.group(1).lower() == "true")

        x_match = re.search(r'"x"\s*:\s*([0-9]+|null)', cleaned, re.IGNORECASE)
        if x_match and x_match.group(1).lower() != "null":
            result["x"] = int(x_match.group(1))

        y_match = re.search(r'"y"\s*:\s*([0-9]+|null)', cleaned, re.IGNORECASE)
        if y_match and y_match.group(1).lower() != "null":
            result["y"] = int(y_match.group(1))

        d_match = re.search(r'"distance_m"\s*:\s*([0-9]+(?:\.[0-9]+)?|null)', cleaned, re.IGNORECASE)
        if d_match and d_match.group(1).lower() != "null":
            result["distance_m"] = float(d_match.group(1))

        return result

    # ============================================================
    # 主流程
    # ============================================================
    def process_image(self, image_bgr):
        target_text = self.current_target_text()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        response_text = self.query_target_position_and_distance(
            image_rgb=image_rgb,
            target_text=target_text
        )

        parsed = self.parse_vlm_response(response_text)

        if not parsed["visible"] or parsed["x"] is None or parsed["y"] is None:
            self.latest_measurement = None
            print(f'[LandmarkDetector] target="{target_text}" not found')
            return

        x = int(parsed["x"])
        y = int(parsed["y"])

        # 防止模型返回越界坐标
        h, w = image_rgb.shape[:2]
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))

        bearing = self.compute_bearing(x, w)

        if self.use_vlm_distance and parsed["distance_m"] is not None:
            distance_m = float(parsed["distance_m"])
        else:
            distance_m = self.default_distance_m

        self.latest_measurement = [distance_m, float(bearing)]

        print('--------------------------------------------------')
        print(f'Current target   : {target_text}')
        print(f'Pixel location   : x={x}, y={y}')
        print(f'Distance / Angle : {self.latest_measurement}')
        print('--------------------------------------------------')

        circled_img_rgb = self.draw_detection_overlay(
            image_rgb, x, y, target_text, bearing, distance_m
        )

        if self.save_debug_plot:
            self.save_images(
                original_img_rgb=image_rgb,
                circled_img_rgb=circled_img_rgb,
                x=x,
                y=y,
                distance=distance_m,
                bearing=bearing,
                target_text=target_text
            )

        self.maybe_advance_to_next_landmark(distance_m)


def main(args=None):
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