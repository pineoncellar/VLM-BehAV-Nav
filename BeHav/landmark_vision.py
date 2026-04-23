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

from PIL import Image as PILImage
import os
from dotenv import load_dotenv

load_dotenv()

class LandmarkDetectorCore:
    def __init__(self, logger=None):
        self.logger = logger
        if self.logger:
            self.logger.info('Initialized LandmarkDetectorCore logic')


        # ========= 读取地标数据 =========
        self.navigation_landmarks = self.load_landmarks_from_file("landmark_data.json")


        # ========= 从 .env 环境变量读取配置 =========
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "sk-e9d7e3da6d6240cd97b4d61af040415d")
        self.vlm_model = os.getenv("DASHSCOPE_VLM_MODEL", "qwen3-vl-plus")
        self.vlm_base_url = os.getenv("DASHSCOPE_VLF_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")

        # 固定导航目标列表：按顺序执行
        # self.navigation_landmarks = ["traffic barrier", "library"]
        self.navigation_actions = ["go to", "then go to"]  # 保留但不参与计算

        # 只订阅图像
        self.image_topic = "/camera_sensor/image_raw"

        # 每10秒执行一次
        self.period_sec = 10.0

        # 水平视场角（你可按自己的相机修改）
        self.horizontal_fov_deg = 69.0

        # 如果模型没给出距离，就用默认值
        self.default_distance_m = 5.0
        self.use_vlm_distance = True

        # 距离小于这个值时，切换到下一个 landmark
        self.goal_reached_threshold_m = 3.0

        # 调试图
        self.save_debug_plot = True
        self.save_debug_dir = "./Image_plots/"

        # 目标点选取模式：
        # "center"       -> bbox中心
        # "bottom_center"-> bbox底部中心（更适合路障/交通锥这类地面目标）
        self.target_point_mode = "center"

        # ========= 运行时状态 =========
        self.current_landmark_index = 0
        self.latest_image = None
        self.latest_measurement = None   # [distance_m, bearing_deg]
        self.is_processing = False

        self.max_retries = 3
        self.delay = 5
        self.latest_measurement = None

        # ROS 订阅和定时器 (Moved to interface node)
        # self.bridge = CvBridge()
        # self.image_sub = ...
        # self.timer = ...

    def load_landmarks_from_file(self, file_path):
        """ 从文件中读取地标数据，确保只提取 'landmarks' 部分 """
        try:
            if self.logger: self.logger.info(f"Loading landmarks from {file_path}")  # 打印日志
            with open(file_path, "r") as f:
                data = json.load(f)
                landmarks = data.get("landmarks", [])
                if self.logger: self.logger.info(f"Loaded landmarks: {landmarks}")  # 打印读取的地标
                return landmarks
        except Exception as e:
            if self.logger: self.logger.error(f"读取地标文件失败: {e}")
            return []  # 如果读取失败，返回空列表

    # ============================================================
    # 基础函数
    # ============================================================
    def current_target_text(self) -> str:
        if self.current_landmark_index < len(self.navigation_landmarks):
            return self.navigation_landmarks[self.current_landmark_index]
        return ""

    def load_image(self, image_np_rgb):
        pil_image = PILImage.fromarray(image_np_rgb)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

    def make_api_request(self, headers, data):
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.vlm_base_url,
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
                    if self.logger: self.logger.error(f'API request failed: {e}')
                    return None

    # ============================================================
    # 几何辅助
    # ============================================================
    def compute_bearing(self, pixel_x, image_width):
        """
        返回角度制（degree）
        图像原点在左上角，x 向右增加
        """
        if image_width <= 1:
            return 0.0

        cx = (image_width - 1) / 2.0
        half_fov_rad = math.radians(self.horizontal_fov_deg / 2.0)

        fx = cx / math.tan(half_fov_rad)
        angle_rad = math.atan((pixel_x - cx) / fx)
        return math.degrees(angle_rad)

    def bbox_to_target_point(self, x_min, y_min, x_max, y_max):
        if self.target_point_mode == "bottom_center":
            x = int((x_min + x_max) / 2)
            y = int(y_max)
        else:
            x = int((x_min + x_max) / 2)
            y = int((y_min + y_max) / 2)
        return x, y

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

    def draw_detection_overlay(self, image_rgb, x_min, y_min, x_max, y_max, x, y):
        vis = image_rgb.copy()
        cv2.rectangle(vis, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        cv2.circle(vis, (int(x), int(y)), 8, (255, 0, 0), -1)
        return vis

    def save_images(self, original_img_rgb, circled_img_rgb):
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
        plt.savefig(save_path)
        plt.close()

    # ============================================================
    # VLM 查询
    # ============================================================
    def query_target_bbox_and_distance(self, image_rgb, target_text):
        h, w = image_rgb.shape[:2]

        prompt = dedent(f"""
            You are given ONE outdoor scene image.

            Target landmark description:
            "{target_text}"

            Image size:
            width = {w} pixels
            height = {h} pixels

            Pixel coordinate system:
            - The origin (0, 0) is at the TOP-LEFT corner of the image
            - x increases to the RIGHT
            - y increases DOWNWARD

            Your task:
            1. Decide whether the target landmark is visible in the image.
            2. If visible, return a TIGHT bounding box around the visible target landmark:
               x_min, y_min, x_max, y_max
            3. If visible, estimate the approximate camera-to-target distance in meters.

            Important rules:
            - x_min and x_max must be integers in [0, {w - 1}]
            - y_min and y_max must be integers in [0, {h - 1}]
            - The box should tightly cover only the visible target landmark
            - If the landmark is partially visible, box only the visible part
            - If the target is not visible, set all coordinates and distance_m to null
            - Do not guess the image center when uncertain
            - Return JSON only
            - Do not add markdown
            - Do not add explanations

            Return exactly this JSON schema:
            {{
              "visible": true or false,
              "x_min": integer or null,
              "y_min": integer or null,
              "x_max": integer or null,
              "y_max": integer or null,
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
            "x_min": None,
            "y_min": None,
            "x_max": None,
            "y_max": None,
            "distance_m": None,
            "raw": response_text
        }

        if not response_text:
            return result

        cleaned = response_text.strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
            result["visible"] = bool(data.get("visible", False))
            result["x_min"] = data.get("x_min", None)
            result["y_min"] = data.get("y_min", None)
            result["x_max"] = data.get("x_max", None)
            result["y_max"] = data.get("y_max", None)
            result["distance_m"] = data.get("distance_m", None)
            return result
        except Exception:
            pass

        visible_match = re.search(r'"visible"\s*:\s*(true|false)', cleaned, re.IGNORECASE)
        if visible_match:
            result["visible"] = (visible_match.group(1).lower() == "true")

        for key in ["x_min", "y_min", "x_max", "y_max"]:
            m = re.search(rf'"{key}"\s*:\s*([0-9]+|null)', cleaned, re.IGNORECASE)
            if m and m.group(1).lower() != "null":
                result[key] = int(m.group(1))

        d_match = re.search(r'"distance_m"\s*:\s*([0-9]+(?:\.[0-9]+)?|null)', cleaned, re.IGNORECASE)
        if d_match and d_match.group(1).lower() != "null":
            result["distance_m"] = float(d_match.group(1))

        return result

    # ============================================================
    # 主流程
    # ============================================================
    def process_image(self, image_bgr):
        if self.logger: self.logger.info("Processing image...")  # 打印日志
        
        target_text = self.current_target_text()
        if not target_text:
            if self.logger: self.logger.info("No targets left to look for.")
            return

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        response_text = self.query_target_bbox_and_distance(
            image_rgb=image_rgb,
            target_text=target_text
        )
        if not response_text:
            if self.logger: self.logger.error(f"Failed to get bounding box and distance for {target_text}")  # 错误日志
            return

        parsed = self.parse_vlm_response(response_text)
        if not parsed.get("visible"):
            if self.logger: 
                self.logger.error(f"Target '{target_text}' not visible or invalid response. Raw VLM response: {response_text}")
            return

        if self.logger: self.logger.info(f"Target bbox: {parsed['x_min']}, {parsed['y_min']}, {parsed['x_max']}, {parsed['y_max']}")  # 打印bounding box

        needed = [
            parsed["visible"],
            parsed["x_min"] is not None,
            parsed["y_min"] is not None,
            parsed["x_max"] is not None,
            parsed["y_max"] is not None
        ]
        if not all(needed):
            self.latest_measurement = None
            print(f'[LandmarkDetector] target="{target_text}" not found')
            return

        h, w = image_rgb.shape[:2]

        x_min = max(0, min(w - 1, int(parsed["x_min"])))
        y_min = max(0, min(h - 1, int(parsed["y_min"])))
        x_max = max(0, min(w - 1, int(parsed["x_max"])))
        y_max = max(0, min(h - 1, int(parsed["y_max"])))

        # 防止框顺序颠倒
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min

        x, y = self.bbox_to_target_point(x_min, y_min, x_max, y_max)

        bearing = self.compute_bearing(x, w)

        if self.use_vlm_distance and parsed["distance_m"] is not None:
            distance_m = float(parsed["distance_m"])
        else:
            distance_m = self.default_distance_m

        self.latest_measurement = [distance_m, float(bearing)]

        print('--------------------------------------------------')
        print(f'Current target         : {target_text}')
        print(f'BBox                   : [{x_min}, {y_min}, {x_max}, {y_max}]')
        print(f'Point used             : x={x}, y={y}')
        print(f'Distance / Angle(deg)  : {self.latest_measurement}')
        print('--------------------------------------------------')

        circled_img_rgb = self.draw_detection_overlay(
            image_rgb, x_min, y_min, x_max, y_max, x, y
        )

        if self.save_debug_plot:
            self.save_images(
                original_img_rgb=image_rgb,
                circled_img_rgb=circled_img_rgb
            )

        self.maybe_advance_to_next_landmark(distance_m)
