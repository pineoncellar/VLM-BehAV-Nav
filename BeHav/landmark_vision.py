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
import torch
from ultralytics import FastSAM

from PIL import Image as PILImage
import os
from dotenv import load_dotenv

load_dotenv()

class LandmarkDetectorCore:
    def __init__(self, logger=None):
        self.logger = logger
        if self.logger:
            self.logger.info('Initialized LandmarkDetectorCore logic')


        self.navigation_landmarks = []

        # ========= 从 .env 环境变量读取配置 =========
        self.api_key = os.getenv("LLM_API_KEY", "sk-e9d7e3da6d6240cd97b4d61af040415d")
        self.vlm_model = os.getenv("LLM_VLM_MODEL", "qwen3-vl-plus")
        self.vlm_base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")

        # 固定导航目标列表：按顺序执行
        # self.navigation_landmarks = ["traffic barrier", "library"]
        self.navigation_actions = ["go to", "then go to"]  # 保留但不参与计算

        self.landmark_refs = {
            "construction cone": {
                "image_path": "./reference_images/ConstructionCone_7m.jpg",
                "gt_distance": 7.0
            },
            "fire hydrant": {
                "image_path": "./reference_images/fire_hydrant_10m.jpg",
                "gt_distance": 10.0
            }
        }

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fastsam_model = FastSAM("FastSAM-x.pt")

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
        图像原点在左上角，x 向右增加。
        遵循 ROS 坐标系约定：车体正前方为 0 度，向左偏为正（逆时针正），向右偏为负。
        """
        if image_width <= 1:
            return 0.0

        cx = (image_width - 1) / 2.0
        half_fov_rad = math.radians(self.horizontal_fov_deg / 2.0)

        fx = cx / math.tan(half_fov_rad)
        # 当 pixel_x > cx (目标在画面右侧) 时，偏角应为负值
        angle_rad = math.atan((cx - pixel_x) / fx)
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
                self.logger.info(f'[LandmarkDetector] reached "{prev_target}", switch to "{self.current_target_text()}"')
            else:
                self.logger.info('[LandmarkDetector] final landmark reached')

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
    def query_fastsam_vlm(self, masked_image_rgb, target_text, ref_img_base64, gt_distance):
        h, w = masked_image_rgb.shape[:2]

        prompt = dedent(f"""
            I have two images:
            1. A ground truth image of '{target_text}' taken from a known distance of {gt_distance} meters.
            2. A masked image with numbered masks, taken from an unknown distance.

            Tasks:
            1. Is the target landmark visible in the second image?
            2. If visible, identify the mask NUMBER containing the landmark.
            3. Estimate the camera distance from the landmark in the test image, based on the relative size compared to the {gt_distance}-meter ground truth image.

            Return exactly this JSON schema:
            {{
              "visible": true or false,
              "mask_number": integer or null,
              "distance_m": number or null
            }}
        """).strip()

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        masked_img_base64 = self.load_image(masked_image_rgb)

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
                                "url": ref_img_base64,
                                "detail": "high"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": masked_img_base64,
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
            "mask_number": None,
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
            result["mask_number"] = data.get("mask_number", None)
            result["distance_m"] = data.get("distance_m", None)
            return result
        except Exception:
            pass

        visible_match = re.search(r'"visible"\s*:\s*(true|false)', cleaned, re.IGNORECASE)
        if visible_match:
            result["visible"] = (visible_match.group(1).lower() == "true")

        m_match = re.search(r'"mask_number"\s*:\s*([0-9]+|null)', cleaned, re.IGNORECASE)
        if m_match and m_match.group(1).lower() != "null":
            result["mask_number"] = int(m_match.group(1))

        d_match = re.search(r'"distance_m"\s*:\s*([0-9]+(?:\.[0-9]+)?|null)', cleaned, re.IGNORECASE)
        if d_match and d_match.group(1).lower() != "null":
            result["distance_m"] = float(d_match.group(1))

        return result

    # ============================================================
    # 主流程
    # ============================================================
    def process_image(self, image_bgr):
        if self.logger: self.logger.info("Processing image...")
        
        target_text = self.current_target_text()
        if not target_text:
            if self.logger: self.logger.info("No targets left to look for.")
            return

        ref_info = self.landmark_refs.get(target_text)
        if not ref_info:
            if self.logger: self.logger.error(f"Missing reference data for {target_text}")
            return
            
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 运行 FastSAM 进行全图目标分割
        results = self.fastsam_model(image_rgb, device=self.device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        
        # 使用原生绘图并在掩膜中心画上数字编号，供VLM读取
        masked_image = results[0].plot()
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []
        
        for i, bbox in enumerate(boxes):
            x1, y1, x2, y2 = bbox
            cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
            # 画序号，白底红字边以保证可见度
            cv2.putText(masked_image, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(masked_image, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
        
        # 读取基准图
        ref_img = cv2.imread(ref_info["image_path"])
        if ref_img is None:
            if self.logger: self.logger.error(f"Failed to load reference image: {ref_info['image_path']}")
            return
        ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        ref_img_base64 = self.load_image(ref_img_rgb)
        
        # 调用 VLM 获取对应的 mask_number 和 精确实例距离
        response_text = self.query_fastsam_vlm(masked_image, target_text, ref_img_base64, ref_info["gt_distance"])
        if not response_text:
            if self.logger: self.logger.error(f"Failed to get mask_number and distance for {target_text}")
            return

        parsed = self.parse_vlm_response(response_text)
        if not parsed.get("visible") or parsed.get("mask_number") is None:
            if self.logger: 
                self.logger.error(f"Target '{target_text}' not visible or mask not found. Raw VLM response: {response_text}")
            return

        # 从 FastSAM 的 merged_ann 字典中直接获取该数字对应的真实像素 Mask
        target_mask_num = parsed["mask_number"]
        
        try:
            if target_mask_num < 0 or target_mask_num >= len(boxes):
                if self.logger: self.logger.error(f"Invalid mask number: {target_mask_num}")
                return
                
            bbox = boxes[target_mask_num]
            x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
            
            x, y = self.bbox_to_target_point(x_min, y_min, x_max, y_max)
            
            distance_m = float(parsed["distance_m"]) if parsed["distance_m"] else self.default_distance_m
            
            # 兜底防撞策略（视觉 looming）：如果长宽占比大于 70%，强制停车
            bbox_w_ratio = (x_max - x_min) / image_rgb.shape[1]
            if bbox_w_ratio > 0.7:
                distance_m = 0.0 # 强制让后续程序认为到达了
                
            bearing = self.compute_bearing(x, image_rgb.shape[1])
            self.latest_measurement = [distance_m, float(bearing)]
            
            if self.logger:
                self.logger.info('--------------------------------------------------')
                self.logger.info(f'Current target         : {target_text}')
                self.logger.info(f'Mask Number            : {target_mask_num}')
                self.logger.info(f'BBox                   : [{x_min}, {y_min}, {x_max}, {y_max}]')
                self.logger.info(f'Point used             : x={x}, y={y}')
                self.logger.info(f'Distance / Angle(deg)  : {self.latest_measurement}')
                self.logger.info('--------------------------------------------------')

            circled_img_rgb = self.draw_detection_overlay(
                image_rgb, x_min, y_min, x_max, y_max, x, y
            )

            if self.save_debug_plot:
                self.save_images(
                    original_img_rgb=image_rgb,
                    circled_img_rgb=circled_img_rgb
                )

            self.maybe_advance_to_next_landmark(distance_m)

        except Exception as e:
            if self.logger: self.logger.error(f"Error extracting mask {target_mask_num}: {e}")
