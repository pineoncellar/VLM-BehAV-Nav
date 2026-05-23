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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image as PILImage
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class LandmarkDetectorCore:
    def __init__(self, logger=None):
        self.logger = logger
        if self.logger:
            self.logger.info('Initialized LandmarkDetectorCore logic')


        self.navigation_landmarks = []

        # ========= 新增 FastSAM 初始化 =========
        # [开关] 设置为 True 使用 FastSAM 分割，False 使用深度直方图测距
        self.use_fastsam = os.getenv("USE_FASTSAM", "False").lower() in ["true", "1", "t", "yes", "y"]
        self.cluster_gap = float(os.getenv("CLUSTER_GAP", "0.8"))

        if self.use_fastsam:
            from ultralytics import FastSAM
            import torch

            if self.logger:
                self.logger.info('Initializing FastSAM model...')
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.fastsam_model = FastSAM("FastSAM-x.pt")
        else:
            if self.logger:
                self.logger.info('FastSAM disabled, using lightweight depth histogram method.')
        
        # 相机内参配置
        # 参考常见 Gazebo 内配，具体需要依照你在模型中的数值，否则测距存在偏差
        self.intrinsics = {
            'fx': 600.0,
            'fy': 600.0,
            'cx': 320.0,
            'cy': 240.0,
        }

        # ========= 从 .env 环境变量读取配置 =========
        self.api_key = os.getenv("LLM_API_KEY", "sk-e9d7e3da6d6240cd97b4d61af040415d")
        self.vlm_model = os.getenv("LLM_VLM_MODEL", "qwen3-vl-plus")
        self.vlm_base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")

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
        self.last_annotated_image_bgr = None  # 保存最后一次成功检测的附带框的图像

        self.max_retries = 3
        self.delay = 5
        self.latest_measurement = None
        
        # 外部回调接口，发布视觉检测带标注的图片
        self.on_vision_image = None

        # ROS 订阅和定时器 (Moved to interface node)
        # self.bridge = CvBridge()
        # self.image_sub = ...
        # self.timer = ...

    # ============================================================
    # 基础函数
    # ============================================================
    def add_status_banner(self, image_bgr, text):
        img_copy = image_bgr.copy()
        h, w = img_copy.shape[:2]
        banner_h = 40
        overlay = img_copy.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img_copy, 0.4, 0, img_copy)
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h - (banner_h - text_size[1]) // 2 + 5
        cv2.putText(img_copy, text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness)
        return img_copy

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

    def calculate_physical_distance(self, bbox, rgb_img, depth_img):
        """
        基于 FastSAM 掩膜提取和深度图中值的 3D 距离测算。
        bbox 格式 [x_min, y_min, x_max, y_max]
        """
        t_start = time.time()
        
        # 第一步：FastSAM 掩膜生成
        results = self.fastsam_model(
            rgb_img,
            device=self.device,
            retina_masks=True,
            conf=0.4,
            iou=0.9,
            bboxes=[bbox],
            verbose=False
        )
        
        if not results or not results[0].masks:
            return None, None, None

        mask = results[0].masks.data[0].cpu().numpy().astype(bool)

        # 把掩膜大小缩放回原图大小
        if mask.shape != rgb_img.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

        # 第二步：深度值提取与清洗
        if depth_img is None:
            return None, mask, None
            
        depths = depth_img[mask]
        # 过滤无效深度点，Gazebo 激光/双目 很多空洞、或距离越界的点
        valid_depths = depths[(depths > 0.0) & (depths < 100.0) & (~np.isnan(depths)) & (~np.isinf(depths))]
        
        if len(valid_depths) == 0:
            return None, mask, None

        # 第三步：统计中值与质心
        z_median = np.median(valid_depths)
        
        y_indices, x_indices = np.where(mask)
        u_c = np.mean(x_indices)
        v_c = np.mean(y_indices)
        
        # 第四步：内参反投影计算真实距离
        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        cx = self.intrinsics['cx']
        cy = self.intrinsics['cy']
        
        # 针孔相机逆投影
        X = (u_c - cx) * z_median / fx
        Y = (v_c - cy) * z_median / fy
        Z = z_median
        
        distance_d = float(np.sqrt(X**2 + Y**2 + Z**2))
        centroid = (int(u_c), int(v_c))
        
        t_end = time.time()
        if self.logger:
            self.logger.info(f'[Timing] FastSAM depth calculation took {t_end - t_start:.4f} seconds')
            
        return distance_d, mask, centroid

    def calculate_physical_distance_histogram(self, bbox, depth_img):
        """
        基于深度直方图峰值的轻量级测距算法，替代 FastSAM
        """
        t_start = time.time()
        
        x_min, y_min, x_max, y_max = bbox
        
        # 截取 BBox 内的深度数据
        crop_depths = depth_img[y_min:y_max, x_min:x_max]
        
        # 过滤无效点 (负数、0、极大值、NaN 等 Gazebo 产生的无效深度)
        valid_mask = (crop_depths > 0.0) & (crop_depths < 100.0) & (~np.isnan(crop_depths)) & (~np.isinf(crop_depths))
        valid_depths = crop_depths[valid_mask]
        
        if len(valid_depths) == 0:
            return None, None, None

        # --- 第一步：动态带宽与密度聚类 (DBSCAN / 均值漂移平替) ---
        # 对于大角度倾斜的大体积物体，深度分布呈现“宽扁”的长尾（可能跨越上米）。
        # 如果仍用固定 0.5m 桶宽的主峰法，一辆巴士可能会被切碎成好几段，掩膜支离破碎。
        # 解决方案：放弃固定直方图分桶，使用基于 KDE (核密度估计) / KNN 的自适应一维滑动聚类。

        # 将深度排序
        sorted_depths = np.sort(valid_depths)
        
        # 寻找平滑区域（最大密集子集群）
        # 参数: cluster_gap 决定两个点之间差多远才算“断层”了（例如人与巴士的断层，或是墙的断层）
        # 因为斜着的巴士内部是“连续渐变”的（跨越再大每次变化也很小），而断层是突变的
        cluster_gap = self.cluster_gap  # 最大允许间隙（米）
        
        clusters = []
        current_cluster = [sorted_depths[0]]
        
        for i in range(1, len(sorted_depths)):
            if sorted_depths[i] - sorted_depths[i-1] <= cluster_gap:
                current_cluster.append(sorted_depths[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sorted_depths[i]]
        clusters.append(current_cluster)
        
        # 找到包含像素最多的那个“连续簇”(通常主导 BBox 面积的)
        largest_cluster = max(clusters, key=len)
        lower_bound = largest_cluster[0]
        upper_bound = largest_cluster[-1]
        
        # 为了防止极端的斜面带来的两端噪点，我们取这个主簇的 20%~80% 核心区来评估代表深度
        p20 = np.percentile(largest_cluster, 20)
        p80 = np.percentile(largest_cluster, 80)
        core_cluster = [d for d in largest_cluster if p20 <= d <= p80]
        
        z_target = float(np.median(core_cluster))

        # --- 第二步：根据连续簇极值提取对应的 Mask 并计算 2D 质心 ---
        # 采用提取到的连续簇上下界 (宽容一部分边缘)
        target_mask_crop = valid_mask & (crop_depths >= lower_bound - 0.2) & (crop_depths >= p20 - 1.0) & (crop_depths <= upper_bound + 0.2)
        
        # 生成基于整个图像大小的 Mask
        mask = np.zeros(depth_img.shape[:2], dtype=bool)
        mask[y_min:y_max, x_min:x_max] = target_mask_crop
        
        # 找到这块掩膜的实际质心 (比用 BBox 中心更稳定，能避开前方的遮挡物)
        y_indices, x_indices = np.where(mask)
        if len(x_indices) > 0 and len(y_indices) > 0:
            u_c = int(np.mean(x_indices))
            v_c = int(np.mean(y_indices))
        else:
            u_c = int((x_min + x_max) / 2)
            v_c = int((y_min + y_max) / 2)
            
        # --- 第三步：通过内参进行针孔逆投影，计算真正的 3D 距离 ---
        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        cx = self.intrinsics['cx']
        cy = self.intrinsics['cy']
        
        X = (u_c - cx) * z_target / fx
        Y = (v_c - cy) * z_target / fy
        Z = z_target
        
        distance_d = float(np.sqrt(X**2 + Y**2 + Z**2))
        
        t_end = time.time()
        if self.logger:
            self.logger.info(f'[Timing] Histogram/Clustering depth calculation took {t_end - t_start:.4f} seconds')
            
        return distance_d, mask, (u_c, v_c)

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

    def apply_blind_action(self):
        # 原本盲开想跑3m，但因为 behav_planner 里固定会减去 2.5m 的 margin 停在目标前
        # 为了抵消那个 margin，把盲开目标设为 3.0 + 2.5 = 5.5m，这样实际行驶就是 3m
        distance_m = 5.5
        bearing_deg = 0.0
        if self.navigation_actions and self.current_landmark_index < len(self.navigation_actions):
            action = str(self.navigation_actions[self.current_landmark_index]).lower()
            if "right" in action or "右转" in action:
                bearing_deg = -75.0
            elif "left" in action or "左转" in action:
                bearing_deg = 75.0
            else:
                bearing_deg = 0.0
        self.latest_measurement = [distance_m, bearing_deg]
        # 盲开行为（向某个方向盲走）是针对当前的，而非历史快照
        self.latest_odom_at_vision = getattr(self, 'current_odom', None)  # Fallback will use current odom in planner if None
        self.new_measurement_ready = True
        if self.logger:
            self.logger.info(f"[LandmarkDetector] Target not visible. Applying blind action: dist={distance_m}m, angle={bearing_deg}deg")

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

    def draw_detection_overlay(self, image_rgb, x_min, y_min, x_max, y_max, x, y, mask=None, real_distance=None):
        vis = image_rgb.copy()
        
        # 绘制半透明 Mask
        if mask is not None:
            color = np.array([255, 0, 0], dtype=np.uint8) # 红色遮罩
            alpha = 0.5
            vis[mask] = vis[mask] * (1 - alpha) + color * alpha
            
        cv2.rectangle(vis, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        cv2.circle(vis, (int(x), int(y)), 8, (255, 0, 0), -1)
        
        if real_distance is not None:
            cv2.putText(vis, f"Dist: {real_distance:.2f}m", (int(x_min), int(max(0, y_min-10))), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
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
        # TODO:优化距离判断
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
            - x_min, y_min, x_max, and y_max MUST be integers normalized to the range [0, 1000]
            - Do not return actual pixel values, output the relative coordinates mapped to [0, 1000]
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
    def process_image(self, image_bgr, depth_image=None, img_odom=None):
        if self.logger: self.logger.info("Processing image...")  # 打印日志
        
        # 决定底图：如果之前有成功的检测框图片则复用（保持框和之前的画面一致），否则用新接收到的干净图
        base_img_for_banner = getattr(self, 'last_annotated_image_bgr', None)
        if base_img_for_banner is None:
            base_img_for_banner = image_bgr

        # 发布处理中图像到回调口
        if self.on_vision_image is not None:
            proc_bgr = self.add_status_banner(base_img_for_banner, "[Processing image...]")
            self.on_vision_image(proc_bgr)
            
        self.img_odom = img_odom
        
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
            if self.on_vision_image is not None:
                self.on_vision_image(self.add_status_banner(base_img_for_banner, "[Detection Failed - VLM Error]"))
            self.apply_blind_action()
            if self.latest_measurement:
                self.maybe_advance_to_next_landmark(self.latest_measurement[0])
            return

        parsed = self.parse_vlm_response(response_text)
        if not parsed.get("visible"):
            if self.logger: 
                self.logger.error(f"Target '{target_text}' not visible or invalid response. Raw VLM response: {response_text}")
            if self.on_vision_image is not None:
                self.on_vision_image(self.add_status_banner(base_img_for_banner, f"[Target '{target_text}' Not Visible]"))
            self.apply_blind_action()
            if self.latest_measurement:
                self.maybe_advance_to_next_landmark(self.latest_measurement[0])
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
            self.logger.info(f'[LandmarkDetector] target="{target_text}" not found')
            if self.on_vision_image is not None:
                self.on_vision_image(self.add_status_banner(base_img_for_banner, f"[Target '{target_text}' not found]"))
            self.apply_blind_action()
            if self.latest_measurement:
                self.maybe_advance_to_next_landmark(self.latest_measurement[0])
            return

        h, w = image_rgb.shape[:2]

        # 从 1000x1000 的归一化坐标系转换回原图的像素坐标
        x_min_px = int(parsed["x_min"] * w / 1000.0)
        y_min_px = int(parsed["y_min"] * h / 1000.0)
        x_max_px = int(parsed["x_max"] * w / 1000.0)
        y_max_px = int(parsed["y_max"] * h / 1000.0)

        x_min = max(0, min(w - 1, x_min_px))
        y_min = max(0, min(h - 1, y_min_px))
        x_max = max(0, min(w - 1, x_max_px))
        y_max = max(0, min(h - 1, y_max_px))

        # 防止框顺序颠倒
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min

        bbox = [x_min, y_min, x_max, y_max]

        # 根据开关选择深度测距方案
        real_distance, target_mask, centroid = None, None, None
        if depth_image is not None:
            if getattr(self, 'use_fastsam', False):
                real_distance, target_mask, centroid = self.calculate_physical_distance(bbox, image_rgb, depth_image)
            else:
                real_distance, target_mask, centroid = self.calculate_physical_distance_histogram(bbox, depth_image)

        if centroid:
            x, y = centroid
        else:
            x, y = self.bbox_to_target_point(x_min, y_min, x_max, y_max)

        bearing = self.compute_bearing(x, w)

        if real_distance is not None:
            distance_m = real_distance
            if self.logger: self.logger.info(f"Using Z_median Real Distance: {distance_m:.2f} m")
        elif self.use_vlm_distance and parsed["distance_m"] is not None:
            distance_m = float(parsed["distance_m"])
            if self.logger: self.logger.info(f"Using VLM Estimated Distance: {distance_m:.2f} m")
        else:
            distance_m = self.default_distance_m
            if self.logger: self.logger.info(f"Using Default Distance: {distance_m:.2f} m")

        self.latest_measurement = [distance_m, float(bearing)]
        self.latest_odom_at_vision = getattr(self, 'img_odom', None)
        self.new_measurement_ready = True

        self.logger.info('--------------------------------------------------')
        self.logger.info(f'Current target         : {target_text}')
        self.logger.info(f'BBox                   : [{x_min}, {y_min}, {x_max}, {y_max}]')
        self.logger.info(f'Point used             : x={x}, y={y}')
        self.logger.info(f'Distance / Angle(deg)  : {self.latest_measurement}')
        self.logger.info('--------------------------------------------------')

        circled_img_rgb = self.draw_detection_overlay(
            image_rgb, x_min, y_min, x_max, y_max, x, y, target_mask, real_distance
        )

        if self.save_debug_plot:
            self.save_images(
                original_img_rgb=image_rgb,
                circled_img_rgb=circled_img_rgb
            )

        # 发布目标已更新图像到回调口
        if self.on_vision_image is not None:
            circled_img_bgr = cv2.cvtColor(circled_img_rgb, cv2.COLOR_RGB2BGR)
            # 成功保存这次带有框和检测标注的完美图片作为“缓存快照”底图
            self.last_annotated_image_bgr = circled_img_bgr.copy()
            updated_bgr = self.add_status_banner(circled_img_bgr, "[Target Updated]")
            self.on_vision_image(updated_bgr)

        self.maybe_advance_to_next_landmark(distance_m)
