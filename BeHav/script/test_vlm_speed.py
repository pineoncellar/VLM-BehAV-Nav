import os
import io
import time
import base64
import requests
import cv2
import numpy as np
from PIL import Image as PILImage
from textwrap import dedent
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def load_image(image_np_rgb):
    """
    将 numpy 数组形式的 RGB 图像转换为 base64 编码的 string，
    逻辑同 landmark_vision.py 中的 load_image 方法一致。
    """
    pil_image = PILImage.fromarray(image_np_rgb)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

def test_vlm_speed():
    # 读取环境变量，默认值与 landmark_vision.py 中一致
    api_key = os.getenv("LLM_API_KEY", "sk-e9d7e3da6d6240cd97b4d61af040415d")
    vlm_model = os.getenv("LLM_VLM_MODEL", "qwen3-vl-plus")
    vlm_base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")

    print(f"Starting VLM Speed Test...")
    print(f"Model: {vlm_model}")
    print(f"Base URL: {vlm_base_url}")
    print("-" * 50)

    # 1. 构造一张随机大小的测试图像，并在中心画一个红色矩形，模拟目标
    width, height = 640, 480
    image_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(image_bgr, (200, 150), (440, 330), (0, 0, 255), -1)
    
    # 因为 VLM 需要的输入通常是 RGB 格式，转换颜色通道
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 测试目标
    target_text = "red rectangular box"

    # 2. 构造 Prompt，与 landmark_vision.py 中完全一致
    prompt = dedent(f"""
        You are given ONE outdoor scene image.

        Target landmark description:
        "{target_text}"

        Image size:
        width = {width} pixels
        height = {height} pixels

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
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    # API 请求报文数据
    data = {
        "model": vlm_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": load_image(image_rgb),
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "temperature": 0,
        "max_tokens": 300
    }

    # 3. 循环发起测试并记录响应时间
    num_trials = 3
    valid_times = []

    for i in range(num_trials):
        print(f"Trial {i + 1}/{num_trials}...")
        start_time = time.time()
        
        try:
            response = requests.post(
                vlm_base_url,
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            
            end_time = time.time()
            duration = end_time - start_time
            valid_times.append(duration)
            
            response_json = response.json()
            content = response_json["choices"][0]["message"]["content"].strip()
            
            print(f"  [SUCCESS] Response received in {duration:.4f} seconds.")
            # 只截取展示部分 response，避免过长
            print(f"  [CONTENT] {content}")
            
        except requests.exceptions.RequestException as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"  [FAILED] Request failed after {duration:.4f} seconds: {e}")
            break

        # 在测试间进行简单休眠，避免触发 API 限流
        if i < num_trials - 1:
            time.sleep(2)

    print("-" * 50)
    if valid_times:
        avg_time = sum(valid_times) / len(valid_times)
        print(f"Test Completed. Valid Trials: {len(valid_times)}/{num_trials}")
        print(f"Average Response Time: {avg_time:.4f} seconds")
    else:
        print("Test Failed. No valid response received.")

if __name__ == "__main__":
    test_vlm_speed()