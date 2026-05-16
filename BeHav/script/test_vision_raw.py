import os
import sys
import cv2
import numpy as np
import base64
import requests
import json
import logging
import torch
from dotenv import load_dotenv

# 加载 .env
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# =========== 配置 ===========
API_KEY = os.getenv("LLM_API_KEY")
VLM_URL = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1/chat/completions")
VLM_MODEL = os.getenv("LLM_VLM_MODEL", "Qwen/Qwen3-VL-32B-Instruct")

def setup_logger():
    logger = logging.getLogger('BehAV_Raw_Logic')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('[%(name)s] [%(levelname)s] %(message)s'))
        logger.addHandler(ch)
    return logger

def encode_image(image_bgr):
    _, buffer = cv2.imencode('.jpg', image_bgr)
    return base64.b64encode(buffer).decode('utf-8')

def main():
    logger = setup_logger()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "input")
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    test_img_path = os.path.join(input_dir, "2.jpg")
    gt_img_path = os.path.join(input_dir, "1.jpg") # 假设1.jpg作为Ground Truth
    
    if not os.path.exists(test_img_path):
        logger.error(f"Test image not found: {test_img_path}")
        return
        
    # 1. 询问需要寻找的目标
    target_object = input(">>> 请输入要寻找的目标 (默认: landmark building): ").strip()
    if not target_object:
        target_object = "landmark building"

    logger.info(">>> 1. Load Images")
    test_img = cv2.imread(test_img_path)
    
    if os.path.exists(gt_img_path):
        gt_img = cv2.imread(gt_img_path)
        logger.info(f"Using {gt_img_path} as Ground Truth.")
    else:
        logger.warning(f"No GT image found at {gt_img_path}. Using test image itself as GT to avoid API failure.")
        gt_img = test_img
        
    logger.info(">>> 2. FastSAM Everything Prompt (Blind Segmentation)")
    # 使用与 BeHav 相同的依赖
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ultralytics import FastSAM
    model_path = os.path.join(os.path.dirname(script_dir), "FastSAM-x.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fastsam_model = FastSAM(model_path)
    
    # 模拟 BehAV_raw 的 "无提示盲分割"
    results = fastsam_model(test_img, device=device, retina_masks=True, conf=0.4, iou=0.9, verbose=False)
    
    # 在图像上绘制带编号的掩膜，以供 VLM 识别
    masked_img = test_img.copy()
    if not results or not results[0].masks:
        logger.error("No masks detected.")
        return
        
    masks = results[0].masks.data.cpu().numpy()
    logger.info(f"Generated {len(masks)} total masks from the scene.")
    
    # 给每个mask一个随机颜色并打上数字标签
    np.random.seed(42)
    colors = np.random.randint(0, 255, (len(masks), 3), dtype=np.uint8)
    
    for i, mask in enumerate(masks):
        mask_bool = mask.astype(bool)
        # 缩放至原图大小
        if mask_bool.shape != (test_img.shape[0], test_img.shape[1]):
            mask_bool = cv2.resize(mask_bool.astype(np.uint8), (test_img.shape[1], test_img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # 涂色叠加
        color_layer = np.zeros_like(test_img, dtype=np.uint8)
        color_layer[mask_bool] = colors[i]
        masked_img = cv2.addWeighted(masked_img, 1.0, color_layer, 0.4, 0)
        
        # 找重心写数字 (模拟 BehAV_raw 中画出编号)
        # y_indices, x_indices = np.where(mask_bool)
        # if len(y_indices) > 0:
        #     cx, cy = int(np.mean(x_indices)), int(np.mean(y_indices))
        #     # VLM 需要看数字，所以数字颜色和大小尽量明显
        #     cv2.putText(masked_img, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        #     cv2.putText(masked_img, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1)
            
    # 输出 Mask 结果保存
    mask_output_path = os.path.join(output_dir, "raw_logic_masked.jpg")
    cv2.imwrite(mask_output_path, masked_img)
    logger.info(f"Masked image saved to {mask_output_path}")
    
    logger.info(">>> 3. Manual Input Selection")
    # 代替 VLM，直接通过终端请求用户输入想要框选的掩膜编号和距离
    user_mask_str = input(f"请输入图像(raw_logic_masked.jpg)中 {target_object} 对应的 Mask Number (1-{len(masks)}): ").strip()
    user_dist_str = input(f"请输入 {target_object} 距离您的预估距离 (单位: 米): ").strip()
    
    try:
        mask_idx = int(user_mask_str) - 1 # 1-based to 0-based
        dist_val = float(user_dist_str)
        import matplotlib.pyplot as plt
        
        if 0 <= mask_idx < len(masks):
            target_mask = masks[mask_idx].astype(bool)
            if target_mask.shape != (test_img.shape[0], test_img.shape[1]):
                target_mask = cv2.resize(target_mask.astype(np.uint8), (test_img.shape[1], test_img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            
            y_indices, x_indices = np.where(target_mask)
            if len(y_indices) > 0:
                x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
                y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
                cx, cy = int(np.mean(x_indices)), int(np.mean(y_indices))
                
                vis_img = test_img.copy()
                # 画框和点
                cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.circle(vis_img, (cx, cy), 8, (255, 0, 0), -1)
                cv2.putText(vis_img, f"Dist: {dist_val}m", (x_min, max(0, y_min-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 输出对比图
                plt.figure(figsize=(12, 5))
                plt.subplot(121)
                plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
                plt.title('Original image')
                plt.axis('off')

                plt.subplot(122)
                plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
                plt.title('Detected target')
                plt.axis('off')

                plt.tight_layout()
                final_plot_path = os.path.join(output_dir, "raw_logic_final_plot.jpg")
                plt.savefig(final_plot_path)
                plt.close()
                logger.info(f"==> Final compared image saved to {final_plot_path}")
            else:
                logger.error("The selected mask is empty.")
        else:
            logger.error(f"Invalid mask number. Please select a number between 1 and {len(masks)}.")
    except Exception as e:
        logger.error(f"Invalid input or plotting failed: {e}")

if __name__ == "__main__":
    main()
