import sys
import os
import cv2
import logging
import glob
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instruction_processor import get_instruction_breakdown, get_ith_key_list
from landmark_vision import LandmarkDetectorCore

def setup_logger():
    logger = logging.getLogger('TestVisionImg')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(name)s] [%(levelname)s] %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def main():
    print("\n" + "="*50)
    print("    [测试模块] 完整视觉管道测试 (6图输出)")
    print("="*50)

    logger = setup_logger()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "input")
    output_dir = os.path.join(script_dir, "output")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 1. 询问指令
    instruction = input(">>> 请输入您的自然语言导航指令: ").strip()
    if not instruction:
        instruction = "Walk to the red car"
        print(f"使用默认指令: {instruction}")

    logger.info(f"=== [Module 1] Instruction Processing ===")
    breakdown = get_instruction_breakdown(instruction)
    landmarks = get_ith_key_list(breakdown, 1)
    nav_actions = get_ith_key_list(breakdown, 2)
    
    logger.info(f"Extracted Landmarks: {landmarks}")
    logger.info(f"Extracted Navigation Acts: {nav_actions}")

    # 2. 选择图像 (RGB图和Depth图配对)
    input_files = glob.glob(os.path.join(input_dir, "*rgb.*"))
    if not input_files:
        input_files = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))
        
    image_files = [f for f in input_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"\n[错误] 请在输入文件夹中放置RGB图像文件: {input_dir}")
        return

    print("\n在 input 文件夹中找到以下图像:")
    for idx, f in enumerate(image_files):
        print(f"  {idx + 1}. {os.path.basename(f)}")
    
    img_choice = input(f">>> 请输入选择的图像序号 (1-{len(image_files)}，默认1): ").strip()
    if not img_choice.isdigit() or int(img_choice) < 1 or int(img_choice) > len(image_files):
        img_choice = 1
    
    rgb_path = image_files[int(img_choice) - 1]
    
    # 尝试找到对应的depth图
    base_name = os.path.basename(rgb_path)
    if "rgb" in base_name:
        depth_name = base_name.replace("rgb", "depth").replace(".jpg", ".npy").replace(".png", ".npy")
        depth_path = os.path.join(input_dir, depth_name)
    else:
        # 默认找 depth_d.npy
        depth_path = os.path.join(input_dir, "depth_d.npy")
        
    logger.info(f"Processing RGB image: {rgb_path}")
    logger.info(f"Searching for Depth image: {depth_path}")

    # 3. 读取图像
    cv_image = cv2.imread(rgb_path)
    if cv_image is None:
        logger.error(f"Failed to read RGB image at {rgb_path}")
        return
        
    rgb_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    depth_img = None
    if os.path.exists(depth_path):
        depth_img = np.load(depth_path)
        logger.info("Found and loaded depth map successfully.")
    else:
        logger.warning(f"Depth map not found at {depth_path}, physical distance estimation may fail.")

    # 4. 初始化 LandmarkDetectorCore
    detector = LandmarkDetectorCore(logger=logger)
    detector.navigation_landmarks = landmarks
    detector.navigation_actions = nav_actions
    detector.save_debug_dir = output_dir

    try:
        # a) 原始RGB
        img_a = rgb_img.copy()
        
        # b) 原始深度图
        if depth_img is not None:
            valid_depths = depth_img[(depth_img > 0) & (depth_img < 100)]
            vmax = np.percentile(valid_depths, 95) if len(valid_depths) > 0 else 10.0
        else:
            vmax = 10.0

        # VLM: 获取目标位置
        target_text = detector.current_target_text()
        logger.info(f"Querying VLM for target: {target_text}")
        response_text = detector.query_target_bbox_and_distance(rgb_img, target_text)
        parsed = detector.parse_vlm_response(response_text)
        
        img_c = None
        img_d = None
        img_e = None
        img_f = img_a.copy()
        real_distance = None
        mask = None
        centroid = None
        
        # c) FastSAM 全图掩膜
        logger.info("Generating Full FastSAM Masks...")
        results_all = detector.fastsam_model(rgb_img, device=detector.device, retina_masks=True, conf=0.4, iou=0.9, verbose=False)
        img_c = img_a.copy()
        if results_all and results_all[0].masks:
            masks_all = results_all[0].masks.data.cpu().numpy()
            for m in masks_all:
                m_resized = cv2.resize(m.astype(np.uint8), (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                color = np.random.randint(0, 255, 3).tolist()
                img_c[m_resized] = img_c[m_resized] * 0.5 + np.array(color) * 0.5

        if parsed.get("visible"):
            h, w = img_a.shape[:2]
            x_min = max(0, min(w - 1, int(parsed["x_min"] * w / 1000.0)))
            y_min = max(0, min(h - 1, int(parsed["y_min"] * h / 1000.0)))
            x_max = max(0, min(w - 1, int(parsed["x_max"] * w / 1000.0)))
            y_max = max(0, min(h - 1, int(parsed["y_max"] * h / 1000.0)))
            
            if x_min > x_max: x_min, x_max = x_max, x_min
            if y_min > y_max: y_min, y_max = y_max, y_min
            
            bbox = [x_min, y_min, x_max, y_max]
            
            # Distance and Target Mask (d, e, f)
            real_distance, mask, centroid = detector.calculate_physical_distance(bbox, rgb_img, depth_img)
            
            # d) 目标切片二值掩膜
            if mask is not None:
                img_d = (mask * 255).astype(np.uint8)
            else:
                img_d = np.zeros(img_a.shape[:2], dtype=np.uint8)
                
            # e) 深度数据融合与中值滤波区
            if depth_img is not None and mask is not None:
                norm_d = np.clip(depth_img, 0, vmax) / vmax * 255
                norm_d = norm_d.astype(np.uint8)
                norm_d_colored = cv2.applyColorMap(norm_d, cv2.COLORMAP_JET)
                
                img_e = norm_d_colored.copy()
                img_e[~mask] = 0
            else:
                img_e = np.zeros_like(img_a)

            # f) 最终位姿与测距输出
            tmode_x, tmode_y = detector.bbox_to_target_point(x_min, y_min, x_max, y_max)
            pt_x = centroid[0] if centroid else tmode_x
            pt_y = centroid[1] if centroid else tmode_y
            
            img_f = detector.draw_detection_overlay(
                img_a, x_min, y_min, x_max, y_max, pt_x, pt_y, mask=mask, real_distance=real_distance
            )
            
        else:
            logger.info("Target not visible.")
            img_d = np.zeros(img_a.shape[:2], dtype=np.uint8)
            img_e = np.zeros_like(img_a)

        # Plot 6 images
        plt.figure(figsize=(18, 10))
        # plt.suptitle(f"Instruction: {instruction} -> Target: {target_text}", fontsize=18)

        plt.subplot(231)
        plt.imshow(img_a)
        plt.title("(a) 原始RGB图像", fontsize=16)
        plt.axis('off')

        plt.subplot(232)
        if depth_img is not None:
            plt.imshow(depth_img, cmap='jet', vmin=0, vmax=vmax)
            plt.colorbar(label='Depth (m)')
        else:
            plt.text(0.5, 0.5, 'No Depth Map', ha='center', va='center')
        plt.title("(b) 原始深度图", fontsize=16)
        plt.axis('off')

        plt.subplot(233)
        if img_c is not None:
            plt.imshow(img_c)
        plt.title("(c) FastSAM全图掩膜", fontsize=16)
        plt.axis('off')

        plt.subplot(234)
        if img_d is not None:
            plt.imshow(img_d, cmap='gray')
        plt.title("(d) 目标切片二值掩膜", fontsize=16)
        plt.axis('off')

        plt.subplot(235)
        if img_e is not None:
            plt.imshow(cv2.cvtColor(img_e, cv2.COLOR_BGR2RGB))
        plt.title("(e) 深度数据融合与中值滤波", fontsize=16)
        plt.axis('off')

        plt.subplot(236)
        if img_f is not None:
            plt.imshow(img_f)
            if real_distance:
                plt.title(f"(f) 最终位姿 (Dist: {real_distance:.2f}m)", fontsize=16)
            else:
                plt.title(f"(f) 最终位姿", fontsize=16)
        plt.axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, "vision_pipeline_6_steps.png")
        plt.savefig(save_path)
        logger.info(f"Pipeline output saved to {save_path}")
        plt.show()

    except Exception as e:
        logger.error(f"Image processing error: {e}")

if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    main()
