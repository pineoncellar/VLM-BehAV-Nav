import sys
import os
import cv2
import logging
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instruction_processor import get_instruction_breakdown, get_ith_key_list
from landmark_vision import LandmarkDetectorCore

def setup_logger():
    # 创建一个简单的Logger来替代ROS的get_logger
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
    print("    [测试模块] 单张图像视觉检测 (Instruction -> Vision)")
    print("="*50)

    logger = setup_logger()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "input")
    output_dir = os.path.join(script_dir, "output")

    # Ensure directories exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 1. 询问指令
    # 模拟Instruction Processing得到的数据
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

    # 2. 选择图像
    input_files = glob.glob(os.path.join(input_dir, "*.*"))
    image_files = [f for f in input_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"\n[错误] 请在输入文件夹中放置图像文件: {input_dir}")
        return

    print("\n在 input 文件夹中找到以下图像:")
    for idx, f in enumerate(image_files):
        print(f"  {idx + 1}. {os.path.basename(f)}")
    
    img_choice = input(f">>> 请输入选择的图像序号 (1-{len(image_files)}，默认1): ").strip()
    if not img_choice.isdigit() or int(img_choice) < 1 or int(img_choice) > len(image_files):
        img_choice = 1
    
    img_path = image_files[int(img_choice) - 1]
    logger.info(f"=== [Module 2] Landmark Vision ===")
    logger.info(f"Processing image: {img_path}")

    # 3. 读取图像
    cv_image = cv2.imread(img_path)
    if cv_image is None:
        logger.error(f"Failed to read image at {img_path}")
        return

    # 4. 初始化和设置 LandmarkDetectorCore
    detector = LandmarkDetectorCore(logger=logger)
    detector.navigation_landmarks = landmarks
    detector.navigation_actions = nav_actions
    
    # 强制将输出图片路径指向本脚本的 output 目录，并开启绘图
    detector.save_debug_dir = output_dir
    detector.save_debug_plot = True

    # 5. 处理图像并预测
    try:
        detector.process_image(cv_image)
        meas = detector.latest_measurement
        if meas:
            distance, bearing = meas
            logger.info(f"==> Detected Target '{landmarks}'! Distance: {distance:.2f}m, Bearing: {bearing:.2f}deg")
            logger.info(f"==> 输出图像可能已保存至: {output_dir}")
        else:
            logger.info(f"Target '{landmarks}' not in view. Searching...")
    except Exception as e:
        logger.error(f"Image processing error: {e}")

if __name__ == '__main__':
    main()
