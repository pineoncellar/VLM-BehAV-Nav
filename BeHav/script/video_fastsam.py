import cv2
import os
import argparse
import sys
import torch
from pathlib import Path

# 添加 BeHav 到模块搜索路径，如果需要的话
script_dir = Path(__file__).resolve().parent
behav_dir = script_dir.parent
sys.path.append(str(behav_dir))

try:
    from ultralytics import FastSAM
except ImportError:
    print("Error: ultralytics package is not installed. Please install it using 'pip install ultralytics'.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run FastSAM on a video and save the output.")
    parser.add_argument("--input_video", type=str, default="input/test.mp4", help="Path to input video file (relative to script/ or absolute).")
    parser.add_argument("--output_video", type=str, default="output/fastsam_out.mp4", help="Path to output video file.")
    parser.add_argument("--model", type=str, default="../FastSAM-x.pt", help="Path to FastSAM model.")
    args = parser.parse_args()

    input_path = script_dir / args.input_video
    output_path = script_dir / args.output_video
    model_path = script_dir / args.model

    if not input_path.exists():
        print(f"Error: Input video not found at {input_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading FastSAM model from: {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = FastSAM(str(model_path))
    except Exception as e:
        print(f"Failed to load FastSAM model: {e}")
        return

    print(f"Opening video: {input_path}")
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    print("Processing frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        # 运行 FastSAM 推理，可以自定义阈值，例如 conf=0.4, iou=0.9
        results = model(frame, device=device, retina_masks=True, imgsz=max(width, height), conf=0.4, iou=0.9, verbose=False)
        
        # 绘制分割结果（无边界框、无标签，手动使用不同颜色的掩码区分不同实例目标）
        res_frame = frame.copy()
        if len(results) > 0 and results[0].masks is not None:
            import numpy as np
            masks = results[0].masks.data.cpu().numpy()
            
            # 使用基于固定种子的随机调色板
            np.random.seed(42)
            color_palette = np.random.randint(0, 255, (200, 3), dtype=np.uint8)
            
            for i, mask in enumerate(masks):
                # 检查并统一掩码的尺寸
                if mask.shape != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                
                color = color_palette[i % len(color_palette)].tolist()
                mask_bool = mask > 0.5
                
                # 半透明混合叠加掩码颜色 (Alpha = 0.5)
                roi = res_frame[mask_bool].astype(np.float32)
                res_frame[mask_bool] = (roi * 0.5 + np.array(color) * 0.5).astype(np.uint8)

        out.write(res_frame)
        
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{total_frames} frames", end='\r')

    print(f"\nFinished processing. Output saved to {output_path}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
