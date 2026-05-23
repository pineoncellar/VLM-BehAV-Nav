import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Track yellow car in video with ROI and progress tracking.")
    parser.add_argument('--select-input', type=int, help='Select input video by index')
    parser.add_argument('--x-min', type=float, default=0.0, help='Min x ratio (0-1), default 0.0')
    parser.add_argument('--x-max', type=float, default=1.0, help='Max x ratio (0-1), default 1.0')
    parser.add_argument('--y-min', type=float, default=0.0, help='Min y ratio (0-1), default 0.0')
    parser.add_argument('--y-max', type=float, default=1.0, help='Max y ratio (0-1), default 1.0')
    parser.add_argument('--draw-circle', type=int, default=0, choices=[0, 1], help='Whether to draw a circle on the car (1 for yes, 0 for no), default 0')
    parser.add_argument('--traj-thickness', type=int, default=2, help='Thickness of the trajectory line, default 2')
    parser.add_argument('--circle-radius', type=int, default=15, help='Radius of the drawn circle, default 15')
    parser.add_argument('--circle-thickness', type=int, default=2, help='Thickness of the drawn circle, default 2')
    args = parser.parse_args()

    input_dir = 'input'
    output_dir = 'output'
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # List video files
    files = [f for f in os.listdir(input_dir) if f.casefold().endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    if not files:
        print(f"没有在 '{input_dir}' 文件夹中找到视频文件。请先将视频放入 '{input_dir}' 文件夹中。")
        return
    
    files.sort()
    
    # Select input file
    if args.select_input is not None:
        idx = args.select_input - 1
        if 0 <= idx < len(files):
            selected_file = files[idx]
        else:
            print(f"选择的序号 {args.select_input} 无效！")
            return
    else:
        print("请选择要处理的视频文件：")
        for i, f in enumerate(files):
            print(f"{i + 1}. {f}")
        try:
            choice = int(input("输入序号: "))
            selected_file = files[choice - 1]
        except (ValueError, IndexError):
            print("输入无效。")
            return
            
    video_path = os.path.join(input_dir, selected_file)
    output_path = os.path.join(output_dir, selected_file)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
        
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    trajectory_points = []
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    
    # Calculate pixel boundaries based on ratios
    x_min_px = int(width * args.x_min)
    x_max_px = int(width * args.x_max)
    y_min_px = int(height * args.y_min)
    y_max_px = int(height * args.y_max)

    print(f"处理视频: {selected_file}")
    print(f"ROI 检测区域: x=[{x_min_px}:{x_max_px}], y=[{y_min_px}:{y_max_px}]")
    
    # Use tqdm for progress bar
    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Create a mask for ROI
            roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            roi_mask[y_min_px:y_max_px, x_min_px:x_max_px] = 255
            
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # Apply ROI mask to only search within desired region
            mask = cv2.bitwise_and(mask, mask, mask=roi_mask)
            
            # Morphological operations
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                M = cv2.moments(c)
                
                if M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    trajectory_points.append((cX, cY))
                    if args.draw_circle == 1:
                        cv2.circle(frame, (cX, cY), args.circle_radius, (0, 255, 255), args.circle_thickness)

            if len(trajectory_points) > 1:
                for i in range(1, len(trajectory_points)):
                    cv2.line(frame, trajectory_points[i-1], trajectory_points[i], (0, 255, 0), thickness=args.traj_thickness)

            out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()
    print(f"\n轨迹视频处理完成！已保存为 {output_path}")

if __name__ == '__main__':
    main()
