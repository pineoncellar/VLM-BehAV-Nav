import cv2
import numpy as np
from ultralytics import FastSAM
import torch
import os

def main():
    print("1. Creating a synthetic test image (a red circle on black background)...")
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(img, (256, 256), 100, (0, 0, 255), -1) 
    cv2.imwrite("test_img.jpg", img)
    
    print("2. Loading FastSAM model...")
    model = FastSAM("FastSAM-x.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"3. Running inference on {device}...")
    results = model("test_img.jpg", device=device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
    
    print("-" * 50)
    masks = results[0].masks
    if masks is not None and len(masks.data) > 0:
        print(f"✅ FastSAM works perfectly! Detected {len(masks.data)} mask(s) in the test image.")
        print(f"Mask tensor shape: {masks.data.shape}")
        
        # 保存带掩码和序号的原图 (Ultralytics 原生的 plot() 功能)
        annotated_img = results[0].plot()
        cv2.imwrite("test_out.jpg", annotated_img)
        print("✅ The output with masked annotations has been saved to test_out.jpg")
    else:
        print("❌ FastSAM failed to detect anything or returned empty.")
    print("-" * 50)
        
    # 清理临时原图
    if os.path.exists("test_img.jpg"):
        os.remove("test_img.jpg")
    if os.path.exists("test.jpg"):
        os.remove("test.jpg")

if __name__ == "__main__":
    main()