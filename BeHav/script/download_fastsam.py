from ultralytics import FastSAM
import torch

def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("Initializing FastSAM-s model...")
    print("If the weights are not found locally, Ultralytics will download them automatically.")
    
    # 初始化模型，这会自动触发权重的下载
    model = FastSAM("FastSAM-s.pt")
    
    print("Download and initialization completed successfully!")

if __name__ == "__main__":
    main()
