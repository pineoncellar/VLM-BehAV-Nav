#!/bin/bash

# 启动 Orbbec 相机
echo "Starting Orbbec camera..."
cd ~/OrbbecSDK_ROS2 || { echo "OrbbecSDK_ROS2 directory not found!"; exit 1; }
source ./install/setup.bash

ros2 launch orbbec_camera gemini_330_series_lowcpu2.launch.py \
    camera_name:=gemini330 \
    config_file_path:=/home/robot/OrbbecSDK_ROS2/orbbec_camera/config/gemini330lowcpu.yaml &

CAMERA_PID=$!
echo "Camera launched with PID $CAMERA_PID"

# 等待相机初始化完成
sleep 7  # 可根据需要调整时间

# 启动 TITA Mapping
echo "Starting TITA mapping..."
cd ~/ZyyPlanner/map || { echo "ZyyPlanner/map directory not found!"; exit 1; }
./run_tita_mapping_geometry_only.sh

# 可选：等待相机进程结束（如果需要）
# wait $CAMERA_PID
