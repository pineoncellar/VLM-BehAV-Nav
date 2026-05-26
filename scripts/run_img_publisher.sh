#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "Starting Image Publisher Node..."

# 可以根据需要在这里修改图片路径和发布频率等参数
# 默认路径使用我们在脚本里设置的:
# RGB_PATH: /home/tita-remote/ZyyPlanner/word-vedio/004-1/rgb
# DEPTH_PATH: /home/tita-remote/ZyyPlanner/word-vedio/004-1/depth

python3 img_publisher.py \
    --ros-args -p fps:=10.0 \
    -p rgb_topic:="/virtual_camera/color/image_raw" \
    -p depth_topic:="/virtual_camera/depth/image_raw"

