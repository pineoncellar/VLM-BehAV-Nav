#!/bin/bash
set -e

echo "Starting BeHav ROS Interface with uv virtual environment..."

# Source 机器人的仿真工作空间环境 (robot_yang) 
WORKSPACE_SETUP="../robot_yang/install/setup.bash"
if [ -f "$WORKSPACE_SETUP" ]; then
    source "$WORKSPACE_SETUP"
    echo "Sourced robot_yang workspace"
else
    echo "Warning: robot_yang workspace setup not found at $WORKSPACE_SETUP. Did you run colcon build?"
fi

# ==============================================================
# 视觉模块参数配置
# ==============================================================
# 是否使用 FastSAM 独立显卡进行分割 (true/false)
export USE_FASTSAM="true"
# 深度连续性突变判定阈值 (米)。相邻深度差大于此值则认为是不同物体 (如人与巴士分离)。
export CLUSTER_GAP="0.8"

# 修复 httpx 不支持 socks:// 前缀的问题 (OpenAI 客户端需要)
export http_proxy="${http_proxy/socks:\/\//socks5://}"
export https_proxy="${https_proxy/socks:\/\//socks5://}"
export all_proxy="${all_proxy/socks:\/\//socks5://}"
export HTTP_PROXY="${HTTP_PROXY/socks:\/\//socks5://}"
export HTTPS_PROXY="${HTTPS_PROXY/socks:\/\//socks5://}"
export ALL_PROXY="${ALL_PROXY/socks:\/\//socks5://}"

# 使用 uv 运行 Python 节点程序
# uv run 会自动识别并使用当前目录下的 .venv 虚拟环境来执行
uv run python3 ros_interface.py
