#!/bin/bash
set -e

# =========================================================================
# VLM-BehAV-Nav 统一整合启动脚本
#
# 此脚本统一整合并启动：
#   1. ROS 2 底层规划管线 (几何与语义建图、Far 路径寻路器、Local 局路探测野簇评分、底层 MPC 控制器)
#   2. VLM-BehAV-Nav 视觉自然语言交互层 (VLM 自然语义控制、地标识别、极坐标解算桥接)
# =========================================================================

# ----------------- 1. 配置与多节点进程管理 -----------------
DISABLE_CONTROL=true       # 设为 true 时，底层 MPC 仅规划不发 /cmd_vel 代码控制，用于安全调试小车
LOG_CMD_VEL_ONLY=false       # 设为 true 时，系统不会真的发布/cmd_vel，而是将时间戳与对应的指令记录到日志文件中
LAUNCH_PLANNER=true         # 是否在当前脚本中一同拉起 Planner 部分
USE_FASTSAM=false            # 是否使用 FastSAM 独立显卡模型进行目标分割
CLUSTER_GAP=0.8             # 深度连续性突变判定阈值 (米)

# 相机话题配置
# 将话题名称结尾加上 /compressed 或 /compressedDepth，代码内的节点会自动识别并切换解码器使用压缩流
CAMERA_RGB_TOPIC="/gemini330/color/image_raw/compressed"
CAMERA_DEPTH_TOPIC="/gemini330/depth/image_raw/compressedDepth"

pids=()
cleanup() {
  echo "Stopping all VLM-BehAV-Nav Nodes..."
  trap - INT TERM EXIT
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  echo "All nodes strictly terminated."
}
trap cleanup INT TERM EXIT

# ----------------- 2. 加载工作空间及 Python 运行虚拟环境 -----------------
# 切换至脚本当前所在目录确保工作区相对路径无冲突
cd "$(dirname "$0")"

START_TIME=$(date +%Y%m%d_%H%M%S)

# 确保 logs 目录存在
mkdir -p logs/far_waypoint_planner logs/local_rollout_selector logs/ackermann_mpc_tracker logs/cmd_vel logs/BeHav

# Source 机器人的仿真或实机运行工作空间 (robot_yang / tita_sdk)
WORKSPACE_SETUP="./robot_yang/install/setup.bash"
if [ -f "$WORKSPACE_SETUP" ]; then
    source "$WORKSPACE_SETUP"
    echo "Sourced robot_yang workspace"
fi

TITA_SETUP="~/tita_sdk/install/setup.bash"
if [ -f "$TITA_SETUP" ]; then
    source "$TITA_SETUP"
    echo "Sourced tita_sdk workspace"
fi

source /opt/ros/humble/setup.bash 2>/dev/null || true

# 激活 uv 管理的虚拟 Python 环境
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated uv (.venv) virtual environment."
fi

# ----------------- 3. 视觉与大语言模型环境代理配置 -----------------
export USE_FASTSAM="$USE_FASTSAM"
export CLUSTER_GAP="$CLUSTER_GAP"

# 修复 httpx 不支持 socks:// 代理前缀的问题 (针对大模型 API 访问的常规适配)
export http_proxy="${http_proxy/socks:\/\//socks5://}"
export https_proxy="${https_proxy/socks:\/\//socks5://}"
export all_proxy="${all_proxy/socks:\/\//socks5://}"
export HTTP_PROXY="${HTTP_PROXY/socks:\/\//socks5://}"
export HTTPS_PROXY="${HTTPS_PROXY/socks:\/\//socks5://}"
export ALL_PROXY="${ALL_PROXY/socks:\/\//socks5://}"

# ----------------- 4. 启动底层规划与控制层 (Planner) -----------------
if [ "$LAUNCH_PLANNER" = "true" ]; then
    echo "=========================================================="
    echo "Step [1/2]: Starting Depth & Semantic Planner Network..."
    echo "=========================================================="

    # 为了避免后台节点日志刷屏干扰前台输入，将输出重定向到 logs/ 对应子文件夹日志文件
    uv run python3 farplanner/far_waypoint_planner.py \
      --current-waypoint-local-topic /far/current_waypoint_local \
      --local-plan-topic /far/local_plan \
      --reference-path-topic /far/reference_path_local \
      --waypoint-queue-topic /far/waypoint_queue_local \
      --current-subgoal-topic /far/current_subgoal_local \
      --route-status-topic /far/route_status \
      --far-goal-reached-topic /far/goal_reached \
      --waypoint-queue-distances 0.8 1.5 2.5 3.3 4.0 \
      --current-subgoal-distance 2.5 \
      --behav-mode > "logs/far_waypoint_planner/${START_TIME}.log" 2>&1 &
    pids+=($!)
    sleep 2

    # 4.3 改良版车辆局部采样横摆规避与打分器 (具有车辆物理、语义占据格高弹性容差)
    uv run python3 localplanner/local_rollout_selector.py \
      --current-waypoint-local-topic /far/current_waypoint_local \
      --far-local-plan-topic /far/local_plan \
      --reference-path-topic /far/reference_path_local \
      --waypoint-queue-topic /far/waypoint_queue_local \
      --current-subgoal-topic /far/current_subgoal_local \
      --route-status-topic /far/route_status \
      --far-goal-reached-topic /far/goal_reached \
      --local-status-topic /local/status \
      --route-tube-radius 0.65 \
      --route-tube-max-deviation 1.10 \
      --ref-mean-weight 4.0 \
      --queue-weight 3.5 \
      --progress-reward-weight 1.4 \
      --switch-smooth-weight 1.2 \
      --arbiter-switch-margin 1.0 \
      --debug-switch \
      --debug-rejections \
      --grid-topic /local_traversability_grid \
      --vehicle-size 0.5 \
      --footprint-check-stride 2 \
      --max-footprint-occupied-hits 30 \
      --max-occupied-hits 50 \
      --max-near-unknown-hits 300 \
      --max-out-of-map-hits 50 \
      --occupied-threshold 80 > "logs/local_rollout_selector/${START_TIME}.log" 2>&1 &
    pids+=($!)
    sleep 2

    # 4.4 车辆物理模型阿克曼 MPC 循迹控制
    MPC_ARGS="--far-goal-reached-topic /far/goal_reached --max-speed 0.5 --cmd-vel-topic /cmd_vel --no-sim-time"
    if [ "$DISABLE_CONTROL" = "true" ]; then
        MPC_ARGS="$MPC_ARGS --disable-control"
        echo "⚠️ [DEBUG MODE] Control commands (/cmd_vel) are DISABLED. The robot will not move."
    fi
    if [ "$LOG_CMD_VEL_ONLY" = "true" ]; then
        LOG_FILE_PATH="$PWD/logs/cmd_vel/${START_TIME}.log"
        MPC_ARGS="$MPC_ARGS --log-cmd-vel-file $LOG_FILE_PATH"
        echo "⚠️ [LOG MODE] Control commands (/cmd_vel) intercept to log: logs/cmd_vel/${START_TIME}.log"
    fi
    uv run python3 localplanner/ackermann_mpc_tracker.py $MPC_ARGS > "logs/ackermann_mpc_tracker/${START_TIME}.log" 2>&1 &
    pids+=($!)
    sleep 2
fi

# ----------------- 5. 启动高层交互与控制模块 (VLM BeHav) -----------------
echo "=========================================================="
echo "Step [2/2]: Starting BeHav ROS Interface (VLM Agent)..."
echo "=========================================================="

echo "==============================================================="
echo "   VLM-BehAV-Nav ALL-IN-ONE PIPELINE IS RUNNING SUCCESSFULLY.  "
echo "   Press [Ctrl+C] to exit and stop all background services.   "
echo "==============================================================="

cd BeHav
uv run python3 ros_interface.py \
    --ros-args \
    -p rgb_topic:="${CAMERA_RGB_TOPIC}" \
    -p depth_topic:="${CAMERA_DEPTH_TOPIC}"
