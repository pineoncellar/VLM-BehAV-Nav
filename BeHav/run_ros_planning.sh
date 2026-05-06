#!/bin/bash

# Source 机器人的仿真工作空间环境 (robot_yang) 
WORKSPACE_SETUP="../robot_yang/install/setup.bash"
if [ -f "$WORKSPACE_SETUP" ]; then
    source "$WORKSPACE_SETUP"
    echo "Sourced robot_yang workspace."
else
    echo "Warning: robot_yang workspace setup not found at $WORKSPACE_SETUP. Did you run colcon build?"
fi

# Activate uv env manually for python dependencies used in planners (e.g. clipseg)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Activated uv virtual environment."
fi

mkdir -p log
LOG_FILE="log/planning_$(date +%Y-%m-%d_%H-%M-%S).log"
echo "Logging all planning output to $LOG_FILE"

exec > >(tee -a "$LOG_FILE") 2>&1

# We use trap to gracefully kill all background ROS nodes if the user presses Ctrl+C
trap 'echo "Terminating ROS Planning Nodes..."; kill $(jobs -p); wait; exit 0' INT

# 1. Start Map building layer
echo "[1/3] Starting Depth & Semantic Grid Standalone..."
cd planner/map
python3 depth_grid_standalone.py \
  --depth-topic /camera_sensor/depth/image_raw \
  --camera-info-topic /camera_sensor/depth/camera_info \
  --rgb-topic /camera_sensor/image_raw \
  --grid-topic /local_traversability_grid \
  --target-frame base_footprint \
  --enable-clipseg \
  --clipseg-model-dir /home/zyy/nvidia/models/clipseg-rd64-refined &
cd ../..
sleep 4

# 2. Start local rollout
echo "[2/3] Starting Local Rollout Selector..."
cd planner/localplanner
python3 local_rollout_selector.py &
cd ../..
sleep 2

# 3. Start MPC Tracker
echo "[3/3] Starting MPC Tracker..."
cd planner/localplanner
python3 ackermann_mpc_tracker.py --max-speed 0.15 &
cd ../..

echo ""
echo "=========================================================="
echo "ROS Base Planning Pipeline is running in the background."
echo "Press [Ctrl+C] in this terminal to strictly KILL ALL nodes."
echo "=========================================================="

wait
