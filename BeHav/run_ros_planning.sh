#!/bin/bash

# ================= Configuration =================
# 设为 true 时，MPC Tracker不会下发 /cmd_vel 控制指令，小车将保持静止，方便纯规划算法的调试
DISABLE_CONTROL=true
# =================================================

WORKSPACE_SETUP="../robot_yang/install/setup.bash"
if [ -f "$WORKSPACE_SETUP" ]; then
    source "$WORKSPACE_SETUP"
    echo "Sourced robot_yang workspace."
else
    echo "Warning: robot_yang workspace setup not found at $WORKSPACE_SETUP. Did you run colcon build?"
fi
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Activated uv virtual environment."
fi
mkdir -p log
LOG_FILE="log/planning_$(date +%Y-%m-%d_%H-%M-%S).log"
echo "Logging all planning output to $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'echo "Terminating ROS Planning Nodes..."; kill $(jobs -p); wait; exit 0' INT TERM EXIT
echo "[1/4] Starting Depth & Semantic Grid Standalone..."
cd planner/map
python3 depth_grid_standalone.py \
  --depth-topic /camera_sensor/depth/image_raw \
  --camera-info-topic /camera_sensor/depth/camera_info \
  --rgb-topic /camera_sensor/image_raw \
  --grid-topic /local_traversability_grid \
  --target-frame base_footprint \
  --enable-clipseg \
  --clipseg-model-dir ~/nvidia/models/clipseg-rd64-refined &
cd ../..
sleep 4
echo "[2/4] Starting Far Waypoint Planner..."
cd planner/farplanner
python3 far_waypoint_planner.py \
  --current-waypoint-local-topic /far/current_waypoint_local \
  --local-plan-topic /far/local_plan \
  --reference-path-topic /far/reference_path_local \
  --waypoint-queue-topic /far/waypoint_queue_local \
  --current-subgoal-topic /far/current_subgoal_local \
  --route-status-topic /far/route_status \
  --far-goal-reached-topic /far/goal_reached \
  --waypoint-queue-distances 0.8 1.5 2.5 3.3 4.0 \
  --current-subgoal-distance 2.5 &
cd ../..
sleep 2
echo "[3/4] Starting Local Rollout Selector..."
cd planner/localplanner
python3 local_rollout_selector.py \
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
  --arbiter-switch-margin 1.0 &
cd ../..
sleep 2
echo "[4/4] Starting MPC Tracker..."
cd planner/localplanner

MPC_ARGS="--far-goal-reached-topic /far/goal_reached --max-speed 0.5"
if [ "$DISABLE_CONTROL" = "true" ]; then
  MPC_ARGS="$MPC_ARGS --disable-control"
  echo "⚠️ [DEBUG MODE] Control commands (/cmd_vel) are DISABLED. The robot will not move."
fi

python3 ackermann_mpc_tracker.py $MPC_ARGS &
cd ../..
echo "=========================================================="
echo "ROS Base Planning Pipeline is running in the background."
echo "Press [Ctrl+C] to strictly KILL ALL nodes."
echo "=========================================================="
wait