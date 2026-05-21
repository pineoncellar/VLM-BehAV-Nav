#!/bin/bash
set -e

echo "Starting BeHav ROS Interface without LLM... (Using preset JSON from env)"

WORKSPACE_SETUP="../robot_yang/install/setup.bash"
if [ -f "$WORKSPACE_SETUP" ]; then
    source "$WORKSPACE_SETUP"
    echo "Sourced robot_yang workspace"
else
    echo "Warning: robot_yang workspace setup not found at $WORKSPACE_SETUP. Did you run colcon build?"
fi

# ==============================================================
# 配置区域：通过修改下面的 JSON 数组，直接注入 LLM 输出的解析结果
# 注意：必须是合法的 JSON 格式列表（如 ["item1", "item2"]）
# ==============================================================
export SKIP_NLP=1

export PRESET_LANDMARKS='["fire hydrant", "bus"]'
export PRESET_NAV_ACTIONS='["go straight", "turn right", "keep walking", "stop"]'
export PRESET_BEHAV_ACTIONS='["avoid stepping on", "walk on"]'
export PRESET_BEHAV_TARGETS='["grass", "road"]'

# 启动节点
uv run python3 ros_interface.py
