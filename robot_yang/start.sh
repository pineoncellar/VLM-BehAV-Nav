#!/bin/bash
source install/setup.bash

# 删除残留的 Gazebo 进程，确保仿真环境干净
killall -9 gzserver gzclient

# 获取当前的显示端口，确保窗口能弹出到你的 RDP 界面
export DISPLAY=$DISPLAY

# 等待时间（秒）
WAIT_TIME=4

# 提取模型（World）名称为变量，方便后续修改
WORLD_NAME="street2.world"
# 动态拼凑绝对路径
WORLD_PATH="$(pwd)/world/$WORLD_NAME"

# 启动仿真
gnome-terminal --title="Gazebo Simulation" -- bash -c "source install/setup.bash && ros2 launch carlike_robot_description gazebo_sim.launch.py world:=$WORLD_PATH; exec bash"


sleep $WAIT_TIME

# 启动键盘控制
gnome-terminal --title="Keyboard Teleop" -- bash -c 'source install/setup.bash && ros2 run teleop_twist_keyboard teleop_twist_keyboard; exec bash'

