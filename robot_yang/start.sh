#!/bin/bash
source install/setup.bash

# 获取当前的显示端口，确保窗口能弹出到你的 RDP 界面
export DISPLAY=:11.0  # 如果你的端口不是10，请根据 echo $DISPLAY 的结果修改

# 等待时间（秒）
WAIT_TIME=4

# 使用 xfce4-terminal 启动仿真
xfce4-terminal --title="Gazebo Simulation" -e "bash -c 'source install/setup.bash && ros2 launch carlike_robot_description gazebo_sim.launch.py; exec bash'"

sleep $WAIT_TIME

# 使用 xfce4-terminal 启动键盘控制
xfce4-terminal --title="Keyboard Teleop" -e "bash -c 'source install/setup.bash && ros2 run teleop_twist_keyboard teleop_twist_keyboard; exec bash'"