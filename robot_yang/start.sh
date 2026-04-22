#!/bin/bash

# 等待时间（秒）
WAIT_TIME=4

gnome-terminal -- bash -c "source install/setup.bash && ros2 launch carlike_robot_description gazebo_sim.launch.py; exec bash"


sleep $WAIT_TIME
gnome-terminal -- bash -c "source install/setup.bash && ros2 run teleop_twist_keyboard teleop_twist_keyboard; exec bash"

