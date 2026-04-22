import os

import launch
import launch.actions
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    robot_name_in_model = "carlike_robot"
    pkg_path = get_package_share_directory('carlike_robot_description')

    default_model_path = os.path.join(pkg_path, 'xacro', 'car_description.xacro')
    default_rviz_config_path = os.path.join(pkg_path, 'config', 'rviz', 'gazebo_sim.rviz')
    default_world_path = '/home/zyy/robot_yang/src/carlike_robot_description/worlds/tree.world'

    declare_model_path = launch.actions.DeclareLaunchArgument(
        name='model',
        default_value=default_model_path,
        description='Absolute path to robot xacro file'
    )

    declare_world_path = launch.actions.DeclareLaunchArgument(
        name='world',
        default_value=default_world_path,
        description='Absolute path to world file'
    )

    robot_description = launch_ros.parameter_descriptions.ParameterValue(
        launch.substitutions.Command(
            ['xacro ', launch.substitutions.LaunchConfiguration('model')]
        ),
        value_type=str
    )

    robot_state_publisher_node = launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
        output='screen'
    )

    joint_state_publisher_node = launch_ros.actions.Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        output='screen'
    )

    launch_gazebo = launch.actions.IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': launch.substitutions.LaunchConfiguration('world')
        }.items()
    )

    spawn_entity_node = launch_ros.actions.Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', '/robot_description',
            '-entity', robot_name_in_model
        ],
        output='screen'
    )

    rviz_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', default_rviz_config_path],
        output='screen'
    )

    load_joint_state_broadcaster = launch.actions.ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'joint_state_broadcaster'],
        output='screen'
    )

    load_ackermann_controller = launch.actions.ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'ackermann_steering_controller'],
        output='screen'
    )

    return launch.LaunchDescription([
        declare_model_path,
        declare_world_path,
        joint_state_publisher_node,
        robot_state_publisher_node,
        launch_gazebo,
        spawn_entity_node,
        load_joint_state_broadcaster,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=load_joint_state_broadcaster,
                on_exit=[load_ackermann_controller]
            )
        ),
        rviz_node
    ])