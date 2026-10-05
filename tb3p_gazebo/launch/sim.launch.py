
"""
Gazebo에 tb3p 설계도로 로봇을 띄운다.

1) Gazebo 실행 (turtlebot3_world 월드)
2) tb3p_description으로 TF 발행 (use_sim_time=true)
3) robot_description 토픽으로 Gazebo에 로봇 스폰
4) Gazebo <-> ROS 토픽 브리지
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    tb3_description = get_package_share_directory('turtlebot3_description')
    tb3p_description = get_package_share_directory('tb3p_description')
    tb3p_gazebo = get_package_share_directory('tb3p_gazebo')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world = os.path.join(tb3_gazebo, 'worlds', 'turtlebot3_world.world')
    bridge_config = os.path.join(tb3p_gazebo, 'config', 'bridge.yaml')

    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')

    return LaunchDescription([
        DeclareLaunchArgument('x_pose', default_value='-2.0'),
        DeclareLaunchArgument('y_pose', default_value='-0.5'),

        # Gazebo가 월드 모델(model://)과 로봇 메시(package://)를 찾을 위치
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(tb3_gazebo, 'models')),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.dirname(tb3_description)),

        # 1) Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
            launch_arguments={'gz_args': f'-r -v2 {world}',
                              'on_exit_shutdown': 'true'}.items(),
        ),

        # 2) 설계도 → TF
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(tb3p_description, 'launch', 'description.launch.py')),
            launch_arguments={'use_sim_time': 'true'}.items(),
        ),

        # 3) 같은 설계도로 Gazebo에 로봇 생성
        Node(
            package='ros_gz_sim',
            executable='create',
            output='screen',
            arguments=['-name', 'tb3p',
                       '-topic', 'robot_description',
                       '-x', x_pose,
                       '-y', y_pose,
                       '-z', '0.01'],
        ),

        # 4) Gazebo <-> ROS 토픽 브리지
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            output='screen',
            parameters=[{'config_file': bridge_config, 'use_sim_time': True}],
        ),
    ])
