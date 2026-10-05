
"""
tb3p 설계도(xacro)를 처리해 robot_state_publisher로 TF를 발행한다.

시뮬레이션과 실제 로봇이 같은 파일을 쓰고, use_sim_time만 다르게 준다.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    xacro_file = os.path.join(
        get_package_share_directory('tb3p_description'), 'urdf', 'tb3p_burger.urdf.xacro')

    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_description = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time',
                              default_value='false',
                              description='시뮬레이션이면 true (Gazebo 시계 사용)'),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description,
                         'use_sim_time': use_sim_time}],
        ),
    ])
