"""
tb3p 설계도(xacro)를 처리해 robot_state_publisher로 TF를 발행한다.

시뮬레이션과 실제 로봇이 같은 파일을 쓰고, use_sim_time만 다르게 준다.
센서 위치는 calib_file(기본: config/calibration.yaml)에서 읽는다.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('tb3p_description')
    xacro_file = os.path.join(share, 'urdf', 'tb3p_burger.urdf.xacro')
    default_calib = os.path.join(share, 'config', 'calibration.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')
    calib_file = LaunchConfiguration('calib_file')
    robot_description = ParameterValue(
        Command(['xacro ', xacro_file, ' calib_file:=', calib_file]), value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='시뮬레이션이면 true (Gazebo 시계 사용)'),
        DeclareLaunchArgument('calib_file', default_value=default_calib,
                              description='센서 위치를 읽을 캘리브레이션 YAML 경로'),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description,
                         'use_sim_time': use_sim_time}],
        ),
    ])
