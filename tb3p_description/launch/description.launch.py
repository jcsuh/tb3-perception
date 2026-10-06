"""
tb3p 설계도(xacro)를 처리해 robot_state_publisher로 TF를 발행한다.

시뮬레이션과 실제 로봇이 같은 파일을 쓰고, use_sim_time과 sim만 다르게 준다.
- calib_file: 센서 위치를 읽을 YAML (기본: config/calibration.yaml)
- sim: true면 D435i 내부 좌표계를 설계도가 만든다 (실제 로봇에서는 드라이버가 발행)
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
    sim = LaunchConfiguration('sim')
    robot_description = ParameterValue(
        Command(['xacro ', xacro_file, ' calib_file:=', calib_file, ' sim:=', sim]),
        value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='시뮬레이션이면 true (Gazebo 시계 사용)'),
        DeclareLaunchArgument('calib_file', default_value=default_calib,
                              description='센서 위치를 읽을 캘리브레이션 YAML 경로'),
        DeclareLaunchArgument('sim', default_value='false',
                              description='true면 D435i 내부 좌표계를 설계도가 발행'),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description,
                         'use_sim_time': use_sim_time}],
        ),
    ])
