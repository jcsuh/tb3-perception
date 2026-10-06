"""
Gazebo에 tb3p 설계도로 로봇을 띄운다.

1) Gazebo 실행 (turtlebot3_world 월드)
2) 공칭 설계도 -> /tf (로봇이 믿는 위치, calibration.yaml)
3) 정답 설계도 -> /truth/robot_description, /truth/tf_static (sim_truth.yaml, 채점 전용)
4) 정답 설계도로 Gazebo에 로봇 스폰 (Gazebo 속 로봇 = 실제로 조립된 상태)
5) Gazebo <-> ROS 토픽 브리지
6) 깊이 영상 -> 점구름 (depth_image_proc, 실제 로봇과 같은 방식)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    tb3_description = get_package_share_directory('turtlebot3_description')
    rs_description = get_package_share_directory('realsense2_description')
    tb3p_description = get_package_share_directory('tb3p_description')
    tb3p_gazebo = get_package_share_directory('tb3p_gazebo')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world = os.path.join(tb3_gazebo, 'worlds', 'turtlebot3_world.world')
    bridge_config = os.path.join(tb3p_gazebo, 'config', 'bridge.yaml')
    xacro_file = os.path.join(tb3p_description, 'urdf', 'tb3p_burger.urdf.xacro')
    truth_calib = os.path.join(tb3p_gazebo, 'config', 'sim_truth.yaml')

    truth_description = ParameterValue(
        Command(['xacro ', xacro_file, ' calib_file:=', truth_calib, ' sim:=true']),
        value_type=str)

    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')

    return LaunchDescription([
        DeclareLaunchArgument('x_pose', default_value='-2.0'),
        DeclareLaunchArgument('y_pose', default_value='-0.5'),

        # Gazebo가 월드 모델(model://)과 로봇·카메라 메시(package://)를 찾을 위치
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(tb3_gazebo, 'models')),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.dirname(tb3_description)),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.dirname(rs_description)),

        # 1) Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
            launch_arguments={'gz_args': f'-r -v2 {world}',
                              'on_exit_shutdown': 'true'}.items(),
        ),

        # 2) 공칭 설계도 → /tf (로봇과 모든 추정기가 보는 값)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(tb3p_description, 'launch', 'description.launch.py')),
            launch_arguments={'use_sim_time': 'true', 'sim': 'true'}.items(),
        ),

        # 3) 정답 설계도 → /truth/... (Gazebo 스폰과 채점 전용, 로봇 /tf와 섞이지 않게)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='truth_state_publisher',
            namespace='truth',
            output='screen',
            parameters=[{'robot_description': truth_description,
                         'use_sim_time': True}],
            remappings=[('/tf', '/truth/tf'), ('/tf_static', '/truth/tf_static')],
        ),

        # 4) 정답 설계도로 Gazebo에 로봇 생성
        Node(
            package='ros_gz_sim',
            executable='create',
            output='screen',
            arguments=['-name', 'tb3p',
                       '-topic', '/truth/robot_description',
                       '-x', x_pose,
                       '-y', y_pose,
                       '-z', '0.01'],
        ),

        # 5) Gazebo <-> ROS 토픽 브리지
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            output='screen',
            parameters=[{'config_file': bridge_config, 'use_sim_time': True}],
        ),

        # 6) 깊이 영상 -> 점구름
        #    Gazebo 점구름은 일반 좌표계 방향이라 optical frame과 맞지 않아 쓰지 않는다
        Node(
            package='depth_image_proc',
            executable='point_cloud_xyz_node',
            name='depth_to_points',
            output='screen',
            parameters=[{'use_sim_time': True}],
            remappings=[('image_rect', '/camera/camera/depth/image_rect_raw'),
                        ('camera_info', '/camera/camera/depth/camera_info'),
                        ('points', '/camera/camera/depth/points')],
        ),
    ])
