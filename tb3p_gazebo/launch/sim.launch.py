"""
Gazebo에 tb3p 설계도로 로봇을 띄운다.

인자
- world: turtlebot3_world(기본), calib (빈 바닥 + AprilGrid 보드),
         lidar_cam (보드를 LiDAR 스캔 높이에 세운 LiDAR-카메라 캘리브레이션용)
         umbmark (빈 바닥, 바퀴 캘리브레이션 UMBmark 주행용)
- enable_infra: true면 D435i IR 카메라 2개를 켠다 (Kalibr용, 시뮬레이션이 느려짐)
- x_pose, y_pose: 로봇 스폰 위치 (비우면 월드별 기본값)
- gui: Gazebo 화면 켜기 (기본 true, calibrate_all.py 같은 자동 실행에서는 false로 서버만)
- calib_file: 로봇이 믿는 캘리브레이션 (기본: 공칭 calibration.yaml).
  calib_merge.py가 만든 추정값을 주면 /tf가 추정값으로 바뀐다. Gazebo 속 로봇(정답)은 그대로

1) Gazebo 실행
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
                            IncludeLaunchDescription, OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# 월드 이름 -> (패키지, worlds 폴더 안 파일, 기본 스폰 x, y)
WORLDS = {
    'turtlebot3_world': ('turtlebot3_gazebo', 'turtlebot3_world.world', '-2.0', '-0.5'),
    'calib': ('tb3p_gazebo', 'calib.sdf', '0.0', '0.0'),
    'lidar_cam': ('tb3p_gazebo', 'lidar_cam.sdf', '0.0', '0.0'),
    'umbmark': ('tb3p_gazebo', 'umbmark.sdf', '0.0', '0.0'),
}


def gazebo_and_spawn(context):
    """월드 인자에 맞춰 Gazebo를 켜고 정답 설계도로 로봇을 스폰한다."""
    name = LaunchConfiguration('world').perform(context)
    if name not in WORLDS:
        raise RuntimeError(f'world는 {list(WORLDS)} 중 하나여야 합니다: {name}')
    pkg, filename, x_default, y_default = WORLDS[name]
    world = os.path.join(get_package_share_directory(pkg), 'worlds', filename)
    x = LaunchConfiguration('x_pose').perform(context) or x_default
    y = LaunchConfiguration('y_pose').perform(context) or y_default
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    server_only = LaunchConfiguration('gui').perform(context).lower() in ('false', '0')
    flags = '-r -s' if server_only else '-r'
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
            launch_arguments={'gz_args': f'{flags} -v2 {world}',
                              'on_exit_shutdown': 'true'}.items(),
        ),
        Node(
            package='ros_gz_sim',
            executable='create',
            output='screen',
            arguments=['-name', 'tb3p',
                       '-topic', '/truth/robot_description',
                       '-x', x,
                       '-y', y,
                       '-z', '0.01'],
        ),
    ]


def generate_launch_description():
    tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    tb3_description = get_package_share_directory('turtlebot3_description')
    rs_description = get_package_share_directory('realsense2_description')
    tb3p_description = get_package_share_directory('tb3p_description')
    tb3p_gazebo = get_package_share_directory('tb3p_gazebo')

    bridge_config = os.path.join(tb3p_gazebo, 'config', 'bridge.yaml')
    xacro_file = os.path.join(tb3p_description, 'urdf', 'tb3p_burger.urdf.xacro')
    truth_calib = os.path.join(tb3p_gazebo, 'config', 'sim_truth.yaml')
    nominal_calib = os.path.join(tb3p_description, 'config', 'calibration.yaml')

    truth_description = ParameterValue(
        Command(['xacro ', xacro_file, ' calib_file:=', truth_calib,
                 ' sim:=true truth:=true enable_infra:=', LaunchConfiguration('enable_infra')]),
        value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='turtlebot3_world',
                              description='turtlebot3_world, calib, lidar_cam, umbmark 중 하나'),
        DeclareLaunchArgument('enable_infra', default_value='false',
                              description='true면 D435i IR 카메라 2개를 켠다'),
        DeclareLaunchArgument('x_pose', default_value='',
                              description='스폰 x (비우면 월드 기본값)'),
        DeclareLaunchArgument('y_pose', default_value='',
                              description='스폰 y (비우면 월드 기본값)'),
        DeclareLaunchArgument('gui', default_value='true',
                              description='false면 Gazebo 화면 없이 서버만 (자동 실행용)'),
        DeclareLaunchArgument('calib_file', default_value=nominal_calib,
                              description='로봇이 믿는 캘리브레이션 YAML (추정값으로 바꿀 수 있음)'),

        # Gazebo가 모델(model://)과 로봇·카메라 메시(package://)를 찾을 위치
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(tb3_gazebo, 'models')),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(tb3p_gazebo, 'models')),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.dirname(tb3_description)),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.dirname(rs_description)),

        # 1) Gazebo + 4) 정답 설계도로 로봇 스폰 (월드에 따라 달라서 함께 처리)
        OpaqueFunction(function=gazebo_and_spawn),

        # 2) 공칭 설계도 → /tf (로봇과 모든 추정기가 보는 값)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(tb3p_description, 'launch', 'description.launch.py')),
            launch_arguments={'use_sim_time': 'true', 'sim': 'true',
                              'calib_file': LaunchConfiguration('calib_file')}.items(),
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
