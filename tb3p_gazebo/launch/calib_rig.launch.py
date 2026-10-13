"""
손에 든 캘리브레이션 리그를 calib 월드에 띄운다 (로봇 없이).

- 리그 = D435i만 떼어 낸 것 (내부 배치, Gazebo 센서, 토픽 이름이 로봇의 D435i와 같다)
- tb3p::Excitation 플러그인이 리그를 보드 앞에서 정해진 궤적으로 흔든다
  (5초 정지 → 3초 동안 서서히 커짐 → 계속 흔들기)
- 리그의 TF는 /rig/tf, /rig/tf_static 으로 발행 (카메라-IMU 외부 파라미터 정답)

인자
- enable_infra: IR 카메라 2개 켜기 (기본 true, Kalibr용)
- gui: Gazebo 화면 켜기 (기본 true, 반복 실험에서는 false로 CPU 절약)
- imu_noise_scale: Gazebo IMU 노이즈 배율 (기본 1 = 데이터시트 값)
"""
import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription, OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# 리그 흔들기 중심 (m): 보드 앞 약 0.45 m, 보드 중심 높이
CENTER = ('0.15', '0.0', '0.1375')


def gazebo(context, world, ros_gz_sim):
    """Launch 인자 gui에 따라 Gazebo를 화면과 함께(-r) 또는 서버만(-r -s) 켠다."""
    server_only = LaunchConfiguration('gui').perform(context).lower() in ('false', '0')
    flags = '-r -s' if server_only else '-r'
    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'{flags} -v2 {world}',
                          'on_exit_shutdown': 'true'}.items(),
    )]


def generate_launch_description():
    rs_description = get_package_share_directory('realsense2_description')
    tb3p_gazebo = get_package_share_directory('tb3p_gazebo')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world = os.path.join(tb3p_gazebo, 'worlds', 'calib.sdf')
    bridge_config = os.path.join(tb3p_gazebo, 'config', 'bridge.yaml')
    rig_xacro = os.path.join(tb3p_gazebo, 'urdf', 'tb3p_rig.urdf.xacro')
    truth_calib = os.path.join(tb3p_gazebo, 'config', 'sim_truth.yaml')
    cx, cy, cz = CENTER

    rig_description = ParameterValue(
        Command(['xacro ', rig_xacro, ' calib_file:=', truth_calib,
                 ' sim:=true enable_infra:=', LaunchConfiguration('enable_infra'),
                 ' imu_noise_scale:=', LaunchConfiguration('imu_noise_scale'),
                 f' cx:={cx} cy:={cy} cz:={cz}']),
        value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument('enable_infra', default_value='true',
                              description='true면 D435i IR 카메라 2개를 켠다'),
        DeclareLaunchArgument('gui', default_value='true',
                              description='false면 Gazebo 서버만 (화면 없음)'),
        DeclareLaunchArgument('imu_noise_scale', default_value='1.0',
                              description='Gazebo IMU 노이즈 배율 (1 = 데이터시트 값)'),

        # Gazebo가 모델, 메시, 시스템 플러그인(libtb3p_excitation.so)을 찾을 위치
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(tb3p_gazebo, 'models')),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.dirname(rs_description)),
        AppendEnvironmentVariable('GZ_SIM_SYSTEM_PLUGIN_PATH',
                                  os.path.join(get_package_prefix('tb3p_gazebo'), 'lib')),

        OpaqueFunction(function=gazebo, args=[world, ros_gz_sim]),

        # 리그 설계도 → /rig/robot_description, /rig/tf_static (정답)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='rig_state_publisher',
            namespace='rig',
            output='screen',
            parameters=[{'robot_description': rig_description,
                         'use_sim_time': True}],
            remappings=[('/tf', '/rig/tf'), ('/tf_static', '/rig/tf_static')],
        ),

        Node(
            package='ros_gz_sim',
            executable='create',
            output='screen',
            arguments=['-name', 'calib_rig',
                       '-topic', '/rig/robot_description',
                       '-x', cx, '-y', cy, '-z', cz],
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            output='screen',
            parameters=[{'config_file': bridge_config, 'use_sim_time': True}],
        ),
    ])
