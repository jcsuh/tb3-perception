#!/usr/bin/env python3
"""
시간 오프셋 추정용 회전 주행을 하며 세 센서를 기록한다 (세션 17).

로봇이 제자리에서 빨라졌다 느려졌다, 왼쪽 오른쪽으로 불규칙하게 돈다 (여러 주파수 사인파의 합).
세 센서가 각자 회전 속도(요 각속도)를 잰다:
  - IMU 자이로 z (200 Hz)
  - 바퀴 오도메트리: 좌우 바퀴 속도 차이 / 간격 (/joint_states)
  - LiDAR: 연속한 두 스캔을 맞춰(스캔 매칭) 구한 회전 / 시간 (10 Hz)
같은 움직임을 재므로 시계가 어긋난 만큼 그래프가 옆으로 밀려 보인다 -> 나중에 그 밀림을 잰다.
LiDAR 스캔 매칭에는 주변 벽이 필요해서 turtlebot3_world(기본 월드)에서 돈다.

기록: <prefix>/imu, <prefix>/joint_states, <prefix>/scan (delay_injector.py 출력), (시뮬레이션) /truth/odom
결과: NPZ 하나 (기본 ~/time_sync_data/session17.npz), 약 70초

사용 예:
  ros2 launch tb3p_gazebo sim.launch.py              (turtlebot3_world)
  python3 delay_injector.py --delays imu=0 joint_states=30 scan=70
  python3 time_sync_collect.py
"""
import argparse
import json
import math
import os

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState, LaserScan

FREQS = (0.11, 0.23, 0.37, 0.53, 0.79, 1.07)   # Hz, 서로 정수배가 아니게


def stamp(msg):
    """메시지 시각 (s)."""
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def yaw_profile(duration, peak, seed=0, ramp=2.0):
    """
    회전 속도 함수 t -> ω (rad/s): 여러 사인파의 합을 최대 peak로 맞추고 앞뒤 ramp초에 걸쳐 키우고 줄인다.

    빠르게 바뀌는 성분(1 Hz 근처)이 있어야 시간 밀림이 잘 보인다.
    """
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, len(FREQS))
    amps = 1.0 / np.sqrt(np.arange(1, len(FREQS) + 1))
    ts = np.linspace(0, duration, int(duration * 100) + 1)
    raw = sum(a * np.sin(2 * np.pi * f * ts + p) for a, f, p in zip(amps, FREQS, phases))
    scale = peak / np.abs(raw).max()

    def omega(t):
        if t <= 0 or t >= duration:
            return 0.0
        env = min(1.0, t / ramp, (duration - t) / ramp)
        val = sum(a * math.sin(2 * math.pi * f * t + p) for a, f, p in zip(amps, FREQS, phases))
        return env * scale * val
    return omega


class Collector(Node):
    """회전 명령을 내고 세 센서와 정답 위치를 모은다."""

    def __init__(self, args):
        super().__init__('time_sync_collect',
                         parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL,
                                                        True)])
        self.args = args
        self.omega = yaw_profile(args.duration, args.peak, args.seed)
        self.imu, self.joint, self.truth = [], [], []
        self.scan_t, self.scan_r = [], []
        self.scan_info = None
        self.t0, self.done = None, False
        sensor = qos_profile_sensor_data
        p = args.prefix
        self.create_subscription(Imu, f'{p}/imu', self.on_imu, sensor)
        self.create_subscription(JointState, f'{p}/joint_states', self.on_joint, sensor)
        self.create_subscription(LaserScan, f'{p}/scan', self.on_scan, sensor)
        self.create_subscription(Odometry, '/truth/odom', self.on_truth, sensor)
        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_timer(0.01, self.on_tick)

    def on_imu(self, m):
        w = m.angular_velocity
        self.imu.append((stamp(m), w.x, w.y, w.z))

    def on_joint(self, m):
        try:
            i, j = m.name.index('wheel_left_joint'), m.name.index('wheel_right_joint')
        except ValueError:
            return
        self.joint.append((stamp(m), m.position[i], m.position[j]))

    def on_scan(self, m):
        if self.scan_info is None:
            self.scan_info = {'angle_min': m.angle_min, 'angle_increment': m.angle_increment,
                              'range_min': m.range_min, 'range_max': m.range_max}
        self.scan_t.append(stamp(m))
        self.scan_r.append(np.asarray(m.ranges, dtype=np.float32))

    def on_truth(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.truth.append((stamp(m), p.x, p.y, q.x, q.y, q.z, q.w))

    def send(self, w):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.twist.angular.z = float(w)
        self.pub.publish(msg)

    def on_tick(self):
        if self.done:
            return
        t = self.get_clock().now().nanoseconds * 1e-9
        if t <= 0 or not self.imu or not self.joint or not self.scan_t:
            return
        if self.t0 is None:
            self.t0 = t + 3.0                     # 3초 정지 후 시작
            self.get_logger().info(f'세 센서 수신 확인, 3초 뒤 {self.args.duration:.0f}초 회전')
        tau = t - self.t0
        self.send(self.omega(tau))
        if tau > self.args.duration + 2.0:
            self.send(0.0)
            self.done = True


def main():
    """명령줄에서 실행할 때의 시작점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prefix', default='/delayed', help='센서 토픽 앞 이름 (delay_injector 출력)')
    ap.add_argument('--duration', type=float, default=60.0, help='회전 시간 (s)')
    ap.add_argument('--peak', type=float, default=1.4, help='최대 회전 속도 (rad/s)')
    ap.add_argument('--seed', type=int, default=0, help='회전 모양을 정하는 난수')
    ap.add_argument('--out', default=os.path.expanduser('~/time_sync_data/session17.npz'))
    args = ap.parse_args()
    rclpy.init()
    node = Collector(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.send(0.0)
        rclpy.try_shutdown()
        print('중단: 저장하지 않음')
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    meta = {'args': vars(args), 'scan_info': node.scan_info}
    np.savez_compressed(args.out, meta=json.dumps(meta), imu=np.array(node.imu),
                        joint=np.array(node.joint), truth=np.array(node.truth),
                        scan_t=np.array(node.scan_t), scan_r=np.array(node.scan_r))
    print(f'저장: {args.out}  (IMU {len(node.imu)}, 바퀴 {len(node.joint)}, '
          f'스캔 {len(node.scan_t)}, 정답 {len(node.truth)})')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
