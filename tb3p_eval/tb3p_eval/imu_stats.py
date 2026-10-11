"""
IMU 토픽을 일정 시간 모아 통계를 낸다.

로봇이 가만히 있을 때 실행하면 다음을 알 수 있다.
- 축별 평균과 표준편차: 표준편차가 노이즈 크기, 자이로 평균이 바이어스
- 실제 수신 주기 (메시지 시각 기준)
- 가속도 평균으로 구한 중력 방향과 기울기 (roll, pitch는 z가 위인 좌표계 가정)

파라미터:
  topic     Imu 토픽 (기본: /imu)
  duration  모을 시간, 초 (기본: 10.0, 메시지 시각 기준)
"""
import math

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Imu


def imu_statistics(stamps, gyro, accel):
    """
    모은 IMU 샘플로 통계를 계산한다.

    stamps: (N,) 초, gyro: (N, 3) rad/s, accel: (N, 3) m/s^2
    """
    a_mean = accel.mean(axis=0)
    a_norm = float(np.linalg.norm(a_mean))
    duration = float(stamps[-1] - stamps[0])
    return {
        'count': len(stamps),
        'rate': (len(stamps) - 1) / duration if duration > 0 else 0.0,
        'gyro_mean': gyro.mean(axis=0),
        'gyro_std': gyro.std(axis=0),
        'accel_mean': a_mean,
        'accel_std': accel.std(axis=0),
        'gravity_norm': a_norm,
        'gravity_dir': a_mean / a_norm,
        'roll': math.atan2(a_mean[1], a_mean[2]),
        'pitch': math.atan2(-a_mean[0], math.hypot(a_mean[1], a_mean[2])),
    }


def _vec(v, fmt):
    return '(' + ', '.join(format(float(x), fmt) for x in v) + ')'


def format_report(topic, s):
    """통계를 사람이 읽기 쉬운 문자열로 만든다."""
    count = s['count']
    rate = s['rate']
    gyro_mean = _vec(s['gyro_mean'], '+.2e')
    gyro_std = _vec(s['gyro_std'], '.2e')
    accel_mean = _vec(s['accel_mean'], '+.4f')
    accel_std = _vec(s['accel_std'], '.4f')
    g_norm = s['gravity_norm']
    g_dir = _vec(s['gravity_dir'], '+.4f')
    roll_deg = math.degrees(s['roll'])
    pitch_deg = math.degrees(s['pitch'])
    return '\n'.join([
        f'topic            {topic}',
        f'samples, rate    {count}, {rate:.1f} Hz',
        f'gyro  mean       {gyro_mean} rad/s',
        f'gyro  std        {gyro_std} rad/s',
        f'accel mean       {accel_mean} m/s^2',
        f'accel std        {accel_std} m/s^2',
        f'gravity |a|      {g_norm:.4f} m/s^2, direction {g_dir}',
        f'tilt (z-up)      roll {roll_deg:+.3f} deg, pitch {pitch_deg:+.3f} deg',
    ])


class ImuStats(Node):
    """IMU 토픽을 정해진 시간 동안 모으는 노드."""

    def __init__(self):
        super().__init__('imu_stats')
        self.topic = self.declare_parameter('topic', '/imu').value
        self.duration = float(self.declare_parameter('duration', 10.0).value)
        self.stamps, self.gyro, self.accel = [], [], []
        self.done = False
        self.create_subscription(Imu, self.topic, self._callback, 200)
        self.get_logger().info(
            f'{self.topic} 를 {self.duration:.0f}초 동안 모읍니다. 로봇을 움직이지 마세요.')

    def _callback(self, msg):
        if self.done:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        w = msg.angular_velocity
        a = msg.linear_acceleration
        self.stamps.append(t)
        self.gyro.append((w.x, w.y, w.z))
        self.accel.append((a.x, a.y, a.z))
        if t - self.stamps[0] >= self.duration:
            self.done = True


def main(args=None):
    """명령줄에서 실행할 때의 시작점."""
    rclpy.init(args=args)
    node = ImuStats()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.done:
            stats = imu_statistics(
                np.array(node.stamps), np.array(node.gyro), np.array(node.accel))
            print(format_report(node.topic, stats))
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
