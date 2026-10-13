#!/usr/bin/env python3
"""
rosbag에 녹화된 IMU를 분석해 흔들기 데이터가 쓸 만한지 확인한다.

- 축마다 가장 강한 주파수와 그 진폭 (흔들기 궤적의 주파수와 비교)
- 이웃한 두 샘플 사이의 최대 변화 (갑자기 튀는 값이 있는지)
- 각속도, 가속도 그래프를 PNG로 저장

사용 예:
  python3 imu_bag_report.py ~/kalibr_data/rig_01 /camera/camera/imu
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from rclpy.serialization import deserialize_message  # noqa: E402
import rosbag2_py  # noqa: E402
from sensor_msgs.msg import Imu  # noqa: E402


def read_imu(bag_dir, topic):
    """Bag에서 IMU 메시지를 읽어 (시각, 각속도, 가속도) 배열로 돌려준다."""
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id='mcap'),
                rosbag2_py.ConverterOptions(input_serialization_format='cdr',
                                            output_serialization_format='cdr'))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    stamps, gyro, accel = [], [], []
    while reader.has_next():
        _, data, _ = reader.read_next()
        msg = deserialize_message(data, Imu)
        stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        w, a = msg.angular_velocity, msg.linear_acceleration
        gyro.append((w.x, w.y, w.z))
        accel.append((a.x, a.y, a.z))
    return np.array(stamps), np.array(gyro), np.array(accel)


def dominant_frequency(t, x):
    """신호에서 가장 강한 주파수(Hz, 0.05 Hz 이상)와 그 주파수 사인파의 진폭을 구한다."""
    dt = np.median(np.diff(t))
    x = x - x.mean()
    n_fft = 16 * len(x)                          # 0을 덧붙여 주파수를 촘촘하게
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)), n_fft))
    freqs = np.fft.rfftfreq(n_fft, dt)
    f = float(freqs[np.argmax(spectrum * (freqs > 0.05))])
    # 그 주파수의 사인, 코사인으로 최소제곱 맞춤 → 진폭
    basis = np.column_stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)])
    coef, *_ = np.linalg.lstsq(basis, x, rcond=None)
    return f, float(np.hypot(*coef))


def main():
    """명령줄에서 실행할 때의 시작점."""
    bag_dir = os.path.expanduser(sys.argv[1])
    topic = sys.argv[2] if len(sys.argv) > 2 else '/camera/camera/imu'
    t, gyro, accel = read_imu(bag_dir, topic)
    rate = (len(t) - 1) / (t[-1] - t[0])
    print(f'{topic}: {len(t)} samples, {t[-1] - t[0]:.1f} s, {rate:.1f} Hz')

    for name, data, unit in (('gyro ', gyro, 'rad/s'), ('accel', accel, 'm/s^2')):
        for k, axis in enumerate('xyz'):
            f, amp = dominant_frequency(t, data[:, k])
            jump = np.abs(np.diff(data[:, k])).max()
            print(f'{name} {axis}: mean {data[:, k].mean():+8.3f}  '
                  f'peak {f:.3f} Hz, amplitude {amp:.3f} {unit}, max step {jump:.3f}')

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for k, axis in enumerate('xyz'):
        axes[0].plot(t - t[0], gyro[:, k], lw=0.6, label=axis)
        axes[1].plot(t - t[0], accel[:, k], lw=0.6, label=axis)
    axes[0].set_ylabel('angular velocity (rad/s)')
    axes[1].set_ylabel('linear acceleration (m/s^2)')
    axes[1].set_xlabel('time (s)')
    for ax in axes:
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
    fig.suptitle(f'{os.path.basename(bag_dir)} {topic}')
    out = bag_dir.rstrip('/') + '_imu.png'
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f'saved {out}')


if __name__ == '__main__':
    main()
