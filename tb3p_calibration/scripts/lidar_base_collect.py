#!/usr/bin/env python3
"""
LiDAR·카메라 -> 몸 캘리브레이션 재료를 돌아다니며 모은다 (세션 24).

로봇은 drift_experiment.py와 같은 방식으로 장애물을 피하며 좌우로 흔들며 돌아다닌다.
  - 앞으로 가기: LiDAR에 비스듬히 보이는 정도 -> LiDAR yaw
  - 돌기: LiDAR가 회전 중심 둘레로 그리는 원 -> LiDAR 위치 (x, y)
  - 깊이 영상의 바닥 -> 카메라 기울기와 높이
  - (선택) --stop-time > 0이면 --stop-every초마다 멈추고, 멈춘 1.5초 뒤부터 0.5초마다 깊이 영상을 더
    저장한다. 세션 24에서 달릴 때와 서 있을 때의 바닥 결과가 같음을 확인해서 기본은 멈추지 않는다
기록: /scan (전부), /joint_states 바퀴 각도, 깊이 영상 (--depth-period초마다 한 장, 픽셀 --stride개마다
하나만 저장해 용량을 줄임), 깊이 camera_info, (시뮬레이션) /truth/odom (나중 확인용)
LiDAR 스캔 매칭에 벽과 기둥이 필요해서 turtlebot3_world(기본 월드)에서 한다.

결과: NPZ 하나 (기본 ~/lidar_base_data/session24.npz), 기본 3분
분석: lidar_base_calib.py

사용 예 (sim.launch.py 실행 중, turtlebot3_world):
  python3 lidar_base_collect.py
"""
import argparse
import json
import os

from drift_experiment import Wander
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState, LaserScan

DEPTH = '/camera/camera/depth/image_rect_raw'


def stamp(msg):
    """메시지 시각 (s)."""
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def depth_array(m, stride):
    """sensor_msgs/Image (32FC1 m 또는 16UC1 mm) -> 깊이 m, 픽셀 stride개마다 하나 (float32)."""
    if m.encoding == '32FC1':
        img = np.frombuffer(m.data, dtype=np.float32).reshape(m.height, m.step // 4)
    elif m.encoding == '16UC1':
        img = np.frombuffer(m.data, dtype=np.uint16).reshape(m.height, m.step // 2)
        img = np.where(img > 0, img / 1000.0, np.nan)
    else:
        raise ValueError(f'모르는 깊이 형식 {m.encoding}')
    return np.ascontiguousarray(img[::stride, :m.width:stride], dtype=np.float32)


class Collector(Node):
    """돌아다니며 스캔, 바퀴, 깊이 영상을 모은다."""

    def __init__(self, args):
        super().__init__('lidar_base_collect',
                         parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL,
                                                        True)])
        self.args = args
        self.wander = Wander(args.seed)
        self.joint, self.truth, self.scan_t, self.scan_r = [], [], [], []
        self.depth, self.depth_t = [], []
        self.scan_info, self.k, self.last_scan = None, None, None
        self.t0, self.done, self.last_depth = None, False, -1e9
        self.stopped_since = None         # 멈춘 시각 (달리는 중이면 None)
        sensor = qos_profile_sensor_data
        self.create_subscription(LaserScan, '/scan', self.on_scan, sensor)
        self.create_subscription(JointState, '/joint_states', self.on_joint, sensor)
        self.create_subscription(Image, DEPTH, self.on_depth, sensor)
        self.create_subscription(CameraInfo, DEPTH.rsplit('/', 1)[0] + '/camera_info',
                                 self.on_info, sensor)
        self.create_subscription(Odometry, '/truth/odom', self.on_truth, sensor)
        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_timer(0.05, self.on_tick)

    def on_scan(self, m):
        if self.scan_info is None:
            self.scan_info = {'angle_min': m.angle_min, 'angle_increment': m.angle_increment}
        self.last_scan = (list(m.ranges), m.angle_min, m.angle_increment)
        if self.t0 is not None and not self.done:
            self.scan_t.append(stamp(m))
            self.scan_r.append(np.asarray(m.ranges, dtype=np.float32))

    def on_joint(self, m):
        try:
            i, j = m.name.index('wheel_left_joint'), m.name.index('wheel_right_joint')
        except ValueError:
            return
        if self.t0 is not None and not self.done:
            self.joint.append((stamp(m), m.position[i], m.position[j]))

    def on_info(self, m):
        self.k = [float(v) for v in m.k]

    def on_depth(self, m):
        t = stamp(m)
        if self.t0 is None or self.done:
            return
        settled = self.stopped_since is not None and t - self.stopped_since > 1.5
        if t - self.last_depth < (0.5 if settled else self.args.depth_period):
            return
        self.depth.append(depth_array(m, self.args.stride))
        self.depth_t.append(t)
        self.last_depth = t

    def on_truth(self, m):
        if self.t0 is not None and not self.done:
            p, q = m.pose.pose.position, m.pose.pose.orientation
            self.truth.append((stamp(m), p.x, p.y, q.x, q.y, q.z, q.w))

    def send(self, v, w):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(w)
        self.pub.publish(msg)

    def on_tick(self):
        if self.done:
            return
        t = self.get_clock().now().nanoseconds * 1e-9
        if t <= 0 or self.last_scan is None or self.k is None:
            return
        if self.t0 is None:
            self.t0 = t
            self.get_logger().info(f'스캔·깊이 수신 확인, {self.args.duration:.0f}초 돌아다니며 기록')
        tau = t - self.t0
        if tau < 2.0:                       # 처음 2초는 정지 (기록 시작 정렬)
            self.send(0.0, 0.0)
            return
        if tau > self.args.duration + 2.0:
            self.send(0.0, 0.0)
            self.done = True
            return
        cycle = self.args.stop_every + self.args.stop_time
        if self.args.stop_time > 0 and (tau - 2.0) % cycle >= self.args.stop_every:
            if self.stopped_since is None:
                self.stopped_since = t
            self.send(0.0, 0.0)
            return
        self.stopped_since = None
        self.send(*self.wander.step(t, self.last_scan))


def main():
    """명령줄에서 실행할 때의 시작점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--duration', type=float, default=180.0, help='돌아다니는 시간 (s)')
    ap.add_argument('--depth-period', type=float, default=3.0, help='깊이 영상 저장 간격 (s)')
    ap.add_argument('--stride', type=int, default=4, help='깊이 영상 픽셀 간격 (용량 줄이기)')
    ap.add_argument('--stop-every', type=float, default=12.0, help='이만큼 달린 뒤 멈춤 (s)')
    ap.add_argument('--stop-time', type=float, default=0.0, help='멈추는 시간 (s, 0 = 안 멈춤)')
    ap.add_argument('--seed', type=int, default=24, help='돌아다니기 무작위 씨앗')
    ap.add_argument('--out', default=os.path.expanduser('~/lidar_base_data/session24.npz'))
    args = ap.parse_args()
    rclpy.init()
    node = Collector(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    for _ in range(5):
        node.send(0.0, 0.0)
    if not node.scan_t or not node.depth:
        raise SystemExit('기록이 비어 있음 (시뮬레이션이 켜져 있나?)')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    meta = {'scan': node.scan_info, 'K': node.k, 'stride': args.stride,
            'duration': args.duration, 'seed': args.seed}
    np.savez_compressed(args.out, meta=json.dumps(meta), joint=np.array(node.joint),
                        scan_t=np.array(node.scan_t), scan_r=np.array(node.scan_r),
                        depth=np.array(node.depth), depth_t=np.array(node.depth_t),
                        truth=np.array(node.truth))
    print(f'saved {args.out}: 스캔 {len(node.scan_t)}개, 바퀴 {len(node.joint)}개, '
          f'깊이 영상 {len(node.depth)}장')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
