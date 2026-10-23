#!/usr/bin/env python3
"""
드리프트 검사 1 시험용 기록: 카메라 브래킷을 단계별로 돌리며 LiDAR 스캔과 깊이 영상을 모은다 (세션 20).

단계마다 (기본 0°, 1°, 2°, 5°, 다시 0°)
  1) /sim/camera_mount_yaw_cmd로 카메라를 그 각도로 돌린다 (Gazebo 정답 모델의 마운트 관절)
     로봇의 TF(공칭)는 그대로라서 로봇은 카메라가 돌아간 것을 모른다 = 실제 드리프트와 같은 상황
  2) 3초 기다린 뒤 2초 정지, 이어서 제자리에서 천천히 왼쪽으로 돌다가 오른쪽으로 돈다
     (돌면서 재야 장면이 바뀌고, 좌우 둘 다 돌면 시각이 어긋났을 때 그 영향이 반대로 나타나 드러난다)
  3) 깊이 영상이 올 때마다 (초당 --rate장) 가장 가까운 스캔과 짝지어 저장
     깊이 영상은 LiDAR 선이 지나가는 가로 띠만 잘라서 저장 (용량)
함께 기록: IMU 자이로 (회전 보정용), /truth/joint_states의 실제 마운트 각도 (채점용)

결과: NPZ 하나 (기본 ~/drift_data/session20.npz), 약 2분 30초

사용 예 (sim.launch.py 실행 중, turtlebot3_world):
  python3 drift_collect.py --calib ~/lidar_cam_data/session13_result.yaml
"""
import argparse
import json
import math
import os

import drift_check as dc
from geometry_msgs.msg import TwistStamped
import lidar_cam_check as lcc
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, Imu, JointState, LaserScan
from std_msgs.msg import Float64
from tf2_msgs.msg import TFMessage
import yaml

DEPTH = '/camera/camera/depth/image_rect_raw'


def stamp(msg):
    """메시지 시각 (s)."""
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def depth_image(msg):
    """깊이 영상 -> m 단위 float32 (32FC1은 m, 16UC1은 mm)."""
    if msg.encoding == '32FC1':
        raw = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.step // 4)
        return raw[:, :msg.width].copy()
    if msg.encoding == '16UC1':
        raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.step // 2)
        d = raw[:, :msg.width].astype(np.float32) / 1000.0
        d[d == 0] = np.nan
        return d
    raise ValueError(f'모르는 깊이 형식 {msg.encoding}')


class Collector(Node):
    """단계 진행, 명령, 기록."""

    def __init__(self, args, t_color_lidar, elev):
        super().__init__('drift_collect',
                         parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL,
                                                        True)])
        self.args, self.t_color_lidar, self.elev = args, t_color_lidar, elev
        latched = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        sensor = qos_profile_sensor_data
        self.info = None
        self.scans = []                   # 최근 스캔 몇 개 (시각, 거리)
        self.scan_meta = None
        self.imu = []
        self.mount = np.nan               # 실제 마운트 각도 (rad)
        self.edges = {}
        self.frames = []
        self.geo = None
        self.create_subscription(Image, DEPTH, self.on_depth, sensor)
        self.create_subscription(CameraInfo, DEPTH.rsplit('/', 1)[0] + '/camera_info',
                                 lambda m: setattr(self, 'info', m), sensor)
        self.create_subscription(LaserScan, '/scan', self.on_scan, sensor)
        self.create_subscription(Imu, '/imu', self.on_imu, sensor)
        self.create_subscription(JointState, '/truth/joint_states', self.on_truth_joint, sensor)
        self.create_subscription(TFMessage, '/tf_static', lambda m: lcc.add_edges(self.edges, m),
                                 latched)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.yaw_pub = self.create_publisher(Float64, '/sim/camera_mount_yaw_cmd', 10)
        self.plan = self.make_plan()
        self.k, self.t0, self.done = -1, None, False
        self.last_cap = -1e9
        self.create_timer(0.02, self.on_tick)

    def make_plan(self):
        """[(단계 번호, 마운트 각도 rad, 이름, 길이 s, ω)] 시간표."""
        a = self.args
        half = a.phase_time / 2
        out = []
        for i, deg in enumerate(a.yaws_deg):
            yaw = math.radians(deg)
            out += [(i, yaw, 'settle', 3.0, 0.0), (i, yaw, 'static', 2.0, 0.0),
                    (i, yaw, 'left', half, a.rotate), (i, yaw, 'right', half, -a.rotate)]
        return out

    def on_scan(self, m):
        if self.scan_meta is None:
            self.scan_meta = {'angle_min': m.angle_min, 'angle_increment': m.angle_increment}
        self.scans.append((stamp(m), np.asarray(m.ranges, dtype=np.float32)))
        self.scans = self.scans[-5:]

    def on_imu(self, m):
        self.imu.append((stamp(m), m.angular_velocity.z))

    def on_truth_joint(self, m):
        if 'camera_mount_yaw_joint' in m.name:
            self.mount = m.position[m.name.index('camera_mount_yaw_joint')]

    def geometry(self):
        """처음 한 번: 공칭 TF와 캘리브레이션으로 깊이 카메라 자세, 마운트 점."""
        need = ('base_scan', 'camera_color_optical_frame', 'camera_depth_optical_frame',
                'camera_bottom_screw_frame')
        if any(f not in self.edges for f in need) or self.info is None:
            return None
        poses = lcc.frame_poses(self.edges)
        t_base_lidar = lcc.relative(poses, 'base_footprint', 'base_scan')
        t_color_depth = lcc.relative(poses, 'camera_color_optical_frame',
                                     'camera_depth_optical_frame')
        t_base_cam0 = t_base_lidar @ np.linalg.inv(self.t_color_lidar) @ t_color_depth
        mount = lcc.relative(poses, 'base_footprint', 'camera_bottom_screw_frame')[:3, 3]
        k = np.array(self.info.k, dtype=np.float64).reshape(3, 3)
        return {'T_base_lidar': t_base_lidar, 'T_base_cam0': t_base_cam0, 'mount': mount,
                'K': k, 'width': int(self.info.width), 'height': int(self.info.height)}

    def band_rows(self, ranges):
        """깊이 영상에서 LiDAR 선이 지나가는 행 범위 ± 여유 (첫 스캔으로 한 번 정함)."""
        g = self.geo
        pts = dc.to_frame(g['T_base_lidar'], dc.lidar_points(
            ranges, self.scan_meta['angle_min'], self.scan_meta['angle_increment'], self.elev))
        pc = dc.to_frame(np.linalg.inv(g['T_base_cam0']), pts)
        pc = pc[pc[:, 2] > 0.2]
        v = g['K'][1, 1] * pc[:, 1] / pc[:, 2] + g['K'][1, 2]
        u = g['K'][0, 0] * pc[:, 0] / pc[:, 2] + g['K'][0, 2]
        v = v[(u >= 0) & (u < g['width'])]
        lo = int(max(0, math.floor(v.min()) - self.args.band_margin))
        hi = int(min(g['height'], math.ceil(v.max()) + self.args.band_margin))
        return lo, hi

    def on_depth(self, m):
        if self.k < 0 or not self.scans or self.plan[self.k][2] == 'settle':
            return
        t = stamp(m)
        if t - self.last_cap < 1.0 / self.args.rate:
            return
        if self.geo is None:
            self.geo = self.geometry()
            if self.geo is None:
                return
            self.band = self.band_rows(self.scans[-1][1])
            self.get_logger().info(f'깊이 영상 띠: {self.band[0]}–{self.band[1]}행 저장')
        ts, rng = min(self.scans, key=lambda s: abs(s[0] - t))
        d = depth_image(m)[self.band[0]:self.band[1]]
        i, yaw, name, _, _ = self.plan[self.k]
        self.frames.append({'t_img': t, 't_scan': ts, 'ranges': rng,
                            'depth': d.astype(np.float16), 'phase': i, 'part': name,
                            'cmd': yaw, 'truth': self.mount})
        self.last_cap = t

    def send(self, w):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.twist.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def on_tick(self):
        if self.done:
            return
        t = self.get_clock().now().nanoseconds * 1e-9
        if t <= 0 or self.info is None or not self.scans or not self.imu:
            return
        if self.k < 0 or t - self.t0 >= self.plan[self.k][3]:
            self.k += 1
            self.t0 = t
            if self.k >= len(self.plan):
                self.send(0.0)
                self.yaw_pub.publish(Float64(data=0.0))
                self.done = True
                return
            i, yaw, name, _, _ = self.plan[self.k]
            if name == 'settle':
                self.get_logger().info(f'단계 {i + 1}/{len(self.args.yaws_deg)}: '
                                       f'카메라 마운트 {math.degrees(yaw):+.1f}°')
        i, yaw, name, _, w = self.plan[self.k]
        self.yaw_pub.publish(Float64(data=float(yaw)))
        self.send(w)


def main():
    """명령줄에서 실행할 때의 시작점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--calib', required=True, help='LiDAR-카메라 캘리브레이션 결과 YAML (세션 13)')
    ap.add_argument('--yaws-deg', type=float, nargs='+', default=[0, 1, 2, 5, 0],
                    help='단계별 카메라 마운트 각도 (°)')
    ap.add_argument('--phase-time', type=float, default=20.0, help='단계마다 회전 시간 (s)')
    ap.add_argument('--rotate', type=float, default=0.3, help='회전 속도 (rad/s)')
    ap.add_argument('--rate', type=float, default=2.0, help='초당 저장할 영상 수')
    ap.add_argument('--band-margin', type=int, default=40, help='띠 위아래 여유 (행)')
    ap.add_argument('--out', default=os.path.expanduser('~/drift_data/session20.npz'))
    args = ap.parse_args()

    with open(os.path.expanduser(args.calib)) as f:
        cal = yaml.safe_load(f)
    t_color_lidar = np.array(cal['matrix'], dtype=np.float64)
    elev = float(cal.get('beam_elev_deg', 0.0))
    rclpy.init()
    node = Collector(args, t_color_lidar, elev)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.send(0.0)
        node.yaw_pub.publish(Float64(data=0.0))
        rclpy.try_shutdown()
        print('중단: 저장하지 않음')
        return
    fr = node.frames
    g = node.geo
    meta = {'args': vars(args), 'scan': node.scan_meta, 'band_v0': node.band[0],
            'elev_deg': elev, 'K': g['K'].tolist(), 'T_base_lidar': g['T_base_lidar'].tolist(),
            'T_base_cam0': g['T_base_cam0'].tolist(), 'mount': g['mount'].tolist(),
            'calib': os.path.abspath(os.path.expanduser(args.calib))}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out, meta=json.dumps(meta), imu=np.array(node.imu),
        t_img=np.array([f['t_img'] for f in fr]), t_scan=np.array([f['t_scan'] for f in fr]),
        ranges=np.array([f['ranges'] for f in fr]), depth=np.array([f['depth'] for f in fr]),
        phase=np.array([f['phase'] for f in fr]), part=np.array([f['part'] for f in fr]),
        cmd=np.array([f['cmd'] for f in fr]), truth=np.array([f['truth'] for f in fr]))
    print(f'저장: {args.out}  (영상 {len(fr)}장, 띠 {node.band[1] - node.band[0]}행)')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
