#!/usr/bin/env python3
"""
UMBmark 주행을 자동으로 한다: 정사각형을 시계·반시계로 번갈아 돌며 기록 (세션 15).

로봇은 /joint_states의 바퀴 각도와 --wheels의 바퀴 값으로 "자기 오도메트리"를 직접 계산하고,
그 오도메트리만 보고 한 변 --side m 정사각형을 돈다 (Gazebo /odom은 쓰지 않는다).
바퀴 값이 틀렸으면 오도메트리로는 출발점에 돌아왔어도 실제로는 어긋난 곳에 선다.
  - 주행 전: --wheels 기본값 = 공칭(calibration.yaml) -> 보정 전 측정
  - 보정 후: --wheels <umbmark_analyze.py가 만든 *_wheels_ls.yaml> -> 보정 후 측정

주행마다 기록 (나중에 umbmark_analyze.py가 계산)
  - 바퀴 각도 (50 Hz), (시뮬레이션) 실제 위치 /truth/odom (50 Hz)
  - 출발·도착 전에 1초씩 멈춰서 정지 상태의 위치를 남긴다
실제 로봇에서는 /truth/odom 대신 도착점을 줄자로 재야 한다 (실제 단계에서 추가).

결과: NPZ 하나 (기본 ~/umbmark_data/before.npz)

사용 예 (sim.launch.py world:=umbmark 실행 중):
  python3 umbmark_run.py --out ~/umbmark_data/before.npz
  python3 umbmark_run.py --wheels ~/umbmark_data/before_wheels_ls.yaml
      --out ~/umbmark_data/after.npz                                 (실제로는 위와 한 줄로)
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
from sensor_msgs.msg import JointState
import wheel_odom as wo

HERE = os.path.dirname(os.path.abspath(__file__))
NOMINAL = os.path.join(HERE, '..', '..', 'tb3p_description', 'config', 'calibration.yaml')
SETTLE = 1.0          # 주행 전후 정지 시간 (s, 시뮬레이션 시각)
DRIVE_TIMEOUT = 180.0  # 정사각형 한 바퀴 제한 시간 (s), 보통 약 70초


def stamp(msg):
    """메시지 시각 (s)."""
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def yaw_of(q):
    """쿼터니언 -> yaw (rad)."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class UmbmarkRunner(Node):
    """주행 순서 관리, 오도메트리 계산, 기록."""

    def __init__(self, args, wheels):
        super().__init__('umbmark_run',
                         parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL,
                                                        True)])
        self.args, self.wheels = args, wheels
        self.dirs = [d for _ in range(args.laps) for d in (wo.CW, wo.CCW)]
        self.phi = None              # 최신 바퀴 각도 (왼쪽, 오른쪽)
        self.t_joint = None
        self.truth = None            # 최신 실제 위치 (t, x, y, yaw)
        self.odo = np.zeros(3)       # 자기 오도메트리
        self.odo_phi = None          # 오도메트리를 마지막으로 계산한 바퀴 각도
        self.runs = []
        self.state, self.t_state, self.run = 'wait', None, None
        self.create_subscription(JointState, '/joint_states', self.on_joint,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, '/truth/odom', self.on_truth, qos_profile_sensor_data)
        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_timer(0.02, self.on_tick)
        self.finished = False

    def on_joint(self, msg):
        try:
            i, j = msg.name.index('wheel_left_joint'), msg.name.index('wheel_right_joint')
        except ValueError:
            return
        self.phi = (msg.position[i], msg.position[j])
        self.t_joint = stamp(msg)
        if self.odo_phi is not None:      # 메시지마다 오도메트리를 조금씩 쌓는다
            pl, pr = self.odo_phi
            self.odo = np.array(wo.odom_step(self.odo, self.phi[0] - pl, self.phi[1] - pr,
                                             self.wheels))
        self.odo_phi = self.phi

    def on_truth(self, msg):
        p = msg.pose.pose
        self.truth = (stamp(msg), p.position.x, p.position.y, yaw_of(p.orientation))
        if self.run is not None:
            self.run['truth'].append(self.truth)

    def send(self, v, w):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(w)
        self.pub.publish(msg)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_tick(self):
        if self.finished:
            return
        t = self.now()
        if self.state == 'wait':
            if self.phi is None or self.truth is None or t <= 0:
                return
            if self.t_state is None:          # 스폰 직후 흔들림이 가라앉게 3초 기다림
                self.t_state = t
            if t - self.t_state < 3.0:
                return
            self.get_logger().info(f'주행 {len(self.dirs)}회 시작 (한 변 {self.args.side} m, '
                                   f'바퀴 값 {self.wheels})')
            self.start_run(t)
            return
        if self.run is not None:
            self.run['joint'].append((self.t_joint, *self.phi))
        if self.state == 'pre':
            self.send(0.0, 0.0)
            if t - self.t_state >= SETTLE:
                self.driver.reset(tuple(self.odo), t)
                self.run['odo_start'] = self.odo.tolist()
                self.state, self.t_state = 'drive', t
        elif self.state == 'drive':
            v, w = self.driver.step(tuple(self.odo), t)
            self.send(v, w)
            if self.driver.done:
                self.state, self.t_state = 'post', t
            elif t - self.t_state > DRIVE_TIMEOUT:
                self.get_logger().error(f'주행이 {DRIVE_TIMEOUT:.0f}초 안에 안 끝남: 멈춤 '
                                        '(바퀴 각도 부호나 /cmd_vel 연결 확인)')
                self.send(0.0, 0.0)
                self.run, self.finished = None, True
        elif self.state == 'post':
            self.send(0.0, 0.0)
            if t - self.t_state >= SETTLE:
                self.finish_run()
                if len(self.runs) < len(self.dirs):
                    self.start_run(t)
                else:
                    self.finished = True

    def start_run(self, t):
        d = self.dirs[len(self.runs)]
        self.driver = wo.SquareDriver(self.args.side, d, v_max=self.args.speed)
        self.run = {'dir': d, 'joint': [], 'truth': []}
        self.state, self.t_state = 'pre', t

    def finish_run(self):
        run, self.run = self.run, None
        tr = np.array(run['truth'])
        start = tr[0, 1:]
        end = wo.relative_pose(start, tr[-1:, 1:])[0]
        odo_end = wo.relative_pose(np.array(run['odo_start']), self.odo[None, :])[0]
        self.runs.append(run)
        n = len(self.runs)
        name = wo.DIRECTION_NAME[run['dir']]
        self.get_logger().info(
            f'[{n}/{len(self.dirs)}] {name:3s}  실제 도착 오차 x {end[0] * 1000:+7.1f} '
            f'y {end[1] * 1000:+7.1f} mm  (오도메트리로는 x {odo_end[0] * 1000:+.1f} '
            f'y {odo_end[1] * 1000:+.1f} mm)')


def save(path, runs, args, wheels):
    """주행 기록을 NPZ로: run{i}_joint (t, φL, φR), run{i}_truth (t, x, y, yaw), 정보 JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays = {}
    for i, r in enumerate(runs):
        arrays[f'run{i}_joint'] = np.array(r['joint'], dtype=np.float64)
        arrays[f'run{i}_truth'] = np.array(r['truth'], dtype=np.float64)
    meta = {'side': args.side, 'speed': args.speed, 'wheels': wheels,
            'wheels_file': os.path.abspath(args.wheels),
            'dirs': [r['dir'] for r in runs]}
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)


def main():
    """명령줄에서 실행할 때의 시작점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wheels', default=NOMINAL, help='오도메트리에 쓸 바퀴 값 YAML (wheels 항목)')
    ap.add_argument('--side', type=float, default=2.0, help='정사각형 한 변 (m)')
    ap.add_argument('--laps', type=int, default=5, help='방향마다 주행 횟수 (시계 + 반시계)')
    ap.add_argument('--speed', type=float, default=0.2, help='최대 직진 속도 (m/s)')
    ap.add_argument('--out', default=os.path.expanduser('~/umbmark_data/before.npz'))
    args = ap.parse_args()

    wheels = wo.load_wheels(args.wheels)
    rclpy.init()
    node = UmbmarkRunner(args, wheels)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        print('중단: 끝난 주행만 저장합니다')
    node.send(0.0, 0.0)
    if node.runs:
        save(args.out, node.runs, args, wheels)
        print(f'{len(node.runs)}회 저장: {args.out}')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
