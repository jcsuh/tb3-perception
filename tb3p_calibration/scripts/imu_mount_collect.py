#!/usr/bin/env python3
"""
IMU 장착 각도 추정용 주행을 하며 기록한다 (세션 16).

정해진 동작을 시간표대로 (오도메트리 없이 명령만으로) 하고, 그동안의 센서 값을 모은다.
  1) 정지 20초          -> 중력 방향 (IMU가 수평에서 얼마나 기울었나: roll, pitch)
  2) 앞뒤 직진 20번 왕복 -> 가속·감속 때 IMU가 느끼는 방향 = 로봇 앞쪽 (yaw)
     가감속이 클수록, 횟수가 많을수록 노이즈에 덜 흔들린다 (가속도 노이즈 σ 0.03 m/s²)
  3) 제자리 회전 ±90° 8쌍 -> 회전축 방향(중력과 교차 검증), 원심 가속도로 IMU 위치(보너스)
  4) 왼쪽·오른쪽 원 돌기 5쌍 (일정 속도로 10초씩) -> 옆 방향 원심 가속도 v·ω로 yaw (세션 16)
     가감속처럼 바퀴 힘이 갑자기 바뀌지 않아 Gazebo 접촉 떨림의 영향을 덜 받는다
  5) 정지 10초          -> 처음 정지와 비교 (재현성)
기록: /imu (200 Hz), /joint_states의 바퀴 각도, (시뮬레이션) /truth/odom (채점용)

결과: NPZ 하나 (기본 ~/imu_mount_data/session16.npz), 약 4분 30초

사용 예 (sim.launch.py world:=umbmark 실행 중):
  python3 imu_mount_collect.py
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
from sensor_msgs.msg import Imu, JointState


def stamp(msg):
    """메시지 시각 (s)."""
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def trapezoid(t, peak, accel, cruise):
    """사다리꼴 속도: 0에서 accel로 peak까지, cruise초 유지, 같은 기울기로 0까지."""
    t_acc = abs(peak) / accel
    if t < t_acc:
        return math.copysign(accel * t, peak)
    if t < t_acc + cruise:
        return peak
    return math.copysign(max(abs(peak) - accel * (t - t_acc - cruise), 0.0), peak)


def schedule(n_moves, n_turns, accel, speed, ang_accel, ang_speed, cruise=0.3, pause=0.7,
             n_arcs=0, arc_speed=0.2, arc_rate=1.2, arc_time=8.0, arc_ramp=0.6):
    """
    동작 시간표: [(이름, 길이 s, 함수 t -> (v, ω))].

    이름: static(정지 측정), move(직진), turn(회전), arc(원 돌기), pause(사이 정지, 분석에 안 씀)
    arc: arc_ramp초 동안 v, ω를 함께 올리고 arc_time초 유지한 뒤 같은 시간에 내린다
    """
    def stop(_):
        return 0.0, 0.0

    move_len = 2 * speed / accel + cruise + 0.1
    turn_cruise = (math.pi / 2 - ang_speed ** 2 / ang_accel) / ang_speed   # 90°가 되는 유지 시간
    turn_len = 2 * ang_speed / ang_accel + turn_cruise + 0.1
    out = [('pause', 3.0, stop), ('static', 20.0, stop)]
    for i in range(n_moves):
        for sign in (1, -1):
            out.append(('move', move_len,
                        lambda t, s=sign: (trapezoid(t, s * speed, accel, cruise), 0.0)))
            out.append(('pause', pause, stop))
    for i in range(n_turns):
        for sign in (1, -1):
            out.append(('turn', turn_len,
                        lambda t, s=sign: (0.0, trapezoid(t, s * ang_speed, ang_accel,
                                                          turn_cruise))))
            out.append(('pause', pause, stop))
    arc_len = 2 * arc_ramp + arc_time
    for i in range(n_arcs):
        for sign in (1, -1):
            out.append(('arc', arc_len, lambda t, s=sign: (
                trapezoid(t, arc_speed, arc_speed / arc_ramp, arc_time),
                trapezoid(t, s * arc_rate, arc_rate / arc_ramp, arc_time))))
            out.append(('pause', pause, stop))
    out.append(('static', 10.0, stop))
    return out


class Collector(Node):
    """시간표대로 명령을 내고 센서 값을 모은다."""

    def __init__(self, plan):
        super().__init__('imu_mount_collect',
                         parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL,
                                                        True)])
        self.plan = plan
        self.imu, self.joint, self.truth = [], [], []
        self.phases = []                 # (이름, 시작 시각, 끝 시각)
        self.k, self.t0, self.done = -1, None, False
        sensor = qos_profile_sensor_data
        self.create_subscription(Imu, '/imu', self.on_imu, sensor)
        self.create_subscription(JointState, '/joint_states', self.on_joint, sensor)
        self.create_subscription(Odometry, '/truth/odom', self.on_truth, sensor)
        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_timer(0.01, self.on_tick)

    def on_imu(self, m):
        a, w = m.linear_acceleration, m.angular_velocity
        self.imu.append((stamp(m), a.x, a.y, a.z, w.x, w.y, w.z))

    def on_joint(self, m):
        try:
            i, j = m.name.index('wheel_left_joint'), m.name.index('wheel_right_joint')
        except ValueError:
            return
        self.joint.append((stamp(m), m.position[i], m.position[j]))

    def on_truth(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.truth.append((stamp(m), p.x, p.y, p.z, q.x, q.y, q.z, q.w))

    def send(self, v, w):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.twist.linear.x, msg.twist.angular.z = float(v), float(w)
        self.pub.publish(msg)

    def on_tick(self):
        if self.done:
            return
        t = self.get_clock().now().nanoseconds * 1e-9
        if t <= 0 or not self.imu or not self.joint:
            return
        if self.k < 0 or t - self.t0 >= self.plan[self.k][1]:
            if self.k >= 0:
                self.phases.append((self.plan[self.k][0], self.t0, t))
            self.k += 1
            self.t0 = t
            if self.k >= len(self.plan):
                self.send(0.0, 0.0)
                self.done = True
                return
            name = self.plan[self.k][0]
            if name != 'pause':
                self.get_logger().info(f'[{self.k + 1}/{len(self.plan)}] {name}')
        self.send(*self.plan[self.k][2](t - self.t0))


def main():
    """명령줄에서 실행할 때의 시작점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--moves', type=int, default=20, help='앞뒤 왕복 수')
    ap.add_argument('--turns', type=int, default=8, help='좌우 회전 쌍 수')
    ap.add_argument('--accel', type=float, default=0.8,
                    help='직진 가감속 (m/s², Gazebo DiffDrive 한계 1.0)')
    ap.add_argument('--speed', type=float, default=0.2, help='직진 최고 속도 (m/s)')
    ap.add_argument('--ang-accel', type=float, default=3.0, help='회전 가감속 (rad/s²)')
    ap.add_argument('--ang-speed', type=float, default=1.5, help='회전 최고 속도 (rad/s)')
    ap.add_argument('--arcs', type=int, default=5, help='왼쪽·오른쪽 원 돌기 쌍 수')
    ap.add_argument('--arc-speed', type=float, default=0.2, help='원 돌기 속도 (m/s)')
    ap.add_argument('--arc-rate', type=float, default=1.2, help='원 돌기 회전 속도 (rad/s)')
    ap.add_argument('--arc-time', type=float, default=10.0, help='원 돌기 유지 시간 (s)')
    ap.add_argument('--out', default=os.path.expanduser('~/imu_mount_data/session16.npz'))
    args = ap.parse_args()

    plan = schedule(args.moves, args.turns, args.accel, args.speed, args.ang_accel,
                    args.ang_speed, n_arcs=args.arcs, arc_speed=args.arc_speed,
                    arc_rate=args.arc_rate, arc_time=args.arc_time)
    total = sum(p[1] for p in plan)
    rclpy.init()
    node = Collector(plan)
    node.get_logger().info(f'동작 {len(plan)}개, 약 {total:.0f}초 (시뮬레이션 시각)')
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        print('중단: 저장하지 않음')
        node.send(0.0, 0.0)
        rclpy.try_shutdown()
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    meta = {'phases': node.phases, 'args': vars(args)}
    np.savez_compressed(args.out, meta=json.dumps(meta), imu=np.array(node.imu),
                        joint=np.array(node.joint), truth=np.array(node.truth))
    print(f'저장: {args.out}  (IMU {len(node.imu)}개, 바퀴 {len(node.joint)}개, '
          f'정답 {len(node.truth)}개)')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
