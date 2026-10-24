#!/usr/bin/env python3
"""
드리프트 모니터 시험 (세션 22): 로봇을 돌아다니게 하면서 고장을 시간표대로 넣고 /diagnostics를 기록한다.

돌아다니기: LiDAR로 장애물을 피하며 앞으로 가고, 좌우로 살짝씩 흔들며(검사 2·3에 회전이 필요) 막히면 제자리에서 돈다.

시나리오 (--scenario)
  camera : 정상 120초로 기준을 배운 뒤, 카메라 마운트를 1° -> 0° -> 2° -> 0° -> 5° -> 0° 로 두 번 (약 16분)
           -> 각도별 감지율 (ERROR 비율), 0° 구간 오경보
  sensor : 정상 120초 뒤 바퀴·LiDAR 시각에 20 ms, 50 ms 지연을 넣었다 빼고,
           이어서 바퀴 회전량 배율 1.03, 1.06, 0.94 (바퀴 반지름이 바뀐 것과 같음) (약 18분)
           -> delay_injector.py 실행 + 모니터가 /delayed/* 를 보게 해야 한다 (아래 사용 예)
  normal : 고장 없이 --duration 초 (기본 1800 = 30분) -> 오경보 수

기록: CSV 한 줄 = /diagnostics 메시지 하나 (1 Hz). 시각, 구간, 넣은 고장, 실제 마운트 각도, 검사 3개의 상태와 값.
분석: drift_experiment_analyze.py

사용 예 (sim.launch.py world:=turtlebot3_world 실행 중, 모니터를 새로 켠 뒤 바로)
  터미널 A: ros2 run tb3p_calibration drift_monitor_node --ros-args -p use_sim_time:=true
              -p calib_file:=$HOME/lidar_cam_data/session13_result.yaml
  터미널 B: python3 drift_experiment.py --scenario camera

  sensor 시험은 지연 주입 노드를 먼저 켜고, 모니터가 지연된 토픽을 보게 한다:
  터미널 C: python3 delay_injector.py --delays joint_states=0 scan=0
  터미널 A: (위 명령 끝에) -p joint_topic:=/delayed/joint_states -p scan_topic:=/delayed/scan
  터미널 B: python3 drift_experiment.py --scenario sensor
"""
import argparse
import csv
import math
import os
import random

from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import TwistStamped
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Float64, String

# CSV 열: 검사별로 /diagnostics에서 꺼낼 값
CHECKS = {
    'camera_lidar': ('cam', ['yaw_deg', 'deviation_deg', 'baseline_deg', 'batches',
                             'skipped_still', 'residual_mm_at_best',
                             'lidar_time_correction_ms']),
    'yaw_rate_consistency': ('rot', ['wheel_scale', 'lidar_scale', 'wheel_scale_baseline',
                                     'lidar_scale_baseline', 'rate_std', 'evaluations']),
    'time_offset': ('time', ['wheel_offset_ms', 'lidar_offset_ms', 'wheel_offset_baseline_ms',
                             'lidar_offset_baseline_ms', 'evaluations']),
}
COLUMNS = (['t', 'phase', 'phase_t', 'label', 'mount_cmd_deg', 'mount_true_deg',
            'delay_wheel_ms', 'delay_lidar_ms', 'joint_scale'] +
           [f'{p}_{k}' for p, keys in CHECKS.values() for k in ['level'] + keys])


def scenario(name, duration):
    """[(이름, 길이 s, 마운트 각도 °, 바퀴 지연 ms, LiDAR 지연 ms, 바퀴 배율)]."""
    def phase(label, dur, deg=0.0, dw=0.0, dl=0.0, js=1.0):
        return (label, dur, deg, dw, dl, js)

    warm = [phase('warmup', 120.0)]
    if name == 'camera':
        steps = []
        for _ in range(2):
            for deg, hold in ((1.0, 100.0), (2.0, 100.0), (5.0, 80.0)):
                steps += [phase(f'cam{deg:g}', hold, deg=deg), phase('normal', 50.0)]
        return warm + steps
    if name == 'sensor':
        steps = []
        for d in (20.0, 50.0):
            steps += [phase(f'wheel{d:g}ms', 80.0, dw=d), phase('normal', 60.0),
                      phase(f'lidar{d:g}ms', 80.0, dl=d), phase('normal', 60.0)]
        for js in (1.03, 1.06, 0.94):
            steps += [phase(f'scale{js:g}', 80.0, js=js), phase('normal', 60.0)]
        return warm + steps
    return [phase('normal', duration)]


def sector_min(ranges, angle_min, inc, lo_deg, hi_deg):
    """lo_deg..hi_deg 방향 (0 = 앞, 왼쪽 +) 안의 가장 가까운 거리 (없으면 큰 값)."""
    best = 10.0
    for i, r in enumerate(ranges):
        if not math.isfinite(r) or r < 0.05:
            continue
        a = math.degrees(math.remainder(angle_min + inc * i, 2 * math.pi))
        if lo_deg <= a <= hi_deg:
            best = min(best, r)
    return best


class Wander:
    """장애물 피하며 돌아다니기. 출력: (v, ω)."""

    STOP, CLEAR, V_MAX, W_TURN = 0.38, 0.55, 0.15, 0.8

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.turn_dir, self.turn_until = 0, 0.0
        self.weave_t0, self.weave_period = 0.0, 9.0

    def step(self, t, scan):
        if scan is None:
            return 0.0, 0.0
        r, amin, inc = scan
        front = sector_min(r, amin, inc, -35, 35)
        left = sector_min(r, amin, inc, 35, 100)
        right = sector_min(r, amin, inc, -100, -35)
        if self.turn_dir:
            # 막혀서 제자리 회전 중: 앞이 트이고 최소 시간이 지나면 다시 출발
            if front > self.CLEAR and t > self.turn_until:
                self.turn_dir = 0
            else:
                return 0.0, self.turn_dir * self.W_TURN
        if front < self.STOP:
            wide_l = sector_min(r, amin, inc, 20, 160)
            wide_r = sector_min(r, amin, inc, -160, -20)
            self.turn_dir = 1 if wide_l >= wide_r else -1
            self.turn_until = t + self.rng.uniform(0.6, 2.0)   # 매번 조금씩 다른 방향으로
            return 0.0, self.turn_dir * self.W_TURN
        if t - self.weave_t0 > self.weave_period:
            self.weave_t0, self.weave_period = t, self.rng.uniform(7.0, 12.0)
        weave = 0.5 * math.sin(2 * math.pi * (t - self.weave_t0) / self.weave_period)
        avoid = 0.08 * (1.0 / max(right, 0.15) - 1.0 / max(left, 0.15))
        v = self.V_MAX * min(1.0, max(0.3, (front - self.STOP) / 0.3))
        return v, max(-1.0, min(1.0, weave + avoid))


class Experiment(Node):
    """시간표대로 고장을 넣고, 돌아다니고, /diagnostics를 CSV로."""

    def __init__(self, args):
        super().__init__('drift_experiment',
                         parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL,
                                                        True)])
        self.plan = scenario(args.scenario, args.duration)
        self.wander = Wander(args.seed)
        self.scan, self.mount_true = None, math.nan
        self.k, self.t0, self.phase_t0, self.done = -1, None, 0.0, False
        self.last_diag = None
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        self.file = open(args.out, 'w', newline='')
        self.csv = csv.writer(self.file)
        self.csv.writerow(COLUMNS)
        self.rows = 0
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(JointState, '/truth/joint_states', self.on_truth,
                                 qos_profile_sensor_data)
        self.create_subscription(DiagnosticArray, '/diagnostics', self.on_diag, 10)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.yaw_pub = self.create_publisher(Float64, '/sim/camera_mount_yaw_cmd', 10)
        self.delay_pub = self.create_publisher(String, '/sim/delay_cmd', 10)
        self.create_timer(0.05, self.on_tick)
        self.create_timer(1.0, self.on_second)
        total = sum(p[1] for p in self.plan)
        self.get_logger().info(f'시나리오 {args.scenario}: 구간 {len(self.plan)}개, '
                               f'{total / 60:.1f}분 -> {args.out}')

    def on_scan(self, m):
        self.scan = (list(m.ranges), m.angle_min, m.angle_increment)

    def on_truth(self, m):
        if 'camera_mount_yaw_joint' in m.name:
            self.mount_true = m.position[m.name.index('camera_mount_yaw_joint')]

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def send_fault(self):
        _, _, deg, dw, dl, js = self.plan[self.k]
        self.yaw_pub.publish(Float64(data=math.radians(deg)))
        self.delay_pub.publish(String(data=f'joint_states={dw:g} scan={dl:g} joint_scale={js:g}'))

    def on_tick(self):
        if self.done:
            return
        t = self.now_s()
        if t <= 0.0:
            return
        if self.t0 is None:
            self.t0 = t
        if self.k < 0 or t - self.phase_t0 >= self.plan[self.k][1]:
            self.k += 1
            if self.k >= len(self.plan):
                self.finish()
                return
            self.phase_t0 = t
            name, dur, deg, dw, dl, js = self.plan[self.k]
            self.get_logger().info(f'[{t - self.t0:6.0f} s] 구간 {self.k} {name}: {dur:.0f}초, '
                                   f'마운트 {deg:+g}°, 지연 바퀴 {dw:g} / LiDAR {dl:g} ms, '
                                   f'바퀴 배율 {js:g}')
            self.send_fault()
        v, w = self.wander.step(t, self.scan)
        self.publish_cmd(v, w)

    def on_second(self):
        if self.done or self.k < 0:
            return
        self.send_fault()          # 놓친 메시지가 없도록 1초마다 다시 보낸다 (같은 값은 무시됨)
        t = self.now_s()
        if self.last_diag is None or t - self.last_diag > 10.0:
            self.get_logger().warn('/diagnostics가 10초 넘게 안 옴: drift_monitor_node 실행 중인가?',
                                   throttle_duration_sec=10.0)

    def publish_cmd(self, v, w):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def on_diag(self, msg):
        if self.done or self.k < 0:
            return
        t = self.now_s()
        self.last_diag = t
        name, _, deg, dw, dl, js = self.plan[self.k]
        row = {'t': f'{t - self.t0:.3f}', 'phase': self.k,
               'phase_t': f'{t - self.phase_t0:.3f}', 'label': name,
               'mount_cmd_deg': deg, 'mount_true_deg': f'{math.degrees(self.mount_true):.4f}',
               'delay_wheel_ms': dw, 'delay_lidar_ms': dl, 'joint_scale': js}
        for st in msg.status:
            key = st.name.replace('tb3p_drift: ', '')
            if key not in CHECKS:
                continue
            prefix, keys = CHECKS[key]
            values = {kv.key: kv.value for kv in st.values}
            row[f'{prefix}_level'] = int.from_bytes(st.level, 'little') \
                if isinstance(st.level, bytes) else int(st.level)
            for k in keys:
                row[f'{prefix}_{k}'] = values.get(k, '')
        self.csv.writerow([row.get(c, '') for c in COLUMNS])
        self.rows += 1
        if self.rows % 30 == 0:
            self.file.flush()

    def finish(self):
        if self.file.closed:
            return
        self.done = True
        self.file.close()          # 기록부터 안전하게 닫는다 (Ctrl+C 때도)
        print(f'끝: {self.rows}줄 기록')
        try:
            self.publish_cmd(0.0, 0.0)
            self.yaw_pub.publish(Float64(data=0.0))
            self.delay_pub.publish(String(data='joint_states=0 scan=0 joint_scale=1'))
        except Exception:          # Ctrl+C로 ROS가 이미 꺼졌으면 못 보낸다
            pass


def main():
    """명령줄에서 실행할 때의 시작점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario', choices=['camera', 'sensor', 'normal'], required=True)
    ap.add_argument('--duration', type=float, default=1800.0, help='normal 시나리오 길이 (s)')
    ap.add_argument('--seed', type=int, default=22, help='돌아다니기 무작위 씨앗')
    ap.add_argument('--out', default=None,
                    help='CSV (기본 ~/drift_data/session22_<시나리오>.csv)')
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.expanduser(f'~/drift_data/session22_{args.scenario}.csv')
    rclpy.init()
    node = Experiment(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    node.finish()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
