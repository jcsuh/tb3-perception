#!/usr/bin/env python3
"""
센서 토픽에 일부러 시간 지연을 넣어 다시 내보낸다 (세션 17, W8–9 지연 주입 노드).

실제 로봇에서는 센서마다 측정한 순간과 메시지에 찍힌 시각이 조금씩 다르다
(USB·드라이버 지연, 버퍼, 시계 차이). 시뮬레이션은 모두 정확해서 이것을 흉내 낸다.
  - 시각 지연: header.stamp = 측정 시각 + 지연  (추정해야 할 "시간 오프셋")
  - 전달 지연: 메시지도 지연만큼 늦게 내보낸다 (실제처럼, --no-transport면 바로 내보냄)
입력 토픽 /X -> 출력 토픽 <prefix>/X (기본 /delayed/X). 추정기는 출력 토픽만 본다.
실행 중 지연 바꾸기 (세션 22): /sim/delay_cmd (std_msgs/String) 에 "joint_states=20 scan=0" 처럼 보낸다.
바퀴 배율 고장 (세션 22): 같은 토픽에 "joint_scale=1.03" -> 바퀴 회전량을 1.03배로 (바퀴 반지름이 바뀐 것과 같음)

calibration.yaml의 time_offset 약속: "센서 시각에 더해야 할 보정값"
  -> 지연 d ms를 넣었으면 정답 time_offset = -d ms

사용 예 (sim.launch.py 실행 중):
  python3 delay_injector.py --delays imu=0 joint_states=30 scan=70
"""
import argparse
import collections
import math

from nav_msgs.msg import Odometry
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu, JointState, LaserScan
from std_msgs.msg import String

TYPES = {'imu': Imu, 'joint_states': JointState, 'scan': LaserScan, 'odom': Odometry}


class DelayInjector(Node):
    """토픽마다 시각을 늦추고 (선택) 전달도 늦춘다."""

    def __init__(self, delays_ms, prefix, transport):
        super().__init__('delay_injector',
                         parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL,
                                                        True)])
        self.queue = collections.deque()          # (내보낼 시각 ns, 발행기, 메시지)
        self.transport = transport
        self.count = collections.Counter()
        self.delays = dict(delays_ms)
        self.joint_scale = 1.0
        self.joint_raw, self.joint_out = {}, {}     # 바퀴 배율: 관절별 이전 원래 값, 내보낸 값
        for name, d in delays_ms.items():
            pub = self.create_publisher(TYPES[name], f'{prefix}/{name}', qos_profile_sensor_data)
            self.create_subscription(TYPES[name], f'/{name}',
                                     lambda m, p=pub, n=name: self.on_msg(m, p, self.delays[n], n),
                                     qos_profile_sensor_data)
            self.get_logger().info(f'/{name} -> {prefix}/{name}: 지연 {d:g} ms')
        self.create_subscription(String, '/sim/delay_cmd', self.on_cmd, 10)
        self.create_timer(0.001, self.flush)
        self.create_timer(10.0, self.report)

    def scale_joints(self, msg):
        """관절 각도의 증가량에 배율을 곱해 이어 붙인다 (±π에서 감기는 것도 처리)."""
        pos = list(msg.position)
        for i, (n, raw) in enumerate(zip(msg.name, pos)):
            if n in self.joint_raw:
                step = math.remainder(raw - self.joint_raw[n], 2 * math.pi)
                self.joint_out[n] += self.joint_scale * step
            else:
                self.joint_out[n] = raw
            self.joint_raw[n] = raw
            pos[i] = self.joint_out[n]
        msg.position = pos

    def on_msg(self, msg, pub, delay_ms, name):
        if name == 'joint_states':
            self.scale_joints(msg)
        d = Duration(nanoseconds=int(round(delay_ms * 1e6)))
        msg.header.stamp = (Time.from_msg(msg.header.stamp) + d).to_msg()
        self.count[name] += 1
        if not self.transport or delay_ms <= 0:
            pub.publish(msg)
            return
        due = self.get_clock().now().nanoseconds + d.nanoseconds
        self.queue.append((due, pub, msg))

    def on_cmd(self, msg):
        items = msg.data.split()
        for it in [x for x in items if x.startswith('joint_scale=')]:
            items.remove(it)
            try:
                scale = float(it.split('=')[1])
            except ValueError:
                self.get_logger().warn(f'바퀴 배율 명령 무시: {it}')
                continue
            if scale != self.joint_scale:
                self.joint_scale = scale
                self.get_logger().info(f'바퀴 배율 변경: {scale:g}')
        try:
            new = parse_delays(items)
        except (ValueError, SystemExit) as e:
            self.get_logger().warn(f'지연 명령 무시 ({msg.data!r}): {e}')
            return
        changed = {k: v for k, v in new.items() if k in self.delays and self.delays[k] != v}
        if changed:
            self.delays.update(changed)
            text = ', '.join(f'{k} {v:g} ms' for k, v in sorted(changed.items()))
            self.get_logger().info(f'지연 변경: {text}')

    def flush(self):
        now = self.get_clock().now().nanoseconds
        while self.queue and self.queue[0][0] <= now:
            _, pub, msg = self.queue.popleft()
            pub.publish(msg)

    def report(self):
        text = ', '.join(f'{k} {v}' for k, v in sorted(self.count.items()))
        self.get_logger().info(f'지난 10초 전달: {text}')
        self.count.clear()


def parse_delays(items):
    """['imu=0', 'scan=70'] -> {'imu': 0.0, 'scan': 70.0}."""
    out = {}
    for it in items:
        name, val = it.split('=')
        if name not in TYPES:
            raise SystemExit(f'모르는 토픽 {name}: {list(TYPES)} 중 하나')
        out[name] = float(val)
    return out


def main():
    """명령줄에서 실행할 때의 시작점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--delays', nargs='+', default=['imu=0', 'joint_states=0', 'scan=0'],
                    help='토픽=지연ms (토픽: imu, joint_states, scan, odom)')
    ap.add_argument('--prefix', default='/delayed', help='출력 토픽 앞에 붙일 이름')
    ap.add_argument('--no-transport', action='store_true',
                    help='시각만 늦추고 메시지는 바로 내보냄')
    args = ap.parse_args()
    rclpy.init()
    node = DelayInjector(parse_delays(args.delays), args.prefix, not args.no_transport)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
