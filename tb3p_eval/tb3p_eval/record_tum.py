"""
Odometry 토픽들을 TUM 궤적 파일로 기록한다.

TUM 형식은 한 줄에 "시각 x y z qx qy qz qw"이고, evo 같은 궤적 평가 도구가 바로 읽는다.
시각은 메시지 header.stamp(시뮬레이션에서는 Gazebo 시계)를 쓴다.

파라미터:
  topics   기록할 Odometry 토픽 목록 (기본: ['/truth/odom', '/odom'])
  out_dir  저장 폴더 (기본: /tmp/tb3p_traj), 파일 이름은 토픽 이름에서 만든다
"""
import os

from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


def odom_to_tum_line(msg):
    """Odometry 메시지 하나를 TUM 형식 한 줄로 바꾼다."""
    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    return f'{t:.9f} {p.x:.6f} {p.y:.6f} {p.z:.6f} {q.x:.9f} {q.y:.9f} {q.z:.9f} {q.w:.9f}\n'


def topic_to_filename(topic):
    """토픽 이름을 파일 이름으로 바꾼다. 예: /truth/odom -> truth_odom.tum."""
    return topic.strip('/').replace('/', '_') + '.tum'


class TumRecorder(Node):
    """여러 Odometry 토픽을 각각의 TUM 파일로 기록하는 노드."""

    def __init__(self):
        super().__init__('record_tum')
        topics = self.declare_parameter('topics', ['/truth/odom', '/odom']).value
        out_dir = self.declare_parameter('out_dir', '/tmp/tb3p_traj').value
        os.makedirs(out_dir, exist_ok=True)

        self._files = {}
        self._counts = {}
        for topic in topics:
            path = os.path.join(out_dir, topic_to_filename(topic))
            self._files[topic] = open(path, 'w', encoding='utf-8')
            self._counts[topic] = 0
            self.create_subscription(Odometry, topic, self._make_callback(topic), 50)
            self.get_logger().info(f'{topic} -> {path}')
        self.create_timer(5.0, self._report)

    def _make_callback(self, topic):
        def callback(msg):
            self._files[topic].write(odom_to_tum_line(msg))
            self._counts[topic] += 1
        return callback

    def _report(self):
        summary = ', '.join(f'{t}: {n}' for t, n in self._counts.items())
        self.get_logger().info(f'기록한 메시지 수 ({summary})')

    def close(self):
        """열어 둔 파일을 모두 닫는다."""
        for f in self._files.values():
            f.close()


def main(args=None):
    """명령줄에서 실행할 때의 시작점."""
    rclpy.init(args=args)
    node = TumRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
