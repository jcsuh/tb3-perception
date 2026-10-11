"""record_tum의 단위 테스트."""
from nav_msgs.msg import Odometry
from tb3p_eval.record_tum import odom_to_tum_line, topic_to_filename


def test_topic_to_filename():
    assert topic_to_filename('/truth/odom') == 'truth_odom.tum'
    assert topic_to_filename('/odom') == 'odom.tum'


def test_odom_to_tum_line():
    msg = Odometry()
    msg.header.stamp.sec = 12
    msg.header.stamp.nanosec = 500000000
    msg.pose.pose.position.x = 1.0
    msg.pose.pose.orientation.w = 1.0
    fields = odom_to_tum_line(msg).split()
    assert len(fields) == 8
    assert float(fields[0]) == 12.5
    assert float(fields[1]) == 1.0
    assert float(fields[7]) == 1.0
