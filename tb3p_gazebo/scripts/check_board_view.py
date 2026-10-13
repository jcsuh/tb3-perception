#!/usr/bin/env python3
"""
카메라 영상 한 장에서 AprilGrid 태그를 검출해 보드가 제대로 보이는지 확인한다.

- 검출된 태그 수와, 0번 태그가 영상에서 왼쪽 아래에 있는지(= 좌우 반전 없음)를 출력한다
- 받은 영상은 /tmp/board_view.png 로 저장한다 (검출된 태그 표시)

사용 예:
  python3 check_board_view.py /camera/camera/infra1/image_rect_raw
"""
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


def image_to_gray(msg):
    """sensor_msgs/Image(mono8, rgb8, bgr8)를 흑백 numpy 배열로 바꾼다."""
    channels = {'mono8': 1, 'rgb8': 3, 'bgr8': 3}[msg.encoding]
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    img = raw[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
    if channels == 1:
        return img[:, :, 0].copy()
    code = cv2.COLOR_RGB2GRAY if msg.encoding == 'rgb8' else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(img, code)


def detector_params():
    """
    AprilGrid용 검출 설정.

    Kalibr AprilGrid의 태그는 검은 테두리가 2칸이다 (OpenCV 기본값은 1칸).
    """
    if hasattr(cv2.aruco, 'ArucoDetector'):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()
    params.markerBorderBits = 2
    return params


def detect_tags(gray):
    """AprilTag 36h11 검출 결과 (corners, ids). OpenCV 4.6과 4.7+ 모두 지원."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params())
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        params = detector_params()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    return corners, ids


class OneShot(Node):
    """영상 토픽에서 메시지 한 개를 받는 노드."""

    def __init__(self, topic):
        super().__init__('check_board_view')
        self.msg = None
        self.create_subscription(Image, topic, self._callback, 1)

    def _callback(self, msg):
        self.msg = msg


def main():
    """명령줄에서 실행할 때의 시작점."""
    topic = sys.argv[1] if len(sys.argv) > 1 else '/camera/camera/infra1/image_rect_raw'
    rclpy.init()
    node = OneShot(topic)
    while rclpy.ok() and node.msg is None:
        rclpy.spin_once(node, timeout_sec=0.5)
    msg = node.msg
    node.destroy_node()
    rclpy.try_shutdown()

    gray = image_to_gray(msg)
    corners, ids = detect_tags(gray)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    n = 0 if ids is None else len(ids)
    print(f'topic {topic}: {msg.width}x{msg.height} {msg.encoding}, tags detected {n} / 24')
    if n:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        centers = {int(i): c[0].mean(axis=0) for c, i in zip(corners, ids.flatten())}
        if 0 in centers and 1 in centers and 6 in centers:
            ok = centers[1][0] > centers[0][0] and centers[6][1] < centers[0][1]
            print('tag 0 bottom-left, 1 to its right, 6 above it:', 'OK' if ok else 'NO (뒤집힘?)')
    cv2.imwrite('/tmp/board_view.png', vis)
    print('saved /tmp/board_view.png')


if __name__ == '__main__':
    main()
