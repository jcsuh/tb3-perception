#!/usr/bin/env python3
"""
LiDAR 점을 카메라 영상 위에 그려 LiDAR-카메라 캘리브레이션을 눈으로 확인한다 (세션 14).

두 가지 방식
  offline: lidar_cam_collect.py가 저장한 JSON과 자세별 영상(<JSON 이름>_img/pXX.png)으로
           자세마다 겹친 그림을 만들고, 한 장의 모음 그림으로 저장한다 (Gazebo 필요 없음)
  live   : 시뮬레이션(또는 실제 로봇)을 돌리면서 /scan과 카메라 영상을 받아
           겹친 영상을 토픽으로 내보낸다 (rqt_image_view로 보기)

그리는 것 (영상 속 글자는 한글 글꼴 문제로 영어)
  - 빨강: 공칭 관계(xacro CAD 값)로 옮긴 LiDAR 점
  - 초록: 보정된 관계(--calib, lidar_cam_calib.py 결과 YAML)로 옮긴 LiDAR 점
  - 노랑 테두리: 정답 관계 (시뮬레이션 전용, offline은 데이터에 있으면, live는 --truth)
  - 하늘색: 카메라가 PnP로 찾은 보드 외곽선 (offline)
  캘리브레이션이 맞으면 초록 점이 노랑 테두리 안에 들어가고, 보드 위에 반듯하게 놓인다.

LiDAR 점의 3D 위치에는 빔 높이각(원뿔)을 반영한다: --beam-elev-deg, 없으면 --calib YAML의 값.

사용 예
  python3 lidar_cam_overlay.py offline ~/lidar_cam_data/session13.json
      --calib ~/lidar_cam_data/session13_result.yaml               (실제로는 위와 한 줄로)
  python3 lidar_cam_overlay.py live --calib ~/lidar_cam_data/session13_result.yaml --truth
      (다른 터미널에서) ros2 run rqt_image_view rqt_image_view /lidar_cam_overlay/image_raw
"""
import argparse
import json
import os

import cv2
import lidar_cam_calib as lcal
import lidar_cam_check as lcc
import numpy as np
import yaml

RED, GREEN, YELLOW, CYAN = (0, 0, 255), (0, 200, 0), (0, 230, 255), (255, 200, 0)


def load_calib(path):
    """lidar_cam_calib.py(또는 lidar_cam_calib_ceres) 결과 YAML -> (4x4 변환, 빔 높이각 °)."""
    with open(path) as f:
        res = yaml.safe_load(f)
    return np.array(res['matrix'], dtype=np.float64), float(res.get('beam_elev_deg', 0.0))


def default_intrinsics(width, height, hfov_deg):
    """
    내부 파라미터가 없는 옛 데이터용: Gazebo 이상적 카메라의 K (왜곡 없음).

    fx = fy = (가로 / 2) / tan(hfov / 2), 주점 = 영상 가운데.
    """
    f = width / 2.0 / np.tan(np.radians(hfov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def project_points(points_lidar, t_cam_lidar, k, dist):
    """
    점들(LiDAR 좌표계)을 영상 픽셀로 옮긴다.

    돌려주는 값: (픽셀 N x 2, 카메라 앞에 있는지 N). 뒤에 있는 점의 픽셀은 nan.
    """
    q = np.asarray(points_lidar, dtype=np.float64) @ t_cam_lidar[:3, :3].T + t_cam_lidar[:3, 3]
    front = q[:, 2] > 0.05
    uv = np.full((len(q), 2), np.nan)
    if front.any():
        px, _ = cv2.projectPoints(q[front], np.zeros(3), np.zeros(3), k, dist)
        uv[front] = px.reshape(-1, 2)
    return uv, front


def draw_points(img, uv, color, radius=3, ring=False):
    """픽셀 좌표들을 점(또는 테두리 원)으로 그린다. 영상 밖의 점은 건너뛴다."""
    h, w = img.shape[:2]
    for u, v in uv:
        if not (np.isfinite(u) and np.isfinite(v)) or not (0 <= u < w and 0 <= v < h):
            continue
        center = (int(round(u)), int(round(v)))
        if ring:
            cv2.circle(img, center, radius + 2, color, 1, cv2.LINE_AA)
        else:
            cv2.circle(img, center, radius, color, -1, cv2.LINE_AA)


def board_outline(target):
    """보드 판의 네 모서리 (보드 좌표계, m): 태그 격자 + 바깥 여백(태그 사이 간격)."""
    size, gap = target['tagSize'], target['tagSize'] * target['tagSpacing']
    w = target['tagCols'] * size + (target['tagCols'] - 1) * gap
    h = target['tagRows'] * size + (target['tagRows'] - 1) * gap
    return np.array([[-gap, -gap, 0], [w + gap, -gap, 0], [w + gap, h + gap, 0],
                     [-gap, h + gap, 0]], dtype=np.float64)


def draw_legend(img, items):
    """왼쪽 위에 범례를 그린다. items: [(글자, 색), ...]."""
    y = 22
    for text, color in items:
        cv2.rectangle(img, (8, y - 14), (8 + 9 * len(text) + 26, y + 6), (0, 0, 0), -1)
        cv2.circle(img, (18, y - 4), 5, color, -1, cv2.LINE_AA)
        cv2.putText(img, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        y += 22


def pixel_gap(uv_a, uv_b):
    """두 투영 결과의 점별 픽셀 거리 평균 (둘 다 카메라 앞에 있는 점만)."""
    ok = np.isfinite(uv_a).all(axis=1) & np.isfinite(uv_b).all(axis=1)
    return float(np.mean(np.linalg.norm(uv_a[ok] - uv_b[ok], axis=1))) if ok.any() else np.nan


def run_offline(args):
    """저장된 데이터로 자세마다 겹친 그림 + 모음 그림을 만든다."""
    with open(args.data) as f:
        data = json.load(f)
    with open(args.target) as f:
        target = yaml.safe_load(f)
    base = os.path.splitext(args.data)[0]
    img_dir, out_dir = base + '_img', base + '_overlay'
    os.makedirs(out_dir, exist_ok=True)

    t_nom = np.array(data['T_nominal'])
    t_true = np.array(data['T_truth']) if 'T_truth' in data else None
    t_cal, elev = (load_calib(args.calib) if args.calib else (None, 0.0))
    if args.beam_elev_deg is not None:
        elev = args.beam_elev_deg
    dist = np.array(data.get('D', []), dtype=np.float64)
    corners = board_outline(target)

    rels = [('nominal', t_nom, RED)] + ([('calibrated', t_cal, GREEN)] if args.calib else [])
    print(f'빔 높이각 {elev:+.3f}°. 픽셀 차이 = 보드 위 LiDAR 점을 정답 관계로 그린 자리와의 거리')
    print('    자세   점 | 잔차 RMS mm: 공칭  보정 | 정답과의 픽셀 차이: 공칭  보정')
    tiles = []
    for p in data['poses']:
        name = p['name']
        path = os.path.join(img_dir, name + '.png')
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f'    {name}  영상 없음: {path}')
            continue
        h, w = gray.shape
        k = (np.array(data['K']) if 'K' in data
             else default_intrinsics(w, h, args.hfov_deg))
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        pts = lcal.apply_beam_elevation(p['points'], elev)

        t_cb = np.array(p['t_cam_board'])
        outline = (corners @ t_cb[:3, :3].T + t_cb[:3, 3]).reshape(-1, 1, 3)
        px, _ = cv2.projectPoints(outline, np.zeros(3), np.zeros(3), k, dist)
        cv2.polylines(img, [np.round(px).astype(np.int32)], True, CYAN, 1, cv2.LINE_AA)

        uv_true = None
        if t_true is not None:
            uv_true, _ = project_points(pts, t_true, k, dist)
            draw_points(img, uv_true, YELLOW, ring=True)
        rms, gap = {}, {}
        for rel, t, color in rels:
            uv, _ = project_points(pts, t, k, dist)
            draw_points(img, uv, color)
            res = lcc.point_plane_residuals(pts, t, np.array(p['n']), p['d'])
            rms[rel] = float(np.sqrt(np.mean(res ** 2)))
            gap[rel] = pixel_gap(uv, uv_true) if uv_true is not None else np.nan

        legend = [('nominal', RED)] + ([('calibrated', GREEN)] if args.calib else [])
        if t_true is not None:
            legend.append(('truth (ring)', YELLOW))
        draw_legend(img, legend + [('board outline (PnP)', CYAN)])
        dist_m, yaw, tilt = p['dist'], p['yaw_deg'], p['tilt_deg']
        label = f'{name}  d={dist_m:.2f} m  yaw {yaw:+d}  tilt {tilt:+d}'
        cv2.putText(img, label, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255),
                    1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, name + '.png'), img)
        tiles.append(cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA))

        rms_nom, gap_nom = rms['nominal'], gap['nominal']
        cal_rms, cal_gap = '    -', '    -'
        if args.calib:
            r_cal, g_cal = rms['calibrated'], gap['calibrated']
            cal_rms, cal_gap = f'{r_cal * 1000:5.1f}', f'{g_cal:5.1f}'
        print(f'    {name}  {len(pts):3d} | {rms_nom * 1000:15.1f} {cal_rms}'
              f' | {gap_nom:22.1f} {cal_gap}')

    if not tiles:
        raise SystemExit(f'그릴 영상이 없음: {img_dir}')
    cols = 3
    blank = np.zeros_like(tiles[0])
    tiles += [blank] * (-len(tiles) % cols)
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(base + '_overlay.png', np.vstack(rows))
    print(f'saved {base}_overlay.png  (자세별 원본 크기: {out_dir}/)')


def image_to_bgr(msg):
    """sensor_msgs/Image(mono8, rgb8, bgr8)를 BGR numpy 배열로."""
    channels = {'mono8': 1, 'rgb8': 3, 'bgr8': 3}[msg.encoding]
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    img = raw[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
    if channels == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    if msg.encoding == 'rgb8':
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img.copy()


def run_live(args):
    """영상과 스캔을 받을 때마다 겹친 영상을 토픽으로 낸다."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (DurabilityPolicy, qos_profile_sensor_data, QoSProfile,
                           ReliabilityPolicy)
    from sensor_msgs.msg import CameraInfo, Image, LaserScan
    from tf2_msgs.msg import TFMessage

    t_cal, elev = (load_calib(args.calib) if args.calib else (None, 0.0))
    if args.beam_elev_deg is not None:
        elev = args.beam_elev_deg

    class Overlay(Node):
        def __init__(self):
            super().__init__('lidar_cam_overlay')
            latched = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=ReliabilityPolicy.RELIABLE)
            sensor = qos_profile_sensor_data
            self.info = self.scan = self.last = None
            self.nominal, self.truth = {}, {}
            self.count = 0
            info_topic = args.image.rsplit('/', 1)[0] + '/camera_info'
            self.create_subscription(Image, args.image, self.on_image, sensor)
            self.create_subscription(CameraInfo, info_topic,
                                     lambda m: setattr(self, 'info', m), sensor)
            self.create_subscription(LaserScan, '/scan', lambda m: setattr(self, 'scan', m),
                                     sensor)
            self.create_subscription(TFMessage, '/tf_static',
                                     lambda m: lcc.add_edges(self.nominal, m), latched)
            if args.truth:
                self.create_subscription(TFMessage, '/truth/tf_static',
                                         lambda m: lcc.add_edges(self.truth, m), latched)
                self.create_subscription(TFMessage, '/truth/tf',
                                         lambda m: lcc.add_edges(self.truth, m), 10)
            self.pub = self.create_publisher(Image, args.topic, 10)

        def relation(self, edges, need=()):
            # 필요한 좌표계를 다 받기 전에는 None (빠진 부모를 뿌리로 착각하지 않게)
            if any(f not in edges for f in (args.cam_frame, args.lidar_frame, *need)):
                return None
            poses = lcc.frame_poses(edges)
            try:
                return lcc.relative(poses, args.cam_frame, args.lidar_frame)
            except KeyError:
                return None

        def on_image(self, msg):
            if self.info is None or self.scan is None:
                return
            img = image_to_bgr(msg)
            k = np.array(self.info.k, dtype=np.float64).reshape(3, 3)
            dist = np.array(self.info.d, dtype=np.float64)
            s = self.scan
            pts = lcc.scan_points(s.ranges, s.angle_min, s.angle_increment,
                                  max(s.range_min, 0.12), min(s.range_max, args.max_range))
            pts = lcal.apply_beam_elevation(pts, elev) if len(pts) else pts
            legend = []
            layers = [('nominal', self.relation(self.nominal), RED, False)]
            if args.truth:
                t_true = self.relation(self.truth, ('camera_mount_link',))
                layers.append(('truth (ring)', t_true, YELLOW, True))
            if t_cal is not None:
                layers.append(('calibrated', t_cal, GREEN, False))
            for name, t, color, ring in layers:
                if t is None or (name == 'nominal' and args.no_nominal):
                    continue
                if len(pts):
                    uv, _ = project_points(pts, t, k, dist)
                    draw_points(img, uv, color, radius=2, ring=ring)
                legend.append((name, color))
            draw_legend(img, legend)
            out = Image()
            out.header = msg.header
            out.height, out.width = img.shape[:2]
            out.encoding, out.step = 'bgr8', img.shape[1] * 3
            out.data = img.tobytes()
            self.pub.publish(out)
            self.last = img
            self.count += 1
            if self.count == 1:
                self.get_logger().info(f'겹친 영상을 {args.topic}에 내보내는 중 '
                                       f'(LiDAR 점 {len(pts)}개, 빔 높이각 {elev:+.3f}°)')

    rclpy.init()
    node = Overlay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if args.snapshot and node.last is not None:
        cv2.imwrite(os.path.expanduser(args.snapshot), node.last)
        print(f'saved {args.snapshot}')
    node.destroy_node()
    rclpy.try_shutdown()


def main():
    """명령줄에서 실행할 때의 시작점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)
    for p in (sub.add_parser('offline', help='저장된 데이터로 그림 만들기'),
              sub.add_parser('live', help='실행 중인 로봇/시뮬레이션에서 겹친 영상 내보내기')):
        p.add_argument('--calib', default='', help='lidar_cam_calib.py 결과 YAML (초록 점)')
        p.add_argument('--beam-elev-deg', type=float, default=None,
                       help='LiDAR 빔 높이각 (°, 아래가 -). 없으면 --calib YAML의 값')
    off = sub.choices['offline']
    off.add_argument('data', help='lidar_cam_collect.py가 만든 JSON (옆에 *_img/ 영상 폴더)')
    off.add_argument('--target', default=lcc.DEFAULT_TARGET, help='AprilGrid YAML')
    off.add_argument('--hfov-deg', type=float, default=69.0,
                     help='JSON에 K가 없을 때(세션 13 이전 데이터) 쓸 카메라 수평 화각')
    live = sub.choices['live']
    live.add_argument('--image', default='/camera/camera/color/image_raw')
    live.add_argument('--cam-frame', default='camera_color_optical_frame')
    live.add_argument('--lidar-frame', default='base_scan')
    live.add_argument('--topic', default='/lidar_cam_overlay/image_raw', help='내보낼 토픽')
    live.add_argument('--truth', action='store_true', help='(시뮬레이션) 정답 관계도 그리기')
    live.add_argument('--no-nominal', action='store_true', help='공칭 관계(빨강)는 안 그리기')
    live.add_argument('--max-range', type=float, default=8.0, help='그릴 LiDAR 점의 최대 거리 m')
    live.add_argument('--snapshot', default='', help='Ctrl+C로 끝낼 때 마지막 장면을 저장할 파일')
    args = ap.parse_args()
    if args.mode == 'offline':
        run_offline(args)
    else:
        run_live(args)


if __name__ == '__main__':
    main()
