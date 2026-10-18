#!/usr/bin/env python3
"""
보드 자세 하나에서 LiDAR-카메라 캘리브레이션 재료를 뽑고, 원리를 정답으로 검증한다 (세션 11).

1) 카메라 -> 보드 평면: AprilGrid 태그 모서리 + solvePnP
   -> 카메라 좌표계에서 보드 평면 (법선 n, 원점에서의 거리 d)
2) LiDAR -> 보드 위 점: 스캔에서 보드에 떨어진 점 한 줄만 골라낸다
   장면이 멈춰 있으므로 스캔 여러 장(--scans)을 빔별로 평균해 노이즈를 √N배 줄인다
3) 원리 검증: LiDAR 점을 카메라 좌표계로 옮겨 점-평면 거리(잔차)를 잰다
   - 정답 관계(/truth/tf_static + /truth/tf): 잔차가 LiDAR 노이즈 수준이어야 한다
     (스캔 1장 σ 10 mm, N장 평균 10/√N mm)
   - 공칭 관계(/tf_static): 잔차가 눈에 띄게 커야 한다
   - 주의: 보드 자세 하나로는 관계 오차의 일부만 잔차에 드러난다 (그래서 여러 자세가 필요)
+ (시뮬레이션 전용) 카메라가 구한 보드 자세를 Gazebo 정답 보드 자세와 비교한다

결과 그림: /tmp/lidar_cam_check.png (LiDAR 점을 영상에 투영, 초록 = 정답 관계, 빨강 = 공칭 관계)

사용 예 (sim.launch.py world:=lidar_cam 실행 중):
  python3 lidar_cam_check.py
"""
import argparse
import os
import time
import xml.etree.ElementTree as ElementTree

from ament_index_python.packages import get_package_share_directory
import cv2
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from tb3p_eval.transforms import rotation_angle, rpy_to_matrix
from tf2_msgs.msg import TFMessage
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TARGET = os.path.join(HERE, '..', 'config', 'aprilgrid_a3.yaml')
LIDAR_SIGMA_MM = 10.0


# ---------- 순수 계산 (ROS 없이 시험 가능) ----------

def tag_corners_board(tag_id, cols, size, spacing):
    """
    태그 하나의 네 모서리를 보드 좌표(x 오른쪽, y 위, z 앞, m)로 돌려준다.

    보드 원점 = 0번 태그 왼쪽 아래, 번호는 오른쪽으로 늘고 다음 줄은 위로 (Kalibr AprilGrid).
    순서는 OpenCV aruco가 돌려주는 순서에 맞춘다. OpenCV는 태그 무늬 기준으로
    "왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래"를 주는데, Kalibr AprilGrid의 태그 무늬는
    OpenCV 기준과 180° 돌아가 있어서 보드에서 보면 오른쪽 아래, 왼쪽 아래, 왼쪽 위, 오른쪽 위가 된다
    (세션 11에서 확인: 이 순서가 틀리면 PnP 재투영 오차가 60 px 이상).
    """
    pitch = size * (1.0 + spacing)
    x0 = (tag_id % cols) * pitch
    y0 = (tag_id // cols) * pitch
    return np.array([[x0 + size, y0, 0.0], [x0, y0, 0.0],
                     [x0, y0 + size, 0.0], [x0 + size, y0 + size, 0.0]])


def board_pose_from_tags(corners, ids, target, k):
    """
    검출된 태그 모서리로 카메라 기준 보드 자세를 구한다.

    돌려주는 값: (T_cam_board 4x4, 재투영 오차 RMS px, 사용한 모서리 수)
    """
    obj, img = [], []
    for c, i in zip(corners, ids.flatten()):
        if 0 <= i < target['tagCols'] * target['tagRows']:
            obj.append(tag_corners_board(int(i), target['tagCols'],
                                         target['tagSize'], target['tagSpacing']))
            img.append(c.reshape(4, 2))
    obj = np.concatenate(obj).astype(np.float64)
    img = np.concatenate(img).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, k, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError('solvePnP 실패')
    proj, _ = cv2.projectPoints(obj, rvec, tvec, k, np.zeros(5))
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    t = np.eye(4)
    t[:3, :3] = cv2.Rodrigues(rvec)[0]
    t[:3, 3] = tvec.ravel()
    return t, rms, len(obj)


def plane_from_pose(t_cam_board):
    """보드 자세에서 평면을 구한다: 법선 n(보드 z축)과 거리 d (n·x = d), d > 0이 되게."""
    n = t_cam_board[:3, 2].copy()
    d = float(n @ t_cam_board[:3, 3])
    if d < 0:
        n, d = -n, -d
    return n, d


def scan_points(ranges, angle_min, angle_inc, rmin, rmax):
    """스캔 거리값을 LiDAR 좌표계의 점(N x 3, z = 0)으로 바꾼다. 각도는 -π~π로 정렬."""
    r = np.asarray(ranges, dtype=np.float64)
    a = angle_min + angle_inc * np.arange(len(r))
    ok = np.isfinite(r) & (r > rmin) & (r < rmax)
    a = np.arctan2(np.sin(a[ok]), np.cos(a[ok]))
    r = r[ok]
    order = np.argsort(a)
    a, r = a[order], r[order]
    return np.column_stack([r * np.cos(a), r * np.sin(a), np.zeros_like(r)])


def board_segment(points, gap=0.05, trim=1):
    """
    점들을 이웃 간격(gap, m)으로 덩어리로 나누고 가장 큰 덩어리를 보드로 고른다.

    양 끝 trim개는 보드 모서리에 걸친 점일 수 있어 버린다.
    빈 월드에서는 보드가 유일한 물체라 이 방법으로 충분하다
    (실제 환경에서는 공칭 관계로 예측한 보드 위치 주변만 보도록 바꿀 것).
    """
    if len(points) == 0:
        return points
    jumps = np.where(np.linalg.norm(np.diff(points, axis=0), axis=1) > gap)[0]
    starts = np.r_[0, jumps + 1]
    ends = np.r_[jumps + 1, len(points)]
    # 각도가 -π/π에서 끊긴 덩어리는 이어 붙인다 (보드가 정확히 뒤쪽에 있을 때)
    groups = [list(range(s, e)) for s, e in zip(starts, ends)]
    if len(groups) > 1 and np.linalg.norm(points[0] - points[-1]) <= gap:
        groups[0] = groups[-1] + groups[0]
        groups.pop()
    best = max(groups, key=len)
    if len(best) > 2 * trim:
        best = best[trim:len(best) - trim]
    return points[best]


def average_ranges(scans):
    """같은 빔끼리 거리값을 평균한다 (유효한 값만). 한 번도 유효하지 않은 빔은 inf."""
    arr = np.array([sc.ranges for sc in scans], dtype=np.float64)
    ok = np.isfinite(arr)
    count = ok.sum(axis=0)
    total = np.where(ok, arr, 0.0).sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), np.inf)


def point_plane_residuals(points_lidar, t_cam_lidar, n, d):
    """점들을 LiDAR 좌표계에서 카메라 좌표계로 옮긴 뒤 평면까지의 부호 있는 거리(m)."""
    q = points_lidar @ t_cam_lidar[:3, :3].T + t_cam_lidar[:3, 3]
    return q @ n - d


def stats_mm(res):
    """잔차 요약 (mm): 평균, 표준편차, RMS, 최대 절댓값."""
    r = np.asarray(res) * 1000.0
    return {'mean': float(r.mean()), 'std': float(r.std()),
            'rms': float(np.sqrt(np.mean(r ** 2))), 'max': float(np.abs(r).max())}


def frame_poses(edges):
    """{child: (parent, T_parent_child)}로부터 각 좌표계의 뿌리 기준 자세 T_root_frame을 구한다."""
    poses = {}

    def pose(frame):
        if frame not in poses:
            if frame not in edges:
                poses[frame] = np.eye(4)
            else:
                parent, t_pc = edges[frame]
                poses[frame] = pose(parent) @ t_pc
        return poses[frame]

    for frame in list(edges):
        pose(frame)
    return poses


def relative(poses, a, b):
    """T_a_b: 좌표계 b를 a 기준으로 (두 좌표계가 같은 나무에 있어야 함)."""
    return np.linalg.inv(poses[a]) @ poses[b]


def pose_to_matrix(x, y, z, roll, pitch, yaw):
    """위치 + rpy(URDF·SDF 약속)를 4x4 변환 행렬로."""
    t = np.eye(4)
    t[:3, :3] = rpy_to_matrix(roll, pitch, yaw)
    t[:3, 3] = (x, y, z)
    return t


def quat_to_matrix(x, y, z, w):
    """단위 쿼터니언을 3x3 회전 행렬로 바꾼다."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def world_board_pose(world_file, model_name='aprilgrid'):
    """월드 SDF에서 보드 모델(include)의 pose를 읽어 T_world_board로."""
    root = ElementTree.parse(world_file).getroot()
    for inc in root.iter('include'):
        if inc.findtext('name') == model_name:
            return pose_to_matrix(*[float(v) for v in inc.findtext('pose').split()])
    raise RuntimeError(f'{world_file}에 {model_name} 모델이 없음')


# ---------- ROS: 필요한 메시지를 한 번씩 모은다 ----------

def image_to_gray(msg):
    """sensor_msgs/Image(mono8, rgb8, bgr8)를 흑백 numpy 배열로 바꾼다."""
    channels = {'mono8': 1, 'rgb8': 3, 'bgr8': 3}[msg.encoding]
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    img = raw[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
    if channels == 1:
        return img[:, :, 0].copy()
    code = cv2.COLOR_RGB2GRAY if msg.encoding == 'rgb8' else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(img, code)


def detect_tags(gray):
    """보드의 태그(AprilTag 36h11, 검은 테두리 2칸)를 검출한다. OpenCV 4.6과 4.7+ 모두 지원."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        params = cv2.aruco.DetectorParameters()
        params.markerBorderBits = 2
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        params.markerBorderBits = 2
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    return corners, ids


def add_edges(edges, msg):
    """받은 TF 메시지의 변환들을 {child: (parent, T)}에 더한다."""
    for ts in msg.transforms:
        t, q = ts.transform.translation, ts.transform.rotation
        m = np.eye(4)
        m[:3, :3] = quat_to_matrix(q.x, q.y, q.z, q.w)
        m[:3, 3] = (t.x, t.y, t.z)
        edges[ts.child_frame_id] = (ts.header.frame_id, m)


class Collector(Node):
    """영상, camera_info, 스캔, 공칭·정답 TF, 정답 로봇 위치를 모은다."""

    def __init__(self, image_topic, n_scans):
        super().__init__('lidar_cam_check')
        self.n_scans = n_scans
        self.scans = []
        latched = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.image = self.info = self.odom = None
        self.nominal, self.truth = {}, {}
        info_topic = image_topic.rsplit('/', 1)[0] + '/camera_info'
        sensor = qos_profile_sensor_data
        self.create_subscription(Image, image_topic, lambda m: setattr(self, 'image', m), sensor)
        self.create_subscription(CameraInfo, info_topic,
                                 lambda m: setattr(self, 'info', m), sensor)
        self.create_subscription(LaserScan, '/scan', self.scans.append, sensor)
        self.create_subscription(Odometry, '/truth/odom',
                                 lambda m: setattr(self, 'odom', m), sensor)
        self.create_subscription(TFMessage, '/tf_static',
                                 lambda m: add_edges(self.nominal, m), latched)
        self.create_subscription(TFMessage, '/truth/tf_static',
                                 lambda m: add_edges(self.truth, m), latched)
        self.create_subscription(TFMessage, '/truth/tf', lambda m: add_edges(self.truth, m), 10)

    def missing(self, cam_frame):
        """아직 못 받은 것들의 이름."""
        need = {'image': self.image, 'camera_info': self.info, '/truth/odom': self.odom}
        out = [k for k, v in need.items() if v is None]
        if len(self.scans) < self.n_scans:
            out.append(f'scan {len(self.scans)}/{self.n_scans}')
        for name, edges in (('/tf_static', self.nominal), ('/truth/tf*', self.truth)):
            if cam_frame not in edges or 'base_scan' not in edges:
                out.append(name)
        if 'camera_mount_link' not in self.truth:
            out.append('/truth/tf (camera_mount_link)')
        return out


def collect(image_topic, cam_frame, n_scans, timeout):
    """필요한 메시지를 모두 받을 때까지 기다린다."""
    rclpy.init()
    node = Collector(image_topic, n_scans)
    # 실제 경과 시간으로 잰다 (spin_once는 메시지가 오면 바로 돌아오므로 횟수로 세면 안 됨)
    start = time.monotonic()
    while rclpy.ok() and node.missing(cam_frame) and time.monotonic() - start < timeout:
        rclpy.spin_once(node, timeout_sec=0.2)
    miss = node.missing(cam_frame)
    node.destroy_node()
    rclpy.try_shutdown()
    if miss:
        raise SystemExit(f'{timeout:.0f}초 안에 못 받음: {miss} (sim.launch.py가 실행 중인지 확인)')
    return node


# ---------- 보고 ----------

def project(points_cam, k):
    """카메라 좌표계 점을 영상 픽셀로 (카메라 앞에 있는 점만)."""
    front = points_cam[:, 2] > 0.05
    p = points_cam[front] @ k.T
    return p[:, :2] / p[:, 2:3]


def main():
    """명령줄에서 실행할 때의 시작점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', default='/camera/camera/color/image_raw')
    ap.add_argument('--cam-frame', default='camera_color_optical_frame')
    ap.add_argument('--target', default=DEFAULT_TARGET, help='AprilGrid YAML (Kalibr 형식)')
    ap.add_argument('--world', default='lidar_cam.sdf', help='정답 보드 자세를 읽을 월드 파일')
    ap.add_argument('--scans', type=int, default=20, help='평균할 스캔 수 (10 Hz)')
    ap.add_argument('--timeout', type=float, default=20.0)
    args = ap.parse_args()

    with open(args.target) as f:
        target = yaml.safe_load(f)
    node = collect(args.image, args.cam_frame, args.scans, args.timeout)
    k = np.array(node.info.k, dtype=np.float64).reshape(3, 3)

    # 1) 카메라 -> 보드 평면
    gray = image_to_gray(node.image)
    corners, ids = detect_tags(gray)
    n_tags = 0 if ids is None else len(ids)
    n_all = target['tagCols'] * target['tagRows']
    print(f'[1] 카메라 → 보드 평면  ({args.image})')
    if n_tags < 4:
        raise SystemExit(f'    태그 {n_tags}/{n_all}개: 너무 적음 (보드가 카메라에 보이는지 확인)')
    t_cam_board, reproj, n_pts = board_pose_from_tags(corners, ids, target, k)
    n, d = plane_from_pose(t_cam_board)
    print(f'    태그 {n_tags}/{n_all}, 모서리 {n_pts}개, PnP 재투영 오차 {reproj:.2f} px')
    print(f'    평면 법선 ({n[0]:+.3f}, {n[1]:+.3f}, {n[2]:+.3f}), 거리 {d * 1000:.1f} mm')

    # (시뮬레이션 전용) 정답 보드 자세와 비교: T_cam_board_truth
    world_file = os.path.join(get_package_share_directory('tb3p_gazebo'), 'worlds', args.world)
    p = node.odom.pose.pose
    t_world_base = np.eye(4)
    t_world_base[:3, :3] = quat_to_matrix(p.orientation.x, p.orientation.y,
                                          p.orientation.z, p.orientation.w)
    t_world_base[:3, 3] = (p.position.x, p.position.y, p.position.z)
    truth_poses = frame_poses(node.truth)
    t_world_cam = t_world_base @ relative(truth_poses, 'base_footprint', args.cam_frame)
    t_cam_board_true = np.linalg.inv(t_world_cam) @ world_board_pose(world_file)
    n_true, d_true = plane_from_pose(t_cam_board_true)
    ang = np.degrees(np.arccos(np.clip(n @ n_true, -1.0, 1.0)))
    pos_err = np.linalg.norm(t_cam_board[:3, 3] - t_cam_board_true[:3, 3]) * 1000
    rot_err = np.degrees(rotation_angle(t_cam_board[:3, :3].T @ t_cam_board_true[:3, :3]))
    print(f'    정답 대비: 법선 {ang:.3f}°, 거리 {(d - d_true) * 1000:+.2f} mm'
          f'  (보드 자세 위치 {pos_err:.1f} mm, 회전 {rot_err:.3f}°)')

    # 2) LiDAR -> 보드 위 점
    s = node.scans[0]
    n_scans = len(node.scans)
    sigma_avg = LIDAR_SIGMA_MM / np.sqrt(n_scans)
    pts = scan_points(average_ranges(node.scans), s.angle_min, s.angle_increment, 0.15, 3.0)
    board = board_segment(pts)
    board1 = board_segment(scan_points(s.ranges, s.angle_min, s.angle_increment, 0.15, 3.0))
    width = np.linalg.norm(board[-1] - board[0]) * 1000 if len(board) > 1 else 0.0
    dist = np.linalg.norm(board.mean(axis=0)) * 1000 if len(board) else 0.0
    print('[2] LiDAR → 보드 위 점')
    print(f'    스캔 {n_scans}장 평균. 0.15–3 m 안의 점 {len(pts)}개 중 보드 {len(board)}개, '
          f'양 끝 간격 {width:.0f} mm (보드 폭 405 mm), 평균 거리 {dist:.0f} mm')
    if len(board) < 5:
        raise SystemExit('    보드 위 점이 너무 적음 (스캔 높이가 보드를 지나는지 확인)')

    # 3) 원리 검증: 정답 관계 vs 공칭 관계
    t_true = relative(truth_poses, args.cam_frame, 'base_scan')
    t_nom = relative(frame_poses(node.nominal), args.cam_frame, 'base_scan')
    s1 = stats_mm(point_plane_residuals(board1, t_true, n, d))
    st = stats_mm(point_plane_residuals(board, t_true, n, d))
    sn = stats_mm(point_plane_residuals(board, t_nom, n, d))
    rel_rot = np.degrees(rotation_angle(t_nom[:3, :3].T @ t_true[:3, :3]))
    rel_pos = np.linalg.norm(t_nom[:3, 3] - t_true[:3, 3]) * 1000
    print(f'[3] 점-평면 잔차 (mm). LiDAR 노이즈 σ: 1장 {LIDAR_SIGMA_MM:.0f} mm, '
          f'{n_scans}장 평균 {sigma_avg:.1f} mm')
    print('                         평균   표준편차      RMS     최대')
    rows = (('정답 관계, 스캔 1장 ', s1), (f'정답 관계, {n_scans}장 평균', st),
            (f'공칭 관계, {n_scans}장 평균', sn))
    for name, r in rows:
        mean, std, rms, mx = r['mean'], r['std'], r['rms'], r['max']
        print(f'    {name}  {mean:+8.1f} {std:8.1f} {rms:8.1f} {mx:8.1f}')
    print(f'    (공칭 vs 정답 LiDAR-카메라 관계 차이: 회전 {rel_rot:.2f}°, 위치 {rel_pos:.1f} mm)')
    # 정답: 평균 잔차가 노이즈 수준 (+ 카메라 평면 오차 여유 2 mm)
    ok_true = st['rms'] < 2 * sigma_avg + 2.0
    ok_nom = sn['rms'] > 2 * st['rms']
    verdict = {True: 'OK', False: 'NO'}
    print(f'    판정: 정답 관계 잔차가 노이즈 수준 {verdict[ok_true]}, '
          f'공칭 관계가 확실히 더 큼 {verdict[ok_nom]}')

    # 그림: LiDAR 점 투영
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    for t, color in ((t_nom, (0, 0, 255)), (t_true, (0, 200, 0))):
        q = board @ t[:3, :3].T + t[:3, 3]
        for u, v in project(q, k):
            cv2.circle(vis, (int(round(u)), int(round(v))), 3, color, -1)
    cv2.imwrite('/tmp/lidar_cam_check.png', vis)
    print('saved /tmp/lidar_cam_check.png  (초록 = 정답 관계, 빨강 = 공칭 관계)')


if __name__ == '__main__':
    main()
