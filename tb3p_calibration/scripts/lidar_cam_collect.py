#!/usr/bin/env python3
"""
보드를 여러 자세로 옮기며 LiDAR-카메라 캘리브레이션 재료를 모은다 (세션 12).

자세마다
1) gz service set_pose로 보드를 옮긴다 (보드 중심은 늘 LiDAR 스캔 높이)
2) 옮긴 뒤에 찍힌 영상 1장 + 스캔 N장만 쓴다 (타임스탬프로 확인)
3) 카메라 -> 보드 평면 (n, d), LiDAR -> 보드 위 점 (스캔 N장 빔별 평균)
4) (시뮬레이션 전용) 카메라 평면을 명령한 보드 자세(정답)와 비교해 이상한 자세를 걸러낸다

결과: JSON 하나 (기본 ~/lidar_cam_data/session12.json)와 자세별 흑백 영상(<JSON 이름>_img/pXX.png)
  - poses[]: 자세 이름, 명령한 보드 중심/기울기, 평면 n·d, 보드 위 점(LiDAR 좌표계), 검사값
  - T_nominal, T_truth: 카메라 <- LiDAR 관계 (초기값과 채점용 정답)
lidar_cam_calib.py가 이 파일로 최적화와 채점을 한다.

사용 예 (sim.launch.py world:=lidar_cam 실행 중):
  python3 lidar_cam_collect.py
"""
import argparse
import json
import os
import subprocess
import time

import cv2
import lidar_cam_check as lcc
import numpy as np
import rclpy
import yaml

SCAN_HEIGHT = 0.233          # 바닥 위 LiDAR 스캔 높이 (m): 보드 중심을 여기에 둔다
BOARD_CENTER = (0.1875, 0.1225, 0.0)   # 보드 원점(0번 태그 왼쪽 아래) 기준 보드 중심

# (이름, 앞 거리 m, 옆 m(+왼쪽), yaw °(수직축), tilt °(보드 가로축, +면 위가 뒤로 젖혀짐))
# upright(tilt 0)만으로는 위아래 방향 오차가 안 보인다 → 앞뒤로 젖힌 자세를 섞는다
# 거리는 보드 네 모서리가 컬러 영상 안에 여유 있게 들어오도록 정함 (카메라가 보드 중심보다 낮음)
POSES = [
    ('p01', 0.75, 0.00, 0, 0),
    ('p02', 0.85, 0.10, 30, 0),
    ('p03', 0.80, -0.10, -30, 0),
    ('p04', 0.75, 0.00, 0, 25),
    ('p05', 0.70, 0.00, 0, -25),
    ('p06', 0.90, 0.15, 20, 20),
    ('p07', 0.90, -0.15, -20, 20),
    ('p08', 0.90, 0.15, -20, -20),
    ('p09', 0.90, -0.15, 20, -20),
    ('p10', 1.10, 0.00, 35, 15),
    ('p11', 1.10, 0.00, -35, -15),
    ('p12', 1.10, 0.20, 0, 25),
    ('p13', 1.10, -0.20, 0, -25),
    ('p14', 0.80, 0.10, 15, -15),
    ('p15', 0.80, -0.10, -15, 15),
    # 먼 거리에서 젖힌 자세: 끄덕임(카메라 x축 회전)과 위아래 위치를 구분하게 해 준다
    # (거리가 다양해야 "각도 x 거리"와 "위아래 이동"이 서로 다른 잔차 모양을 만든다)
    ('p16', 1.25, 0.00, 0, 25),
    ('p17', 1.25, 0.00, 0, -25),
    ('p18', 1.25, 0.10, 20, -20),
    ('p19', 1.25, -0.10, -20, 20),
]


def board_world_pose(dist, side, yaw_deg, tilt_deg):
    """
    보드 중심 위치와 기울기로 보드 원점의 월드 자세 T_world_board를 만든다.

    기본 방향은 lidar_cam.sdf와 같다: 보드 y(위) → 월드 +z, 앞면이 로봇(-x) 쪽.
    tilt는 보드 가로축(보드 x) 둘레, yaw는 월드 수직축 둘레 회전이다.
    """
    r = (lcc.rpy_to_matrix(0.0, 0.0, np.radians(yaw_deg))
         @ lcc.rpy_to_matrix(np.pi / 2, 0.0, -np.pi / 2)
         @ lcc.rpy_to_matrix(np.radians(tilt_deg), 0.0, 0.0))
    center = np.array([dist, side, SCAN_HEIGHT])
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = center - r @ np.array(BOARD_CENTER)
    return t


def matrix_to_quat(r):
    """3x3 회전 행렬을 쿼터니언 (x, y, z, w)로 바꾼다."""
    w = np.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2.0
    x = np.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])) / 2.0
    x = np.copysign(x, r[2, 1] - r[1, 2])
    y = np.copysign(y, r[0, 2] - r[2, 0])
    z = np.copysign(z, r[1, 0] - r[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def set_board_pose(world, model, t_world_board):
    """Gazebo 서비스로 보드 모델을 옮긴다. 성공하면 True."""
    x, y, z = t_world_board[:3, 3]
    qx, qy, qz, qw = matrix_to_quat(t_world_board[:3, :3])
    req = (f'name: "{model}", position: {{x: {x:.6f}, y: {y:.6f}, z: {z:.6f}}}, '
           f'orientation: {{x: {qx:.8f}, y: {qy:.8f}, z: {qz:.8f}, w: {qw:.8f}}}')
    cmd = ['gz', 'service', '-s', f'/world/{world}/set_pose', '--reqtype', 'gz.msgs.Pose',
           '--reptype', 'gz.msgs.Boolean', '--timeout', '3000', '--req', req]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    return 'true' in out.stdout


def stamp(msg):
    """메시지 헤더 시각(초, 시뮬레이션 시계)."""
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def spin_until(node, done, timeout):
    """done()이 참이 될 때까지 실제 시간으로 최대 timeout초 돌린다."""
    start = time.monotonic()
    while rclpy.ok() and not done() and time.monotonic() - start < timeout:
        rclpy.spin_once(node, timeout_sec=0.2)
    return done()


def capture(node, n_scans, settle, timeout):
    """
    보드를 옮긴 뒤의 영상 1장과 스캔 n장을 모은다.

    set_pose가 끝난 뒤 0.5초 동안 밀려 있던 메시지를 비우고, 그때까지 받은 가장 최근
    스캔 시각(= 지금 시뮬레이션 시각) + settle 이후에 찍힌 것만 쓴다.
    """
    node.scans.clear()
    node.image = None
    spin_until(node, lambda: False, 0.5)
    t_now = max((stamp(s) for s in node.scans), default=0.0)
    t_min = t_now + settle

    def fresh():
        ok_img = node.image is not None and stamp(node.image) > t_min
        return ok_img and sum(stamp(s) > t_min for s in node.scans) >= n_scans

    if not spin_until(node, fresh, timeout):
        return None, None
    scans = [s for s in node.scans if stamp(s) > t_min][:n_scans]
    return node.image, scans


def measure(image, scans, info_k, target):
    """영상과 스캔으로 평면과 보드 위 점을 구한다. 실패하면 (None, 이유)."""
    gray = lcc.image_to_gray(image)
    corners, ids = lcc.detect_tags(gray)
    n_tags = 0 if ids is None else len(ids)
    if n_tags < 4:
        return None, f'태그 {n_tags}개'
    t_cam_board, reproj, _ = lcc.board_pose_from_tags(corners, ids, target, info_k)
    n, d = lcc.plane_from_pose(t_cam_board)
    s = scans[0]
    pts = lcc.scan_points(lcc.average_ranges(scans), s.angle_min, s.angle_increment, 0.15, 3.0)
    board = lcc.board_segment(pts)
    if len(board) < 5:
        return None, f'보드 위 점 {len(board)}개'
    return {'tags': n_tags, 'reproj_px': reproj, 'n': n, 'd': d,
            't_cam_board': t_cam_board, 'points': board}, ''


def truth_plane(node, cam_frame, t_world_board):
    """(시뮬레이션 전용) 명령한 보드 자세로 카메라 기준 정답 평면을 구한다."""
    p = node.odom.pose.pose
    t_world_base = np.eye(4)
    t_world_base[:3, :3] = lcc.quat_to_matrix(p.orientation.x, p.orientation.y,
                                              p.orientation.z, p.orientation.w)
    t_world_base[:3, 3] = (p.position.x, p.position.y, p.position.z)
    poses = lcc.frame_poses(node.truth)
    t_world_cam = t_world_base @ lcc.relative(poses, 'base_footprint', cam_frame)
    return lcc.plane_from_pose(np.linalg.inv(t_world_cam) @ t_world_board)


def main():
    """명령줄에서 실행할 때의 시작점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', default='/camera/camera/color/image_raw')
    ap.add_argument('--cam-frame', default='camera_color_optical_frame')
    ap.add_argument('--target', default=lcc.DEFAULT_TARGET)
    ap.add_argument('--world', default='lidar_cam', help='Gazebo 월드 이름')
    ap.add_argument('--model', default='aprilgrid', help='보드 모델 이름')
    ap.add_argument('--scans', type=int, default=20)
    ap.add_argument('--settle', type=float, default=0.3, help='옮긴 뒤 버릴 시간 (s, sim)')
    ap.add_argument('--out', default=os.path.expanduser('~/lidar_cam_data/session12.json'))
    args = ap.parse_args()

    with open(args.target) as f:
        target = yaml.safe_load(f)
    rclpy.init()
    node = lcc.Collector(args.image, 1)
    if not spin_until(node, lambda: not node.missing(args.cam_frame), 20.0):
        raise SystemExit(f'메시지를 못 받음: {node.missing(args.cam_frame)}')
    k = np.array(node.info.k, dtype=np.float64).reshape(3, 3)
    t_nom = lcc.relative(lcc.frame_poses(node.nominal), args.cam_frame, 'base_scan')
    t_true = lcc.relative(lcc.frame_poses(node.truth), args.cam_frame, 'base_scan')

    print('    자세  거리   옆   yaw tilt | 태그 재투영px 점수 | 평면 오차 °     mm')
    img_dir = os.path.splitext(args.out)[0] + '_img'
    os.makedirs(img_dir, exist_ok=True)
    poses = []
    for name, dist, side, yaw, tilt in POSES:
        t_wb = board_world_pose(dist, side, yaw, tilt)
        head = f'    {name}  {dist:.2f} {side:+.2f} {yaw:+4d} {tilt:+4d} |'
        if not set_board_pose(args.world, args.model, t_wb):
            print(f'{head} set_pose 실패 (gz service, 월드 이름 {args.world} 확인)')
            continue
        image, scans = capture(node, args.scans, args.settle, 20.0)
        if image is None:
            print(f'{head} 새 영상·스캔을 못 받음')
            continue
        cv2.imwrite(os.path.join(img_dir, f'{name}.png'), lcc.image_to_gray(image))
        m, why = measure(image, scans, k, target)
        if m is None:
            print(f'{head} 건너뜀: {why}')
            continue
        n_true, d_true = truth_plane(node, args.cam_frame, t_wb)
        ang = float(np.degrees(np.arccos(np.clip(m['n'] @ n_true, -1.0, 1.0))))
        dd = (m['d'] - d_true) * 1000
        tags, reproj, npts = m['tags'], m['reproj_px'], len(m['points'])
        print(f'{head} {tags:3d} {reproj:7.2f} {npts:4d} | {ang:9.3f} {dd:+6.2f}')
        poses.append({'name': name, 'dist': dist, 'side': side, 'yaw_deg': yaw,
                      'tilt_deg': tilt, 'tags': m['tags'], 'reproj_px': m['reproj_px'],
                      'n': m['n'].tolist(), 'd': m['d'], 'points': m['points'].tolist(),
                      'plane_err_deg': ang, 'plane_err_mm': dd})

    # 보드를 원래 자리로 (lidar_cam.sdf와 같은 자세)
    set_board_pose(args.world, args.model, board_world_pose(0.75, 0.0, 15, 0))
    node.destroy_node()
    rclpy.try_shutdown()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    data = {'cam_frame': args.cam_frame, 'lidar_frame': 'base_scan', 'scans_per_pose': args.scans,
            'T_nominal': t_nom.tolist(), 'T_truth': t_true.tolist(), 'poses': poses}
    with open(args.out, 'w') as f:
        json.dump(data, f, indent=1)
    print(f'{len(poses)}/{len(POSES)} 자세 저장: {args.out}  (영상: {img_dir})')


if __name__ == '__main__':
    main()
