#!/usr/bin/env python3
"""
(시뮬레이션 전용) Gazebo LiDAR가 정답 TF가 말하는 평면을 정말로 재는지 검사한다 (세션 13).

배경: LiDAR-카메라 캘리브레이션 결과가 정답과 약 0.6° 다른데, 잔차를 보면 "정답 쪽"이
Gazebo LiDAR의 실제 스캔과 어긋나 있을 가능성이 컸다. 카메라 없이 확인한다.

방법
  보드를 로봇 둘레 여러 방향(앞, 옆, 뒤)에 세우고(똑바로, 뒤로 젖힘, 앞으로 숙임),
  LiDAR 점을 Gazebo 정답 보드 평면(명령한 보드 자세 + /truth/odom + 정답 TF)과 비교한다.
  어긋남의 모양으로 원인을 가린다.
    - 스캔 평면이 기울어짐: 앞쪽 보드는 위로, 뒤쪽 보드는 아래로 (방향에 따라 부호가 바뀜)
    - 모든 빔이 조금 위(또는 아래)를 향함(원뿔): 방향과 상관없이 같은 부호, 거리에 비례
      (거리 하나만 쓰면 원뿔과 "높이만 다름"이 구분되지 않아 두 거리에서 잰다)
    - 거리값 자체의 치우침: 똑바로 선 보드에서도 평균 잔차가 0이 아님

결과 (2026-10-22): cone. 모든 방향에서 빔이 약 0.44–0.48° 아래를 향함 (빔 400개, 404개 모두)
  - tilt(스캔 평면 기울기)와 range(거리 치우침)는 아님
  - 면당 픽셀 수 짝수/홀수 가설(400 -> 404)로는 사라지지 않음 -> Gazebo GPU LiDAR의 고정된 특성
  - 캘리브레이션에서는 lidar_cam_calib.py --beam-elev-deg -0.444 로 반영 (정답 대비 0.61° -> 0.21°)

사용 예 (sim.launch.py world:=lidar_cam 실행 중):
  python3 lidar_truth_check.py
"""
import argparse

import lidar_cam_check as lcc
import lidar_cam_collect as col
import numpy as np
import rclpy
from scipy.optimize import least_squares

AZIMUTHS = [0, 90, 180, 270]       # 보드 방향 (°, 로봇 기준 반시계, 0 = 앞)
TILTS = [0, 25, -25]               # 보드 기울기 (°)
DISTS = [0.6, 1.3]                 # 보드 중심까지 거리 (m): 둘 이상이어야 원뿔과 높이 차이를 가린다


def lidar_plane(node, t_world_board):
    """정답 보드 평면을 LiDAR 좌표계(base_scan)로: 법선 n, 거리 d (n·x = d)."""
    p = node.odom.pose.pose
    t_world_base = np.eye(4)
    t_world_base[:3, :3] = lcc.quat_to_matrix(p.orientation.x, p.orientation.y,
                                              p.orientation.z, p.orientation.w)
    t_world_base[:3, 3] = (p.position.x, p.position.y, p.position.z)
    t_world_lidar = t_world_base @ lcc.relative(lcc.frame_poses(node.truth),
                                                'base_footprint', 'base_scan')
    return lcc.plane_from_pose(np.linalg.inv(t_world_lidar) @ t_world_board)


def model_residuals(kind, x, pts, ns, ds):
    """
    어긋남 모형을 적용한 점-평면 거리 (m).

    tilt: 스캔 평면이 roll x[0], pitch x[1]만큼 기울고 높이가 x[2]만큼 다름
    cone: 모든 빔이 위로 x[0] rad 들림 + 높이 x[1]
    range: 거리값에 x[0] m가 더해짐
    """
    r = np.linalg.norm(pts[:, :2], axis=1)
    a = np.arctan2(pts[:, 1], pts[:, 0])
    if kind == 'tilt':
        cr, sr, cp, sp = np.cos(x[0]), np.sin(x[0]), np.cos(x[1]), np.sin(x[1])
        rot = np.array([[cp, sr * sp, cr * sp], [0, cr, -sr], [-sp, sr * cp, cr * cp]])
        q = pts @ rot.T + np.array([0.0, 0.0, x[2]])
    elif kind == 'cone':
        q = np.column_stack([r * np.cos(x[0]) * np.cos(a), r * np.cos(x[0]) * np.sin(a),
                             r * np.sin(x[0]) + x[1]])
    else:
        q = np.column_stack([(r + x[0]) * np.cos(a), (r + x[0]) * np.sin(a), np.zeros_like(r)])
    return np.sum(q * ns, axis=1) - ds


def main():
    """명령줄에서 실행할 때의 시작점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--world', default='lidar_cam')
    ap.add_argument('--model', default='aprilgrid')
    ap.add_argument('--scans', type=int, default=20)
    args = ap.parse_args()

    rclpy.init()
    node = lcc.Collector('/camera/camera/color/image_raw', 1)
    if not col.spin_until(node, lambda: not node.missing('camera_color_optical_frame'), 20.0):
        miss = node.missing('camera_color_optical_frame')
        raise SystemExit(f'메시지를 못 받음: {miss}')

    print('  거리  방향  기울기 |  점  평균잔차mm  RMS mm | 평균잔차/(거리·sin기울기) °')
    all_pts, all_n, all_d = [], [], []
    poses = [(dist, az, tilt) for dist in DISTS for az in AZIMUTHS for tilt in TILTS]
    for dist, az, tilt in poses:
        ca, sa = np.cos(np.radians(az)), np.sin(np.radians(az))
        t_wb = col.board_world_pose(dist * ca, dist * sa, az, tilt)
        head = f'  {dist:.1f}  {az:4d}°  {tilt:+4d}° |'
        if not col.set_board_pose(args.world, args.model, t_wb):
            print(f'{head} set_pose 실패')
            continue
        image, scans = col.capture(node, args.scans, 0.3, 20.0)
        if scans is None:
            print(f'{head} 새 스캔을 못 받음')
            continue
        s = scans[0]
        pts = lcc.scan_points(lcc.average_ranges(scans), s.angle_min, s.angle_increment,
                              0.15, 3.0)
        board = lcc.board_segment(pts)
        n, d = lidar_plane(node, t_wb)
        res = (board @ n - d) * 1000
        rms = np.sqrt(np.mean(res ** 2))
        line = f'{head} {len(board):3d} {res.mean():+10.2f} {rms:7.2f} |'
        if tilt:
            rng = np.linalg.norm(board[:, :2], axis=1).mean()
            elev = np.degrees(res.mean() / 1000 / (rng * np.sin(np.radians(tilt))))
            line += f' {elev:+.3f}'
        print(line)
        all_pts.append(board)
        all_n.append(np.tile(n, (len(board), 1)))
        all_d.append(np.full(len(board), d))

    col.set_board_pose(args.world, args.model, col.board_world_pose(0.75, 0.0, 15, 0))
    node.destroy_node()
    rclpy.try_shutdown()

    pts, ns, ds = np.vstack(all_pts), np.vstack(all_n), np.concatenate(all_d)
    r0 = np.sum(pts * ns, axis=1) - ds
    rms0 = np.sqrt(np.mean(r0 ** 2)) * 1000
    print(f'\n[어긋남 모형 맞추기]  점 {len(pts)}개, 모형 없음 RMS {rms0:.2f} mm')
    for kind, x0, names in (('tilt', [0, 0, 0], ('roll °', 'pitch °', '높이 mm')),
                            ('cone', [0, 0], ('빔 높이각 °', '높이 mm')),
                            ('range', [0], ('거리 치우침 mm',))):
        sol = least_squares(lambda x: model_residuals(kind, x, pts, ns, ds), x0)
        vals = [np.degrees(v) if '°' in nm else v * 1000 for v, nm in zip(sol.x, names)]
        desc = ', '.join(f'{nm} {v:+.3f}' for nm, v in zip(names, vals))
        rms = np.sqrt(np.mean(sol.fun ** 2)) * 1000
        print(f'  {kind:5s}: RMS {rms:.2f} mm  ({desc})')


if __name__ == '__main__':
    main()
