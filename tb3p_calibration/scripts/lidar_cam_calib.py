#!/usr/bin/env python3
"""
점-평면 잔차를 최소화해 LiDAR-카메라 관계(카메라 <- LiDAR)를 추정하고 정답으로 채점한다 (세션 12).

문제
  자세 i의 보드 평면(카메라 좌표계): n_i · x = d_i
  자세 i의 보드 위 LiDAR 점 p_ij (LiDAR 좌표계)
  찾는 것: R, t (6개 숫자) -> 모든 i, j에 대해 n_i · (R p_ij + t) - d_i = 0 에 가장 가깝게
  풀이: 비선형 최소제곱 (scipy.optimize.least_squares), 초기값 = 공칭 관계(xacro CAD 값)

출력
  - 추정 전(공칭)과 추정 후의 정답 대비 오차: 회전 °, 위치 mm (카메라 축별)
  - 추정 불확실성(1σ): 잔차 크기와 야코비안으로 계산 -> 어느 방향이 잘 안 보이는지
  - 자세별 잔차 RMS (추정 전 -> 후)
  - 결과 YAML (기본: 입력 JSON 옆 *_result.yaml)

가중치 (--weighted, 세션 13)
  점 하나의 잔차가 믿을 만한 정도 = LiDAR 거리 노이즈 + 카메라 평면의 불확실성
  w = 1 / sqrt(σ_LiDAR² + σ_평면²): 먼 보드처럼 평면이 덜 정확한 자세의 점은 덜 반영된다
  (σ_평면은 PnP 공분산에서 점마다 계산, 데이터에 pose_cov가 있어야 함)

LiDAR 빔 높이각 (--beam-elev-deg, 세션 13)
  2D LiDAR의 빔이 회전축에 정확히 수직이 아니면 스캔은 평면이 아니라 얕은 원뿔이 된다.
  Gazebo GPU LiDAR는 모든 방향에서 빔이 약 0.44° 아래를 향함을 lidar_truth_check.py로 확인했다.
  카메라가 보는 앞쪽에서는 이 원뿔이 "LiDAR가 숙여 붙음"과 똑같이 보여서, 데이터만으로는
  구분할 수 없다 (잔차가 그대로). 알려진 값을 주면 점을 원뿔 위의 실제 3D 위치로 바꿔 쓴다.

사용 예:
  python3 lidar_cam_calib.py ~/lidar_cam_data/session12.json
  python3 lidar_cam_calib.py ~/lidar_cam_data/session12.json --only-upright   # 실험: 똑바로 선 자세만
  python3 lidar_cam_calib.py ~/lidar_cam_data/session13.json --weighted

C++(Ceres)와 비교 (--export-problem, 세션 14)
  풀기 직전의 문제(점, 평면, 공칭·정답, 가중치)와 이 스크립트의 답을 YAML로 저장한다.
  같은 파일을 C++ 프로그램 lidar_cam_calib_ceres가 읽어 Ceres로 다시 풀고 두 답을 비교한다.
    python3 lidar_cam_calib.py ~/lidar_cam_data/session13.json --beam-elev-deg -0.444
        --export-problem ~/lidar_cam_data/session13_problem.yaml     (실제로는 위와 한 줄로)
    ros2 run tb3p_calibration lidar_cam_calib_ceres ~/lidar_cam_data/session13_problem.yaml
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
import yaml

PASS_ROT_DEG = 0.5
PASS_POS_MM = 10.0


# 파라미터 = 초기값(공칭)에 더하는 작은 보정 6개, 모두 카메라 좌표계 기준
#   x[:3] 회전 보정 (rad, 카메라 x·y·z축 둘레), x[3:] 위치 보정 (m)
#   T = [Rot(x[:3]) R0 | t0 + x[3:]]
# 이렇게 하면 불확실성을 "카메라 x축 둘레 회전 몇 도"처럼 바로 읽을 수 있다
# 약한 방향을 비교할 때의 단위: 합격 기준(0.5°, 10 mm)을 1로 본다
UNIT = np.r_[np.radians(PASS_ROT_DEG) * np.ones(3), PASS_POS_MM / 1000 * np.ones(3)]


def apply(t0, x):
    """초기 변환 t0에 보정 x(6개)를 적용한 4x4 변환."""
    t = np.eye(4)
    t[:3, :3] = cv2.Rodrigues(np.asarray(x[:3], dtype=np.float64))[0] @ t0[:3, :3]
    t[:3, 3] = t0[:3, 3] + x[3:]
    return t


def plane_sigma(t_cam_board, cov, q):
    """
    카메라 평면의 불확실성이 점 q(카메라 좌표계, M x 3)의 점-평면 거리에 주는 1σ (m).

    보드 자세 6개 숫자(회전 벡터, 위치)를 조금씩 바꿨을 때 거리가 변하는 정도(기울기 g)를
    수치 미분으로 구하고, σ² = g Σ gᵀ 로 계산한다.
    """
    base = np.r_[cv2.Rodrigues(np.asarray(t_cam_board)[:3, :3])[0].ravel(),
                 np.asarray(t_cam_board)[:3, 3]]

    def dist(params):
        n = cv2.Rodrigues(params[:3])[0][:, 2]
        return q @ n - n @ params[3:]

    eps = 1e-6
    g = np.stack([(dist(base + eps * e) - dist(base - eps * e)) / (2 * eps)
                  for e in np.eye(6)], axis=1)
    return np.sqrt(np.einsum('ij,jk,ik->i', g, np.asarray(cov), g))


def point_weights(poses, t_init, sigma_lidar):
    """자세별 점 가중치 1/sqrt(σ_LiDAR² + σ_평면²) (1/m)와 자세별 평균 σ_평면 (m)."""
    ws, sig = [], []
    for p in poses:
        q = np.asarray(p['points'], dtype=np.float64)
        qc = q @ t_init[:3, :3].T + t_init[:3, 3]
        sp = plane_sigma(p['t_cam_board'], p['pose_cov'], qc)
        ws.append(1.0 / np.sqrt(sigma_lidar ** 2 + sp ** 2))
        sig.append(float(np.mean(sp)))
    return ws, sig


def apply_beam_elevation(points, elev_deg):
    """
    스캔 점(z = 0)을 빔 높이각 elev_deg(°, 아래가 -)인 원뿔 위의 실제 3D 위치로 바꾼다.

    거리 r과 방위각 a는 그대로 두고 (r cosε cos a, r cosε sin a, r sinε)로 만든다.
    """
    q = np.asarray(points, dtype=np.float64)
    if not elev_deg:
        return q
    r = np.linalg.norm(q[:, :2], axis=1)
    a = np.arctan2(q[:, 1], q[:, 0])
    e = np.radians(elev_deg)
    return np.column_stack([r * np.cos(e) * np.cos(a), r * np.cos(e) * np.sin(a), r * np.sin(e)])


def stack(poses, weights=None):
    """
    자세들을 점 단위 배열로 펼친다.

    돌려주는 값: 점(N x 3), 평면 법선(N x 3), 거리(N), 자세 번호(N), 가중치(N, 없으면 1)
    """
    pts, ns, ds, idx, ws = [], [], [], [], []
    for i, p in enumerate(poses):
        q = np.asarray(p['points'], dtype=np.float64)
        pts.append(q)
        ns.append(np.tile(p['n'], (len(q), 1)))
        ds.append(np.full(len(q), p['d']))
        idx.append(np.full(len(q), i))
        ws.append(np.ones(len(q)) if weights is None else weights[i])
    return (np.vstack(pts), np.vstack(ns), np.concatenate(ds), np.concatenate(idx),
            np.concatenate(ws))


def residuals(x, t0, pts, ns, ds, ws=1.0):
    """보정 x를 적용했을 때 모든 점의 점-평면 거리 (m), 가중치 ws를 곱해서."""
    t = apply(t0, x)
    q = pts @ t[:3, :3].T + t[:3, 3]
    return (np.sum(q * ns, axis=1) - ds) * ws


def rotation_angle_deg(r):
    """회전 행렬이 나타내는 회전의 크기 (°)."""
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


def errors(t_est, t_true):
    """정답 대비 오차: 회전 (°), 위치 크기 (mm), 위치 축별 (mm, 카메라 축 x 오른쪽 y 아래 z 앞)."""
    rot = rotation_angle_deg(t_est[:3, :3].T @ t_true[:3, :3])
    dpos = (t_est[:3, 3] - t_true[:3, 3]) * 1000
    return rot, float(np.linalg.norm(dpos)), dpos


def calibrate(poses, t_init, weights=None):
    """
    점-평면 잔차 최소화 (weights가 있으면 가중 최소제곱).

    돌려주는 값: (추정 변환, least_squares 결과, (점, 법선, 거리, 자세 번호, 가중치))
    """
    pts, ns, ds, idx, ws = stack(poses, weights)
    sol = least_squares(residuals, np.zeros(6), args=(t_init, pts, ns, ds, ws), method='lm',
                        x_scale=UNIT)
    return apply(t_init, sol.x), sol, (pts, ns, ds, idx, ws)


def uncertainty(sol, n_points):
    """
    추정값의 1σ 불확실성 (카메라 축별 회전 °, 위치 mm)과 가장 약한 방향.

    공분산 ≈ s² (JᵀJ)⁻¹, s² = (가중) 잔차 제곱합 / (점 수 - 6).
    JᵀJ의 고유값이 0에 가까운 방향 = 데이터로 거의 정해지지 않는 방향
    (회전과 위치를 비교하려고 합격 기준 단위 UNIT으로 맞춘 뒤 계산).
    """
    j = sol.jac
    s2 = float(np.sum(sol.fun ** 2) / max(n_points - 6, 1))
    cov = s2 * np.linalg.pinv(j.T @ j)
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    js = j * UNIT
    eigval, eigvec = np.linalg.eigh(js.T @ js)
    return np.degrees(std[:3]), std[3:] * 1000, eigval, eigvec[:, 0]


def axis_name(v):
    """6차원 방향 벡터가 주로 어느 파라미터인지 사람이 읽는 이름으로."""
    names = ['회전 x축(끄덕임)', '회전 y축(좌우 돌림)', '회전 z축(옆 기울임)',
             '위치 x(좌우)', '위치 y(위아래)', '위치 z(앞뒤)']
    order = np.argsort(-np.abs(v))
    return ', '.join(f'{names[i]} {v[i]:+.2f}' for i in order[:2])


def export_problem(path, data, poses, weights, elev_deg, result, nfev, solve_ms):
    """
    C++(Ceres) 프로그램이 같은 문제를 풀 수 있게 문제와 이 스크립트의 답을 YAML로 저장한다.

    점은 빔 높이각을 이미 적용한 3D 위치(LiDAR 좌표계, m)다. 가중치는 --weighted일 때만 넣는다.
    reference_python은 비교용일 뿐 C++ 풀이에는 쓰지 않는다.
    """
    out = {
        'cam_frame': data['cam_frame'], 'lidar_frame': data['lidar_frame'],
        'beam_elev_deg': float(elev_deg), 'weighted': weights is not None,
        'pass_rot_deg': PASS_ROT_DEG, 'pass_pos_mm': PASS_POS_MM,
        'T_nominal': np.asarray(data['T_nominal'], dtype=float).tolist(),
        'T_truth': np.asarray(data['T_truth'], dtype=float).tolist(),
        'poses': [],
        'reference_python': {
            'solver': 'scipy.optimize.least_squares (lm, numeric jacobian)',
            'matrix': result['matrix'], 'residual_rms_mm': result['residual_rms_mm'],
            'std_rot_deg': result['std_rot_deg'], 'std_pos_mm': result['std_pos_mm'],
            'error_vs_truth': result['error_vs_truth'],
            'nfev': int(nfev), 'solve_ms': float(solve_ms),
        },
    }
    for i, p in enumerate(poses):
        item = {'name': p['name'], 'n': [float(v) for v in p['n']], 'd': float(p['d']),
                'points': np.asarray(p['points'], dtype=float).round(9).tolist()}
        if weights is not None:
            item['weights'] = [float(w) for w in weights[i]]
        out['poses'].append(item)
    with open(path, 'w') as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=None, width=1000)


def main():
    """명령줄에서 실행할 때의 시작점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data', help='lidar_cam_collect.py가 만든 JSON')
    ap.add_argument('--only-upright', action='store_true',
                    help='실험: 젖히지 않은(tilt 0) 자세만 사용')
    ap.add_argument('--max-poses', type=int, default=0, help='앞에서부터 이 개수만 사용 (0 = 전부)')
    ap.add_argument('--weighted', action='store_true',
                    help='LiDAR 노이즈와 카메라 평면 불확실성으로 점마다 가중치')
    ap.add_argument('--sigma-lidar', type=float, default=0.0,
                    help='점 하나의 LiDAR 거리 노이즈 (mm). 0이면 10 / √(자세당 스캔 수)')
    ap.add_argument('--beam-elev-deg', type=float, default=0.0,
                    help='LiDAR 빔 높이각 (°, 아래가 -). Gazebo GPU LiDAR는 약 -0.44')
    ap.add_argument('--out', default='', help='결과 YAML (기본: 입력 옆 *_result.yaml)')
    ap.add_argument('--export-problem', default='',
                    help='C++(Ceres) 비교용: 문제와 이 스크립트의 답을 이 YAML에 저장')
    args = ap.parse_args()

    with open(args.data) as f:
        data = json.load(f)
    poses = data['poses']
    if args.only_upright:
        poses = [p for p in poses if abs(p['tilt_deg']) < 1e-6]
    if args.max_poses:
        poses = poses[:args.max_poses]
    if len(poses) < 2:
        raise SystemExit(f'자세가 {len(poses)}개뿐: 최소 2개, 권장 10개 이상')
    t_nom = np.array(data['T_nominal'])
    t_true = np.array(data['T_truth'])
    if args.beam_elev_deg:
        poses = [dict(p, points=apply_beam_elevation(p['points'], args.beam_elev_deg))
                 for p in poses]

    weights, sig_plane = None, [float('nan')] * len(poses)
    if args.weighted:
        sigma_l = (args.sigma_lidar or 10.0 / np.sqrt(data.get('scans_per_pose', 1))) / 1000
        weights, sig_plane = point_weights(poses, t_nom, sigma_l)
    start = time.perf_counter()
    t_est, sol, (pts, ns, ds, idx, _) = calibrate(poses, t_nom, weights)
    solve_ms = (time.perf_counter() - start) * 1000
    r0 = residuals(np.zeros(6), t_nom, pts, ns, ds) * 1000
    r1 = residuals(sol.x, t_nom, pts, ns, ds) * 1000
    rot_std, pos_std, eigval, weakest = uncertainty(sol, len(pts))

    note = '  (실험: 똑바로 선 자세만)' if args.only_upright else ''
    if args.weighted:
        note += f'  (가중치: σ_LiDAR {sigma_l * 1000:.1f} mm + 자세별 σ_평면)'
    if args.beam_elev_deg:
        note += f'  (빔 높이각 {args.beam_elev_deg:+.3f}°)'
    print(f'자세 {len(poses)}개, 점 {len(pts)}개, 잔차 계산 {sol.nfev}회, {solve_ms:.1f} ms{note}')
    print('[자세별 잔차 RMS, mm]  추정 전(공칭) -> 추정 후')
    for i, p in enumerate(poses):
        m = idx == i
        before = np.sqrt(np.mean(r0[m] ** 2))
        after = np.sqrt(np.mean(r1[m] ** 2))
        name, dist, yaw, tilt = p['name'], p['dist'], p['yaw_deg'], p['tilt_deg']
        sp = f'   σ_평면 {sig_plane[i] * 1000:.1f} mm' if args.weighted else ''
        print(f'    {name}  거리 {dist:.2f} yaw {yaw:+4d} tilt {tilt:+4d}  '
              f'{before:6.1f} -> {after:5.1f}   (점 {int(m.sum())}){sp}')
    print(f'    전체                          {np.sqrt(np.mean(r0 ** 2)):6.1f} -> '
          f'{np.sqrt(np.mean(r1 ** 2)):5.1f}')

    print('[정답 대비 오차]  카메라 축: x 오른쪽, y 아래, z 앞')
    rows = (('추정 전 (공칭)', t_nom), ('추정 후', t_est))
    for name, t in rows:
        rot, pos, d = errors(t, t_true)
        print(f'    {name:12s} 회전 {rot:6.3f}°  위치 {pos:6.1f} mm  '
              f'(x {d[0]:+6.1f}, y {d[1]:+6.1f}, z {d[2]:+6.1f})')
    rot, pos, _ = errors(t_est, t_true)
    ok = rot <= PASS_ROT_DEG and pos <= PASS_POS_MM
    verdict = '합격' if ok else '불합격'
    print(f'    합격 기준 회전 {PASS_ROT_DEG}°, 위치 {PASS_POS_MM:.0f} mm -> {verdict}')

    print('[추정 불확실성 1σ]  데이터만으로 계산 (정답을 안 씀)')
    print(f'    회전 (°)   x축 {rot_std[0]:.3f}  y축 {rot_std[1]:.3f}  z축 {rot_std[2]:.3f}')
    print(f'    위치 (mm)  x {pos_std[0]:.1f}  y {pos_std[1]:.1f}  z {pos_std[2]:.1f}')
    cond = eigval[-1] / max(eigval[0], 1e-30)
    print(f'    가장 약한 방향: {axis_name(weakest)}  (조건수 {cond:.1e})')

    out = args.out or os.path.splitext(args.data)[0] + '_result.yaml'
    q = cv2.Rodrigues(t_est[:3, :3])[0].ravel()
    result = {
        'parent': data['cam_frame'], 'child': data['lidar_frame'],
        'xyz': [float(v) for v in t_est[:3, 3]],
        'rotation_vector': [float(v) for v in q],
        'matrix': t_est.tolist(),
        'poses_used': [p['name'] for p in poses],
        'beam_elev_deg': args.beam_elev_deg,
        'residual_rms_mm': float(np.sqrt(np.mean(r1 ** 2))),
        'std_rot_deg': [float(v) for v in rot_std],
        'std_pos_mm': [float(v) for v in pos_std],
        'error_vs_truth': {'rot_deg': rot, 'pos_mm': pos},
    }
    with open(out, 'w') as f:
        yaml.safe_dump(result, f, sort_keys=False, allow_unicode=True)
    print(f'saved {out}')
    if args.export_problem:
        export_problem(args.export_problem, data, poses, weights, args.beam_elev_deg, result,
                       sol.nfev, solve_ms)
        print(f'saved {args.export_problem}  (C++: ros2 run tb3p_calibration '
              f'lidar_cam_calib_ceres {args.export_problem})')


if __name__ == '__main__':
    main()
