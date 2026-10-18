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

사용 예:
  python3 lidar_cam_calib.py ~/lidar_cam_data/session12.json
  python3 lidar_cam_calib.py ~/lidar_cam_data/session12.json --only-upright   # 실험: 똑바로 선 자세만
"""
import argparse
import json
import os

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


def stack(poses):
    """자세들을 점 단위 배열로 펼친다: 점(N x 3), 평면 법선(N x 3), 거리(N), 자세 번호(N)."""
    pts, ns, ds, idx = [], [], [], []
    for i, p in enumerate(poses):
        q = np.asarray(p['points'], dtype=np.float64)
        pts.append(q)
        ns.append(np.tile(p['n'], (len(q), 1)))
        ds.append(np.full(len(q), p['d']))
        idx.append(np.full(len(q), i))
    return np.vstack(pts), np.vstack(ns), np.concatenate(ds), np.concatenate(idx)


def residuals(x, t0, pts, ns, ds):
    """보정 x를 적용했을 때 모든 점의 점-평면 거리 (m)."""
    t = apply(t0, x)
    q = pts @ t[:3, :3].T + t[:3, 3]
    return np.sum(q * ns, axis=1) - ds


def rotation_angle_deg(r):
    """회전 행렬이 나타내는 회전의 크기 (°)."""
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


def errors(t_est, t_true):
    """정답 대비 오차: 회전 (°), 위치 크기 (mm), 위치 축별 (mm, 카메라 축 x 오른쪽 y 아래 z 앞)."""
    rot = rotation_angle_deg(t_est[:3, :3].T @ t_true[:3, :3])
    dpos = (t_est[:3, 3] - t_true[:3, 3]) * 1000
    return rot, float(np.linalg.norm(dpos)), dpos


def calibrate(poses, t_init):
    """점-평면 잔차 최소화. 돌려주는 값: (추정 변환, least_squares 결과, 점 배열들)."""
    pts, ns, ds, idx = stack(poses)
    sol = least_squares(residuals, np.zeros(6), args=(t_init, pts, ns, ds), method='lm',
                        x_scale=UNIT)
    return apply(t_init, sol.x), sol, (pts, ns, ds, idx)


def uncertainty(sol, n_points):
    """
    추정값의 1σ 불확실성 (카메라 축별 회전 °, 위치 mm)과 가장 약한 방향.

    공분산 ≈ s² (JᵀJ)⁻¹, s² = 잔차 제곱합 / (점 수 - 6).
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


def main():
    """명령줄에서 실행할 때의 시작점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data', help='lidar_cam_collect.py가 만든 JSON')
    ap.add_argument('--only-upright', action='store_true',
                    help='실험: 젖히지 않은(tilt 0) 자세만 사용')
    ap.add_argument('--max-poses', type=int, default=0, help='앞에서부터 이 개수만 사용 (0 = 전부)')
    ap.add_argument('--out', default='', help='결과 YAML (기본: 입력 옆 *_result.yaml)')
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

    t_est, sol, (pts, ns, ds, idx) = calibrate(poses, t_nom)
    r0 = residuals(np.zeros(6), t_nom, pts, ns, ds) * 1000
    r1 = sol.fun * 1000
    rot_std, pos_std, eigval, weakest = uncertainty(sol, len(pts))

    note = '  (실험: 똑바로 선 자세만)' if args.only_upright else ''
    print(f'자세 {len(poses)}개, 점 {len(pts)}개, 반복 {sol.nfev}회{note}')
    print('[자세별 잔차 RMS, mm]  추정 전(공칭) -> 추정 후')
    for i, p in enumerate(poses):
        m = idx == i
        before = np.sqrt(np.mean(r0[m] ** 2))
        after = np.sqrt(np.mean(r1[m] ** 2))
        name, dist, yaw, tilt = p['name'], p['dist'], p['yaw_deg'], p['tilt_deg']
        print(f'    {name}  거리 {dist:.2f} yaw {yaw:+4d} tilt {tilt:+4d}  '
              f'{before:6.1f} -> {after:5.1f}   (점 {int(m.sum())})')
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
        'residual_rms_mm': float(np.sqrt(np.mean(r1 ** 2))),
        'std_rot_deg': [float(v) for v in rot_std],
        'std_pos_mm': [float(v) for v in pos_std],
        'error_vs_truth': {'rot_deg': rot, 'pos_mm': pos},
    }
    with open(out, 'w') as f:
        yaml.safe_dump(result, f, sort_keys=False, allow_unicode=True)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
