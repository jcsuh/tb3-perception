#!/usr/bin/env python3
"""
LiDAR와 카메라가 로봇 몸(base_link) 기준으로 어디에 어떻게 붙었는지 추정한다 (세션 24).

왜 필요한가
  LiDAR-카메라 캘리브레이션(세션 12–14)은 두 센서 사이 관계만 준다. 몸 기준 자세는
  "몸 -> LiDAR"를 알아야 이어서 계산할 수 있는데, 지금까지 이 고리는 공칭값이었다 (세션 23).

두 가지 관측 (입력: lidar_base_collect.py가 만든 NPZ)
  1) 손-눈 캘리브레이션 (평면): 같은 1초 동안의 움직임을
       바퀴 오도메트리 A (몸이 움직인 것)  과  LiDAR 스캔 매칭 B (LiDAR가 움직인 것)
     로 각각 잰다. LiDAR 자세 X (yaw φ, 위치 p) 라면  A·X = X·B  이어야 한다.
       회전 부분은 자동으로 같고, 이동 부분:  (R_A - I)·p - R_φ·t_B = -t_A
     -> p와 (cos φ, sin φ)에 대해 선형 -> 최소제곱. 앞으로 갈 때 LiDAR에 비스듬히 보이는 만큼이 φ,
        돌 때 LiDAR가 회전 중심 둘레로 그리는 원이 p.  (LiDAR의 yaw, x, y)
  2) 바닥 평면: 깊이 영상에서 가장 큰 수평 평면(RANSAC)을 찾으면, 카메라 기준 바닥의 방향과 거리가 나온다.
     바닥은 몸 기준으로 알고 있다 (바퀴 접지점을 지나고, 캐스터 때문에 몸이 약간 뒤로 기움).
     -> 카메라의 roll, pitch(바닥 기준), 높이
     벽에 붙어 바닥이 조금만 보이는 장은 엉뚱한 평면이 잡히므로, 바닥 점이 충분하고 서로 일치하는
     장만 골라 평균한다 (select_floor)
  3) 둘을 LiDAR-카메라 관계(T_color_lidar)로 이어 카메라 자세 6개를 풀면 (방정식 6개, 미지수 6개)
     LiDAR와 카메라 자세가 모두 정해진다.

결과: YAML (calibration.yaml의 sensors.lidar, sensors.camera 형식) -> calib_merge.py --lidar-base
채점 (시뮬레이션): --truth sim_truth.yaml

사용 예:
  python3 lidar_base_calib.py ~/lidar_base_data/session24.npz
      --wheels ~/umbmark_data/before2_wheels_ls.yaml
      --lidar-cam ~/lidar_cam_data/session13_result.yaml
      --truth ../../tb3p_gazebo/config/sim_truth.yaml                     (실제로는 한 줄로)
"""
import argparse
import json
import math
import os

import calib_merge as cm
import numpy as np
import scan_match as sm
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import wheel_odom as wo
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
NOMINAL = os.path.join(HERE, '..', '..', 'tb3p_description', 'config', 'calibration.yaml')
DEPTH = 'camera_depth_optical_frame'
CASTER_X, CASTER_GAP = 0.081, 0.001        # imu_mount_calib.py와 같은 설계 치수
PIVOT = np.array([0.0, 0.0, -0.010])       # 바퀴 접지점 (base_link 기준, base_joint z 0.010)


# ===== 2D 자세 =====

def se2(x, y, th):
    """2D 자세 -> 3x3."""
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def se2_params(t):
    """3x3 -> (x, y, θ)."""
    return float(t[0, 2]), float(t[1, 2]), math.atan2(t[1, 0], t[0, 0])


def wrap(a):
    """각도를 -π~π로."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def odom_at(t_query, t_odom, poses):
    """오도메트리 자세 (N x 3)를 원하는 시각들로 선형 보간 (θ는 풀린 값이라 그대로 보간)."""
    return np.column_stack([np.interp(t_query, t_odom, poses[:, k]) for k in range(3)])


# ===== 1) 손-눈 =====

def solve_hand_eye_linear(motions):
    """
    motions: [(A, B)] 3x3 쌍. A = 몸 움직임, B = LiDAR 움직임.

    돌려주는 값: (φ, px, py) — (R_A - I) p - R_φ t_B = -t_A 의 선형 최소제곱 해
    """
    rows, rhs = [], []
    for a, b in motions:
        ra, ta, tb = a[:2, :2], a[:2, 2], b[:2, 2]
        rows.append([ra[0, 0] - 1, ra[0, 1], -tb[0], tb[1]])
        rows.append([ra[1, 0], ra[1, 1] - 1, -tb[1], -tb[0]])
        rhs += [-ta[0], -ta[1]]
    z, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return math.atan2(z[3], z[2]), float(z[0]), float(z[1])


def hand_eye_residuals(params, motions):
    """비선형 다듬기용 잔차: A·X 와 X·B 의 이동 차이 (m), 쌍마다 2개."""
    phi, px, py = params
    x = se2(px, py, phi)
    out = []
    for a, b in motions:
        d = a @ x - x @ b
        out += [d[0, 2], d[1, 2]]
    return np.array(out)


def solve_hand_eye(motions):
    """선형 해 -> 튀는 쌍에 덜 휘둘리게(soft L1) 다듬기 + 1σ. 돌려주는 값: dict."""
    init = solve_hand_eye_linear(motions)
    sol = least_squares(hand_eye_residuals, init, args=(motions,), loss='soft_l1',
                        f_scale=0.005)
    r = sol.fun
    dof = max(1, len(r) - 3)
    s2 = float(r @ r) / dof
    cov = np.linalg.pinv(sol.jac.T @ sol.jac) * s2
    std = np.sqrt(np.diag(cov))
    return {'phi': float(sol.x[0]), 'p': (float(sol.x[1]), float(sol.x[2])),
            'std_phi': float(std[0]), 'std_p': (float(std[1]), float(std[2])),
            'rms_mm': float(np.sqrt(np.mean(r ** 2)) * 1000), 'n': len(motions),
            'linear': init}


def scan_motions(scan_t, scans, info, odo_t, odo, x0, gap=10, step=5, min_points=80,
                 max_rms=0.02):
    """
    스캔 gap개 간격(약 1초)마다 몸 움직임 A(오도메트리)와 LiDAR 움직임 B(ICP)를 짝짓는다.

    ICP 초기값은 오도메트리 움직임을 공칭 LiDAR 자세 x0로 옮긴 값 (B ≈ x0⁻¹ A x0).
    돌려주는 값: ([(A, B)], 쓴 쌍 수, 버린 쌍 수)
    """
    pose = odom_at(scan_t, odo_t, odo)
    pts = [sm.scan_points(r, info['angle_min'], info['angle_increment']) for r in scans]
    x0_inv = np.linalg.inv(x0)
    out, bad = [], 0
    for i in range(0, len(scans) - gap, step):
        j = i + gap
        a = np.linalg.inv(se2(*pose[i])) @ se2(*pose[j])
        guess = se2_params(x0_inv @ a @ x0)
        res, n, rms = sm.icp(pts[j], pts[i], init=guess)
        if n < min_points or not np.isfinite(rms) or rms > max_rms:
            bad += 1
            continue
        out.append((a, se2(*res)))
    return out, len(out), bad


# ===== 2) 바닥 =====

def depth_points(depth, k, stride):
    """줄여 저장한 깊이 영상 (픽셀 stride마다 하나) -> 깊이 광학 좌표계 점 N x 3."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    u, v = u.ravel() * stride, v.ravel() * stride
    z = depth.ravel().astype(np.float64)
    ok = np.isfinite(z) & (z > 0.25) & (z < 3.0)
    u, v, z = u[ok], v[ok], z[ok]
    return np.column_stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z])


def fit_floor(points, n_guess, max_angle_deg=15.0, thresh=0.01, iters=300, rng=None):
    """
    RANSAC으로 n_guess와 비슷한 방향의 가장 큰 평면 -> (n, d, 안쪽 점 수, 잔차 RMS m).

    평면: n·x = d (n은 단위 벡터, n_guess 쪽을 향함). 벽·기둥은 방향이 달라 걸러진다.
    """
    rng = rng or np.random.default_rng(0)
    n_guess = np.asarray(n_guess, dtype=float) / np.linalg.norm(n_guess)
    if len(points) < 100:
        return None
    cos_lim = math.cos(math.radians(max_angle_deg))
    best, best_cnt = None, 0
    for _ in range(iters):
        p = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if n @ n_guess < 0:
            n = -n
        if n @ n_guess < cos_lim:
            continue
        cnt = int((np.abs(points @ n - n @ p[0]) < thresh).sum())
        if cnt > best_cnt:
            best, best_cnt = (n, float(n @ p[0])), cnt
    if best is None:
        return None
    n, d = best
    for th in (thresh, thresh / 2):          # 안쪽 점으로 다시 맞추기 (두 번)
        inl = points[np.abs(points @ n - d) < th]
        c = inl.mean(axis=0)
        _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
        n = vt[2] if vt[2] @ n_guess > 0 else -vt[2]
        d = float(n @ c)
    r = inl @ n - d
    return n, d, len(inl), float(np.sqrt(np.mean(r ** 2)))


def select_floor(fits, min_ratio=0.6, max_angle_deg=0.15, max_dist=0.002):
    """
    장마다 찾은 바닥 평면 중 믿을 만한 것만 고른다 -> (n, d, 고른 표시, 버린 이유 셈).

    세션 24 Gazebo 기록: 벽에 붙어 바닥이 조금만 보이면 RANSAC이 엉뚱한 평면을 고르는 장이 있었다
    (바닥 점이 적고, 거리·방향이 크게 벗어남). 바닥이 잘 보인 장끼리는 ±0.5 mm, 0.03° 안에서 같았다.
      1) 바닥 점 수가 중앙값의 min_ratio 배보다 적은 장을 뺀다
      2) 남은 장의 중앙값 평면에서 방향 max_angle_deg 또는 거리 max_dist 넘게 벗어난 장을 뺀다
      3) 남은 장의 평균 (n은 다시 단위 벡터로)
    """
    ns = np.array([f[0] for f in fits])
    ds = np.array([f[1] for f in fits])
    cnt = np.array([f[2] for f in fits])
    few = cnt < min_ratio * np.median(cnt)
    ok = ~few
    n_med = np.median(ns[ok], axis=0)
    n_med /= np.linalg.norm(n_med)
    d_med = float(np.median(ds[ok]))
    ang = np.degrees(np.arccos(np.clip(ns @ n_med, -1.0, 1.0)))
    off = ok & ((ang > max_angle_deg) | (np.abs(ds - d_med) > max_dist))
    ok &= ~off
    n = ns[ok].mean(axis=0)
    return n / np.linalg.norm(n), float(ds[ok].mean()), ok, {'few': int(few.sum()),
                                                             'off': int(off.sum())}


def level_rotation(body_tilt):
    """몸 좌표 -> 수평 좌표 회전 (body_tilt = (roll, pitch) rad, imu_mount_calib.py와 같은 약속)."""
    return cm.rpy_to_matrix(0.0, body_tilt[1], 0.0) @ cm.rpy_to_matrix(body_tilt[0], 0.0, 0.0)


def floor_in_base(body_tilt):
    """몸(base_link) 기준 바닥 평면 (n_b, d_b): 바퀴 접지점을 지나고 수평 위 방향이 n_b."""
    n_b = level_rotation(body_tilt).T @ np.array([0.0, 0.0, 1.0])
    return n_b, float(n_b @ PIVOT)


KIND_LABELS = {'still': '서 있음', 'turn': '제자리 회전', 'move': '달리는 중'}


def frame_kinds(t_frames, t_odom, odo, settle=1.0):
    """
    깊이 영상마다 로봇 상태: 'still'(앞선 settle초 동안 거의 안 움직임), 'turn'(제자리 회전), 'move'.

    오도메트리 자세 차이로 속도를 구한다 (바퀴 기록이 있는 구간만).
    """
    out = []
    for t in t_frames:
        p0, p1 = odom_at([t - settle, t], t_odom, odo)
        dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        turn = abs(p1[2] - p0[2])
        if dist < 0.002 and turn < 0.005:
            out.append('still')
        elif dist < 0.01 * settle:
            out.append('turn')
        else:
            out.append('move')
    return out


# ===== 3) 잇기 =====

def lidar_in_level(t_base_lidar, body_tilt):
    """몸 기준 LiDAR 자세 -> 수평 평면에서 본 (yaw, x, y) (손-눈이 재는 것)."""
    r_l = level_rotation(body_tilt)
    q = r_l @ (t_base_lidar[:3, 3] - PIVOT)
    r = r_l @ t_base_lidar[:3, :3]
    return math.atan2(r[1, 0], r[0, 0]), float(q[0]), float(q[1])


def make_pose(x):
    """[회전 벡터 3, 위치 3] -> 4x4."""
    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
    t[:3, 3] = x[3:]
    return t


def combine_residuals(x, floor_c, he, t_color_lidar, t_color_depth, body_tilt):
    """카메라 자세 x (base <- color)에 대한 잔차 6개: 바닥 방향 3(실제 2), 바닥 거리 1, 손-눈 3."""
    t_bc = make_pose(x)
    t_bd = t_bc @ t_color_depth
    n_b, d_b = floor_in_base(body_tilt)
    n_pred = t_bd[:3, :3].T @ n_b
    d_pred = d_b - n_b @ t_bd[:3, 3]
    yaw, qx, qy = lidar_in_level(t_bc @ t_color_lidar, body_tilt)
    return np.concatenate([n_pred - floor_c[0], [d_pred - floor_c[1]],
                           [wrap(yaw - he['phi']), qx - he['p'][0], qy - he['p'][1]]])


def combine(floor_c, he, t_color_lidar, t_color_depth, body_tilt, t_base_color0):
    """바닥(카메라 기준)과 손-눈(LiDAR)을 이어 base <- color 자세를 푼다."""
    x0 = np.concatenate([Rotation.from_matrix(t_base_color0[:3, :3]).as_rotvec(),
                         t_base_color0[:3, 3]])
    sol = least_squares(combine_residuals, x0,
                        args=(floor_c, he, t_color_lidar, t_color_depth, body_tilt))
    return make_pose(sol.x), sol.fun


# ===== 읽기 · 채점 · 쓰기 =====

def load(path):
    """lidar_base_collect.py의 NPZ -> dict."""
    raw = np.load(path)
    d = {k: raw[k] for k in raw.files if k != 'meta'}
    d['meta'] = json.loads(str(raw['meta']))
    return d


def pose_entry(parent, child, t):
    """4x4 -> calibration.yaml 센서 항목."""
    xyz, rpy = cm.from_matrix(t)
    return {'parent': parent, 'child': child, 'xyz': [round(v, 7) for v in xyz],
            'rpy': [round(v, 7) for v in rpy]}


def errors(est, truth):
    """두 자세 차이: 회전 °, 위치 mm, (roll, pitch, yaw) °, (x, y, z) mm (추정 - 정답)."""
    rot, pos = cm.difference(truth, est)
    drpy = np.degrees(cm.matrix_to_rpy(truth[:3, :3].T @ est[:3, :3]))
    return rot, pos, drpy, (est[:3, 3] - truth[:3, 3]) * 1000


def main():
    """명령줄에서 실행할 때의 시작점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data', help='lidar_base_collect.py가 만든 NPZ')
    ap.add_argument('--wheels', default=NOMINAL, help='바퀴 값 YAML (세션 15–16 결과 권장)')
    ap.add_argument('--lidar-cam', required=True, help='lidar_cam_calib.py의 *_result.yaml')
    ap.add_argument('--calib', default=NOMINAL, help='공칭 캘리브레이션 (초기값과 비교)')
    ap.add_argument('--time', default='', help='time_offset_estimate.py 결과 (시각 보정, 선택)')
    ap.add_argument('--body-tilt-deg', type=float, nargs=2, default=None,
                    help='몸 기울기 roll pitch (°, 기본: 캐스터 설계 치수로 계산)')
    ap.add_argument('--gap', type=int, default=10, help='손-눈 비교 간격 (스캔 수, 10 = 약 1초)')
    ap.add_argument('--floor-frames', choices=['all', 'still'], default='all',
                    help='바닥 평면 후보 영상 (still: 서 있을 때만). 어느 쪽이든 믿을 만한 장만 고름')
    ap.add_argument('--truth', default='', help='(시뮬레이션 채점 전용) sim_truth.yaml')
    ap.add_argument('--out', default='', help='결과 YAML (기본: 입력 옆 *_lidar_base.yaml)')
    args = ap.parse_args()

    data = load(args.data)
    meta = data['meta']
    nominal = cm.load(args.calib)
    body = (tuple(math.radians(v) for v in args.body_tilt_deg) if args.body_tilt_deg
            else (0.0, -math.atan2(CASTER_GAP, CASTER_X)))
    lc = cm.load(args.lidar_cam)
    t_color_lidar = np.array(lc['matrix'], dtype=float)
    edges = cm.urdf_edges(cm.run_xacro(args.calib))
    t_color_depth = cm.relative(edges, cm.COLOR, DEPTH)
    t_screw_color = cm.relative(edges, cm.SCREW, cm.COLOR)
    t_base_lidar0 = cm.sensor_matrix(nominal, 'lidar')
    t_base_color0 = cm.sensor_matrix(nominal, 'camera') @ t_screw_color

    # 시각 보정 (calibration.yaml 약속: 센서 시각 + time_offset = IMU 기준 시각)
    off_w = off_l = 0.0
    if args.time:
        off = cm.load(args.time)['time_offset_s']
        off_w, off_l = float(off['wheels']), float(off['lidar'])

    # 1) 손-눈
    wheels = wo.load_wheels(args.wheels)
    jt = data['joint'][:, 0] + off_w
    odo = wo.integrate(data['joint'][:, 1], data['joint'][:, 2], wheels)
    yaw0, x0, y0 = lidar_in_level(t_base_lidar0, body)
    motions, used, bad = scan_motions(data['scan_t'] + off_l, data['scan_r'], meta['scan'],
                                      jt, odo, se2(x0, y0, yaw0), gap=args.gap)
    he = solve_hand_eye(motions)
    print(f'[1] 손-눈 (스캔 {args.gap}개 간격 쌍 {used}개, ICP 실패로 뺀 쌍 {bad}개)')
    phi_d, sphi_d = math.degrees(he['phi']), math.degrees(he['std_phi'])
    (px, py), (spx, spy) = [v * 1000 for v in he['p']], [v * 1000 for v in he['std_p']]
    print(f'    LiDAR yaw {phi_d:+.3f}° ± {sphi_d:.3f}°,  x {px:+.1f} ± {spx:.1f} mm,'
          f'  y {py:+.1f} ± {spy:.1f} mm   (공칭 {math.degrees(yaw0):+.3f}°, '
          f'{x0 * 1000:+.1f}, {y0 * 1000:+.1f} mm)')
    rms = he['rms_mm']
    print(f'    잔차 RMS {rms:.2f} mm (A·X와 X·B의 이동 차이)')

    # 2) 바닥
    k = np.array(meta['K']).reshape(3, 3)
    n_b, _ = floor_in_base(body)
    n_guess = (t_base_color0 @ t_color_depth)[:3, :3].T @ n_b
    kinds = frame_kinds(data['depth_t'], jt, odo)
    fits, fit_kinds = [], []
    for depth, kind in zip(data['depth'], kinds):
        f = fit_floor(depth_points(depth, k, meta['stride']), n_guess)
        if f is not None and f[2] > 300:
            fits.append(f)
            fit_kinds.append(kind)
    if not fits:
        raise SystemExit('깊이 영상에서 바닥을 못 찾음')
    fit_kinds = np.array(fit_kinds)
    ns = np.array([f[0] for f in fits])
    ds = np.array([abs(f[1]) for f in fits])
    n_img = len(data['depth'])
    print(f'[2] 바닥 평면 (영상 {len(fits)}/{n_img}장, 장당 바닥 점 중앙값 '
          f'{int(np.median([f[2] for f in fits]))}개, 평면 잔차 중앙값 '
          f'{np.median([f[3] for f in fits]) * 1000:.1f} mm)')
    print('    로봇 상태별 (오도메트리로 구분)  장 수 | 바닥까지 mm 평균 ± 표준편차 | 방향 흩어짐 ° '
          '| 전체 평균과의 방향 차 °')
    n_all = ns.mean(axis=0) / np.linalg.norm(ns.mean(axis=0))
    for kind, label in KIND_LABELS.items():
        m = fit_kinds == kind
        if not m.any():
            continue
        nk = ns[m].mean(axis=0) / np.linalg.norm(ns[m].mean(axis=0))
        sp = np.degrees(np.arccos(np.clip(ns[m] @ nk, -1, 1))).mean()
        off = math.degrees(math.acos(min(1.0, float(nk @ n_all))))
        print(f'      {label:16s} {m.sum():4d} | {ds[m].mean() * 1000:7.1f} ± '
              f'{ds[m].std() * 1000:5.1f} | {sp:6.3f} | {off:6.3f}')
    pool = fit_kinds == 'still' if args.floor_frames == 'still' else np.ones(len(fits), bool)
    if pool.sum() < 3:
        print(f'    (서 있는 영상이 {pool.sum()}장뿐이라 모든 영상에서 고름)')
        pool = np.ones(len(fits), bool)
    idx = np.flatnonzero(pool)
    n_c, d_c, ok, why = select_floor([fits[i] for i in idx])
    use = np.zeros(len(fits), bool)
    use[idx[ok]] = True
    sel_ang = np.degrees(np.arccos(np.clip(ns[use] @ n_c, -1, 1)))
    few, off = why['few'], why['off']
    print(f'    고른 영상 {use.sum()}/{pool.sum()}장 (바닥 점이 적어 뺌 {few}장, '
          f'중앙값에서 벗어나 뺌 {off}장): 바닥까지 {-d_c * 1000:.1f} ± '
          f'{ds[use].std() * 1000:.1f} mm, 방향 흩어짐 {sel_ang.mean():.3f}°')
    print(f'    몸 기울기 roll {math.degrees(body[0]):+.3f}°, pitch {math.degrees(body[1]):+.3f}°')

    # 3) 잇기
    t_base_color, res = combine((n_c, d_c), he, t_color_lidar, t_color_depth, body,
                                t_base_color0)
    t_base_lidar = t_base_color @ t_color_lidar
    t_base_screw = t_base_color @ np.linalg.inv(t_screw_color)
    print(f'[3] 잇기: 남은 잔차 최대 {np.abs(res).max():.1e} (0이면 여섯 조건을 모두 만족)')
    for name, t0, t1 in (('lidar', t_base_lidar0, t_base_lidar),
                         ('camera', cm.sensor_matrix(nominal, 'camera'), t_base_screw)):
        rot, pos = cm.difference(t0, t1)
        print(f'    {name:7s} 공칭 대비 변화 회전 {rot:.3f}°, 위치 {pos:.1f} mm')

    if args.truth:
        truth = cm.load(args.truth)
        print('[정답과 비교] (시뮬레이션 채점 전용)  오차 = 추정 - 정답')
        print('    센서     |    회전 °  (roll, pitch, yaw)        |  위치 mm (x, y, z)')
        for name, est, nom in (('lidar', t_base_lidar, t_base_lidar0),
                               ('camera', t_base_screw, cm.sensor_matrix(nominal, 'camera'))):
            tr = cm.sensor_matrix(truth, name)
            for label, t in (('공칭', nom), ('추정', est)):
                rot, pos, drpy, dxyz = errors(t, tr)
                print(f'    {name:6s} {label} | {rot:6.3f}  ({drpy[0]:+.2f}, {drpy[1]:+.2f}, '
                      f'{drpy[2]:+.2f}) | {pos:5.1f}  ({dxyz[0]:+.1f}, {dxyz[1]:+.1f}, '
                      f'{dxyz[2]:+.1f})')
        tyaw, tx, ty = lidar_in_level(cm.sensor_matrix(truth, 'lidar'), body)
        e_yaw = math.degrees(wrap(he['phi'] - tyaw))
        e_x, e_y = (he['p'][0] - tx) * 1000, (he['p'][1] - ty) * 1000
        print(f'    손-눈만: yaw 오차 {e_yaw:+.3f}°, x {e_x:+.1f} mm, y {e_y:+.1f} mm')

    out = args.out or os.path.splitext(args.data)[0] + '_lidar_base.yaml'
    result = {
        'source': 'estimated',
        'note': (f'lidar_base_calib.py ({args.data}): 손-눈(LiDAR yaw, x, y) + 바닥 평면'
                 f'(카메라 roll, pitch, 높이) + LiDAR-카메라 관계 ({args.lidar_cam})'),
        'sensors': {'lidar': pose_entry('base_link', cm.LIDAR, t_base_lidar),
                    'camera': pose_entry('base_link', cm.SCREW, t_base_screw)},
        'hand_eye': {'yaw_deg': math.degrees(he['phi']), 'x_mm': he['p'][0] * 1000,
                     'y_mm': he['p'][1] * 1000, 'std_yaw_deg': math.degrees(he['std_phi']),
                     'std_x_mm': he['std_p'][0] * 1000, 'std_y_mm': he['std_p'][1] * 1000,
                     'pairs': used, 'rms_mm': he['rms_mm']},
        'floor': {'distance_mm': -d_c * 1000, 'normal_depth_frame': [float(v) for v in n_c],
                  'frames': int(use.sum()), 'frames_kind': args.floor_frames,
                  'body_tilt_deg': [math.degrees(v) for v in body]},
    }
    with open(out, 'w') as f:
        yaml.safe_dump(result, f, sort_keys=False, allow_unicode=True)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
