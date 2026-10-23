#!/usr/bin/env python3
"""
IMU 장착 각도(와 위치)를 정지·직진·회전 기록으로 추정한다 (세션 16).

입력: imu_mount_collect.py가 만든 NPZ

무엇으로 무엇을 아나
  1) 정지 중 가속도 = 중력 반대 방향 -> IMU가 "수평"에서 얼마나 기울었나 (roll, pitch)
     하지만 이 기울기 = 로봇 몸이 기운 각도 + IMU를 비뚤게 붙인 각도. 중력만으로는 못 나눈다.
  2) 몸 기울기는 따로 안다: 캐스터가 바퀴 바닥보다 1 mm 높아 몸이 뒤로 기우는 각도를
     설계 치수로 계산 (실제 로봇: 몸 판 위에 디지털 수평계, --body-tilt-deg로 입력)
     -> IMU 장착 roll, pitch = IMU 기울기에서 몸 기울기를 뺀 것
  3) yaw는 중력으로 안 보인다 (중력은 수직이라 수평으로 돌려도 그대로).
     직진 가감속 때 IMU가 느끼는 가속도 방향 = 로봇 앞쪽 -> IMU 축이 앞쪽에서 돌아간 각도
     (가속도를 바퀴 오도메트리로 구한 앞쪽 가속도에 직선 회귀: 기울기 벡터 = 앞쪽 방향,
      절편 = 중력, 그래서 중력이 섞여도 방향이 흐려지지 않는다)
     세션 16 Gazebo에서는 가감속 때 바퀴 접촉이 떨려 IMU 가속도가 엉망이었다 (검사로 표시)
  3b) 그래서 원 돌기를 추가: 일정 속도로 돌면 옆 방향 가속도 v·ω가 꾸준히 생긴다.
     왼쪽 원과 오른쪽 원의 평균을 빼면 중력, IMU 위치, 회전 중심 어긋남의 몫이 사라지고
     base y축(왼쪽) 방향만 남는다. 원 돌기 기록이 있으면 이것을 최종 yaw로 쓴다
  4) 위(1)와 앞(3) 또는 옆(3b) 두 방향으로 회전 행렬 하나를 만든다 (TRIAD: 두 벡터 맞추기)
  5) 교차 검증: 회전 중 자이로 축 방향 = 수직 -> 중력으로 구한 기울기와 같아야 한다
  6) 보너스: 제자리 회전 중 원심 가속도 (ω² × 회전 중심에서의 거리) -> IMU의 x, y 위치
     정확히는 "실제 회전 중심에서 본" 위치. 회전 중심이 바퀴 사이 가운데(base_link 원점)라고
     가정하고 base_link 기준으로 쓴다 (세션 16: Gazebo 바퀴 축이 0.046° 기울어 회전 중심이
     9 mm 옆으로 밀려 있던 것을 이 값으로 발견)

출력: 표, 그림 <입력>_imu.png, 추정값 YAML <입력>_imu.yaml (calibration.yaml의 sensors.imu 형식)
  채점: ros2 run tb3p_eval calib_error --estimate <입력>_imu.yaml --truth <sim_truth.yaml>

사용 예:
  python3 imu_mount_calib.py ~/imu_mount_data/session16.npz
      --truth ../../tb3p_gazebo/config/sim_truth.yaml                (실제로는 위와 한 줄로)
"""
import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import wheel_odom as wo
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
NOMINAL = os.path.join(HERE, '..', '..', 'tb3p_description', 'config', 'calibration.yaml')
# 몸 기울기 (설계 치수, tb3p_burger.urdf.xacro): 캐스터 공 중심 base_link 기준 x -0.081, z -0.004,
# 반지름 0.005 -> 공 바닥 z -0.009. 바퀴 바닥은 z -0.010 -> 캐스터가 1 mm 떠 있어
# 로봇이 바퀴 접지선을 축으로 뒤로 기울어 캐스터가 닿는다: pitch = -atan(0.001 / 0.081)
CASTER_X, CASTER_GAP = 0.081, 0.001
SMOOTH = 0.05              # 가속도 비교 전 이동 평균 창 (s), 두 번 적용


def rot_x(a):
    """X축 회전 행렬."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a):
    """Y축 회전 행렬."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rpy_to_matrix(roll, pitch, yaw):
    """URDF 약속 R = Rz(yaw) Ry(pitch) Rx(roll)."""
    c, s = math.cos(yaw), math.sin(yaw)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    return rz @ rot_y(pitch) @ rot_x(roll)


def matrix_to_rpy(r):
    """rpy_to_matrix의 역 (rad)."""
    return (math.atan2(r[2, 1], r[2, 2]), math.asin(-max(-1.0, min(1.0, r[2, 0]))),
            math.atan2(r[1, 0], r[0, 0]))


def quat_to_matrix(x, y, z, w):
    """쿼터니언 -> 회전 행렬."""
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tilt_of(up):
    """'위' 방향 벡터(센서 좌표계) -> 그 센서의 roll, pitch (rad, 수평 기준)."""
    return math.atan2(up[1], up[2]), math.atan2(-up[0], math.hypot(up[1], up[2]))


def triad(u_a, f_a, u_b, f_b):
    """
    두 벡터 맞추기(TRIAD): 좌표계 b의 벡터를 a로 옮기는 회전 R_ab (u는 정확히, f는 최대한).

    u: 주 벡터(위), f: 보조 벡터(앞). 각 좌표계에서 [u, u×f, u×(u×f)] 정규 직교 축을 만들어 맞춘다.
    """
    def frame(u, f):
        t1 = u / np.linalg.norm(u)
        t2 = np.cross(t1, f)
        t2 /= np.linalg.norm(t2)
        return np.column_stack([t1, t2, np.cross(t1, t2)])
    return frame(u_a, f_a) @ frame(u_b, f_b).T


def smooth(x, n):
    """중심 이동 평균 (n개, 양 끝은 있는 만큼), 열마다."""
    x = np.asarray(x, dtype=float)
    if n <= 1:
        return x
    k = np.ones(n) / n
    if x.ndim == 1:
        return np.convolve(x, k, mode='same') / np.convolve(np.ones(len(x)), k, mode='same')
    return np.column_stack([smooth(x[:, i], n) for i in range(x.shape[1])])


def phase_mask(t, phases, name, skip=0.0):
    """시각 t 중 이름이 name인 구간에 드는 것 (각 구간 앞 skip초는 뺀다)."""
    m = np.zeros(len(t), dtype=bool)
    for nm, t0, t1 in phases:
        if nm == name:
            m |= (t >= t0 + skip) & (t < t1)
    return m


def forward_velocity(t_imu, joint, wheels, n):
    """바퀴 각도 -> IMU 시각의 로봇 앞쪽 속도 (m/s) (보간, 미분, 이동 평균)."""
    t_j, idx = np.unique(joint[:, 0], return_index=True)
    phi_l = np.interp(t_imu, t_j, np.unwrap(joint[idx, 1]))
    phi_r = np.interp(t_imu, t_j, np.unwrap(joint[idx, 2]))
    dist = (phi_l * wheels['radius_left'] + phi_r * wheels['radius_right']) / 2
    return smooth(np.gradient(dist, t_imu), n)


def forward_accel(t_imu, joint, wheels, n):
    """바퀴 각도 -> IMU 시각의 로봇 앞쪽 가속도 (m/s²)."""
    return smooth(np.gradient(forward_velocity(t_imu, joint, wheels, n), t_imu), n)


def fit_forward(t, phases, a_fwd, acc):
    """
    직진 구간에서 IMU 가속도(3축) = 절편(중력) + k × 앞쪽 가속도 를 맞춘다.

    돌려주는 값: (k, 회귀 계수 2 x 3, 잔차 RMS, 직진마다 따로 구한 yaw 목록)
    """
    m = phase_mask(t, phases, 'move')
    design = np.column_stack([np.ones(m.sum()), a_fwd[m]])
    coef, *_ = np.linalg.lstsq(design, acc[m], rcond=None)
    rms = float(np.sqrt(np.mean((acc[m] - design @ coef) ** 2)))
    yaws = []
    for nm, t0, t1 in phases:
        if nm != 'move':
            continue
        mm = (t >= t0) & (t < t1)
        c, *_ = np.linalg.lstsq(np.column_stack([np.ones(mm.sum()), a_fwd[mm]]), acc[mm],
                                rcond=None)
        yaws.append(math.atan2(-c[1, 1], c[1, 0]))
    return coef[1], coef, rms, yaws


def analyze(data, wheels, body_tilt, imu_z):
    """
    기록 -> 추정 결과 dict.

    body_tilt: (roll, pitch) rad, 로봇 몸(base_link)이 수평에서 기운 각도
    imu_z: IMU 높이 (base_link 기준, m, 공칭값) - 위치(보너스) 계산에만 사용
    """
    imu, joint, phases = data['imu'], data['joint'], data['phases']
    t = imu[:, 0]
    acc, gyr = imu[:, 1:4], imu[:, 4:7]
    dt = np.median(np.diff(t))
    n = max(1, int(round(SMOOTH / dt)))
    out = {'n_imu': len(t), 'rate': 1.0 / dt}

    # 1) 정지: 중력 방향
    each = [acc[(t >= t0 + 1.0) & (t < t1)].mean(axis=0)
            for nm, t0, t1 in phases if nm == 'static']
    f_static = np.mean(each, axis=0)
    up_imu = f_static / np.linalg.norm(f_static)
    out['static_each'] = [tuple(np.degrees(tilt_of(f / np.linalg.norm(f)))) for f in each]
    out['g_norm'] = float(np.linalg.norm(f_static))
    out['tilt_imu'] = tilt_of(up_imu)

    # 몸 기울기로 base 좌표계의 위·앞 방향
    r_level_base = rot_y(body_tilt[1]) @ rot_x(body_tilt[0])
    up_base = r_level_base.T @ np.array([0.0, 0.0, 1.0])
    fwd_base = r_level_base.T @ np.array([1.0, 0.0, 0.0])

    a_fwd = forward_accel(t, joint, wheels, n)
    acc_s = smooth(smooth(acc, n), n)             # 앞쪽 가속도와 같은 정도로 부드럽게
    wz = smooth(gyr[:, 2], n)
    alpha = smooth(np.gradient(wz, t), n)         # 자이로로 잰 각가속도

    mt = phase_mask(t, phases, 'turn', skip=0.2)
    steady = mt & (np.abs(wz) > 0.8 * np.abs(wz[mt]).max()) & (np.abs(np.gradient(wz, t)) < 0.5)
    w2 = wz[steady] ** 2

    def lever(r_base_imu):
        # 원심 가속도 -ω² × (수평 위치) -> 수평 위치 h (x, y), 회전마다 따로 구한 값들
        a_h = (r_level_base @ r_base_imu @ (acc_s[steady] - f_static).T).T
        h = -(w2 @ a_h[:, :2]) / (w2 @ w2)
        per = []
        for nm, t0, t1 in phases:
            sel = (t[steady] >= t0) & (t[steady] < t1)
            if nm == 'turn' and sel.sum() > 5:
                ww = w2[sel]
                per.append(-(ww @ a_h[sel, :2]) / (ww @ ww))
        return h, np.array(per), a_h

    # 검사: 직진 한 번이 끝나면 다시 멈추므로 IMU 앞쪽 가속도를 적분한 속도 변화는 0이어야 한다.
    #   0이 아니면 가속도 기록이 실제 움직임과 안 맞는 것 (세션 16: 바퀴 모서리 접촉 떨림이
    #   1 kHz 물리를 200 Hz로 뽑는 사이 한쪽으로 치우쳐 남음)
    dv_end, v_peak = [], []
    for nm, t0, t1 in phases:
        if nm != 'move':
            continue
        mm = (t >= t0) & (t < t1 + 0.3)
        dv = np.cumsum(acc[mm, 0] - f_static[0]) * dt
        dv_end.append(abs(dv[-1]))
        v_peak.append(np.abs(np.cumsum(a_fwd[mm]) * dt).max())
    out['dv_check'] = (float(np.mean(dv_end)), float(np.mean(v_peak)))

    # 2) 직진 가감속 -> 앞쪽 방향, 3) TRIAD로 R_base_imu
    #    직진 중에도 좌우 바퀴 차이로 조금씩 돌기 시작하면(각가속도 α ∝ 가속도) IMU가 회전
    #    중심에서 떨어진 만큼 옆 가속도 α × x가 생겨 yaw처럼 보인다 (α가 가속도에 비례해서
    #    회귀로는 못 가른다). 그래서 위치(5)를 먼저 구해 그 몫을 빼고 다시 맞춘다 (2번 반복)
    acc_fit = acc_s
    for _ in range(3):
        k, coef, rms, yaws = fit_forward(t, phases, a_fwd, acc_fit)
        fwd_imu = k / np.linalg.norm(k)
        r_base_imu = triad(up_base, fwd_base, up_imu, fwd_imu)
        h, per, a_h = lever(r_base_imu)
        tan_level = np.column_stack([-alpha * h[1], alpha * h[0], np.zeros_like(alpha)])
        acc_fit = acc_s - (r_level_base @ r_base_imu).T.dot(tan_level.T).T
    out.update({'k': k, 'fwd_imu': fwd_imu, 'r_base_imu': r_base_imu,
                'rpy': matrix_to_rpy(r_base_imu),
                'yaw_only': math.atan2(-fwd_imu[1], fwd_imu[0]),
                'move_fit': (a_fwd[phase_mask(t, phases, 'move')],
                             acc_fit[phase_mask(t, phases, 'move')], coef, rms),
                'yaw_sem': float(np.std(yaws, ddof=1) / math.sqrt(len(yaws))) if len(yaws) > 1
                else 0.0,
                'n_moves': len(yaws)})

    # 3b) 원 돌기: 일정한 속도 v, 회전 ω로 돌면 IMU 가속도 = v·ω × (base y축) - ω² × h
    #     (h = 회전 중심에서 본 IMU 수평 위치, 5에서 제자리 회전으로 구함).
    #     ω² × h 몫을 빼고, 왼쪽 원 평균 - 오른쪽 원 평균 = 2 v|ω| × (base y축) (중력은 빼면 사라짐)
    #     -> IMU 좌표계에서 base y축 방향. 좌우 바퀴가 달라 두 방향의 ω가 조금 다르므로
    #     ω² 몫을 원마다 빼야 한다 (안 빼면 약 0.06° 치우침)
    out['arc'] = None
    arcs = [(t0, t1) for nm, t0, t1 in phases if nm == 'arc']
    if len(arcs) >= 2:
        v_fwd = forward_velocity(t, joint, wheels, n)
        raw, signs, vw, w2_arc = [], [], [], []
        for t0, t1 in arcs:
            mm = (t >= t0 + 1.1) & (t < t1 - 0.7)          # 올리고 내리는 구간과 여유 0.5 s 제외
            raw.append((acc[mm] - f_static).mean(axis=0))
            signs.append(np.sign(gyr[mm, 2].mean()))
            vw.append(abs(v_fwd[mm].mean() * gyr[mm, 2].mean()))
            w2_arc.append(np.mean(gyr[mm, 2] ** 2))
        raw, signs, w2_arc = np.array(raw), np.array(signs), np.array(w2_arc)
        lat_base = r_level_base.T @ np.array([0.0, 1.0, 0.0])
        r_arc = r_base_imu
        for _ in range(3):
            lever_imu = (r_level_base @ r_arc).T @ np.r_[h, 0.0]   # h를 IMU 축으로
            means = raw + w2_arc[:, None] * lever_imu[None, :]     # - (-ω² h) 몫 제거
            d = means[signs > 0].mean(axis=0) - means[signs < 0].mean(axis=0)
            lat_imu = d / np.linalg.norm(d)
            r_arc = triad(up_base, lat_base, up_imu, lat_imu)
            h, per, a_h = lever(r_arc)
        pair_yaws = []                                  # 왼쪽·오른쪽 한 쌍씩 -> 흩어짐
        for i in range(0, len(means) - 1, 2):
            if signs[i] != signs[i + 1]:
                dd = (means[i] - means[i + 1]) * signs[i]
                pair_yaws.append(math.atan2(dd[0], dd[1]))
        sem = (float(np.std(pair_yaws, ddof=1) / math.sqrt(len(pair_yaws)))
               if len(pair_yaws) > 1 else 0.0)
        out['arc'] = {'lat_imu': lat_imu, 'r': r_arc, 'rpy': matrix_to_rpy(r_arc),
                      'yaw_only': math.atan2(lat_imu[0], lat_imu[1]), 'yaw_sem': sem,
                      'measured': float(np.linalg.norm(d) / 2), 'expected': float(np.mean(vw)),
                      'n': len(arcs)}
        # 원 돌기 결과를 최종으로 (가감속은 Gazebo 접촉 떨림에 약함), 위치도 다시 계산
        out['accel_rpy'] = out['rpy']
        out['r_base_imu'], out['rpy'] = r_arc, matrix_to_rpy(r_arc)

    # 4) 교차 검증: 회전 중 자이로 축 = 수직 방향
    w = gyr[mt]
    big = np.abs(w[:, 2]) > 0.5 * np.abs(w[:, 2]).max()
    axis = (w[big] * np.sign(w[big, 2])[:, None]).sum(axis=0)
    out['tilt_gyro'] = tilt_of(axis / np.linalg.norm(axis))

    # 5) 보너스: IMU 위치. 원심 가속도는 수평면에서 생기므로 수평 위치가 나온다.
    #    몸이 기울어 있으면 IMU 높이 z가 수평 위치를 z·sin(pitch)만큼 바꾸므로
    #    (70 mm × sin 0.7° ≈ 0.9 mm) 알려진 z로 base 좌표로 되돌린다
    def to_base(hh):
        return np.linalg.solve(r_level_base[:2, :2], hh - r_level_base[:2, 2] * imu_z)

    out['lever'] = tuple(to_base(h))
    per_b = np.array([to_base(p) for p in per])
    out['lever_sem'] = (per_b.std(axis=0, ddof=1) / math.sqrt(len(per_b)) if len(per_b) > 1
                        else np.zeros(2))
    out['lever_fit'] = (w2, a_h)
    out['n_steady'] = int(steady.sum())
    return out


def truth_tilt(data, phases):
    """(시뮬레이션) 정지 구간의 실제 몸 기울기 (roll, pitch rad), /truth/odom."""
    tr = data['truth']
    if len(tr) == 0:
        return None
    m = phase_mask(tr[:, 0], phases, 'static', skip=1.0)
    q = tr[m, 4:8].mean(axis=0)
    q /= np.linalg.norm(q)
    r = quat_to_matrix(*q)
    up = r.T @ np.array([0.0, 0.0, 1.0])
    return tilt_of(up)


def main():
    """명령줄에서 실행할 때의 시작점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data', help='imu_mount_collect.py가 만든 NPZ')
    ap.add_argument('--wheels', default=NOMINAL, help='앞쪽 가속도 계산에 쓸 바퀴 값 YAML')
    ap.add_argument('--body-tilt-deg', type=float, nargs=2, default=None,
                    metavar=('ROLL', 'PITCH'),
                    help='몸 기울기 실측값 (°). 없으면 캐스터 설계 치수로 계산')
    ap.add_argument('--nominal', default=NOMINAL, help='공칭 캘리브레이션 (IMU z 위치, 비교용)')
    ap.add_argument('--truth', default='', help='(시뮬레이션 채점 전용) sim_truth.yaml')
    args = ap.parse_args()

    raw = np.load(args.data)
    data = {'imu': raw['imu'], 'joint': raw['joint'], 'truth': raw['truth'],
            'phases': json.loads(str(raw['meta']))['phases']}
    if args.body_tilt_deg:
        body = tuple(math.radians(v) for v in args.body_tilt_deg)
        body_src = '실측값 (--body-tilt-deg)'
    else:
        body = (0.0, -math.atan2(CASTER_GAP, CASTER_X))
        body_src = f'설계 치수 (캐스터 {CASTER_GAP * 1000:.0f} mm 뜸, 바퀴에서 {CASTER_X * 1000:.0f} mm)'
    with open(args.nominal) as f:
        nominal = yaml.safe_load(f)['sensors']['imu']
    res = analyze(data, wo.load_wheels(args.wheels), body, nominal['xyz'][2])
    deg = math.degrees

    n_imu, rate, g_norm, k = res['n_imu'], res['rate'], res['g_norm'], res['k']
    print(f'IMU {n_imu}개 ({rate:.0f} Hz)')
    ti = res['tilt_imu']
    print('[1] 정지 중 중력 -> IMU의 수평 대비 기울기 (몸 기울기 + 장착 각도가 섞인 값)')
    for i, (r, p) in enumerate(res['static_each']):
        print(f'    정지 {i + 1}: roll {r:+.3f}°, pitch {p:+.3f}°')
    print(f'    평균  : roll {deg(ti[0]):+.3f}°, pitch {deg(ti[1]):+.3f}°, '
          f'|g| {g_norm:.4f} m/s²')
    tg = res['tilt_gyro']
    print(f'    교차 검증 (회전 중 자이로 축): roll {deg(tg[0]):+.3f}°, pitch {deg(tg[1]):+.3f}°')

    print(f'[2] 로봇 몸 기울기: roll {deg(body[0]):+.3f}°, pitch {deg(body[1]):+.3f}°  ({body_src})')
    t_body = truth_tilt(data, data['phases'])
    if t_body is not None:
        print(f'    (시뮬레이션 확인) Gazebo 실제 몸 기울기: roll {deg(t_body[0]):+.3f}°, '
              f'pitch {deg(t_body[1]):+.3f}°')

    fwd = res['fwd_imu']
    _, _, coef, rms = res['move_fit']
    print('[3] 직진 가감속 -> 로봇 앞쪽이 IMU 좌표계에서 가리키는 방향')
    dv_end, v_peak = res['dv_check']
    ratio = dv_end / max(v_peak, 1e-9)
    verdict = '정상' if ratio < 0.2 else '이상: IMU 가속도가 움직임과 안 맞음 (yaw를 믿지 말 것)'
    print(f'    검사: 직진마다 IMU 가속도 적분 속도 변화 {dv_end:.3f} m/s '
          f'(최고 속도 {v_peak:.3f} m/s의 {ratio * 100:.0f}%, 0이어야 함) -> {verdict}')
    print(f'    앞쪽 1 m/s²당 IMU 가속도 ({k[0]:+.4f}, {k[1]:+.4f}, {k[2]:+.4f}), '
          f'크기 {np.linalg.norm(k):.4f}, 회귀 잔차 {rms:.4f} m/s²')
    yaw_only, yaw_sem, n_mv = deg(res['yaw_only']), deg(res['yaw_sem']), res['n_moves']
    print(f'    앞쪽 방향 ({fwd[0]:+.5f}, {fwd[1]:+.5f}, {fwd[2]:+.5f}) '
          f'-> yaw만 보면 {yaw_only:+.3f}° ± {yaw_sem:.3f}° (직진 {n_mv}번의 흩어짐으로 1σ)')

    arc = res['arc']
    if arc:
        lat = arc['lat_imu']
        ratio = arc['measured'] / max(arc['expected'], 1e-9)
        verdict = '정상' if abs(ratio - 1) < 0.1 else '이상: 옆 가속도가 v·ω와 안 맞음'
        y_only, y_sem = deg(arc['yaw_only']), deg(arc['yaw_sem'])
        print(f'[3b] 원 돌기 {arc["n"]}번 -> base y축(왼쪽)이 IMU 좌표계에서 가리키는 방향')
        print(f'    검사: 옆 가속도 {arc["measured"]:.4f} m/s², 바퀴·자이로로 계산한 v·ω '
              f'{arc["expected"]:.4f} m/s² (비율 {ratio:.3f}, 1이어야 함) -> {verdict}')
        print(f'    옆 방향 ({lat[0]:+.5f}, {lat[1]:+.5f}, {lat[2]:+.5f}) '
              f'-> yaw만 보면 {y_only:+.3f}° ± {y_sem:.3f}° (왼쪽·오른쪽 쌍의 흩어짐으로 1σ)')
        ar = res['accel_rpy']
        print(f'    (참고: 가감속으로 구한 장착 각도 roll {deg(ar[0]):+.3f}°, '
              f'pitch {deg(ar[1]):+.3f}°, yaw {deg(ar[2]):+.3f}°)')
    rpy = res['rpy']
    src = '원 돌기' if arc else '직진 가감속'
    print(f'[4] IMU 장착 각도 (base_link 기준, TRIAD: 중력 + {src})')
    print(f'    roll {deg(rpy[0]):+.3f}°, pitch {deg(rpy[1]):+.3f}°, yaw {deg(rpy[2]):+.3f}°')
    rx, ry = res['lever']
    sx, sy = res['lever_sem'] * 1000
    n_st = res['n_steady']
    print(f'[5] 보너스: 회전 중심에서 본 IMU 위치 (원심 가속도, 정상 회전 {n_st}개 샘플): '
          f'x {rx * 1000:+.1f} ± {sx:.1f} mm, y {ry * 1000:+.1f} ± {sy:.1f} mm')

    if args.truth:
        with open(args.truth) as f:
            truth = yaml.safe_load(f)['sensors']['imu']
        print('[정답과 비교] (시뮬레이션 채점 전용)')
        print('    항목             공칭      추정      정답    추정 오차')
        nom_rpy = nominal['rpy']
        rows = [(f'{nm} °', deg(nom_rpy[i]), deg(rpy[i]), deg(truth['rpy'][i]))
                for i, nm in enumerate(('roll', 'pitch', 'yaw'))]
        rows += [(f'{nm} mm', nominal['xyz'][i] * 1000, (rx, ry)[i] * 1000,
                  truth['xyz'][i] * 1000) for i, nm in enumerate(('x', 'y'))]
        for name, nom, est, tru in rows:
            print(f'    {name:10s} {nom:+9.3f} {est:+9.3f} {tru:+9.3f} {est - tru:+11.3f}')
        r_true = rpy_to_matrix(*truth['rpy'])

        def angle_deg(r):
            return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2))))
        err = angle_deg(r_true.T @ res['r_base_imu'])
        err_nom = angle_deg(r_true.T @ rpy_to_matrix(*nom_rpy))
        print(f'    회전 오차 크기: 공칭 {err_nom:.3f}° -> 추정 {err:.3f}°')

    base = os.path.splitext(args.data)[0]
    est = {'source': 'estimated',
           'note': f'imu_mount_calib.py ({args.data}), 몸 기울기 {body_src}, z는 공칭',
           'sensors': {'imu': {'parent': 'base_link', 'child': 'imu_link',
                               'xyz': [round(float(rx), 5), round(float(ry), 5),
                                       float(nominal['xyz'][2])],
                               'rpy': [round(float(v), 6) for v in rpy],
                               'time_offset': 0.0}}}
    with open(base + '_imu.yaml', 'w') as f:
        yaml.safe_dump(est, f, sort_keys=False, allow_unicode=True)
    print(f'saved {base}_imu.yaml')
    plot(base + '_imu.png', res)
    print(f'saved {base}_imu.png')


def plot(path, res):
    """왼쪽: 직진 중 IMU 가속도 vs 앞쪽 가속도, 오른쪽: 회전 중 원심 가속도 vs ω²."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    a_fwd, acc, coef, _ = res['move_fit']
    xs = np.linspace(a_fwd.min(), a_fwd.max(), 2)
    for i, (name, color) in enumerate((('x', 'tab:red'), ('y', 'tab:green'))):
        ax1.scatter(a_fwd[::5], acc[::5, i] - coef[0, i], s=3, color=color, alpha=0.3)
        ax1.plot(xs, coef[1, i] * xs, color=color,
                 label=f'IMU {name}: slope {coef[1, i]:+.4f}')
    yaw = math.degrees(math.atan2(-coef[1, 1], coef[1, 0]))
    ax1.set_title(f'straight moves: IMU accel vs wheel forward accel (yaw {yaw:+.2f} deg)',
                  fontsize=10)
    ax1.set_xlabel('forward acceleration from wheels (m/s^2)')
    ax1.set_ylabel('IMU acceleration minus gravity part (m/s^2)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    w2, a_b = res['lever_fit']
    rx, ry = res['lever']
    xs = np.linspace(0, w2.max(), 2)
    for i, (name, r, color) in enumerate((('x', rx, 'tab:red'), ('y', ry, 'tab:green'))):
        ax2.scatter(w2[::3], a_b[::3, i], s=3, color=color, alpha=0.3)
        ax2.plot(xs, -r * xs, color=color, label=f'base {name}: IMU at {r * 1000:+.1f} mm (fit)')
    ax2.set_title('in-place turns: centripetal acceleration = -omega^2 x position', fontsize=10)
    ax2.set_xlabel('omega^2 (rad^2/s^2)')
    ax2.set_ylabel('IMU acceleration, horizontal axes, gravity removed (m/s^2)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == '__main__':
    main()
