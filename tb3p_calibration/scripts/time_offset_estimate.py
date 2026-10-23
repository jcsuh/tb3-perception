#!/usr/bin/env python3
"""
IMU·바퀴·LiDAR의 회전 기록을 맞춰 센서 사이 시간 오프셋을 추정한다 (세션 17).

입력: time_sync_collect.py가 만든 NPZ

세 센서의 "누적 회전 각도"
  - IMU: 자이로 z를 시간으로 적분
  - 바퀴: (오른쪽 반지름 × 오른쪽 바퀴 각도 - 왼쪽 반지름 × 왼쪽 바퀴 각도) / 간격
  - LiDAR: 연속 스캔 매칭(scan_match.py)으로 구한 회전을 차례로 더함

두 가지 방법으로 각 센서가 IMU보다 얼마나 밀렸나(δ)를 찾는다
  1) 상호상관(빠른 확인): 회전 속도 그래프 두 개를 옆으로 밀어 가며 가장 잘 겹치는 밀림
  2) 구간 증가량 맞추기(최종): 센서가 잰 짧은 구간(약 0.1 s)의 회전 증가량과, 같은 구간을 δ만큼
     밀어서 잰 IMU 증가량을 비교해 차이 제곱합이 가장 작은 δ (배율·자이로 바이어스도 함께 맞춤).
     LiDAR처럼 10 Hz로 띄엄띄엄 재는 센서도 그 구간을 정확히 비교해서 몇 ms까지 맞출 수 있다

부호 약속 (calibration.yaml의 time_offset = "센서 시각에 더해야 할 보정값")
  센서 S의 시각이 실제보다 d만큼 늦게 찍혔다면 S(t) = 실제(t - d) = IMU(t - d) -> δ_S = -d
  즉 결과 δ_S가 바로 S의 time_offset (IMU 기준, IMU = 0)

출력: 표, 그림 <입력>_sync.png, YAML <입력>_time_offset.yaml
  (시뮬레이션) --injected-ms로 delay_injector.py에 넣은 지연을 주면 정답과 비교한다

사용 예:
  python3 time_offset_estimate.py ~/time_sync_data/session17.npz
      --wheels ~/umbmark_data/before_wheels_ls.yaml --injected-ms imu=0 joint_states=30 scan=70
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
import scan_match as sm
import wheel_odom as wo
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
NOMINAL = os.path.join(HERE, '..', '..', 'tb3p_description', 'config', 'calibration.yaml')
MAX_LAG = 0.3          # 찾는 범위 ±0.3 s
WINDOW = 0.1           # 구간 증가량 길이 (s)


def cumulative(t, rate):
    """속도 기록을 사다리꼴 적분한 누적 각도."""
    return np.concatenate([[0.0], np.cumsum(np.diff(t) * (rate[1:] + rate[:-1]) / 2)])


def angle_signals(data, wheels, log=print):
    """세 센서의 누적 회전 각도: {'imu': (t, θ), 'wheel': (t, θ), 'lidar': (t, θ)} + 진단 정보."""
    imu = data['imu']
    t_g, w_g = imu[:, 0], imu[:, 3]
    out = {'imu': (t_g, cumulative(t_g, w_g))}

    joint = data['joint']
    t_j, idx = np.unique(joint[:, 0], return_index=True)
    phi_l, phi_r = np.unwrap(joint[idx, 1]), np.unwrap(joint[idx, 2])
    th_w = (wheels['radius_right'] * phi_r - wheels['radius_left'] * phi_l) / wheels['separation']
    keep = np.r_[0, np.nonzero(np.diff(np.floor(t_j / 0.005)))[0] + 1]   # 200 Hz 정도로 솎기
    out['wheel'] = (t_j[keep], th_w[keep] - th_w[0])

    info = data['scan_info']
    n_pairs = len(data['scan_t']) - 1
    log(f'LiDAR 스캔 매칭 {n_pairs}쌍 ...')
    dth, used, res = sm.scan_rotations(data['scan_r'], info['angle_min'],
                                       info['angle_increment'])
    out['lidar'] = (data['scan_t'], np.concatenate([[0.0], np.cumsum(dth)]))
    diag = {'icp_points': float(np.median(used)), 'icp_rms_mm': float(np.nanmedian(res) * 1000)}
    return out, diag


def increment_cost(t_s, th_s, ref, delta, step):
    """
    구간 증가량 비교: 센서 S의 [t_k, t_k+step] 증가량 vs 기준의 [t_k+δ, t_k+step+δ] 증가량.

    증가량_S = 배율 × 증가량_기준 + 바이어스 × 구간 길이 를 최소제곱으로 맞춘 잔차 제곱합 (rad²).
    """
    t_r, th_r = ref
    d_s = th_s[step:] - th_s[:-step]
    dt = t_s[step:] - t_s[:-step]
    d_r = np.interp(t_s[step:] + delta, t_r, th_r) - np.interp(t_s[:-step] + delta, t_r, th_r)
    a = np.column_stack([d_r, dt])
    coef, *_ = np.linalg.lstsq(a, d_s, rcond=None)
    r = d_s - a @ coef
    return float(r @ r), coef


def fit_offset(sig, ref, window=WINDOW):
    """센서 신호를 기준에 맞추는 δ (s), 비용 곡선 (그림용), 배율."""
    t_s, th_s = sig
    t_r = ref[0]
    m = (t_s > t_r[0] + MAX_LAG) & (t_s < t_r[-1] - MAX_LAG)    # 밀어도 기준 범위 안
    t_s, th_s = t_s[m], th_s[m]
    step = max(1, int(round(window / np.median(np.diff(t_s)))))
    lags = np.arange(-MAX_LAG, MAX_LAG + 1e-9, 0.002)
    costs = np.array([increment_cost(t_s, th_s, ref, d, step)[0] for d in lags])
    i = int(np.argmin(costs))
    lo, hi = lags[max(i - 2, 0)], lags[min(i + 2, len(lags) - 1)]
    sol = minimize_scalar(lambda d: increment_cost(t_s, th_s, ref, d, step)[0],
                          bounds=(lo, hi), method='bounded', options={'xatol': 1e-5})
    _, coef = increment_cost(t_s, th_s, ref, sol.x, step)
    return float(sol.x), (lags, costs), float(coef[0])


def rate_on_grid(sig, grid, smooth_s=0.05):
    """누적 각도 -> 균일 시각의 회전 속도 (구간 기울기, 이동 평균)."""
    t, th = sig
    rate = np.interp(grid, (t[1:] + t[:-1]) / 2, np.diff(th) / np.diff(t))
    n = max(1, int(round(smooth_s / (grid[1] - grid[0]))))
    return np.convolve(rate, np.ones(n) / n, mode='same')


def xcorr_offset(sig, ref, fs=200.0):
    """상호상관: 회전 속도 그래프를 밀어 가며 가장 잘 겹치는 δ (s), 포물선 보간."""
    t0 = max(sig[0][0], ref[0][0]) + MAX_LAG + 1.0
    t1 = min(sig[0][-1], ref[0][-1]) - MAX_LAG - 1.0
    grid = np.arange(t0, t1, 1.0 / fs)
    a = rate_on_grid(sig, grid)
    a -= a.mean()
    lags = np.arange(-int(MAX_LAG * fs), int(MAX_LAG * fs) + 1)
    vals = []
    for k in lags:
        b = rate_on_grid(ref, grid + k / fs)
        b -= b.mean()
        vals.append(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    vals = np.array(vals)
    i = int(np.argmax(vals))
    frac = 0.0
    if 0 < i < len(vals) - 1:
        y0, y1, y2 = vals[i - 1], vals[i], vals[i + 1]
        frac = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return (lags[i] + frac) / fs, float(vals[i])


def truth_signal(data):
    """(시뮬레이션) /truth/odom의 누적 yaw."""
    tr = data['truth']
    if len(tr) == 0:
        return None
    x, y, z, w = tr[:, 3], tr[:, 4], tr[:, 5], tr[:, 6]
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return tr[:, 0], yaw - yaw[0]


def parse_injected(items):
    """['imu=0', 'scan=70'] -> {'imu': 0.0, 'lidar': 70.0, ...} (센서 이름으로)."""
    names = {'imu': 'imu', 'joint_states': 'wheel', 'scan': 'lidar'}
    out = {}
    for it in items or []:
        k, v = it.split('=')
        out[names[k]] = float(v)
    return out


def main():
    """명령줄에서 실행할 때의 시작점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data', help='time_sync_collect.py가 만든 NPZ')
    ap.add_argument('--wheels', default=NOMINAL, help='바퀴 값 YAML (세션 15 결과 권장)')
    ap.add_argument('--injected-ms', nargs='*', default=None,
                    help='(시뮬레이션 채점) delay_injector.py에 준 지연, 예: imu=0 scan=70')
    args = ap.parse_args()

    raw = np.load(args.data)
    meta = json.loads(str(raw['meta']))
    data = {k: raw[k] for k in ('imu', 'joint', 'truth', 'scan_t', 'scan_r')}
    data['scan_info'] = meta['scan_info']
    sigs, diag = angle_signals(data, wo.load_wheels(args.wheels))
    n_pts, icp_rms = diag['icp_points'], diag['icp_rms_mm']
    print(f'    ICP 맞춘 점 (중앙값) {n_pts:.0f}개, 잔차 {icp_rms:.1f} mm')
    ref = sigs['imu']

    results = {}
    print('[IMU 기준 시간 오프셋]  δ = 그 센서 시각에 더해야 할 보정값 (ms)')
    print('    센서      상호상관   구간 맞추기   배율(센서/IMU)')
    for name in ('wheel', 'lidar'):
        d_x, corr = xcorr_offset(sigs[name], ref)
        d_f, curve, scale = fit_offset(sigs[name], ref)
        results[name] = {'xcorr': d_x, 'fit': d_f, 'curve': curve, 'scale': scale}
        print(f'    {name:6s} {d_x * 1000:+10.1f} {d_f * 1000:+12.2f} {scale:13.4f}  '
              f'(상관 {corr:.3f})')
    d_lw, _, _ = fit_offset(sigs['lidar'], sigs['wheel'])
    diff = results['lidar']['fit'] - results['wheel']['fit']
    print(f'    교차 확인: LiDAR를 바퀴에 바로 맞추면 {d_lw * 1000:+.2f} ms, '
          f'두 결과의 차이 {diff * 1000:+.2f} ms (어긋남 {(d_lw - diff) * 1000:+.2f} ms)')

    tru = truth_signal(data)
    if tru is not None:
        print('[(시뮬레이션) Gazebo 정답 회전 기준 오프셋] 지연을 안 넣은 센서도 0이 맞는지')
        for name in ('imu', 'wheel', 'lidar'):
            d, _, _ = fit_offset(sigs[name], tru)
            print(f'    {name:6s} {d * 1000:+8.2f} ms')

    injected = parse_injected(args.injected_ms)
    if injected:
        base = injected.get('imu', 0.0)
        print('[정답과 비교] 넣은 지연 d -> 정답 δ = -(d_센서 - d_IMU)')
        print('    센서      추정 ms    정답 ms    오차 ms')
        for name in ('wheel', 'lidar'):
            if name in injected:
                true = -(injected[name] - base)
                est = results[name]['fit'] * 1000
                print(f'    {name:6s} {est:+9.2f} {true:+10.2f} {est - true:+10.2f}')

    stem = os.path.splitext(args.data)[0]
    out = {'source': 'estimated', 'note': f'time_offset_estimate.py ({args.data}), IMU 기준',
           'time_offset_s': {'imu': 0.0, 'wheels': round(results['wheel']['fit'], 5),
                             'lidar': round(results['lidar']['fit'], 5)}}
    with open(stem + '_time_offset.yaml', 'w') as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)
    print(f'saved {stem}_time_offset.yaml')
    plot(stem + '_sync.png', sigs, results)
    print(f'saved {stem}_sync.png')


def plot(path, sigs, results):
    """위: 회전 속도 (원래 시각), 가운데: 보정한 시각, 아래: δ에 따른 비용 곡선."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    t_g = sigs['imu'][0]
    t0 = t_g[0] + 0.45 * (t_g[-1] - t_g[0])
    grid = np.arange(t0, t0 + 6.0, 0.005)
    colors = {'imu': 'k', 'wheel': 'tab:blue', 'lidar': 'tab:red'}
    for ax, shifted in ((axes[0], False), (axes[1], True)):
        for name in ('imu', 'wheel', 'lidar'):
            d = results[name]['fit'] if shifted and name != 'imu' else 0.0
            t, th = sigs[name]
            ax.plot(grid, rate_on_grid((t + d, th), grid, 0.0 if name == 'lidar' else 0.02),
                    color=colors[name], lw=1.0, label=f'{name} (shift {d * 1000:+.1f} ms)')
        ax.set_ylabel('yaw rate (rad/s)')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    axes[0].set_title('yaw rate from each sensor, raw time stamps')
    axes[1].set_title('after adding the estimated time offset')
    axes[1].set_xlabel('time (s)')
    for name in ('wheel', 'lidar'):
        lags, costs = results[name]['curve']
        axes[2].plot(lags * 1000, costs / costs.min(), color=colors[name], label=name)
        axes[2].axvline(results[name]['fit'] * 1000, color=colors[name], ls=':')
    axes[2].set_yscale('log')
    axes[2].set_xlabel('time offset delta (ms)')
    axes[2].set_ylabel('increment mismatch (relative)')
    axes[2].set_title('increment mismatch vs time offset (minimum = estimate)')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == '__main__':
    main()
