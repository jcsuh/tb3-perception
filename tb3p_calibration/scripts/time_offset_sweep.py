#!/usr/bin/env python3
"""
주입 지연 10–100 ms와 기록 길이에 따라 시간 오프셋 복원 오차가 어떻게 변하는지 잰다 (세션 18).

기록 하나(time_sync_collect.py)로 조건을 바꿔 가며 time_offset_estimate.py의 추정을 반복한다.
  - 지연: 바퀴·LiDAR 시각에 d ms를 더 늦춰 찍은 것처럼 밀어 놓고 (d = 0, 10, ..., 100) 다시 추정
    지연을 안 넣은 기록(delay_injector.py --delays imu=0 joint_states=0 scan=0)을 주면
    d가 곧 넣은 지연이 된다 (지연을 넣은 기록이면 그만큼 더해진 범위를 본다)
    시각만 밀리는 문제라 Gazebo에서 delay_injector.py로 넣은 것과 같다
    (--online으로 Gazebo에서 직접 넣어 잰 기록을 함께 주면 같은 그래프에 점으로 찍어 확인)
  - 기록 길이: 회전 구간을 10, 20, 30초 조각과 전체로 나눠 조각마다 따로 추정 -> 흩어짐
  - 시각 흔들림(jitter): 실제 로봇은 메시지마다 시각이 몇 ms씩 들쭉날쭉 찍힌다.
    바퀴·LiDAR 시각에 σ = 1, 3, 10 ms 무작위 흔들림을 더해 20초 조각에서 반복 (지연 50 ms)

채점 기준 (시뮬레이션)
  각 센서를 Gazebo 정답 회전(/truth/odom)에 직접 맞춰 센서 고유의 시각 어긋남을 먼저 잰다
  (세션 17: IMU 자체가 약 1.3 ms 늦게 찍힘). IMU 기준 정답 = (센서 고유 - IMU 고유) - 더 민 d
  -> 추정 - 정답 = 복원 오차. 완료 기준: 모든 지연에서 ±5 ms 이내

출력: 표, 그림 <입력>_sweep.png, 원자료 <입력>_sweep.csv

사용 예:
  python3 time_offset_sweep.py ~/time_sync_data/base.npz
      --wheels ~/umbmark_data/before2_wheels_ls.yaml                (실제로는 위와 한 줄로)
  (Gazebo 직접 주입 결과도 함께)
  python3 time_offset_sweep.py ~/time_sync_data/base.npz
      --online ~/time_sync_data/on_10_100.npz:joint_states=10,scan=100       (실제로는 위와 한 줄로)
"""
import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import time_offset_estimate as toe
import wheel_odom as wo

DELAYS_MS = list(range(0, 101, 10))
LENGTHS_S = [10.0, 20.0, 30.0, 0.0]      # 0 = 전체
PASS_MS = 5.0
SENSORS = ('wheel', 'lidar')


def load(path, wheels):
    """NPZ -> (각도 신호들, 정답 회전 신호, 회전 구간 (시작, 끝))."""
    raw = np.load(path)
    meta = json.loads(str(raw['meta']))
    data = {k: raw[k] for k in ('imu', 'joint', 'truth', 'scan_t', 'scan_r')}
    data['scan_info'] = meta['scan_info']
    sigs, _ = toe.angle_signals(data, wheels, log=lambda s: None)
    tru = toe.truth_signal(data)
    imu = data['imu']
    moving = np.nonzero(np.abs(imu[:, 3]) > 0.05)[0]
    active = (imu[moving[0], 0] + 2.0, imu[moving[-1], 0] - 2.0)     # 처음·끝 2초(키우기·줄이기) 제외
    return sigs, tru, active


def clip(sig, t0, t1):
    """신호에서 [t0, t1] 구간만."""
    t, th = sig
    m = (t >= t0) & (t <= t1)
    return t[m], th[m]


def inherent_offsets(sigs, tru):
    """(시뮬레이션) 센서마다 정답 회전에 직접 맞춘 고유 오프셋 (s)."""
    return {name: toe.fit_offset(sigs[name], tru)[0] for name in ('imu',) + SENSORS}


def estimate(sigs, t0, t1, shift):
    """
    [t0, t1] 구간에서 바퀴·LiDAR 시각을 shift(s)만큼 늦춘 뒤 IMU 기준 오프셋 추정.

    IMU는 밀어 볼 여유를 위해 구간 앞뒤로 조금 더 쓴다.
    """
    ref = clip(sigs['imu'], t0 - toe.MAX_LAG - 0.1, t1 + toe.MAX_LAG + 0.1)
    out = {}
    for name in SENSORS:
        t, th = clip(sigs[name], t0, t1)
        out[name] = toe.fit_offset((t + shift, th), ref)[0]
    return out


def chunks(active, length):
    """회전 구간을 length초 조각으로 (0이면 전체 하나)."""
    t0, t1 = active
    if length <= 0 or length >= t1 - t0:
        return [(t0, t1)]
    n = int((t1 - t0) // length)
    return [(t0 + i * length, t0 + (i + 1) * length) for i in range(n)]


JITTER_MS = [1.0, 3.0, 10.0]


def jitter_test(sigs, active, seeds=5, delay_ms=50.0, length=20.0):
    """시각 흔들림 σ마다 20초 조각 × 무작위 반복의 추정값 (ms) 목록 {σ: {센서: [...]}}."""
    out = {}
    for sigma in JITTER_MS:
        out[sigma] = {name: [] for name in SENSORS}
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            noisy = dict(sigs)
            for name in SENSORS:
                t, th = sigs[name]
                tj = np.sort(t + rng.normal(0.0, sigma / 1000, len(t)))   # 순서는 유지
                noisy[name] = (tj, th)
            for t0, t1 in chunks(active, length):
                est = estimate(noisy, t0, t1, delay_ms / 1000)
                for name in SENSORS:
                    out[sigma][name].append(est[name] * 1000)
    return out


def parse_online(item):
    """'path.npz:joint_states=10,scan=100' -> (path, {'wheel': 10, 'lidar': 100, 'imu': 0})."""
    path, spec = item.rsplit(':', 1)
    names = {'imu': 'imu', 'joint_states': 'wheel', 'scan': 'lidar'}
    inj = {'imu': 0.0}
    for kv in spec.split(','):
        k, v = kv.split('=')
        inj[names[k]] = float(v)
    return os.path.expanduser(path), inj


def main():
    """명령줄에서 실행할 때의 시작점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data', help='time_sync_collect.py가 만든 NPZ (지연은 아무 값이어도 됨)')
    ap.add_argument('--wheels', default=toe.NOMINAL, help='바퀴 값 YAML')
    ap.add_argument('--online', nargs='*', default=[],
                    help='Gazebo에서 직접 지연을 넣어 잰 기록: 파일:joint_states=ms,scan=ms')
    args = ap.parse_args()
    wheels = wo.load_wheels(args.wheels)

    print('스캔 매칭과 정답 기준 고유 오프셋 계산 ...')
    sigs, tru, active = load(args.data, wheels)
    if tru is None:
        raise SystemExit('/truth/odom 기록이 없음: 시뮬레이션 기록이 필요 (채점 기준)')
    inh = inherent_offsets(sigs, tru)
    print('[센서 고유 오프셋 (정답 회전 기준, 이 기록에 넣은 지연 포함)]')
    for name, v in inh.items():
        print(f'    {name:6s} {v * 1000:+8.2f} ms')
    print(f'    회전 구간 {active[1] - active[0]:.0f}초')

    rows = []      # (길이, 조각, 지연 ms, 센서, 추정 ms, 정답 ms, 오차 ms)
    for length in LENGTHS_S:
        for ci, (t0, t1) in enumerate(chunks(active, length)):
            for d in DELAYS_MS:
                est = estimate(sigs, t0, t1, d / 1000)
                for name in SENSORS:
                    true = ((inh[name] - inh['imu']) * 1000) - d
                    rows.append((length or (active[1] - active[0]), ci, d, name,
                                 est[name] * 1000, true, est[name] * 1000 - true))
        lab = f'{length:.0f}초' if length else '전체'
        print(f'  기록 길이 {lab} 완료')

    lengths = sorted({r[0] for r in rows})
    print(f'\n[복원 오차 ms] 오차 = 추정 - 정답, 조각별 평균 ± 표준편차 (최대 |오차|), 기준 ±{PASS_MS:g} ms')
    print('    (열 = 기록에 더 민 지연 d ms)')
    print('    길이     센서   ' + ''.join(f'{d:>8d}' for d in DELAYS_MS[::2]) + '   전체 최대  판정')
    for length in lengths:
        for name in SENSORS:
            cells = []
            for d in DELAYS_MS[::2]:
                e = np.array([r[6] for r in rows if r[0] == length and r[3] == name and r[2] == d])
                cells.append(f'{e.mean():+8.2f}')
            allerr = np.array([r[6] for r in rows if r[0] == length and r[3] == name])
            worst = np.abs(allerr).max()
            n = len({r[1] for r in rows if r[0] == length})
            verdict = '합격' if worst <= PASS_MS else '불합격'
            print(f'    {length:4.0f}s({n:d}) {name:6s}' + ''.join(cells)
                  + f'   {worst:8.2f}  {verdict}')

    print('\n[시각 흔들림] 20초 조각, 지연 50 ms, 무작위 5번 -> 오차 평균 ± 표준편차 (최대 |오차|)')
    jit = jitter_test(sigs, active)
    for sigma, per in jit.items():
        cells = []
        for name in SENSORS:
            true = (inh[name] - inh['imu']) * 1000 - 50.0
            e = np.array(per[name]) - true
            cells.append(f'{name} {e.mean():+.2f} ± {e.std():.2f} ({np.abs(e).max():.2f})')
        print(f'    σ {sigma:4.1f} ms:  ' + '   '.join(cells))

    online = []
    for item in args.online:
        path, inj = parse_online(item)
        s2, tr2, _ = load(path, wheels)
        inh2 = inherent_offsets(s2, tr2)
        for name in SENSORS:
            est = toe.fit_offset(s2[name], s2['imu'])[0] * 1000
            true_inj = -(inj.get(name, 0.0) - inj['imu'])
            true_gz = (inh2[name] - inh2['imu']) * 1000
            online.append((os.path.basename(path), name, inj.get(name, 0.0), est, true_gz,
                           true_inj))
    if online:
        print('\n[Gazebo에서 직접 넣은 지연] 정답 = 정답 회전 기준 (넣은 지연만으로 계산한 값은 괄호)')
        for f, name, d, est, tgz, tin in online:
            print(f'    {f:24s} {name:6s} 넣은 지연 {d:5.0f} ms  추정 {est:+8.2f}  정답 {tgz:+8.2f} '
                  f'({tin:+.0f})  오차 {est - tgz:+.2f} ms')

    stem = os.path.splitext(args.data)[0]
    with open(stem + '_sweep.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['length_s', 'chunk', 'added_delay_ms', 'sensor', 'est_ms', 'true_ms',
                    'err_ms'])
        for r in rows:
            w.writerow([f'{r[0]:.1f}', r[1], r[2], r[3], f'{r[4]:.3f}', f'{r[5]:.3f}',
                        f'{r[6]:.3f}'])
    plot(stem + '_sweep.png', rows, lengths, online)
    print(f'\nsaved {stem}_sweep.png, {stem}_sweep.csv')


def plot(path, rows, lengths, online):
    """센서별로 지연 vs 복원 오차 (기록 길이마다 평균 ± 표준편차), ±5 ms 띠."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ax, name in zip(axes, SENSORS):
        for li, length in enumerate(lengths):
            xs, mean, std = [], [], []
            for d in DELAYS_MS:
                sel = [r for r in rows if r[0] == length and r[3] == name and r[2] == d]
                e = np.array([r[6] for r in sel])
                xs.append(-sel[0][5])                 # 센서가 IMU보다 늦게 찍힌 정도 = -정답
                mean.append(e.mean())
                std.append(e.std())
            n = len({r[1] for r in rows if r[0] == length})
            ax.errorbar(np.array(xs) + li * 0.8, mean, yerr=std, marker='o', ms=4,
                        capsize=3, color=colors[li], label=f'{length:.0f} s windows (n={n})')
        pts = [o for o in online if o[1] == name]
        if pts:
            ax.scatter([-o[4] for o in pts], [o[3] - o[4] for o in pts], marker='*', s=160,
                       color='k', zorder=5, label='injected live in Gazebo')
        ax.axhspan(-PASS_MS, PASS_MS, color='tab:green', alpha=0.08, label='pass band +-5 ms')
        ax.axhline(0, color='gray', lw=0.6)
        ax.set_title(f'{name}: time offset recovery error vs injected delay')
        ax.set_xlabel('sensor stamp delay relative to IMU (ms) = -true offset')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('estimate - truth (ms)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == '__main__':
    main()
