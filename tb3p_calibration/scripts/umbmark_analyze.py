#!/usr/bin/env python3
"""
UMBmark 기록으로 바퀴 오차를 구하고 보정 전후를 비교한다 (세션 15).

입력: umbmark_run.py가 만든 NPZ (보정 전), 선택으로 --after NPZ (보정 후 다시 잰 것)

계산
  1) 주행마다 실제 도착 오차 (출발 자세 기준 x, y)
     방향별 무게중심과 E_max,syst = 두 무게중심 중 출발점에서 먼 쪽의 거리 (Borenstein의 지표)
  2) Borenstein 방법: 무게중심 4개 숫자만으로 A형(좌우 지름 비 Ed)과 B형(간격 비 Eb) 오차
     -> 줄자로 도착점만 재면 되는 방법. 평균 반지름(거리 배율)은 알 수 없음
  3) 전체 경로 맞추기: 바퀴 각도로 다시 계산한 경로가 실제 경로와 가장 잘 맞는 반지름 2개 + 간격
     -> 실제 경로를 아는 경우 (시뮬레이션 정답, 실제에서는 외부 측정). 거리 배율까지 정해짐
  4) (시뮬레이션) --truth sim_truth.yaml이면 정답 바퀴 값과 비교 (채점 전용)

출력: 표, 그림 <입력>_umbmark.png, 바퀴 값 YAML 두 개 (*_wheels_ls.yaml, *_wheels_borenstein.yaml)

사용 예:
  python3 umbmark_analyze.py ~/umbmark_data/before.npz
      --truth ../../tb3p_gazebo/config/sim_truth.yaml               (실제로는 위와 한 줄로)
  python3 umbmark_analyze.py ~/umbmark_data/before.npz --after ~/umbmark_data/after.npz
      --truth ../../tb3p_gazebo/config/sim_truth.yaml               (실제로는 위와 한 줄로)
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import wheel_odom as wo

HERE = os.path.dirname(os.path.abspath(__file__))
NOMINAL = os.path.join(HERE, '..', '..', 'tb3p_description', 'config', 'calibration.yaml')
KEYS = ('radius_left', 'radius_right', 'separation')


def load_runs(path):
    """
    NPZ -> (정보 dict, 주행 목록).

    주행마다 실제 위치 기록 시각에 맞춰 바퀴 각도를 보간해 둔다:
    {'dir', 'phi_l', 'phi_r', 'truth' (N x 3, x y yaw), 'end' (출발 기준 도착 x, y, θ)}
    """
    data = np.load(path)
    meta = json.loads(str(data['meta']))
    runs = []
    for i, d in enumerate(meta['dirs']):
        joint = data[f'run{i}_joint']
        truth = data[f'run{i}_truth']
        t_j, first = np.unique(joint[:, 0], return_index=True)   # 같은 메시지 중복 제거
        phi_l, phi_r = np.unwrap(joint[first, 1]), np.unwrap(joint[first, 2])
        keep = (truth[:, 0] >= t_j[0]) & (truth[:, 0] <= t_j[-1])
        tr = truth[keep]
        runs.append({
            'dir': d,
            'phi_l': np.interp(tr[:, 0], t_j, phi_l),
            'phi_r': np.interp(tr[:, 0], t_j, phi_r),
            'truth': tr[:, 1:],
            'end': wo.relative_pose(tr[0, 1:], tr[-1:, 1:])[0],
        })
    return meta, runs


def summarize(runs):
    """방향별 도착점 무게중심, E_max,syst, 무게중심 둘레 흩어짐 (모두 m)."""
    out = {}
    for d in (wo.CW, wo.CCW):
        ends = np.array([r['end'][:2] for r in runs if r['dir'] == d])
        cg = ends.mean(axis=0)
        out[d] = {'ends': ends, 'cg': cg, 'r': float(np.linalg.norm(cg)),
                  'spread': float(np.mean(np.linalg.norm(ends - cg, axis=1)))}
    out['emax'] = max(out[wo.CW]['r'], out[wo.CCW]['r'])
    return out


def print_runs(label, meta, runs, summ):
    """주행별 도착 오차와 요약을 출력."""
    n_cw = sum(r['dir'] == wo.CW for r in runs)
    side, wheels = meta['side'], fmt_wheels(meta['wheels'])
    print(f'[{label}] 한 변 {side} m, 주행 {len(runs)}회 (시계 {n_cw}, '
          f'반시계 {len(runs) - n_cw}), 오도메트리 바퀴 값 {wheels}')
    print('     #  방향   도착 오차 x mm    y mm   거리 mm   방향 °')
    for i, r in enumerate(runs):
        e = r['end']
        name = wo.DIRECTION_NAME[r['dir']]
        print(f'    {i + 1:2d}  {name:4s} {e[0] * 1000:+12.1f} {e[1] * 1000:+7.1f} '
              f'{np.hypot(e[0], e[1]) * 1000:9.1f} {np.degrees(e[2]):+8.2f}')
    for d in (wo.CW, wo.CCW):
        cg, r_mm, spread = summ[d]['cg'] * 1000, summ[d]['r'] * 1000, summ[d]['spread'] * 1000
        print(f'    무게중심 {wo.DIRECTION_NAME[d]:3s} ({cg[0]:+.1f}, {cg[1]:+.1f}) mm, '
              f'거리 {r_mm:.1f} mm, 흩어짐 {spread:.1f} mm')
    emax = summ['emax'] * 1000
    print(f'    E_max,syst = {emax:.1f} mm')


def fmt_wheels(w):
    """바퀴 값을 mm 단위 짧은 글자로."""
    rl, rr, sep = (w[k] * 1000 for k in KEYS)
    return f'(왼 {rl:.3f}, 오른 {rr:.3f}, 간격 {sep:.2f} mm)'


def main():
    """명령줄에서 실행할 때의 시작점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data', help='umbmark_run.py가 만든 NPZ (보정 전)')
    ap.add_argument('--after', default='', help='보정한 바퀴 값으로 다시 잰 NPZ (선택)')
    ap.add_argument('--nominal', default=NOMINAL, help='공칭 바퀴 값 YAML (경로 맞추기 초기값)')
    ap.add_argument('--truth', default='', help='(시뮬레이션 채점 전용) sim_truth.yaml')
    args = ap.parse_args()

    meta, runs = load_runs(args.data)
    summ = summarize(runs)
    print_runs('보정 전', meta, runs, summ)
    used = meta['wheels']

    bor = wo.borenstein(summ[wo.CW]['cg'], summ[wo.CCW]['cg'], meta['side'], used)
    print('[Borenstein 방법] 도착점 무게중심만 사용')
    a_deg, b_deg = np.degrees(bor['alpha']), np.degrees(bor['beta'])
    fit, e_d, e_b, bor_wheels = bor['fit_rms_mm'], bor['Ed'], bor['Eb'], fmt_wheels(bor['wheels'])
    print(f'    회전마다 더 돈 각도 α {a_deg:+.3f}°, 직진 한 변에서 휜 각도 β {b_deg:+.3f}° '
          f'(모형 맞춤 잔차 {fit:.1f} mm)')
    print(f'    Ed (오른쪽/왼쪽 지름) {e_d:.5f}, Eb (실제/공칭 간격) {e_b:.5f}')
    print(f'    -> 바퀴 값 {bor_wheels}  (평균 반지름은 그대로)')

    nominal = wo.load_wheels(args.nominal)
    ls, sem, rms = wo.fit_wheels(runs, nominal)
    print('[전체 경로 맞추기] 실제 경로 전체 사용')
    v = [ls[k] * 1000 for k in KEYS]
    e = [sem[k] * 1000 for k in KEYS]
    print(f'    왼 {v[0]:.4f} ± {e[0]:.4f} mm, 오른 {v[1]:.4f} ± {e[1]:.4f} mm, '
          f'간격 {v[2]:.3f} ± {e[2]:.3f} mm')
    print(f'    (± = 주행별로 따로 맞춘 값의 표준오차), 경로 잔차 RMS {rms:.1f} mm')

    rows = [('공칭', nominal), ('Borenstein', bor['wheels']), ('경로 맞추기', ls)]
    if args.truth:
        truth = wo.load_wheels(args.truth)
        print('[정답과 비교] (시뮬레이션 채점 전용, 괄호 = 정답 대비 오차 %)')
        print('    값            ' + ''.join(f'{name:>20s}' for name, _ in rows) + '      정답')
        labels = {'radius_left': '왼쪽 반지름 mm', 'radius_right': '오른쪽 반지름 mm',
                  'separation': '간격 mm'}
        for k in KEYS:
            cells = ''.join(f'{w[k] * 1000:11.3f} ({(w[k] / truth[k] - 1) * 100:+5.2f}%)'
                            for _, w in rows)
            print(f'    {labels[k]:14s}{cells}  {truth[k] * 1000:8.3f}')
        ed_t = truth['radius_right'] / truth['radius_left']
        ed_ls = ls['radius_right'] / ls['radius_left']
        print(f'    Ed 정답 {ed_t:.5f}, Borenstein {e_d:.5f} / 경로 맞추기 {ed_ls:.5f}')

    base = os.path.splitext(args.data)[0]
    wo.save_wheels(base + '_wheels_ls.yaml', ls, f'umbmark_analyze.py 전체 경로 맞추기 ({args.data})')
    wo.save_wheels(base + '_wheels_borenstein.yaml', bor['wheels'],
                   f'umbmark_analyze.py Borenstein ({args.data}), 평균 반지름은 주행 때 값')
    print(f'saved {base}_wheels_ls.yaml, {base}_wheels_borenstein.yaml')

    after = None
    if args.after:
        meta_a, runs_a = load_runs(args.after)
        after = summarize(runs_a)
        print()
        print_runs('보정 후', meta_a, runs_a, after)
        print('[보정 전후]                 보정 전     보정 후')
        e_b_mm, e_a_mm = summ['emax'] * 1000, after['emax'] * 1000
        ratio = e_b_mm / max(e_a_mm, 1e-6)
        print(f'    E_max,syst (mm)      {e_b_mm:10.1f} {e_a_mm:11.1f}   ({ratio:.0f}배 줄어듦)')
        for d in (wo.CW, wo.CCW):
            for kind, key in (('무게중심', 'r'), ('흩어짐', 'spread')):
                nm = f'{kind} {wo.DIRECTION_NAME[d]} (mm)'
                b_mm, a_mm = summ[d][key] * 1000, after[d][key] * 1000
                print(f'    {nm:20s} {b_mm:10.1f} {a_mm:11.1f}')
        # 남은 오차도 계통 오차면 Borenstein으로 한 번 더 (주행에 쓴 바퀴 값 기준, 거리 배율 유지)
        bor_a = wo.borenstein(after[wo.CW]['cg'], after[wo.CCW]['cg'], meta_a['side'],
                              meta_a['wheels'])
        a_deg, b_deg = np.degrees(bor_a['alpha']), np.degrees(bor_a['beta'])
        w2 = bor_a['wheels']
        print('[보정 후 남은 오차 -> Borenstein 한 번 더]')
        ed_left, eb_left = bor_a['Ed'], bor_a['Eb']
        print(f'    α {a_deg:+.4f}°, β {b_deg:+.4f}°, 남은 Ed {ed_left:.6f}, 남은 Eb {eb_left:.6f}')
        print(f'    -> 바퀴 값 {fmt_wheels(w2)}')
        if args.truth:
            t_ed = truth['radius_right'] / truth['radius_left']
            ed2 = w2['radius_right'] / w2['radius_left']
            print(f'    Ed: 정답 {t_ed:.6f}, 경로 맞추기 {ed_ls:.6f}, 한 번 더 {ed2:.6f}')
        base_a = os.path.splitext(args.after)[0]
        wo.save_wheels(base_a + '_wheels_borenstein.yaml', bor_a['wheels'],
                       f'umbmark_analyze.py 반복 보정 ({args.after}), 거리 배율은 주행 때 값')
        print(f'saved {base_a}_wheels_borenstein.yaml  (원하면 이 값으로 한 번 더 주행)')

    plot(base + '_umbmark.png', meta, runs, summ, after, ls)
    print(f'saved {base}_umbmark.png')


def plot(path, meta, runs, summ, after, ls):
    """왼쪽: 도착점 산점도 (보정 전후), 오른쪽: 실제 경로와 다시 계산한 오도메트리."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2))
    colors = {wo.CW: 'tab:blue', wo.CCW: 'tab:red'}
    sets = [('before', summ, 'o', True)] + ([('after', after, 'o', False)] if after else [])
    for label, s, mk, filled in sets:
        for d in (wo.CW, wo.CCW):
            e = s[d]['ends'] * 1000
            name = wo.DIRECTION_NAME[d]
            ax1.scatter(e[:, 0], e[:, 1], marker=mk, s=40, color=colors[d] if filled else 'none',
                        edgecolors=colors[d], label=f'{label} {name}')
            ax1.scatter(*(s[d]['cg'] * 1000), marker='x', s=120, color=colors[d])
    ax1.axhline(0, color='gray', lw=0.5)
    ax1.axvline(0, color='gray', lw=0.5)
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.set_xlabel('return error x (mm, start frame)')
    ax1.set_ylabel('return error y (mm)')
    e_before = summ['emax'] * 1000
    title = f'UMBmark return errors (x = centroid), E_max,syst before {e_before:.0f} mm'
    if after:
        e_after = after['emax'] * 1000
        title += f', after {e_after:.0f} mm'
    ax1.set_title(title, fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    side = meta['side']
    for d in (wo.CW, wo.CCW):
        ideal = np.array([[0, 0], [side, 0], [side, d * side], [0, d * side], [0, 0]])
        ax2.plot(ideal[:, 0], ideal[:, 1], color='gray', lw=0.8, ls=':')
    for i, r in enumerate(runs):
        tr = wo.relative_pose(r['truth'][0], r['truth'])
        lab = 'actual path (odometry with wheels used)' if i == 0 else None
        ax2.plot(tr[:, 0], tr[:, 1], color=colors[r['dir']], lw=0.7, alpha=0.6, label=lab)
    r0 = runs[0]
    odo = wo.integrate(r0['phi_l'], r0['phi_r'], meta['wheels'])
    fix = wo.integrate(r0['phi_l'], r0['phi_r'], ls)
    ax2.plot(odo[:, 0], odo[:, 1], 'k--', lw=1, label='run 1 odometry (wheels used)')
    ax2.plot(fix[:, 0], fix[:, 1], 'g-', lw=1, label='run 1 odometry (fitted wheels)')
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.set_xlabel('x (m, start frame)')
    ax2.set_ylabel('y (m)')
    ax2.set_title('paths: robot follows its odometry square, reality drifts', fontsize=10)
    ax2.legend(fontsize=8, loc='center left')
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == '__main__':
    main()
