#!/usr/bin/env python3
"""
LiDAR 노이즈와 보드 자세 수에 따라 LiDAR-카메라 캘리브레이션 정확도가 어떻게 변하는지 잰다 (세션 13).

Gazebo를 다시 돌리지 않고, lidar_cam_collect.py가 저장한 데이터 하나로 조건을 바꿔 가며
캘리브레이션을 반복한다.
  - 자세 수 N: 저장된 자세 중 N개를 무작위로 골라서 (조건마다 --trials번)
  - LiDAR 노이즈: 자세마다 저장한 스캔 20장의 원래 거리값에서 k장을 골라 평균
    (스캔 1장 σ 10 mm -> k장 평균 10/√k mm). 더 시끄러운 LiDAR는 원래 값에
    가우스 노이즈를 더해 흉내 낸다 (스캔 1장 σ 20 mm = 10 mm + 추가 17.3 mm)
  - 가중치: 없음 vs 있음 (lidar_cam_calib.py --weighted와 같은 방식)

출력
  - 조건별 표: 회전·위치 오차 중앙값, 합격률(회전 0.5° 이하 그리고 위치 10 mm 이하)
  - 조건별로 합격률 95% 이상이 되는 가장 작은 자세 수 (실제 단계에서 필요한 자세 수)
  - 그래프 PNG, 원자료 CSV (입력 JSON 옆)

사용 예:
  python3 lidar_cam_sweep.py ~/lidar_cam_data/session13.json
"""
import argparse
import csv
import json
import os

import lidar_cam_calib as lcal
import matplotlib.pyplot as plt
import numpy as np

# (이름, 스캔 1장 노이즈 σ mm, 평균할 스캔 수 k)
CONDITIONS = [
    ('σ10 mm, 스캔 1장', 10.0, 1),
    ('σ10 mm, 스캔 5장', 10.0, 5),
    ('σ10 mm, 스캔 20장', 10.0, 20),
    ('σ20 mm, 스캔 1장', 20.0, 1),
    ('σ20 mm, 스캔 20장', 20.0, 20),
]
SIM_SIGMA_MM = 10.0     # Gazebo LiDAR 설정 (tb3p_burger.urdf.xacro)


def resample_points(pose, k, sigma_mm, rng, elev_deg=0.0):
    """
    자세 하나의 보드 위 점을 다시 만든다: 스캔 k장을 골라 (노이즈를 더한 뒤) 빔별로 평균.

    돌려주는 값: 점 (M x 3, LiDAR 좌표계). 유효한 값이 하나도 없는 빔은 뺀다.
    """
    raw = np.asarray(pose['scan_ranges'], dtype=np.float64)
    pick = rng.choice(len(raw), size=min(k, len(raw)), replace=False)
    r = raw[pick]
    extra = np.sqrt(max(sigma_mm ** 2 - SIM_SIGMA_MM ** 2, 0.0)) / 1000
    if extra > 0:
        r = r + rng.normal(0.0, extra, r.shape)
    ok = np.isfinite(r)
    cnt = ok.sum(axis=0)
    mean = np.where(ok, r, 0.0).sum(axis=0) / np.maximum(cnt, 1)
    a = np.asarray(pose['beam_angles'], dtype=np.float64)
    keep = cnt > 0
    pts = np.column_stack([mean[keep] * np.cos(a[keep]), mean[keep] * np.sin(a[keep]),
                           np.zeros(keep.sum())])
    return lcal.apply_beam_elevation(pts, elev_deg)


def one_trial(poses, n_poses, k, sigma_mm, t_nom, t_true, rng, elev_deg=0.0):
    """자세 n개를 고르고 점을 다시 만든 뒤, 가중치 없음/있음 두 가지로 추정해 오차를 돌려준다."""
    chosen = rng.choice(len(poses), size=n_poses, replace=False)
    trial = []
    for i in chosen:
        p = dict(poses[i])
        p['points'] = resample_points(poses[i], k, sigma_mm, rng, elev_deg)
        trial.append(p)
    out = {}
    t_est, _, _ = lcal.calibrate(trial, t_nom)
    out['가중치 없음'] = lcal.errors(t_est, t_true)[:2]
    ws, _ = lcal.point_weights(trial, t_nom, sigma_mm / np.sqrt(k) / 1000)
    t_est, _, _ = lcal.calibrate(trial, t_nom, ws)
    out['가중치 있음'] = lcal.errors(t_est, t_true)[:2]
    return out


def main():
    """명령줄에서 실행할 때의 시작점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data', help='lidar_cam_collect.py가 만든 JSON (scan_ranges 포함)')
    ap.add_argument('--counts', type=int, nargs='+', default=[4, 6, 8, 12, 0],
                    help='자세 수 목록 (0 = 전부)')
    ap.add_argument('--trials', type=int, default=40, help='조건마다 반복 횟수')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--beam-elev-deg', type=float, default=0.0,
                    help='LiDAR 빔 높이각 (°, 아래가 -). Gazebo GPU LiDAR는 약 -0.44')
    args = ap.parse_args()

    with open(args.data) as f:
        data = json.load(f)
    poses = [p for p in data['poses'] if 'scan_ranges' in p]
    if len(poses) < 4:
        raise SystemExit('scan_ranges가 있는 자세가 4개 미만: 세션 13 lidar_cam_collect.py로 다시 수집')
    t_nom = np.array(data['T_nominal'])
    t_true = np.array(data['T_truth'])
    counts = sorted({c if c else len(poses) for c in args.counts if c <= len(poses)})
    rng = np.random.default_rng(args.seed)
    modes = ('가중치 없음', '가중치 있음')

    rows = []          # (조건, 자세 수, 방식, 시행, 회전°, 위치 mm)
    for cname, sigma, k in CONDITIONS:
        for n in counts:
            for t in range(args.trials):
                res = one_trial(poses, n, k, sigma, t_nom, t_true, rng, args.beam_elev_deg)
                for mode in modes:
                    rows.append((cname, n, mode, t, *res[mode]))
        print(f'  {cname} 완료')

    def summary(cname, n, mode):
        sel = [r for r in rows if r[0] == cname and r[1] == n and r[2] == mode]
        rot = np.array([r[4] for r in sel])
        pos = np.array([r[5] for r in sel])
        ok = np.mean((rot <= lcal.PASS_ROT_DEG) & (pos <= lcal.PASS_POS_MM)) * 100
        return float(np.median(rot)), float(np.median(pos)), float(ok)

    print(f'\n빔 높이각 {args.beam_elev_deg:+.3f}°. '
          f'자세 {len(poses)}개 중 무작위 선택, 조건마다 {args.trials}회. '
          f'합격 = 회전 {lcal.PASS_ROT_DEG}° 이하 그리고 위치 {lcal.PASS_POS_MM:.0f} mm 이하')
    print('조건(점 하나의 노이즈)          자세수   [가중치 없음] 회전°  위치mm 합격%'
          '   [가중치 있음] 회전°  위치mm 합격%')
    for cname, sigma, k in CONDITIONS:
        for n in counts:
            a = summary(cname, n, modes[0])
            b = summary(cname, n, modes[1])
            label = f'{cname} (≈{sigma / np.sqrt(k):.1f} mm)'
            print(f'  {label:28s} {n:4d}   {a[0]:13.3f} {a[1]:7.1f} {a[2]:5.0f}'
                  f'   {b[0]:13.3f} {b[1]:7.1f} {b[2]:5.0f}')

    print('\n합격률 95% 이상이 되는 가장 작은 자세 수 (가중치 있음 기준)')
    for cname, sigma, k in CONDITIONS:
        need = [n for n in counts if summary(cname, n, modes[1])[2] >= 95]
        answer = str(need[0]) if need else '없음 (시험한 범위 안에서)'
        print(f'  {cname:16s}: {answer}')

    base = os.path.splitext(args.data)[0]
    with open(base + '_sweep.csv', 'w', newline='') as f:
        w = csv.writer(f)          # 조건 이름에 쉼표가 있어 csv 모듈로 따옴표 처리
        w.writerow(['condition', 'n_poses', 'mode', 'trial', 'rot_err_deg', 'pos_err_mm'])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], f'{r[4]:.4f}', f'{r[5]:.3f}'])

    # 그래프 글자는 영어 (한글 글꼴이 없는 PC에서도 깨지지 않게)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    panels = (('pass rate (%)', 2, None), ('median rotation error (deg)', 0, lcal.PASS_ROT_DEG),
              ('median translation error (mm)', 1, lcal.PASS_POS_MM))
    for ax, (ylabel, j, limit) in zip(axes, panels):
        for ci, (cname, sigma, k) in enumerate(CONDITIONS):
            for mode, style in zip(modes, ('--', '-')):
                ys = [summary(cname, n, mode)[j] for n in counts]
                lab = f'sigma {sigma:.0f} mm, {k} scan(s)' if mode == modes[1] else None
                ax.plot(counts, ys, style, marker='o', color=colors[ci], label=lab)
        if limit is not None:
            ax.axhline(limit, color='gray', ls=':', label='pass limit')
            ax.set_yscale('log')
        ax.set_xlabel('number of board poses')
        ax.set_ylabel(ylabel)
        ax.set_xticks(counts)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, title='solid = weighted, dashed = unweighted', title_fontsize=8)
    fig.suptitle(f'LiDAR-camera extrinsic accuracy vs LiDAR noise and pose count '
                 f'({args.trials} random trials each, beam elevation '
                 f'{args.beam_elev_deg:+.2f} deg)')
    fig.tight_layout()
    fig.savefig(base + '_sweep.png', dpi=150)
    print(f'\nsaved {base}_sweep.png, {base}_sweep.csv')


if __name__ == '__main__':
    main()
