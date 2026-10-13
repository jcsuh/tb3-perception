#!/usr/bin/env python3
"""
IMU 노이즈 반복 실험(sweep_imu_noise.sh) 결과를 모아 표와 그래프로 만든다.

각 실험 폴더(<sweep>/scale_<배율>/score.json)를 읽어
노이즈 배율에 따른 카메라-IMU 회전·위치 오차와 시간 오프셋을 보여준다.

사용 예:
  python3 plot_sweep.py ~/kalibr_data/sweep_imu_noise
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def load(sweep_dir):
    """실험 폴더의 scale_<배율>/score.json 들을 읽어 배율 순서로 돌려준다."""
    rows = []
    for path in glob.glob(os.path.join(sweep_dir, 'scale_*', 'score.json')):
        scale = float(os.path.basename(os.path.dirname(path))[len('scale_'):])
        with open(path, encoding='utf-8') as f:
            rows.append((scale, json.load(f)))
    return sorted(rows, key=lambda r: r[0])


def main():
    """명령줄에서 실행할 때의 시작점."""
    sweep_dir = os.path.expanduser(sys.argv[1])
    rows = load(sweep_dir)
    if not rows:
        sys.exit(f'{sweep_dir} 에 scale_*/score.json 이 없습니다')

    print('scale   cam0 rot(deg) pos(mm)   cam1 rot(deg) pos(mm)   time shift cam0 (ms)')
    for scale, s in rows:
        c0, c1 = s['cam0_from_imu'], s['cam1_from_imu']
        print(f'{scale:5.1f}   {c0["rot_err_deg"]:10.3f} {c0["pos_err_mm"]:7.2f}   '
              f'{c1["rot_err_deg"]:10.3f} {c1["pos_err_mm"]:7.2f}   '
              f'{s["cam0_timeshift_ms"]:+10.3f}')

    scales = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = (('rot_err_deg', 'rotation error (deg)', 0.5),
              ('pos_err_mm', 'translation error (mm)', 5.0))
    for ax, (key, label, limit) in zip(axes, panels):
        for cam in ('cam0', 'cam1'):
            ax.plot(scales, [r[1][f'{cam}_from_imu'][key] for r in rows], 'o-', label=cam)
        ax.axhline(limit, color='gray', ls='--', lw=1, label='pass limit')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('IMU noise scale (x datasheet)')
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, which='both')
        ax.legend()
    axes[2].plot(scales, [r[1]['cam0_timeshift_ms'] for r in rows], 'o-', label='cam0')
    axes[2].plot(scales, [r[1]['cam1_timeshift_ms'] for r in rows], 'o-', label='cam1')
    axes[2].axhline(0.0, color='gray', lw=1)
    axes[2].set_xscale('log')
    axes[2].set_xlabel('IMU noise scale (x datasheet)')
    axes[2].set_ylabel('estimated time shift (ms), truth 0')
    axes[2].grid(alpha=0.3, which='both')
    axes[2].legend()
    fig.suptitle('Kalibr camera-IMU accuracy vs IMU noise (simulation)')
    out = os.path.join(sweep_dir, 'sweep_imu_noise.png')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f'saved {out}')


if __name__ == '__main__':
    main()
