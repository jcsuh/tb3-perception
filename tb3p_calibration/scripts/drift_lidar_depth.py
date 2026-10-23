#!/usr/bin/env python3
"""
드리프트 검사 1 분석: 카메라가 돈 각도를 LiDAR-깊이 영상 비교로 찾아 실제 각도와 비교한다 (세션 20).

입력: drift_collect.py가 만든 NPZ

영상마다
  1) 짝지은 스캔의 점을 base 좌표로 (공칭 TF, 빔 높이각 반영)
  2) 스캔 시각과 영상 시각 사이 로봇이 돈 만큼 자이로로 되돌림 (회전 중이라 수십 ms도 0.5° 넘게 차이)
  3) 카메라가 마운트 축 둘레로 δ 돌았다고 놓고 잔차가 가장 작은 δ를 찾음 (drift_check.py)
영상 1장씩, 그리고 같은 구간의 --batch장을 묶어서 찾는다 (실제 모니터는 묶음으로 판단)

출력
  - 단계별 표: 명령·실제 마운트 각도, 찾은 δ (평균 ± 표준편차), 잔차 (δ=0일 때 / 찾은 δ에서)
    왼쪽으로 돌 때와 오른쪽으로 돌 때를 따로: 영상·스캔 시각이 어긋나 있으면 두 값이 반대로 벌어진다
  - 그림 <입력>_drift.png

사용 예:
  python3 drift_lidar_depth.py ~/drift_data/session20.npz
"""
import argparse
import json
import math
import os

import drift_check as dc
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    """NPZ -> (정보, 영상별 dict 목록, IMU 누적 yaw (t, yaw))."""
    raw = np.load(path)
    meta = json.loads(str(raw['meta']))
    imu = raw['imu']
    yaw = dc.cumulative_yaw(imu[:, 0], imu[:, 1])
    arrays = {k: raw[k] for k in ('t_img', 't_scan', 'ranges', 'depth', 'phase', 'part', 'cmd',
                                  'truth')}                      # NPZ는 꺼낼 때마다 다시 읽어서 한 번만
    frames = [{k: v[i] for k, v in arrays.items()} for i in range(len(arrays['t_img']))]
    return meta, frames, (imu[:, 0], yaw)


def prepare(meta, frames, imu, compensate=True):
    """영상마다 (base 좌표 점, 깊이 띠 float32)와 보정한 회전 각도."""
    t_base_lidar = np.array(meta['T_base_lidar'])
    sc = meta['scan']
    out, dpsis = [], []
    for f in frames:
        pts = dc.to_frame(t_base_lidar, dc.lidar_points(
            f['ranges'], sc['angle_min'], sc['angle_increment'], meta['elev_deg']))
        dpsi = 0.0
        if compensate:
            pts, dpsi = dc.motion_compensate(pts, f['t_scan'], f['t_img'], *imu)
        out.append((pts, f['depth'].astype(np.float32)))
        dpsis.append(dpsi)
    return out, np.array(dpsis)


def main():
    """명령줄에서 실행할 때의 시작점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data', help='drift_collect.py가 만든 NPZ')
    ap.add_argument('--batch', type=int, default=6, help='묶어서 판단할 영상 수')
    ap.add_argument('--no-compensate', action='store_true', help='실험: 회전 보정 끄기')
    args = ap.parse_args()

    meta, frames, imu = load(args.data)
    k = np.array(meta['K'])
    t0 = np.array(meta['T_base_cam0'])
    mount = np.array(meta['mount'])
    v0 = int(meta['band_v0'])
    data, dpsis = prepare(meta, frames, imu, not args.no_compensate)
    gap_ms = np.median([abs(f['t_img'] - f['t_scan']) for f in frames]) * 1000
    print(f'영상 {len(frames)}장, 스캔-영상 시각 차이 중앙값 {gap_ms:.0f} ms, '
          f'그 사이 회전 보정 최대 {np.degrees(np.abs(dpsis).max()):.2f}°')

    single = []
    for (pts, band), f in zip(data, frames):
        est, c0, cb, _ = dc.estimate_yaw([(pts, band)], t0, mount, k, v0)
        n_pts = len(dc.residuals(pts, t0, k, band, v0))
        single.append((est, c0, cb, n_pts))
    single = np.array(single)

    # 같은 단계·같은 회전 방향 안에서 batch장씩 묶기
    batches = []
    keys = [(int(f['phase']), str(f['part'])) for f in frames]
    for key in dict.fromkeys(keys):
        idx = [i for i, kk in enumerate(keys) if kk == key]
        for s in range(0, len(idx) - args.batch + 1, args.batch):
            sel = idx[s:s + args.batch]
            est, c0, cb, _ = dc.estimate_yaw([data[i] for i in sel], t0, mount, k, v0)
            batches.append((key, sel, est, c0, cb))

    print('\n[단계별] 각도 단위 °, 잔차 = 잘린 절댓값 평균 mm (δ=0 -> 찾은 δ)')
    print('  단계  명령   실제 |  영상 1장 δ (평균 ± 표준편차)  | 묶음 δ 평균 ± 표준편차 (개수)'
          ' | 왼쪽 회전  오른쪽 회전 | 잔차 mm')
    phases = sorted({int(f['phase']) for f in frames})
    for p in phases:
        m = np.array([int(f['phase']) == p for f in frames])
        cmd = math.degrees(np.mean([f['cmd'] for f, mm in zip(frames, m) if mm]))
        tru = math.degrees(np.nanmean([f['truth'] for f, mm in zip(frames, m) if mm]))
        e1 = np.degrees(single[m, 0])
        bs = [b for b in batches if b[0][0] == p]
        eb = np.degrees([b[2] for b in bs])
        lr = []
        for part in ('left', 'right'):
            sel = [b for b in bs if b[0][1] == part]
            lr.append(np.degrees(np.mean([b[2] for b in sel])) if sel else np.nan)
        c0, cb = single[m, 1].mean() * 1000, single[m, 2].mean() * 1000
        print(f'  {p + 1:3d} {cmd:+6.1f} {tru:+6.2f} | {e1.mean():+8.2f} ± {e1.std():5.2f}'
              f'               | {eb.mean():+8.2f} ± {eb.std():5.2f} ({len(eb):2d})       '
              f'| {lr[0]:+8.2f} {lr[1]:+10.2f}   | {c0:5.1f} -> {cb:5.1f}')
    print(f'  (영상 1장당 비교한 점 수 중앙값 {np.median(single[:, 3]):.0f}개)')

    stem = os.path.splitext(args.data)[0]
    plot(stem + '_drift.png', frames, single, batches, data, t0, mount, k, v0)
    print(f'saved {stem}_drift.png')


def plot(path, frames, single, batches, data, t0, mount, k, v0):
    """위: 영상별·묶음별 찾은 각도와 실제 각도, 아래: 단계별 비용 곡선 예시."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8.5))
    idx = np.arange(len(frames))
    ax1.plot(idx, np.degrees([f['truth'] for f in frames]), 'k-', lw=2, label='actual mount yaw')
    ax1.plot(idx, np.degrees(single[:, 0]), '.', color='tab:blue', ms=4,
             label='estimate, single frame')
    for j, b in enumerate(batches):
        sel = b[1]
        ax1.plot([sel[0], sel[-1]], [math.degrees(b[2])] * 2, color='tab:red', lw=3,
                 label='estimate, batch' if j == 0 else None)
    ax1.set_xlabel('frame')
    ax1.set_ylabel('camera yaw (deg)')
    ax1.set_title('camera mount yaw: actual vs estimated from LiDAR-depth consistency')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    phases = sorted({int(f['phase']) for f in frames})
    for p in phases:
        i = [n for n, f in enumerate(frames) if int(f['phase']) == p][0]
        _, _, _, (dl, cs) = dc.estimate_yaw([data[i]], t0, mount, k, v0)
        actual = math.degrees(frames[i]['truth'])
        lab = f'step {p + 1} (actual {actual:+.1f} deg)'
        ax2.plot(np.degrees(dl), np.array(cs) * 1000, label=lab)
    ax2.set_xlabel('assumed camera yaw delta (deg)')
    ax2.set_ylabel('truncated mean |residual| (mm)')
    ax2.set_title('cost curve, first frame of each step (minimum = estimate)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == '__main__':
    main()
