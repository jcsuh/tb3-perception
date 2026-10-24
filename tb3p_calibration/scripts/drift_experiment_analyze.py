#!/usr/bin/env python3
"""
드리프트 모니터 시험 결과 분석 (세션 22): drift_experiment.py가 만든 CSV -> 감지율, 오경보, 그래프.

판단 단위
  검사 1: 묶음 하나 (/diagnostics의 batches가 늘어난 줄) = 판정 한 번
  검사 2·3: 계산 한 번 (evaluations가 늘어난 줄)
  구간이 바뀐 직후는 앞 구간 데이터가 섞여 있으므로 뺀다 (검사 1 --cam-settle초, 검사 2·3 --rot-settle초)

감지율 = 고장 구간 판정 중 ERROR 비율.  오경보 = 고장 없는 구간(섞인 직후 제외)의 ERROR 판정 수.
완료 기준 (W10–11): 2° 이상 감지율 90% 이상, 정상 주행 30분 오경보 0번

사용 예
  python3 drift_experiment_analyze.py ~/drift_data/session22_camera.csv
"""
import argparse
import collections
import csv
import os

import numpy as np

LEVEL = {0: 'OK', 1: 'WARN', 2: 'ERROR', 3: 'STALE'}


def num(x):
    """CSV 칸 -> float (빈 칸은 NaN)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def load(path):
    """CSV -> 열 이름별 리스트."""
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f'{path}: 기록이 없음')
    return rows


def events(rows, counter):
    """판정이 새로 나온 줄만 (counter 열 값이 늘어난 줄)."""
    out, last = [], None
    for r in rows:
        c = num(r[counter])
        if np.isfinite(c) and (last is None or c > last):
            if last is not None:
                out.append(r)
            last = c
    return out


def fault_name(r):
    """줄의 고장 설명 (없으면 '')."""
    parts = []
    deg, dw, dl = num(r['mount_cmd_deg']), num(r['delay_wheel_ms']), num(r['delay_lidar_ms'])
    if deg != 0:
        parts.append(f'카메라 {deg:+g}°')
    if dw != 0:
        parts.append(f'바퀴 지연 {dw:g} ms')
    if dl != 0:
        parts.append(f'LiDAR 지연 {dl:g} ms')
    js = num(r.get('joint_scale', 1.0))
    if np.isfinite(js) and js != 1:
        parts.append(f'바퀴 배율 {js:g}')
    return ', '.join(parts)


def first_error_times(evs, level_col):
    """고장 구간마다 구간 시작 -> 첫 ERROR 시각 (s). 끝까지 없으면 NaN."""
    out = {}
    for r in evs:
        k = int(r['phase'])
        if k not in out:
            out[k] = float('nan')
        if np.isnan(out[k]) and int(num(r[level_col])) == 2:
            out[k] = num(r['phase_t'])
    return out


def summarize(evs, level_col, settle, value_cols, title):
    """고장별 판정 표. 돌려주는 값: {고장: (판정 수, ERROR 수, WARN 이상 수)}."""
    groups = collections.OrderedDict()
    for r in evs:
        lv = int(num(r[level_col]))
        if lv == 3 or num(r['phase_t']) < settle:
            continue
        groups.setdefault(fault_name(r), []).append(r)
    print(f'\n[{title}] (구간 시작 {settle:g}초 안의 판정과 STALE은 뺌)')
    head = '  고장                     | 판정 | ERROR %  WARN이상 % | ' + \
        ' | '.join(f'{name} 평균 ± 표준편차' for name, _ in value_cols)
    print(head)
    out = {}
    for name, rs in groups.items():
        lv = np.array([int(num(r[level_col])) for r in rs])
        n, ne, nw = len(lv), int((lv == 2).sum()), int((lv >= 1).sum())
        vals = []
        for _, fn in value_cols:
            v = np.array([fn(r) for r in rs])
            vals.append(f'{np.nanmean(v):+8.3f} ± {np.nanstd(v):6.3f}')
        label = name or '없음 (정상)'
        print(f'  {label:24s} | {n:4d} | {100 * ne / n:6.1f}  {100 * nw / n:8.1f}   | '
              + ' | '.join(vals))
        out[name] = (n, ne, nw)
    return out


def plot(path, rows, cam, rot, args):
    """시간에 따른 세 검사의 값 + 판정 색."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {0: 'tab:green', 1: 'tab:orange', 2: 'tab:red', 3: 'tab:gray'}
    t = np.array([num(r['t']) for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for ax in axes:
        start = None
        for r, tt in zip(rows + [None], list(t) + [None]):
            on = r is not None and fault_name(r) != ''
            if on and start is None:
                start = tt
            if not on and start is not None:
                ax.axvspan(start, tt if tt is not None else t[-1], color='0.92')
                start = None
        ax.grid(alpha=0.3)
    ax = axes[0]
    base = np.array([num(r['cam_baseline_deg']) for r in rows])
    ax.plot(t, np.array([num(r['mount_true_deg']) for r in rows]), 'k-', lw=1.5,
            label='actual mount yaw')
    for lv, c in colors.items():
        sel = [r for r in cam if int(num(r['cam_level'])) == lv]
        ax.plot([num(r['t']) for r in sel], [num(r['cam_deviation_deg']) for r in sel], 'o',
                color=c, ms=4, label=f'batch deviation from baseline: {LEVEL[lv]}')
    for y, s in ((args.cam_warn, ':'), (args.cam_error, '--')):
        ax.axhline(y, color='tab:red', ls=s, lw=1)
        ax.axhline(-y, color='tab:red', ls=s, lw=1)
    ax.set_ylabel('camera yaw (deg)')
    ax.set_title(f'check 1 camera-LiDAR (baseline {np.nanmedian(base):+.3f} deg)')
    ax.legend(fontsize=8, loc='upper left')
    ax = axes[1]
    judged = [r for r in rot if int(num(r['rot_level'])) != 3]     # 기준 학습 중(STALE) 제외
    for key, m in (('wheel', 'o'), ('lidar', 's')):
        y = [num(r[f'rot_{key}_scale']) - num(r[f'rot_{key}_scale_baseline']) for r in judged]
        c = [colors[int(num(r['rot_level']))] for r in judged]
        ax.scatter([num(r['t']) for r in judged], y, c=c, marker=m, s=14, label=f'{key} scale')
    if 'joint_scale' in rows[0]:
        ax.plot(t, [num(r['joint_scale']) - 1 for r in rows], '-', lw=1,
                label='injected wheel scale - 1')
    for y, s in ((args.scale_warn, ':'), (args.scale_error, '--')):
        ax.axhline(y, color='tab:red', ls=s, lw=1)
        ax.axhline(-y, color='tab:red', ls=s, lw=1)
    ax.set_ylabel('scale - baseline')
    ax.set_title('check 2 yaw-rate consistency (color = level)')
    ax.legend(fontsize=8, loc='upper left')
    ax = axes[2]
    judged = [r for r in rot if int(num(r['time_level'])) != 3]
    for key, m, d in (('wheel', 'o', 'delay_wheel_ms'), ('lidar', 's', 'delay_lidar_ms')):
        y = [num(r[f'time_{key}_offset_ms']) - num(r[f'time_{key}_offset_baseline_ms'])
             for r in judged]
        c = [colors[int(num(r['time_level']))] for r in judged]
        ax.scatter([num(r['t']) for r in judged], y, c=c, marker=m, s=14,
                   label=f'{key} offset')
        ax.plot(t, [-num(r[d]) for r in rows], '-', lw=1, label=f'- injected {key} delay')
    for y, s in ((args.offset_warn, ':'), (args.offset_error, '--')):
        ax.axhline(y, color='tab:red', ls=s, lw=1)
        ax.axhline(-y, color='tab:red', ls=s, lw=1)
    ax.set_ylabel('offset - baseline (ms)')
    ax.set_xlabel('time (s), gray = fault injected')
    ax.set_title('check 3 time offset (color = level)')
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f'saved {path}')


def main():
    """명령줄에서 실행할 때의 시작점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv')
    ap.add_argument('--cam-settle', type=float, default=5.0, help='검사 1: 구간 시작 뒤 뺄 시간 (s)')
    ap.add_argument('--rot-settle', type=float, default=25.0,
                    help='검사 2·3: 구간 시작 뒤 뺄 시간 (s, 계산 창 20초 + 여유)')
    ap.add_argument('--cam-warn', type=float, default=0.5)
    ap.add_argument('--cam-error', type=float, default=1.0)
    ap.add_argument('--scale-warn', type=float, default=0.02)
    ap.add_argument('--scale-error', type=float, default=0.05)
    ap.add_argument('--offset-warn', type=float, default=5.0)
    ap.add_argument('--offset-error', type=float, default=10.0)
    args = ap.parse_args()

    rows = load(args.csv)
    t = [num(r['t']) for r in rows]
    print(f'{args.csv}: {len(rows)}줄, {t[-1] / 60:.1f}분')
    cam = events(rows, 'cam_batches')
    rot = events(rows, 'rot_evaluations')
    skipped = num(rows[-1]['cam_skipped_still'])
    print(f'검사 1 묶음 {len(cam)}개 (서 있어서 건너뛴 묶음 {skipped:.0f}개), 검사 2·3 계산 {len(rot)}번')

    s1 = summarize(cam, 'cam_level', args.cam_settle, [
        ('기준 대비 차이 °', lambda r: num(r['cam_deviation_deg'])),
        ('실제 마운트 °', lambda r: num(r['mount_true_deg']))], '검사 1 카메라-LiDAR')
    s2 = summarize(rot, 'rot_level', args.rot_settle, [
        ('바퀴 배율 차이', lambda r: num(r['rot_wheel_scale']) - num(r['rot_wheel_scale_baseline'])),
        ('LiDAR 배율 차이',
         lambda r: num(r['rot_lidar_scale']) - num(r['rot_lidar_scale_baseline']))],
        '검사 2 회전 일관성')
    s3 = summarize(rot, 'time_level', args.rot_settle, [
        ('바퀴 오프셋 차이 ms',
         lambda r: num(r['time_wheel_offset_ms']) - num(r['time_wheel_offset_baseline_ms'])),
        ('LiDAR 오프셋 차이 ms',
         lambda r: num(r['time_lidar_offset_ms']) - num(r['time_lidar_offset_baseline_ms']))],
        '검사 3 시간 오프셋')

    # 고장 구간마다 첫 ERROR까지 걸린 시간
    print('\n[감지 시간] 고장 구간 시작 -> 첫 ERROR (s)')
    # 검사마다 자기가 맡은 고장만 (다른 고장에 번진 경우는 위 표에서 본다)
    owned = (('검사 1', cam, 'cam_level', '카메라'), ('검사 2', rot, 'rot_level', '배율'),
             ('검사 3', rot, 'time_level', '지연'))
    for title, evs, level_col, word in owned:
        fe = first_error_times([r for r in evs if word in fault_name(r)], level_col)
        names = {int(r['phase']): fault_name(r) for r in evs}
        for k, v in sorted(fe.items()):
            text = f'{v:5.1f}' if np.isfinite(v) else '  없음'
            print(f'  {title} 구간 {k:2d} {names[k]:18s}: {text}')

    print('\n[판정] 감지율 = 고장 구간 판정 중 ERROR 비율')
    targets = [
        ('검사 1', s1, '2° 이상 카메라 드리프트', lambda k: k.startswith('카메라') and
         abs(float(k.split()[1].rstrip('°'))) >= 2),
        ('검사 2', s2, '바퀴 배율 6% 이상', lambda k: k.startswith('바퀴 배율') and
         abs(float(k.split()[2]) - 1) >= 0.06 - 1e-9),
        ('검사 3', s3, '시각 지연 20 ms 이상', lambda k: '지연' in k),
    ]
    for check, summ, what, pick in targets:
        sel = [v for k, v in summ.items() if pick(k)]
        if not sel:
            continue
        n, ne = sum(v[0] for v in sel), sum(v[1] for v in sel)
        rate = 100 * ne / n
        verdict = '합격' if rate >= 90 else '불합격'
        print(f'  {check} {what} 감지율 {rate:.1f}% ({ne}/{n}), 기준 90% -> {verdict}')
    false = sum(s.get('', (0, 0, 0))[1] for s in (s1, s2, s3))
    warns = sum(s.get('', (0, 0, 0))[2] - s.get('', (0, 0, 0))[1] for s in (s1, s2, s3))
    verdict = '합격' if false == 0 else '불합격'
    print(f'  고장 없는 구간의 오경보: ERROR {false}번 (기준 0번 -> {verdict}), WARN {warns}번')
    plot(os.path.splitext(args.csv)[0] + '.png', rows, cam, rot, args)


if __name__ == '__main__':
    main()
