#!/usr/bin/env python3
"""
Plot IMU-noise sweep results with several trials per noise scale.

Reads <trials>/scale_<s>/t<k>/score.json (written by kalibr_score.py --json),
prints a per-trial table and a per-scale summary, and saves
sweep_imu_noise_trials.png and sweep_imu_noise_trials.csv next to <trials>.
"""

import argparse
import glob
import json
import os
import statistics

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

ROT_LIMIT_DEG = 0.5
POS_LIMIT_MM = 5.0
NICE_TICKS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]


def load(trials_dir):
    """Return {scale: [row, ...]} with one row per finished trial."""
    data = {}
    pattern = os.path.join(trials_dir, 'scale_*', 't*', 'score.json')
    for path in sorted(glob.glob(pattern)):
        scale_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
        trial = os.path.basename(os.path.dirname(path))
        try:
            scale = float(scale_dir[len('scale_'):])
        except ValueError:
            continue
        with open(path) as f:
            score = json.load(f)
        ext = score['cam0_from_imu']
        data.setdefault(scale, []).append({
            'trial': trial,
            'rot': ext['rot_err_deg'],
            'pos': ext['pos_err_mm'],
            'z': ext['pos_err_xyz_mm'][2],
            'ts': score.get('cam0_timeshift_ms', float('nan')),
        })
    return data


def summarize(values):
    """Return median, mean, standard deviation and max of a list."""
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.median(values), statistics.mean(values), std, max(values)


def print_tables(data):
    """Print every trial, then a per-scale summary of cam0<-imu errors."""
    print('scale  trial   rot(deg)  pos(mm)   z(mm)   time shift(ms)')
    for scale in sorted(data):
        for r in data[scale]:
            trial, rot, pos, z, ts = r['trial'], r['rot'], r['pos'], r['z'], r['ts']
            print(f'{scale:5.1f}  {trial:>5}  {rot:8.3f}  {pos:7.2f}  {z:+7.2f}  {ts:+8.3f}')
    print()
    print('scale  n   rot(deg) median/max     pos(mm) median/mean+-std/max'
          '   pass(<0.5deg,<5mm)')
    for scale in sorted(data):
        rows = data[scale]
        r_med, _, _, r_max = summarize([r['rot'] for r in rows])
        p_med, p_mean, p_std, p_max = summarize([r['pos'] for r in rows])
        ok = sum(r['rot'] < ROT_LIMIT_DEG and r['pos'] < POS_LIMIT_MM for r in rows)
        print(f'{scale:5.1f}  {len(rows)}   {r_med:6.3f} / {r_max:6.3f}'
              f'     {p_med:6.2f} / {p_mean:6.2f}+-{p_std:5.2f} / {p_max:6.2f}'
              f'     {ok}/{len(rows)}')


def save_csv(data, path):
    """Write one line per trial so the numbers can go into the report."""
    with open(path, 'w') as f:
        f.write('scale,trial,rot_err_deg,pos_err_mm,pos_err_z_mm,timeshift_ms\n')
        for scale in sorted(data):
            for r in data[scale]:
                trial, rot, pos, z, ts = r['trial'], r['rot'], r['pos'], r['z'], r['ts']
                f.write(f'{scale},{trial},{rot:.4f},{pos:.3f},{z:.3f},{ts:.4f}\n')


def plot(data, path):
    """Draw each trial as a dot and the median as a line, per error type."""
    scales = sorted(data)
    panels = [
        ('rot', 'rotation error (deg)', ROT_LIMIT_DEG, True),
        ('pos', 'translation error (mm)', POS_LIMIT_MM, True),
        ('ts', 'time shift (ms, truth 0)', None, False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, label, limit, log_y) in zip(axes, panels):
        for scale in scales:
            vals = [r[key] for r in data[scale]]
            n = len(vals)
            # spread dots a little sideways so equal values do not overlap
            xs = [scale * (1 + 0.06 * (i - (n - 1) / 2)) for i in range(n)]
            ax.scatter(xs, vals, color='tab:blue', alpha=0.6, s=30,
                       label='each trial' if scale == scales[0] else None)
        medians = [statistics.median([r[key] for r in data[s]]) for s in scales]
        ax.plot(scales, medians, 'o-', color='tab:red', label='median')
        if limit is not None:
            ax.axhline(limit, color='gray', ls='--', label=f'pass limit {limit:g}')
        if key == 'ts':
            ax.axhline(0.0, color='black', lw=0.8)
        ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
            lo, hi = ax.get_ylim()
            ax.set_yticks([t for t in NICE_TICKS if lo <= t <= hi])
            ax.set_yticklabels([f'{t:g}' for t in NICE_TICKS if lo <= t <= hi])
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(scales)
        ax.set_xticklabels([f'x{s:g}' for s in scales])
        ax.set_xlabel('IMU noise scale')
        ax.set_ylabel(label)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8)
    counts = ', '.join(f'x{s:g}: {len(data[s])}' for s in scales)
    fig.suptitle(f'Kalibr cam0<-imu error vs IMU noise (trials per scale - {counts})')
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trials_dir', nargs='?',
                        default=os.path.expanduser('~/kalibr_data/sweep_imu_noise/trials'))
    args = parser.parse_args()
    data = load(args.trials_dir)
    if not data:
        raise SystemExit(f'no score.json under {args.trials_dir}')
    print_tables(data)
    out_dir = os.path.dirname(os.path.abspath(args.trials_dir))
    csv_path = os.path.join(out_dir, 'sweep_imu_noise_trials.csv')
    png_path = os.path.join(out_dir, 'sweep_imu_noise_trials.png')
    save_csv(data, csv_path)
    plot(data, png_path)
    print(f'saved {csv_path}')
    print(f'saved {png_path}')


if __name__ == '__main__':
    main()
