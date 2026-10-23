#!/usr/bin/env python3
"""
2D LiDAR 스캔 매칭: 연속한 두 스캔 사이에 로봇이 얼마나 돌고 움직였나 (세션 17).

ROS 없이 쓰는 순수 계산 (time_offset_estimate.py가 사용).
  1) 첫 추정: 제자리 회전이면 스캔(거리 400개)이 통째로 옆으로 밀린다 -> 몇 칸 밀렸는지 찾는다
  2) ICP(점-선): 새 스캔의 점마다 이전 스캔의 가장 가까운 점과 그 점의 표면 방향(법선)을 찾고,
     "점이 그 표면 위에 오도록" 회전·이동을 조금씩 고치기를 반복한다
돌려주는 값: 새 스캔 좌표 -> 이전 스캔 좌표 변환 (dx, dy, dθ). dθ = 그 사이 로봇(LiDAR)이 돈 각도
"""
import math

import numpy as np
from scipy.spatial import cKDTree


def scan_points(ranges, angle_min, angle_inc, rmin=0.12, rmax=8.0):
    """거리 배열 -> 점 (N x 2, LiDAR 좌표계). 유효 범위 밖은 뺀다."""
    r = np.asarray(ranges, dtype=np.float64)
    a = angle_min + angle_inc * np.arange(len(r))
    ok = np.isfinite(r) & (r > rmin) & (r < rmax)
    return np.column_stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])])


def shift_guess(r_prev, r_new, angle_inc, max_shift=40):
    """
    제자리 회전 첫 추정: r_new[i] ≈ r_prev[i + s]가 가장 잘 맞는 칸 수 s -> 회전 s × 간격 (rad).

    로봇이 +θ 돌면 같은 벽이 새 스캔에서는 θ만큼 작은 각도에 보인다.
    """
    a = np.where(np.isfinite(r_prev), r_prev, np.nan)
    b = np.where(np.isfinite(r_new), r_new, np.nan)
    best, best_s = np.inf, 0
    for s in range(-max_shift, max_shift + 1):
        d = np.abs(b - np.roll(a, -s))
        ok = np.isfinite(d)
        if ok.sum() < 0.3 * len(a):
            continue
        cost = np.median(d[ok])
        if cost < best:
            best, best_s = cost, s
    return best_s * angle_inc


def normals(points, tree, k=6):
    """점마다 이웃 k개로 구한 표면 법선 (N x 2)과 표면이 얼마나 곧은지(0~1)."""
    _, idx = tree.query(points, k=k)
    nb = points[idx]                                  # N x k x 2
    c = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', c, c)
    w, v = np.linalg.eigh(cov)
    n = v[:, :, 0]                                    # 가장 작은 고유값 방향 = 법선
    flat = 1.0 - w[:, 0] / np.maximum(w[:, 1], 1e-12)
    return n, flat


def rot(th):
    """2D 회전 행렬."""
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


def icp(src, dst, init=(0.0, 0.0, 0.0), iters=30, max_dist=0.3):
    """
    점-선 ICP: src(새 스캔)를 dst(이전 스캔)에 맞추는 (dx, dy, dθ)와 맞춘 점 수, 잔차 RMS (m).

    매 반복: 가까운 점 짝짓기 -> 법선 방향 거리 n·(R p + t - q)를 작게 하는 작은 보정을
    선형 최소제곱으로 -> 짝짓기 거리 한도를 조금씩 줄인다.
    """
    tree = cKDTree(dst)
    nrm, flat = normals(dst, tree)
    x = np.array(init, dtype=np.float64)
    lim = max_dist
    n_used, rms = 0, np.nan
    for it in range(iters):
        p = src @ rot(x[2]).T + x[:2]
        dist, j = tree.query(p)
        ok = (dist < lim) & (flat[j] > 0.9)
        if ok.sum() < 20:
            break
        n, q, pp = nrm[j[ok]], dst[j[ok]], p[ok]
        r = np.sum(n * (pp - q), axis=1)
        # d/dθ (R p) = (-p_y, p_x) (현재 회전 적용 후)
        jac = np.column_stack([n[:, 0], n[:, 1], n[:, 0] * -pp[:, 1] + n[:, 1] * pp[:, 0]])
        # 큰 잔차에 덜 휘둘리게 (Huber 비슷한 가중치)
        w = 1.0 / np.maximum(1.0, np.abs(r) / 0.02)
        step, *_ = np.linalg.lstsq(jac * w[:, None], -r * w, rcond=None)
        x[0] += step[0]
        x[1] += step[1]
        x[2] += step[2]
        n_used, rms = int(ok.sum()), float(np.sqrt(np.mean(r ** 2)))
        lim = max(0.05, lim * 0.8)
        if np.abs(step).max() < 1e-7:
            break
    return x, n_used, rms


def scan_rotations(scans, angle_min, angle_inc):
    """
    연속한 스캔 쌍마다 회전 dθ (rad)와 맞춘 점 수.

    scans: M x N 거리 배열. 돌려주는 값: (dθ M-1개, 점 수 M-1개, 잔차 RMS M-1개)
    """
    out, used, res = [], [], []
    prev_pts = scan_points(scans[0], angle_min, angle_inc)
    for k in range(1, len(scans)):
        pts = scan_points(scans[k], angle_min, angle_inc)
        th0 = shift_guess(scans[k - 1], scans[k], angle_inc)
        x, n, r = icp(pts, prev_pts, init=(0.0, 0.0, th0))
        out.append(x[2])
        used.append(n)
        res.append(r)
        prev_pts = pts
    return np.array(out), np.array(used), np.array(res)
