#!/usr/bin/env python3
"""
드리프트 검사 1: LiDAR 점과 깊이 영상이 같은 곳을 같은 거리로 보는지 (세션 20, ROS 없는 계산).

캘리브레이션이 맞으면 LiDAR 점을 깊이 카메라 영상에 투영했을 때 그 픽셀의 깊이가 점의 깊이와 같다.
카메라 브래킷이 돌아가면(드리프트) 점이 엉뚱한 픽셀에 떨어져 거리 차이(잔차)가 커진다.
잔차가 커졌다는 것만이 아니라 "얼마나 돌았나"를 직접 찾는다:
  카메라가 마운트 나사의 수직축 둘레로 δ만큼 돌았다고 가정하고, 잔차가 가장 작은 δ를 찾는다.
  (Gazebo에서 /sim/camera_mount_yaw_cmd로 돌리는 축과 같다)

좌표계 (모두 4x4 변환, T_a_b = b를 a 기준으로)
  T_base_lidar : 공칭 TF (base_footprint <- base_scan)
  T_base_cam0  : 캘리브레이션된 깊이 카메라 자세 = T_base_lidar · T_color_lidar⁻¹ · T_color_depth
  mount        : 마운트 나사 위치 (base 좌표, 공칭 TF의 camera_bottom_screw_frame)
"""
import math

import numpy as np

TRUNC = 0.10       # 잔차를 이 값(m)에서 자른다: 벽 경계에서 생기는 큰 잔차에 휘둘리지 않게
MIN_POINTS = 30


def rot_z(a):
    """Z축 회전 4x4."""
    c, s = math.cos(a), math.sin(a)
    t = np.eye(4)
    t[:2, :2] = [[c, -s], [s, c]]
    return t


def translate(p):
    """평행이동 4x4."""
    t = np.eye(4)
    t[:3, 3] = p
    return t


def camera_pose_with_yaw(t_base_cam0, mount, delta):
    """마운트 나사의 수직축(base z) 둘레로 δ (rad) 돈 카메라 자세 T_base_cam."""
    return translate(mount) @ rot_z(delta) @ translate(-np.asarray(mount)) @ t_base_cam0


def lidar_points(ranges, angle_min, angle_inc, elev_deg=0.0, rmin=0.15, rmax=8.0):
    """스캔 -> LiDAR 좌표계 3D 점 (빔 높이각 원뿔 반영), N x 3."""
    r = np.asarray(ranges, dtype=np.float64)
    a = angle_min + angle_inc * np.arange(len(r))
    ok = np.isfinite(r) & (r > rmin) & (r < rmax)
    r, a = r[ok], a[ok]
    e = math.radians(elev_deg)
    return np.column_stack([r * math.cos(e) * np.cos(a), r * math.cos(e) * np.sin(a),
                            r * math.sin(e) * np.ones_like(r)])


def to_frame(t, pts):
    """점들(N x 3)에 4x4 변환 적용."""
    return pts @ t[:3, :3].T + t[:3, 3]


def residuals(pts_base, t_base_cam, k, band, v0, margin=2):
    """
    점들(base 좌표)을 깊이 영상(band: v0행부터 잘라 둔 띠)에 투영해 잔차를 구한다 (영상 깊이 - 점 깊이, m).

    영상 밖, 띠 밖, 영상에 값이 없는 점은 뺀다. 돌려주는 값: 잔차 배열
    """
    pc = to_frame(np.linalg.inv(t_base_cam), pts_base)
    front = pc[:, 2] > 0.2
    pc = pc[front]
    u = k[0, 0] * pc[:, 0] / pc[:, 2] + k[0, 2]
    v = k[1, 1] * pc[:, 1] / pc[:, 2] + k[1, 2]
    ui, vi = np.round(u).astype(int), np.round(v).astype(int) - v0
    h, w = band.shape
    ok = (ui >= margin) & (ui < w - margin) & (vi >= 0) & (vi < h)
    d = band[vi[ok], ui[ok]].astype(np.float64)
    good = np.isfinite(d) & (d > 0.1)
    return d[good] - pc[ok, 2][good]


def robust_cost(r):
    """잘린 절댓값 평균 (m): 가까운 점은 그대로, 먼(어긋난) 점은 TRUNC로."""
    if len(r) < MIN_POINTS:
        return np.nan
    return float(np.mean(np.minimum(np.abs(r), TRUNC)))


def yaw_cost_curve(frames, t_base_cam0, mount, k, v0, deltas):
    """
    여러 프레임을 함께 놓고 δ마다 비용 (프레임별 비용의 평균).

    frames: [(base 좌표 점, 깊이 띠), ...]
    """
    out = []
    for d in deltas:
        tc = camera_pose_with_yaw(t_base_cam0, mount, d)
        costs = [robust_cost(residuals(p, tc, k, band, v0)) for p, band in frames]
        costs = [c for c in costs if np.isfinite(c)]
        out.append(np.mean(costs) if costs else np.nan)
    return np.array(out)


def estimate_yaw(frames, t_base_cam0, mount, k, v0, max_deg=8.0, step_deg=0.1):
    """
    카메라가 돈 각도 δ (rad)를 찾는다: 넓게 0.1° 간격으로 훑고 가장 작은 곳에서 포물선 보간.

    돌려주는 값: (δ, δ=0일 때 비용, 최소 비용, (각도들, 비용들))
    """
    deltas = np.radians(np.arange(-max_deg, max_deg + 1e-9, step_deg))
    costs = yaw_cost_curve(frames, t_base_cam0, mount, k, v0, deltas)
    if not np.isfinite(costs).any():
        return np.nan, np.nan, np.nan, (deltas, costs)
    i = int(np.nanargmin(costs))
    best = deltas[i]
    if 0 < i < len(costs) - 1 and np.isfinite(costs[i - 1]) and np.isfinite(costs[i + 1]):
        y0, y1, y2 = costs[i - 1], costs[i], costs[i + 1]
        den = y0 - 2 * y1 + y2
        if den > 0:
            best += 0.5 * (y0 - y2) / den * (deltas[1] - deltas[0])
    zero = costs[np.argmin(np.abs(deltas))]
    return float(best), float(zero), float(costs[i]), (deltas, costs)


def cumulative_yaw(t, gyro_z):
    """자이로 z 적분 누적 yaw (사다리꼴)."""
    return np.concatenate([[0.0], np.cumsum(np.diff(t) * (gyro_z[1:] + gyro_z[:-1]) / 2)])


def motion_compensate(pts_base, t_scan, t_img, imu_t, imu_yaw):
    """스캔 시각과 영상 시각 사이 로봇이 돈 만큼 점을 되돌린다 (base z 둘레)."""
    dpsi = np.interp(t_img, imu_t, imu_yaw) - np.interp(t_scan, imu_t, imu_yaw)
    return to_frame(rot_z(-dpsi), pts_base), dpsi
