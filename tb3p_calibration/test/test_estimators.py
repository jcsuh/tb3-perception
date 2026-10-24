"""추정기 시험 (세션 25): LiDAR-카메라 점-평면 최적화, 시간 오프셋 추정 (Python 원본)."""
import importlib.util
import math
import os
import sys

import numpy as np

_SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'scripts')
sys.path.insert(0, _SCRIPTS)          # 스크립트가 같은 폴더의 모듈(scan_match 등)을 불러온다


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_SCRIPTS, name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lcc = _load('lidar_cam_calib')
toe = _load('time_offset_estimate')


def pose(rvec, t):
    """회전 벡터 + 위치 -> 4x4."""
    m = np.eye(4)
    m[:3, :3] = lcc.cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    m[:3, 3] = t
    return m


def board_poses(t_cam_lidar, rng, n=12, noise=0.0):
    """
    보드 자세 n개: LiDAR 스캔 평면(z = 0) 위 선분 점들과, 그 점들을 지나는 카메라 기준 평면 (n, d).

    보드는 거리 0.7–1.3 m, 옆으로 돌리고(yaw) 앞뒤로 젖혀서(tilt) 놓는다 (세션 12의 자세와 비슷하게).
    """
    poses = []
    for _ in range(n):
        dist, yaw = rng.uniform(0.7, 1.3), rng.uniform(-0.6, 0.6)
        tilt, bearing = rng.uniform(-0.45, 0.45), rng.uniform(-0.4, 0.4)
        c = dist * np.array([math.cos(bearing), math.sin(bearing), 0.0])
        along = np.array([-math.sin(bearing + yaw), math.cos(bearing + yaw), 0.0])
        s = np.linspace(-0.15, 0.15, 25)
        pts = c + s[:, None] * along                       # LiDAR 좌표계, 보드 위 점들
        facing = -c / np.linalg.norm(c)                     # 보드가 LiDAR 쪽을 봄
        n_l = math.cos(tilt) * np.cross(along, [0.0, 0.0, 1.0]) * np.sign(
            np.cross(along, [0.0, 0.0, 1.0]) @ facing) + math.sin(tilt) * np.array([0, 0, 1.0])
        n_l -= (n_l @ along) * along                       # 선분 방향과 수직 (점이 평면 위)
        n_l /= np.linalg.norm(n_l)
        n_c = t_cam_lidar[:3, :3] @ n_l
        d_c = float(n_c @ (t_cam_lidar[:3, :3] @ c + t_cam_lidar[:3, 3]))
        poses.append({'points': (pts + rng.normal(0, noise, pts.shape)).tolist(),
                      'n': n_c.tolist(), 'd': d_c})
    return poses


def test_point_to_plane_recovers_exact_relation():
    rng = np.random.default_rng(0)
    # 카메라 광학 좌표(z 앞, x 오른쪽, y 아래) <- LiDAR(x 앞, y 왼쪽, z 위)
    base = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    t_true = np.eye(4)
    t_true[:3, :3] = lcc.cv2.Rodrigues(np.array([0.02, -0.03, 0.04]))[0] @ base
    t_true[:3, 3] = [0.01, 0.07, -0.08]
    t_init = t_true @ pose([0.03, -0.02, 0.05], [0.004, -0.008, 0.006])   # 약 3.5°, 1 cm 틀림
    t_est, sol, _ = lcc.calibrate(board_poses(t_true, rng), t_init)
    rot, pos, _ = lcc.errors(t_est, t_true)
    assert rot < 1e-6 and pos < 1e-6 and sol.cost < 1e-16


def test_point_to_plane_with_noise_and_uncertainty():
    rng = np.random.default_rng(1)
    base = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    t_true = np.eye(4)
    t_true[:3, :3] = base
    t_true[:3, 3] = [0.0, 0.07, -0.08]
    poses = board_poses(t_true, rng, n=18, noise=0.002)        # 세션 13: 20장 평균 후 약 2 mm
    t_init = t_true @ pose([0.02, 0.0, -0.02], [0.005, 0.0, 0.0])
    t_est, sol, (pts, _, _, _, _) = lcc.calibrate(poses, t_init)
    rot, pos, _ = lcc.errors(t_est, t_true)
    rot_std, pos_std, eigval, _ = lcc.uncertainty(sol, len(pts))
    assert rot < 0.5 and pos < 10.0                          # 합격 기준 안
    assert rot < 4 * np.linalg.norm(rot_std)                 # 실제 오차가 불확실성과 비슷한 크기
    assert np.all(eigval > 0)


def rotation_record(duration=40.0, rate=200.0, seed=0):
    """여러 사인파를 더한 불규칙 회전의 누적 각도 (t, θ) (세션 17 회전 기록과 비슷하게)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, 1.0 / rate)
    w = sum(a * np.sin(2 * math.pi * f * t + p) for a, f, p in
            zip(rng.uniform(0.2, 0.4, 6), (0.13, 0.29, 0.53, 0.71, 1.1, 1.7),
                rng.uniform(0, 2 * math.pi, 6)))
    return t, toe.cumulative(t, w)


def test_time_offset_increment_fit():
    t, th = rotation_record()
    # 센서: 10 Hz, 시각이 30 ms 늦게 찍힘, 배율 1.006, 바이어스 0.01 rad/s
    ts = np.arange(1.0, 39.0, 0.1)
    sensor = (ts, 1.006 * np.interp(ts - 0.030, t, th) + 0.01 * ts)
    delta, _, scale = toe.fit_offset(sensor, (t, th))
    assert abs(delta - (-0.030)) < 2e-4                       # 정답 time_offset = -30 ms
    assert abs(scale - 1.006) < 1e-3


def test_time_offset_xcorr_agrees():
    t, th = rotation_record(seed=3)
    ts = np.arange(0.5, 39.5, 0.005)                          # 200 Hz 센서, 70 ms 늦음
    sensor = (ts, np.interp(ts - 0.070, t, th))
    lag, peak = toe.xcorr_offset(sensor, (t, th))
    assert abs(lag - (-0.070)) < 2e-3 and peak > 0.99
