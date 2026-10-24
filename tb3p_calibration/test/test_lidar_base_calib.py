"""lidar_base_calib.py 시험 (세션 24): 손-눈 풀이, 바닥 평면, 바닥 + 손-눈 잇기."""
import importlib.util
import math
import os
import sys

import numpy as np

_SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'scripts')
sys.path.insert(0, _SCRIPTS)          # lidar_base_calib가 같은 폴더의 모듈을 불러온다
_SPEC = importlib.util.spec_from_file_location('lidar_base_calib',
                                               os.path.join(_SCRIPTS, 'lidar_base_calib.py'))
lb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lb)
cm = lb.cm

BODY = (0.0, -math.atan2(lb.CASTER_GAP, lb.CASTER_X))


def motions(x_true, n, rng, noise=0.0):
    """무작위 몸 움직임 A와 그때 LiDAR가 본 움직임 B = X⁻¹ A X (+ 잡음)."""
    out = []
    x_inv = np.linalg.inv(x_true)
    for _ in range(n):
        a = lb.se2(rng.uniform(-0.15, 0.15), rng.uniform(-0.02, 0.02), rng.uniform(-0.8, 0.8))
        b = x_inv @ a @ x_true
        b[:2, 2] += rng.normal(0, noise, 2)
        out.append((a, b))
    return out


def test_hand_eye_exact_and_noisy():
    rng = np.random.default_rng(0)
    x_true = lb.se2(-0.035, 0.004, math.radians(2.0))
    phi, px, py = lb.solve_hand_eye_linear(motions(x_true, 50, rng))
    assert abs(phi - math.radians(2.0)) < 1e-9
    assert abs(px + 0.035) < 1e-9 and abs(py - 0.004) < 1e-9
    he = lb.solve_hand_eye(motions(x_true, 300, rng, noise=0.002))
    assert abs(math.degrees(he['phi']) - 2.0) < 0.1
    assert abs(he['p'][0] + 0.035) < 0.001 and abs(he['p'][1] - 0.004) < 0.001
    assert 0 < math.degrees(he['std_phi']) < 0.1        # 불확실성도 그럴듯한 크기


def test_fit_floor_ignores_walls():
    rng = np.random.default_rng(1)
    n_true = np.array([0.0, -0.95, -0.3])
    n_true /= np.linalg.norm(n_true)
    d_true = -0.17
    # 바닥 점 3000개 + 바닥과 수직인 벽 점 2000개 (잡음 3 mm)
    a, b = np.linalg.svd(n_true[None])[2][1:]
    uv = rng.uniform(-1, 1, (3000, 2))
    floor = d_true * n_true + uv[:, :1] * a + uv[:, 1:] * b
    wall = np.column_stack([rng.uniform(-1, 1, 2000), rng.uniform(-1, 1, 2000),
                            np.full(2000, 1.5)])
    pts = np.vstack([floor, wall]) + rng.normal(0, 0.003, (5000, 3))
    n, d, cnt, rms = lb.fit_floor(pts, n_true + [0.0, 0.05, 0.0], rng=rng)
    assert math.degrees(math.acos(min(1.0, n @ n_true))) < 0.2
    assert abs(d - d_true) < 0.001 and cnt > 2500 and rms < 0.005


def test_combine_recovers_lidar_and_camera():
    rng = np.random.default_rng(2)
    t_bc_true = cm.to_matrix([0.06, -0.01, 0.16],
                             [-math.pi / 2 - 0.02, 0.01, -math.pi / 2 + 0.03])
    t_color_lidar = np.linalg.inv(t_bc_true) @ cm.to_matrix([-0.035, 0.004, 0.223],
                                                            [0.005, -0.003, 0.035])
    t_color_depth = cm.to_matrix([0.0, 0.015, 0.0], [0.0, 0.0, 0.0])
    t_bd = t_bc_true @ t_color_depth
    n_b, d_b = lb.floor_in_base(BODY)
    floor_c = (t_bd[:3, :3].T @ n_b, d_b - n_b @ t_bd[:3, 3])
    yaw, qx, qy = lb.lidar_in_level(t_bc_true @ t_color_lidar, BODY)
    he = {'phi': yaw, 'p': (qx, qy)}
    start = t_bc_true @ cm.to_matrix(rng.uniform(-0.01, 0.01, 3), rng.uniform(-0.05, 0.05, 3))
    got, res = lb.combine(floor_c, he, t_color_lidar, t_color_depth, BODY, start)
    rot, pos = cm.difference(got, t_bc_true)
    assert rot < 1e-6 and pos < 1e-6 and np.abs(res).max() < 1e-9


def test_floor_passes_through_wheel_contact():
    n_b, d_b = lb.floor_in_base(BODY)
    assert abs(n_b @ lb.PIVOT - d_b) < 1e-12
    assert n_b[0] > 0          # 몸이 뒤로 기울면(pitch < 0) 몸 기준 위 방향은 앞쪽으로 기운다


def test_select_floor_drops_bad_frames():
    rng = np.random.default_rng(3)
    n0 = np.array([0.0, -0.95, -0.31])
    n0 /= np.linalg.norm(n0)
    good = [(n0 + rng.normal(0, 2e-4, 3), -0.1694 + rng.normal(0, 3e-4), 8000, 0.002)
            for _ in range(20)]
    good = [(f[0] / np.linalg.norm(f[0]),) + f[1:] for f in good]
    tilted = n0 + [0.0, 0.0, 0.05]
    bad = [(tilted / np.linalg.norm(tilted), -0.20, 2100, 0.003),   # 바닥 점이 적음
           (n0, -0.140, 7000, 0.002)]                                # 거리가 벗어남
    n, d, ok, why = lb.select_floor(good + bad)
    assert ok[:20].all() and not ok[20:].any()
    assert why == {'few': 1, 'off': 1}
    assert math.degrees(math.acos(min(1.0, n @ n0))) < 0.02 and abs(d + 0.1694) < 0.0003
