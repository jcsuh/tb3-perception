#!/usr/bin/env python3
"""
바퀴 오도메트리와 UMBmark 캘리브레이션 공용 함수 (세션 15).

ROS 없이 쓸 수 있는 순수 계산만 모았다 (umbmark_run.py, umbmark_analyze.py가 사용).

바퀴 파라미터 (dict): radius_left, radius_right (m), separation (m, 두 바퀴 접지점 사이)
  바퀴 각도 φ (rad)가 변하면 그 바퀴가 굴러간 거리 = 반지름 × Δφ
  로봇 전진 = (dR + dL) / 2, 회전 = (dR - dL) / separation

UMBmark (Borenstein & Feng, 1996)
  로봇이 "자기 오도메트리로" 한 변 L인 정사각형을 시계(CW)·반시계(CCW) 방향으로 돈다.
  오도메트리가 완벽하면 출발점으로 돌아오지만, 바퀴 값이 틀리면 실제로는 어긋난 곳에 선다.
  - A형 오차(좌우 바퀴 지름이 다름, Ed = 오른쪽/왼쪽): 직진이 한쪽으로 휜다 -> 두 방향에서 같은 쪽
  - B형 오차(바퀴 간격이 틀림, Eb = 실제/공칭): 90° 회전이 넘치거나 모자란다 -> 두 방향에서 반대
  두 방향의 도착점을 보면 두 오차를 따로 구할 수 있다.
  평균 반지름(크기 배율)은 정사각형이 닫히는지에 영향이 없어서 UMBmark로는 알 수 없다.
"""
import math

import numpy as np
from scipy.optimize import least_squares
import yaml

CW, CCW = -1, 1
DIRECTION_NAME = {CW: 'CW', CCW: 'CCW'}


def wrap(a):
    """각도를 -π~π로."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def load_wheels(path):
    """캘리브레이션 YAML의 wheels 항목 -> dict (radius_left, radius_right, separation)."""
    with open(path) as f:
        data = yaml.safe_load(f)
    w = data['wheels']
    return {k: float(w[k]) for k in ('radius_left', 'radius_right', 'separation')}


def odom_step(pose, dphi_l, dphi_r, wheels):
    """바퀴 각도 변화 한 번으로 오도메트리 자세 (x, y, θ)를 한 걸음 진행 (실시간용)."""
    dl = float(wrap(dphi_l)) * wheels['radius_left']
    dr = float(wrap(dphi_r)) * wheels['radius_right']
    ds, dth = (dr + dl) / 2.0, (dr - dl) / wheels['separation']
    mid = pose[2] + dth / 2.0
    return (pose[0] + ds * math.cos(mid), pose[1] + ds * math.sin(mid), pose[2] + dth)


def integrate(phi_l, phi_r, wheels, pose0=(0.0, 0.0, 0.0)):
    """
    바퀴 각도 기록으로 오도메트리 자세 (N x 3: x, y, θ)를 계산한다.

    한 걸음마다 중간 방향(θ + Δθ/2)으로 전진량만큼 이동 (작은 호를 잘 근사).
    각도가 ±π에서 넘어가도 되게 변화량을 -π~π로 감는다 (기록 간격이 짧아 한 번에 π 이상 안 돎).
    """
    phi_l = np.asarray(phi_l, dtype=np.float64)
    phi_r = np.asarray(phi_r, dtype=np.float64)
    dl = wrap(np.diff(phi_l)) * wheels['radius_left']
    dr = wrap(np.diff(phi_r)) * wheels['radius_right']
    ds = (dr + dl) / 2.0
    dth = (dr - dl) / wheels['separation']
    th = pose0[2] + np.concatenate([[0.0], np.cumsum(dth)])
    mid = th[:-1] + dth / 2.0
    x = pose0[0] + np.concatenate([[0.0], np.cumsum(ds * np.cos(mid))])
    y = pose0[1] + np.concatenate([[0.0], np.cumsum(ds * np.sin(mid))])
    return np.column_stack([x, y, th])


def relative_pose(pose0, poses):
    """자세들을 pose0 기준 좌표로 (x, y, θ). pose0 = 출발 자세."""
    poses = np.atleast_2d(poses)
    c, s = math.cos(pose0[2]), math.sin(pose0[2])
    dx, dy = poses[:, 0] - pose0[0], poses[:, 1] - pose0[1]
    return np.column_stack([c * dx + s * dy, -s * dx + c * dy, wrap(poses[:, 2] - pose0[2])])


class SquareDriver:
    """
    오도메트리 자세를 받아 정사각형 주행 명령(v, ω)을 내는 상태 기계.

    단계: 직진 0, 회전 0, 직진 1, 회전 1, ... 회전 3 (모두 8단계), 단계 사이에 settle초 정지.
    직진은 오도메트리 좌표의 이상적인 정사각형 변을 따라가고(옆 어긋남·방향을 바로잡음),
    끝에 가까워질수록 천천히 가서 거리 tol_dist 안에서 멈춘다.
    회전도 끝에 가까워질수록 천천히, 방향 오차 tol_yaw 안이거나 지나치면 멈춘다.
    """

    def __init__(self, side, direction, v_max=0.2, w_max=0.6, v_min=0.01, w_min=0.03,
                 accel=0.3, ang_accel=1.0, k_dist=0.8, k_turn=1.5, k_lat=4.0, k_head=3.0,
                 tol_dist=0.0005, tol_yaw=math.radians(0.03), settle=0.5):
        self.side, self.direction = side, direction
        self.v_max, self.w_max, self.v_min, self.w_min = v_max, w_max, v_min, w_min
        self.accel, self.ang_accel = accel, ang_accel
        self.k_dist, self.k_turn, self.k_lat, self.k_head = k_dist, k_turn, k_lat, k_head
        self.tol_dist, self.tol_yaw, self.settle = tol_dist, tol_yaw, settle
        self.phase = -1

    def reset(self, pose, t):
        """출발 자세(오도메트리)와 시각을 정하고 첫 단계로."""
        self.x0, self.y0, self.th0 = pose
        self.phase, self.t_phase, self.waiting = 0, t, False
        self.last_err = None

    def heading(self, k):
        """k번째 변의 이상적인 방향."""
        return self.th0 + self.direction * k * math.pi / 2

    def corner(self, k):
        """k번째 변이 시작하는 이상적인 꼭짓점 (오도메트리 좌표)."""
        x, y = self.x0, self.y0
        for i in range(k):
            x += self.side * math.cos(self.heading(i))
            y += self.side * math.sin(self.heading(i))
        return x, y

    @property
    def done(self):
        """8단계를 모두 마쳤는지."""
        return self.phase >= 8

    def _next(self, t):
        self.phase += 1
        self.t_phase, self.waiting, self.last_err = t, True, None

    def step(self, pose, t):
        """현재 오도메트리 자세와 시각 -> (v, ω). 끝나면 (0, 0)."""
        if self.done:
            return 0.0, 0.0
        if self.waiting:                     # 단계 사이 정지
            if t - self.t_phase < self.settle:
                return 0.0, 0.0
            self.waiting, self.t_phase = False, t
        elapsed = t - self.t_phase
        k = self.phase // 2
        x, y, th = pose
        if self.phase % 2 == 0:              # 직진
            cx, cy = self.corner(k)
            h = self.heading(k)
            ux, uy = math.cos(h), math.sin(h)
            along = (x - cx) * ux + (y - cy) * uy
            lat = -(x - cx) * uy + (y - cy) * ux
            remain = self.side - along
            if remain <= self.tol_dist:
                self._next(t)
                return 0.0, 0.0
            v = min(self.v_max, self.v_min + self.k_dist * remain,
                    self.v_min + self.accel * elapsed)
            w = -self.k_lat * lat - self.k_head * float(wrap(th - h))
            return v, float(np.clip(w, -self.w_max, self.w_max))
        err = float(wrap(self.heading(k + 1) - th))  # 회전
        overshoot = self.last_err is not None and err * self.last_err < 0
        if abs(err) <= self.tol_yaw or overshoot:
            self._next(t)
            return 0.0, 0.0
        self.last_err = err
        w = min(self.w_max, self.w_min + self.k_turn * abs(err),
                self.w_min + self.ang_accel * elapsed)
        return 0.0, math.copysign(w, err)


def wheel_speeds(v, w, wheels):
    """명령 (v, ω) -> 바퀴 각속도 (rad/s) (Gazebo DiffDrive와 같은 식, 공칭 값 사용)."""
    r = (wheels['radius_left'] + wheels['radius_right']) / 2.0
    half = wheels['separation'] / 2.0
    return (v - w * half) / r, (v + w * half) / r


def simulate_run(true_wheels, odo_wheels, side, direction, dt=0.01, slip=0.0, rng=None):
    """
    (검증용) 운동학 시뮬레이션: 로봇이 odo_wheels로 계산한 오도메트리로 정사각형을 돈다.

    명령은 odo_wheels로 바퀴 속도로 바뀌고(DiffDrive), 실제 움직임은 true_wheels로 생긴다.
    slip: 바퀴 각속도에 곱하는 무작위 오차의 표준편차 (비율). 돌려주는 값: 기록 dict
    """
    rng = rng or np.random.default_rng(0)
    drv = SquareDriver(side, direction)
    phi = np.zeros(2)
    truth = np.zeros(3)
    odo = np.zeros(3)
    drv.reset(odo, 0.0)
    t = 0.0
    rec = {'t': [0.0], 'phi_l': [0.0], 'phi_r': [0.0], 'truth': [truth.copy()]}
    while not drv.done and t < 600:
        v, w = drv.step(odo, t)
        wl, wr = wheel_speeds(v, w, odo_wheels)
        if slip:
            wl *= 1 + slip * rng.standard_normal()
            wr *= 1 + slip * rng.standard_normal()
        dphi = np.array([wl, wr]) * dt
        for pose, wh in ((truth, true_wheels), (odo, odo_wheels)):
            p = integrate([0, dphi[0]], [0, dphi[1]], wh, pose)[-1]
            pose[:] = p
        phi += dphi
        t += dt
        rec['t'].append(t)
        rec['phi_l'].append(phi[0])
        rec['phi_r'].append(phi[1])
        rec['truth'].append(truth.copy())
    return {k: np.asarray(v) for k, v in rec.items()}


def square_model(alpha, beta, side, direction):
    """
    정사각형 주행의 오차 모형 (UMBmark): 직진마다 방향이 β만큼 휘고(일정한 곡률), 회전마다 α만큼 더 돈다.

    α는 회전 방향으로 더 돈 각도 (B형, 반시계에서는 왼쪽으로 +α, 시계에서는 오른쪽으로 +α),
    β는 직진 한 변 동안 왼쪽으로 휜 각도 (A형, 방향과 상관없이 같은 쪽).
    돌려주는 값: 출발 자세 기준 도착점 (x, y)
    """
    x = y = th = 0.0
    for _ in range(4):
        if abs(beta) < 1e-12:
            x += side * math.cos(th)
            y += side * math.sin(th)
        else:
            rad = side / beta                # 호의 반지름 (부호 있음)
            x += rad * (math.sin(th + beta) - math.sin(th))
            y += rad * (-math.cos(th + beta) + math.cos(th))
        th += beta
        th += direction * (math.pi / 2 + alpha)
    return x, y


def borenstein(end_cw, end_ccw, side, wheels):
    """
    방향별 도착점 무게중심(UMBmark 결과)으로 A·B형 오차를 구하고 바퀴 값을 고친다.

    Borenstein의 닫힌 식 대신, 같은 오차 모형(square_model)을 두 방향 무게중심 4개 숫자에
    최소제곱으로 맞춘다 (좌표 부호 규약을 틀릴 일이 없고, 오차가 커도 정확).
    Ed, Eb는 "주행에 쓴 바퀴 값" 대비 남은 오차다 (공칭으로 돌았으면 공칭 대비).
    그 뒤 Borenstein 식으로 바꾼다:
      R = (L/2) / sin(β/2),  Ed = (R + b/2) / (R - b/2)       (오른쪽/왼쪽 지름 비)
      Eb = (π/2) / (π/2 + α)                                  (실제/공칭 바퀴 간격)
    고친 값: 평균 반지름은 주행 때 값 그대로, 좌우 비율에 Ed를 곱하고, 간격은 Eb배
    (보정 후 주행에 다시 쓰면 남은 오차를 한 번 더 줄이는 반복 보정이 된다)
    """
    target = np.r_[end_cw, end_ccw]

    def res(p):
        return np.r_[square_model(p[0], p[1], side, CW),
                     square_model(p[0], p[1], side, CCW)] - target

    sol = least_squares(res, np.zeros(2))
    alpha, beta = sol.x
    b = wheels['separation']
    if abs(beta) > 1e-12:
        r_curve = (side / 2) / math.sin(beta / 2)
        e_d = (r_curve + b / 2) / (r_curve - b / 2)
    else:
        r_curve, e_d = math.inf, 1.0
    e_b = (math.pi / 2) / (math.pi / 2 + alpha)
    # 주행에 쓴 바퀴 값이 이미 좌우가 다르면(두 번째 보정) 비율을 곱해서 쌓는다
    e_total = wheels['radius_right'] / wheels['radius_left'] * e_d
    r_mean = (wheels['radius_left'] + wheels['radius_right']) / 2
    fixed = {'radius_left': 2 * r_mean / (e_total + 1),
             'radius_right': 2 * r_mean * e_total / (e_total + 1),
             'separation': e_b * b}
    return {'alpha': alpha, 'beta': beta, 'R': r_curve, 'Ed': e_d, 'Eb': e_b,
            'fit_rms_mm': float(np.sqrt(np.mean(sol.fun ** 2)) * 1000), 'wheels': fixed}


def interp_angles(t_joint, phi, t_query):
    """바퀴 각도 기록을 다른 시각들(t_query)로 선형 보간."""
    return np.interp(t_query, t_joint, phi)


def trajectory_residuals(params, runs, stride=5, head_scale=0.5):
    """
    바퀴 값 params로 다시 계산한 오도메트리와 정답 경로의 차이.

    params = [radius_left, radius_right, separation]. 모든 주행, 출발 자세 기준, stride개마다 한 점.

    위치 차이 (m)와 방향 차이 × head_scale (rad -> m 환산)를 이어 붙인다.
    """
    wheels = dict(zip(('radius_left', 'radius_right', 'separation'), params))
    out = []
    for run in runs:
        odo = integrate(run['phi_l'], run['phi_r'], wheels)
        tru = relative_pose(run['truth'][0], run['truth'])
        d = (odo - tru)[::stride]
        out.append(np.r_[d[:, 0], d[:, 1], wrap(d[:, 2]) * head_scale])
    return np.concatenate(out)


def fit_wheels(runs, init):
    """
    전체 경로 맞추기: 정답 경로(시뮬레이션은 Gazebo, 실제는 외부 측정)에 가장 잘 맞는 바퀴 값 3개.

    runs: [{'phi_l', 'phi_r', 'truth'(N x 3, 같은 시각)}, ...]
    돌려주는 값: (바퀴 dict, 주행별로 따로 맞춘 값들의 표준오차 dict, 잔차 RMS mm)
    UMBmark와 달리 평균 반지름(이동 거리 배율)도 정해진다.
    """
    x0 = np.array([init['radius_left'], init['radius_right'], init['separation']])
    scale = np.array([1e-4, 1e-4, 1e-3])
    sol = least_squares(trajectory_residuals, x0, args=(runs,), x_scale=scale, method='lm')
    each = []
    for run in runs:                     # 주행마다 따로 맞춰 흩어짐으로 불확실성을 본다
        s = least_squares(trajectory_residuals, sol.x, args=([run],), x_scale=scale,
                          method='lm')
        each.append(s.x)
    each = np.array(each)
    sem = each.std(axis=0, ddof=1) / np.sqrt(len(each)) if len(each) > 1 else np.full(3, np.nan)
    keys = ('radius_left', 'radius_right', 'separation')
    rms = float(np.sqrt(np.mean(sol.fun ** 2)) * 1000)
    return dict(zip(keys, map(float, sol.x))), dict(zip(keys, map(float, sem))), rms


def save_wheels(path, wheels, note):
    """바퀴 값을 캘리브레이션 YAML 형식(wheels 항목)으로 저장."""
    data = {'source': 'estimated', 'note': note,
            'wheels': {k: round(float(v), 7) for k, v in wheels.items()}}
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
