"""calib_error와 transforms의 단위 테스트."""
import math

import numpy as np
from tb3p_eval.calib_error import sensor_error
from tb3p_eval.transforms import matrix_to_rpy, rpy_to_matrix


def _sensor(xyz, rpy, t=0.0):
    return {'xyz': list(xyz), 'rpy': list(rpy), 'time_offset': t}


def test_rpy_round_trip():
    rpy = (0.1, -0.2, 0.3)
    assert np.allclose(matrix_to_rpy(rpy_to_matrix(*rpy)), rpy)


def test_identical_is_zero():
    s = _sensor((0.1, 0.0, 0.2), (0.01, 0.02, 0.03))
    e = sensor_error(s, s)
    assert e['trans'] < 1e-12
    assert e['rot'] < 1e-6


def test_yaw_only_error():
    est = _sensor((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    truth = _sensor((0.0, 0.0, 0.0), (0.0, 0.0, math.radians(2.0)))
    e = sensor_error(est, truth)
    assert math.isclose(math.degrees(e['rot']), 2.0, rel_tol=1e-6)
    assert math.isclose(math.degrees(e['drpy'][2]), 2.0, rel_tol=1e-6)


def test_translation_and_time():
    est = _sensor((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), t=0.010)
    truth = _sensor((0.003, 0.004, 0.0), (0.0, 0.0, 0.0), t=0.0)
    e = sensor_error(est, truth)
    assert math.isclose(e['trans'], 0.005, rel_tol=1e-9)  # 3-4-5 직각삼각형
    assert math.isclose(e['dtime'], 0.010, rel_tol=1e-9)
