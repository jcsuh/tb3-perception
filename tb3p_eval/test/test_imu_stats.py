"""imu_stats의 단위 테스트."""
import math

import numpy as np
from tb3p_eval.imu_stats import imu_statistics


def test_rate_tilt_and_noise():
    n = 2000
    stamps = np.arange(n) * 0.005
    rng = np.random.default_rng(0)
    gyro = rng.normal(0.0, 3.7e-3, size=(n, 3))
    accel = np.tile([0.12, 0.0, 9.8], (n, 1))
    s = imu_statistics(stamps, gyro, accel)
    assert math.isclose(s['rate'], 200.0, rel_tol=1e-9)
    assert math.isclose(s['pitch'], math.atan2(-0.12, 9.8), rel_tol=1e-9)
    assert abs(s['roll']) < 1e-12
    assert np.all(np.abs(s['gyro_std'] / 3.7e-3 - 1.0) < 0.05)
