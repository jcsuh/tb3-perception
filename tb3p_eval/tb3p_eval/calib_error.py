"""
캘리브레이션 YAML 두 개를 비교해 센서별 오차를 채점한다.

추정값(또는 공칭값) 파일과 정답 파일(sim_truth.yaml)의 sensors 항목을 이름이 같은 것끼리
비교해 위치 오차(mm), 회전 오차(deg), 시간 오프셋 오차(ms)를 표로 보여준다.
오차는 모두 "추정 - 정답" 방향이다.

사용 예:
  ros2 run tb3p_eval calib_error --estimate calibration.yaml --truth sim_truth.yaml
"""
import argparse
import sys

import numpy as np
from tb3p_eval.transforms import matrix_to_rpy, rotation_angle, rpy_to_matrix
import yaml


def load_sensors(path):
    """YAML 파일에서 sensors 항목을 읽는다."""
    with open(path, encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['sensors']


def sensor_error(est, truth):
    """
    센서 하나의 추정값과 정답을 비교한다.

    반환하는 dict의 키:
      dxyz  위치 오차 벡터 (m), trans 위치 오차 크기 (m)
      rot   회전 오차 크기 (rad), drpy 회전 오차의 roll, pitch, yaw 성분 (rad, 추정 - 정답:
            정답 자세에서 추정 자세로 가는 회전)
      dtime 시간 오프셋 오차 (s)
    """
    dxyz = np.array(est['xyz'], dtype=float) - np.array(truth['xyz'], dtype=float)
    r_est = rpy_to_matrix(*est['rpy'])
    r_true = rpy_to_matrix(*truth['rpy'])
    r_err = r_true.T @ r_est     # 정답 -> 추정 (위치와 같은 방향: 추정 - 정답)
    dtime = float(est.get('time_offset', 0.0)) - float(truth.get('time_offset', 0.0))
    return {
        'dxyz': dxyz,
        'trans': float(np.linalg.norm(dxyz)),
        'rot': rotation_angle(r_err),
        'drpy': np.array(matrix_to_rpy(r_err)),
        'dtime': dtime,
    }


def compare(est_path, truth_path):
    """두 파일에서 이름이 같은 센서를 모두 비교한다."""
    est = load_sensors(est_path)
    truth = load_sensors(truth_path)
    return {name: sensor_error(est[name], truth[name]) for name in truth if name in est}


def format_table(results):
    """비교 결과를 표 문자열로 만든다."""
    header = ('sensor    dpos mm (x, y, z)           |dp| mm   '
              'drot deg (r, p, y)       |dth| deg   dt ms')
    lines = [header, '-' * len(header)]
    for name, e in results.items():
        dx, dy, dz = e['dxyz'] * 1000.0
        rr, rp, ry = np.degrees(e['drpy'])
        trans_mm = e['trans'] * 1000.0
        rot_deg = np.degrees(e['rot'])
        dt_ms = e['dtime'] * 1000.0
        lines.append(
            f'{name:<9} ({dx:+6.1f}, {dy:+6.1f}, {dz:+6.1f})  {trans_mm:7.1f}   '
            f'({rr:+5.2f}, {rp:+5.2f}, {ry:+5.2f})  {rot_deg:8.2f}  {dt_ms:6.1f}')
    return '\n'.join(lines)


def main(argv=None):
    """명령줄에서 실행할 때의 시작점."""
    parser = argparse.ArgumentParser(description='캘리브레이션 YAML을 정답과 비교해 채점한다.')
    parser.add_argument('--estimate', required=True, help='추정값 또는 공칭값 YAML')
    parser.add_argument('--truth', required=True, help='정답 YAML (sim_truth.yaml)')
    args = parser.parse_args(argv)

    print(f'추정: {args.estimate}')
    print(f'정답: {args.truth}')
    print('오차 = 추정 - 정답\n')
    print(format_table(compare(args.estimate, args.truth)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
