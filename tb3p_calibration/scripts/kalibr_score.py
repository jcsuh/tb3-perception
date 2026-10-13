#!/usr/bin/env python3
"""
Kalibr 결과를 시뮬레이션 정답과 비교해 채점한다.

정답은 녹화 파일(rosbag)에서 읽는다.
- 카메라 내부 파라미터: camera_info의 K (Gazebo 카메라 설정에서 나온 값, 왜곡 없음)
- 카메라-IMU, 카메라-카메라 외부 파라미터: /rig/tf_static (리그 설계도)
- 시간 오프셋: 0 (Gazebo는 모든 센서가 같은 시계)

--json 파일을 주면 같은 결과를 기계가 읽을 수 있는 형태로도 저장한다 (반복 실험 집계용).

사용 예:
  python3 kalibr_score.py ~/kalibr_data/rig_01 ~/kalibr_data/rig_01-camchain-imucam.yaml \
      --imu-frame camera_imu_optical_frame \
      --cam-frames camera_infra1_optical_frame camera_infra2_optical_frame
"""
import argparse
import json
import os

import numpy as np
from rclpy.serialization import deserialize_message
import rosbag2_py
from sensor_msgs.msg import CameraInfo
from tf2_msgs.msg import TFMessage
import yaml


def quat_to_matrix(x, y, z, w):
    """단위 쿼터니언을 3x3 회전 행렬로 바꾼다."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def transform_to_matrix(tf):
    """geometry_msgs/Transform을 4x4 변환 행렬로 바꾼다."""
    t, q = tf.translation, tf.rotation
    m = np.eye(4)
    m[:3, :3] = quat_to_matrix(q.x, q.y, q.z, q.w)
    m[:3, 3] = (t.x, t.y, t.z)
    return m


def frame_poses(edges):
    """{child: (parent, T_parent_child)}로부터 각 좌표계의 뿌리 기준 자세 T_root_frame을 구한다."""
    poses = {}

    def pose(frame):
        if frame not in poses:
            if frame not in edges:
                poses[frame] = np.eye(4)              # 뿌리
            else:
                parent, t_pc = edges[frame]
                poses[frame] = pose(parent) @ t_pc
        return poses[frame]

    for frame in list(edges):
        pose(frame)
    return poses


def read_truth(bag_dir, cam_topics):
    """Bag에서 정답 TF 트리와 카메라별 camera_info를 읽는다."""
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id='mcap'),
                rosbag2_py.ConverterOptions(input_serialization_format='cdr',
                                            output_serialization_format='cdr'))
    info_topics = {t.rsplit('/', 1)[0] + '/camera_info': t for t in cam_topics}
    reader.set_filter(rosbag2_py.StorageFilter(topics=['/rig/tf_static', *info_topics]))
    edges, infos = {}, {}
    while reader.has_next() and (not edges or len(infos) < len(info_topics)):
        topic, data, _ = reader.read_next()
        if topic == '/rig/tf_static':
            for ts in deserialize_message(data, TFMessage).transforms:
                edges[ts.child_frame_id] = (ts.header.frame_id, transform_to_matrix(ts.transform))
        elif info_topics[topic] not in infos:
            infos[info_topics[topic]] = deserialize_message(data, CameraInfo)
    return frame_poses(edges), infos


def rotation_angle_deg(r):
    """회전 행렬이 나타내는 회전의 크기(deg)."""
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0))))


def compare(name, t_est, t_true):
    """두 4x4 변환을 비교해 한 줄로 출력하고, 오차를 dict로 돌려준다."""
    d_rot = rotation_angle_deg(t_est[:3, :3].T @ t_true[:3, :3])
    d_pos = (t_est[:3, 3] - t_true[:3, 3]) * 1000
    print(f'  {name:<14} rot err {d_rot:6.3f} deg   pos err {np.linalg.norm(d_pos):6.2f} mm '
          f'({d_pos[0]:+.2f}, {d_pos[1]:+.2f}, {d_pos[2]:+.2f})   '
          f'true t (mm) ({t_true[0, 3] * 1000:+.2f}, {t_true[1, 3] * 1000:+.2f}, '
          f'{t_true[2, 3] * 1000:+.2f})')
    return {'rot_err_deg': d_rot, 'pos_err_mm': float(np.linalg.norm(d_pos)),
            'pos_err_xyz_mm': d_pos.tolist()}


def main():
    """명령줄에서 실행할 때의 시작점."""
    p = argparse.ArgumentParser(description='Kalibr 결과를 시뮬레이션 정답과 비교한다.')
    p.add_argument('bag', help='정답이 담긴 ROS 2 bag 폴더')
    p.add_argument('result', help='Kalibr 결과 YAML (camchain.yaml 또는 camchain-imucam.yaml)')
    p.add_argument('--imu-frame', default='camera_imu_optical_frame')
    p.add_argument('--cam-frames', nargs='+',
                   default=['camera_infra1_optical_frame', 'camera_infra2_optical_frame'])
    p.add_argument('--json', help='결과를 저장할 JSON 파일 (선택)')
    a = p.parse_args()
    score = {}

    with open(os.path.expanduser(a.result), encoding='utf-8') as f:
        result = yaml.safe_load(f)
    cams = sorted(k for k in result if k.startswith('cam'))
    topics = [result[c]['rostopic'] for c in cams]
    poses, infos = read_truth(os.path.expanduser(a.bag), topics)

    print('[카메라 내부 파라미터]  Kalibr - 정답(camera_info), 단위 px')
    for cam, topic in zip(cams, topics):
        fu, fv, pu, pv = result[cam]['intrinsics']
        k = infos[topic].k
        score[f'{cam}_intrinsics_err_px'] = [fu - k[0], fv - k[4], pu - k[2], pv - k[5]]
        print(f'  {cam} fx {fu:8.2f} ({fu - k[0]:+.2f})  fy {fv:8.2f} ({fv - k[4]:+.2f})  '
              f'cx {pu:7.2f} ({pu - k[2]:+.2f})  cy {pv:7.2f} ({pv - k[5]:+.2f})  '
              f'dist {np.round(result[cam]["distortion_coeffs"], 4).tolist()}')

    print('[외부 파라미터]  Kalibr vs 정답(/rig/tf_static)')
    for i, cam in enumerate(cams):
        t_root_cam = poses[a.cam_frames[i]]
        if 'T_cn_cnm1' in result[cam]:
            t_root_prev = poses[a.cam_frames[i - 1]]
            score[f'{cam}_from_cam{i - 1}'] = compare(
                f'{cam}<-cam{i - 1}', np.array(result[cam]['T_cn_cnm1']),
                np.linalg.inv(t_root_cam) @ t_root_prev)
        if 'T_cam_imu' in result[cam]:
            score[f'{cam}_from_imu'] = compare(
                f'{cam}<-imu', np.array(result[cam]['T_cam_imu']),
                np.linalg.inv(t_root_cam) @ poses[a.imu_frame])
        if 'timeshift_cam_imu' in result[cam]:
            shift_ms = result[cam]['timeshift_cam_imu'] * 1000
            score[f'{cam}_timeshift_ms'] = shift_ms
            print(f'  {cam} time shift {shift_ms:+.3f} ms (정답 0)')

    if a.json:
        with open(os.path.expanduser(a.json), 'w', encoding='utf-8') as f:
            json.dump(score, f, indent=2)
        print(f'saved {a.json}')


if __name__ == '__main__':
    main()
