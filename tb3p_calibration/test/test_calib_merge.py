"""calib_merge.py 시험 (세션 23): 좌표 변환, URDF 읽기, 결과 합치기, 설계도 반영 확인."""
import importlib.util
import os

import numpy as np
import pytest

_PATH = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'calib_merge.py')
_SPEC = importlib.util.spec_from_file_location('calib_merge', _PATH)
cm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cm)


def random_pose(rng, scale=0.1):
    """작은 무작위 자세 4x4."""
    return cm.to_matrix(rng.uniform(-scale, scale, 3), rng.uniform(-0.3, 0.3, 3))


def nominal():
    """시험용 기준 캘리브레이션 dict."""
    def sensor(child, xyz):
        return {'parent': 'base_link', 'child': child, 'xyz': xyz, 'rpy': [0.0, 0.0, 0.0],
                'time_offset': 0.0}
    return {'source': 'nominal',
            'sensors': {'imu': sensor('imu_link', [-0.032, 0.0, 0.068]),
                        'lidar': sensor('base_scan', [-0.032, 0.0, 0.219]),
                        'camera': sensor('camera_bottom_screw_frame', [0.05, 0.0, 0.145])},
            'wheels': {'radius_left': 0.033, 'radius_right': 0.033, 'separation': 0.16}}


def joint_xml(name, parent, child, t):
    """4x4 -> URDF 고정 조인트 한 줄."""
    xyz, rpy = (' '.join(map(str, v)) for v in cm.from_matrix(t))
    return (f'<joint name="{name}" type="fixed"><parent link="{parent}"/>'
            f'<child link="{child}"/><origin xyz="{xyz}" rpy="{rpy}"/></joint>')


def fake_urdf(calib, t_screw_color):
    """시험용 작은 URDF: calib 값대로 (D435i 안쪽은 나사 -> 컬러 렌즈 하나로)."""
    s = calib['sensors']
    joints = [joint_xml('imu_joint', 'base_link', 'imu_link', cm.sensor_matrix(calib, 'imu')),
              joint_xml('scan_joint', 'base_link', 'base_scan', cm.sensor_matrix(calib, 'lidar')),
              joint_xml('cam_joint', 'base_link', s['camera']['child'],
                        cm.sensor_matrix(calib, 'camera')),
              joint_xml('lens_joint', s['camera']['child'], cm.COLOR, t_screw_color)]
    links = ''.join(f'<link name="{n}"/>' for n in
                    ('base_link', 'imu_link', 'base_scan', s['camera']['child'], cm.COLOR))
    # transmission 안의 joint (parent/child 없음)는 무시되어야 한다
    extra = '<transmission name="t"><joint name="imu_joint"/></transmission>'
    body = ''.join(joints)
    return f'<robot name="r">{links}{body}{extra}</robot>'


def test_rpy_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        rpy = rng.uniform(-1.2, 1.2, 3)
        back = cm.matrix_to_rpy(cm.rpy_to_matrix(*rpy))
        assert np.allclose(back, rpy, atol=1e-12)


def test_camera_from_lidar_recovers_screw_pose():
    rng = np.random.default_rng(1)
    for _ in range(20):
        t_base_lidar, t_base_screw = random_pose(rng), random_pose(rng)
        t_screw_color = random_pose(rng, 0.03)
        t_base_color = t_base_screw @ t_screw_color
        t_color_lidar = np.linalg.inv(t_base_color) @ t_base_lidar
        got = cm.camera_from_lidar(t_base_lidar, t_color_lidar, t_screw_color)
        assert np.allclose(got, t_base_screw, atol=1e-12)


def test_urdf_relative_and_transmission_ignored():
    rng = np.random.default_rng(2)
    calib = nominal()
    t_sc = random_pose(rng, 0.03)
    edges = cm.urdf_edges(fake_urdf(calib, t_sc))
    assert set(edges) == {'imu_link', 'base_scan', 'camera_bottom_screw_frame', cm.COLOR}
    assert np.allclose(cm.relative(edges, 'camera_bottom_screw_frame', cm.COLOR), t_sc)
    with pytest.raises(KeyError):
        cm.relative(edges, 'base_link', 'no_such_link')


def test_merge_fills_every_section():
    rng = np.random.default_rng(3)
    base = nominal()
    t_sc = random_pose(rng, 0.03)
    t_base_screw_true = random_pose(rng)
    t_color_lidar = np.linalg.inv(t_base_screw_true @ t_sc) @ cm.sensor_matrix(base, 'lidar')
    out = cm.merge(
        base,
        wheels={'path': 'w.yaml', 'wheels': {'radius_left': 0.0331, 'radius_right': 0.033,
                                             'separation': 0.162}},
        imu={'path': 'i.yaml', 'sensors': {'imu': {'xyz': [-0.028, 0.003, 0.068],
                                                   'rpy': [0.01, -0.012, 0.017]}}},
        lidar_cam={'path': 'lc.yaml', 'parent': cm.COLOR, 'child': cm.LIDAR,
                   'matrix': t_color_lidar.tolist(), 'beam_elev_deg': -0.444},
        time={'path': 't.yaml', 'time_offset_s': {'imu': 0.0, 'wheels': 0.0013,
                                                  'lidar': -0.0687}},
        t_screw_color=t_sc)
    assert out['source'] == 'estimated' and base['source'] == 'nominal'   # 기준은 그대로
    assert out['wheels']['separation'] == pytest.approx(0.162)
    assert out['wheels']['time_offset'] == pytest.approx(0.0013)
    assert out['sensors']['lidar']['time_offset'] == pytest.approx(-0.0687)
    assert out['sensors']['imu']['rpy'] == [0.01, -0.012, 0.017]
    rot, pos = cm.difference(cm.sensor_matrix(out, 'camera'), t_base_screw_true)
    assert rot < 1e-4 and pos < 1e-3            # 소수 7자리 반올림만큼
    assert set(out['estimated_from']) >= {'wheels', 'imu', 'camera', 'time_offset', 'lidar'}


def test_merge_rejects_wrong_frames():
    bad = {'path': 'x', 'parent': cm.LIDAR, 'child': cm.COLOR, 'matrix': np.eye(4).tolist()}
    with pytest.raises(ValueError):
        cm.merge(nominal(), lidar_cam=bad, t_screw_color=np.eye(4))


def test_check_urdf_zero_for_matching_urdf():
    rng = np.random.default_rng(4)
    calib = nominal()
    cm.set_pose(calib, 'camera', random_pose(rng))
    t_sc = random_pose(rng, 0.03)
    urdf = fake_urdf(calib, t_sc)
    t_color_lidar = (np.linalg.inv(cm.sensor_matrix(calib, 'camera') @ t_sc)
                     @ cm.sensor_matrix(calib, 'lidar'))
    rows = cm.check_urdf(urdf, calib, t_color_lidar)
    assert len(rows) == 4
    assert max(r[1] for r in rows) < 1e-5 and max(r[2] for r in rows) < 1e-5
    # 값이 다르면 차이가 보여야 한다
    calib['sensors']['imu']['rpy'] = [0.0, 0.0, 0.01]
    assert cm.check_urdf(urdf, calib)[0][1] == pytest.approx(np.degrees(0.01), abs=1e-6)
