#!/usr/bin/env python3
"""
각 캘리브레이션 결과를 calibration.yaml 하나로 합치고, 설계도(URDF)에 그대로 반영되는지 확인한다 (세션 23).

합치는 것 (모두 선택, 안 주면 기준 파일 값을 그대로 둔다)
  --wheels     umbmark_analyze.py의 *_wheels_ls.yaml      -> wheels
  --imu        imu_mount_calib.py의 *_imu.yaml            -> sensors.imu (xyz, rpy)
  --lidar-cam  lidar_cam_calib.py의 *_result.yaml         -> sensors.camera (xyz, rpy)
  --time       time_offset_estimate.py의 *_time_offset.yaml -> 각 센서 time_offset, wheels.time_offset

카메라 자세는 바로 나오지 않는다
  LiDAR-카메라 캘리브레이션이 구한 것은 두 센서 사이 관계(T_color_lidar)뿐이다.
  calibration.yaml에는 base_link 기준 카메라 브래킷 나사(camera_bottom_screw_frame) 자세가 들어가므로
    T_base_screw = T_base_lidar · T_color_lidar⁻¹ · T_screw_color⁻¹
  T_base_lidar는 calibration.yaml의 LiDAR 값, T_screw_color(나사 -> 컬러 렌즈)는 D435i 설계값으로
  설계도(xacro)를 직접 돌려 읽는다. 따라서 LiDAR 자세가 틀리면 카메라도 같이 틀린다.

설계도 반영 확인 (--check, 기본으로 함)
  새 YAML로 xacro를 돌려 URDF를 만들고, 그 URDF에서 다시 읽은
    base_link -> imu_link, base_scan, camera_bottom_screw_frame 와
    LiDAR-카메라 관계 T_color_lidar
  가 넣은 값과 같은지 본다. 같으면 robot_state_publisher가 발행하는 /tf에 그대로 들어간다.

결과: YAML 하나 (calibration.yaml과 같은 형식, source: estimated, 어디서 왔는지 estimated_from에 기록)
채점 (시뮬레이션): ros2 run tb3p_eval calib_error --estimate <결과> --truth sim_truth.yaml
시뮬레이션에서 로봇이 이 값을 믿게 하려면: ros2 launch tb3p_gazebo sim.launch.py calib_file:=<결과>

사용 예:
  python3 calib_merge.py --wheels ~/umbmark_data/before_wheels_ls.yaml
      --imu ~/imu_mount_data/session16_imu.yaml --lidar-cam ~/lidar_cam_data/session13_result.yaml
      --time ~/time_sync_data/session23_time_offset.yaml --out ~/calib_runs/session23.yaml
      (실제로는 한 줄로)
"""
import argparse
import copy
import datetime
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DESC = os.path.join(HERE, '..', '..', 'tb3p_description')
SCREW, COLOR, LIDAR, IMU = ('camera_bottom_screw_frame', 'camera_color_optical_frame',
                            'base_scan', 'imu_link')


# ===== 좌표 변환 (URDF 약속: R = Rz(yaw) Ry(pitch) Rx(roll)) =====

def rpy_to_matrix(roll, pitch, yaw):
    """Roll, pitch, yaw (rad) -> 3x3 회전 행렬."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def matrix_to_rpy(r):
    """3x3 회전 행렬 -> roll, pitch, yaw (rad)."""
    pitch = np.arcsin(-np.clip(r[2, 0], -1.0, 1.0))
    return float(np.arctan2(r[2, 1], r[2, 2])), float(pitch), float(np.arctan2(r[1, 0], r[0, 0]))


def to_matrix(xyz, rpy):
    """위치 + rpy -> 4x4."""
    t = np.eye(4)
    t[:3, :3] = rpy_to_matrix(*rpy)
    t[:3, 3] = xyz
    return t


def from_matrix(t):
    """4x4 -> (xyz 리스트, rpy 리스트)."""
    return [float(v) for v in t[:3, 3]], list(matrix_to_rpy(t[:3, :3]))


def difference(a, b):
    """두 4x4의 차이: (회전 °, 위치 mm)."""
    d = np.linalg.inv(a) @ b
    c = np.clip((np.trace(d[:3, :3]) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(c))), float(np.linalg.norm(d[:3, 3]) * 1000)


# ===== 설계도 (URDF) =====

def run_xacro(calib_file, xacro_file=None):
    """xacro로 tb3p 설계도(sim:=true, D435i 내부 좌표계 포함)를 만들어 URDF 문자열로."""
    if xacro_file is None:
        try:
            from ament_index_python.packages import get_package_share_directory
            share = get_package_share_directory('tb3p_description')
        except Exception:      # 빌드 전 소스 폴더에서 바로 쓸 때
            share = SRC_DESC
        xacro_file = os.path.join(share, 'urdf', 'tb3p_burger.urdf.xacro')
    cmd = ['xacro', xacro_file, f'calib_file:={os.path.abspath(calib_file)}', 'sim:=true']
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def urdf_edges(urdf):
    """URDF 문자열 -> {child 링크: (parent 링크, T_parent_child)} (조인트 origin만, 관절 각도 0)."""
    edges = {}
    for joint in ET.fromstring(urdf).findall('joint'):    # 맨 위 joint만 (transmission 안 것 제외)
        parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
        origin = joint.find('origin')
        xyz = [float(v) for v in (origin.get('xyz', '0 0 0') if origin is not None
                                  else '0 0 0').split()]
        rpy = [float(v) for v in (origin.get('rpy', '0 0 0') if origin is not None
                                  else '0 0 0').split()]
        edges[child] = (parent, to_matrix(xyz, rpy))
    return edges


def relative(edges, a, b):
    """T_a_b: 링크 b를 링크 a 기준으로 (같은 나무 안)."""
    def root_pose(frame, depth=0):
        if frame not in edges:
            return np.eye(4)
        if depth > 100:
            raise RuntimeError('URDF 나무에 고리가 있음')
        parent, t = edges[frame]
        return root_pose(parent, depth + 1) @ t
    for f in (a, b):
        if f not in edges and not any(p == f for p, _ in edges.values()):
            raise KeyError(f'URDF에 {f} 링크가 없음')
    return np.linalg.inv(root_pose(a)) @ root_pose(b)


# ===== 합치기 =====

def camera_from_lidar(t_base_lidar, t_color_lidar, t_screw_color):
    """브래킷 나사 자세 T_base_screw: LiDAR 자세와 LiDAR-카메라 관계로 계산."""
    return t_base_lidar @ np.linalg.inv(t_color_lidar) @ np.linalg.inv(t_screw_color)


def load(path):
    """YAML 읽기."""
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def sensor_matrix(calib, name):
    """캘리브레이션 dict의 sensors.<name> -> base 기준 4x4 (parent가 base_link여야 함)."""
    s = calib['sensors'][name]
    if s['parent'] != 'base_link':
        parent = s['parent']
        raise ValueError(f'{name}의 parent가 base_link가 아님: {parent}')
    return to_matrix(s['xyz'], s['rpy'])


def set_pose(calib, name, t):
    """sensors.<name>의 xyz, rpy를 4x4 값으로 (소수 7자리 = 0.1 µm, 0.1 µrad)."""
    xyz, rpy = from_matrix(t)
    calib['sensors'][name]['xyz'] = [round(v, 7) for v in xyz]
    calib['sensors'][name]['rpy'] = [round(v, 7) for v in rpy]


def merge(base, wheels=None, imu=None, lidar_cam=None, time=None, t_screw_color=None):
    """
    기준 calibration dict에 결과들을 덮어쓴 새 dict와 기록(estimated_from)을 돌려준다.

    wheels, imu, lidar_cam, time: 각 결과 파일을 읽은 dict (없으면 None) + 'path' 키
    t_screw_color: 나사 -> 컬러 렌즈 4x4 (lidar_cam이 있을 때 필요)
    """
    out = copy.deepcopy(base)
    out['source'] = 'estimated'
    log = {}
    if wheels is not None:
        for k in ('radius_left', 'radius_right', 'separation'):
            out['wheels'][k] = float(wheels['wheels'][k])
        log['wheels'] = wheels['path']
    if imu is not None:
        s = imu['sensors']['imu']
        out['sensors']['imu']['xyz'] = [float(v) for v in s['xyz']]
        out['sensors']['imu']['rpy'] = [float(v) for v in s['rpy']]
        log['imu'] = imu['path']
    if time is not None:
        off = time['time_offset_s']
        out['sensors']['imu']['time_offset'] = float(off.get('imu', 0.0))
        out['sensors']['lidar']['time_offset'] = float(off['lidar'])
        out['wheels']['time_offset'] = float(off['wheels'])
        log['time_offset'] = time['path']
    if lidar_cam is not None:
        if lidar_cam.get('parent') != COLOR or lidar_cam.get('child') != LIDAR:
            par, chi = lidar_cam.get('parent'), lidar_cam.get('child')
            raise ValueError(f'LiDAR-카메라 결과는 {COLOR} <- {LIDAR} 관계여야 함: {par} <- {chi}')
        t_cl = np.array(lidar_cam['matrix'], dtype=float)
        t_bs = camera_from_lidar(sensor_matrix(out, 'lidar'), t_cl, t_screw_color)
        set_pose(out, 'camera', t_bs)
        log['camera'] = lidar_cam['path'] + ' (LiDAR-카메라 관계 + LiDAR 자세 + D435i 설계값)'
        log['lidar_beam_elev_deg'] = float(lidar_cam.get('beam_elev_deg', 0.0))
    log['lidar'] = '기준 파일 값 그대로 (LiDAR -> base_link는 아직 추정하지 않음)'
    out['estimated_from'] = log
    return out


def check_urdf(urdf, calib, t_color_lidar=None):
    """
    URDF에서 다시 읽은 값이 calib과 같은지. 돌려주는 값: [(이름, 회전 차이 °, 위치 차이 mm)].

    t_color_lidar를 주면 URDF로 계산한 카메라-LiDAR 관계와도 비교한다.
    """
    edges = urdf_edges(urdf)
    rows = []
    for name, frame in (('imu', IMU), ('lidar', LIDAR), ('camera', SCREW)):
        rows.append((f'base_link -> {frame}', *difference(
            relative(edges, 'base_link', frame), sensor_matrix(calib, name))))
    if t_color_lidar is not None:
        rows.append((f'{COLOR} <- {LIDAR}', *difference(
            relative(edges, COLOR, LIDAR), np.asarray(t_color_lidar, dtype=float))))
    return rows


class FlowListDumper(yaml.SafeDumper):
    """목록(xyz, rpy)은 한 줄로, 나머지는 줄을 나눠 쓰는 YAML 출력."""


FlowListDumper.add_representer(
    list, lambda d, v: d.represent_sequence('tag:yaml.org,2002:seq', v, flow_style=True))


HEADER = """# tb3p 센서 캘리브레이션 (추정값, calib_merge.py가 만듦: {date})
#
# - 형식은 tb3p_description/config/calibration.yaml과 같다 (단위: m, rad, s)
# - estimated_from: 각 항목이 어느 결과 파일에서 왔는지
# - 시뮬레이션에서 로봇이 이 값을 믿게 하려면: ros2 launch tb3p_gazebo sim.launch.py calib_file:=<이 파일>

"""


def main():
    """명령줄에서 실행할 때의 시작점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', default=os.path.join(SRC_DESC, 'config', 'calibration.yaml'),
                    help='덮어쓸 기준 (기본: 공칭 calibration.yaml)')
    ap.add_argument('--wheels', help='umbmark_analyze.py의 *_wheels_ls.yaml')
    ap.add_argument('--imu', help='imu_mount_calib.py의 *_imu.yaml')
    ap.add_argument('--lidar-cam', help='lidar_cam_calib.py의 *_result.yaml')
    ap.add_argument('--time', help='time_offset_estimate.py의 *_time_offset.yaml')
    ap.add_argument('--xacro', default=None, help='설계도 파일 (기본: 설치된 tb3p_description)')
    ap.add_argument('--no-check', action='store_true', help='설계도 반영 확인을 건너뜀')
    ap.add_argument('--out', default=os.path.expanduser('~/calib_runs/calibration_estimated.yaml'))
    args = ap.parse_args()

    def read(path):
        if not path:
            return None
        d = load(path)
        d['path'] = os.path.abspath(path)
        return d

    base = load(args.base)
    lidar_cam = read(args.lidar_cam)
    t_screw_color = None
    if lidar_cam is not None or not args.no_check:
        urdf_base = run_xacro(args.base, args.xacro)
        t_screw_color = relative(urdf_edges(urdf_base), SCREW, COLOR)
    calib = merge(base, read(args.wheels), read(args.imu), lidar_cam, read(args.time),
                  t_screw_color)

    print('[합친 결과]  기준 파일 대비 변화 (위치 mm, 회전 °)')
    for name in ('imu', 'lidar', 'camera'):
        rot, pos = difference(sensor_matrix(base, name), sensor_matrix(calib, name))
        src = calib['estimated_from'].get(name, '기준 파일 값 그대로')
        print(f'    {name:7s} 회전 {rot:6.3f}°  위치 {pos:6.2f} mm   <- {os.path.basename(src)}')
    w0, w1 = base['wheels'], calib['wheels']
    print('    wheels  ' + ', '.join(f'{k} {w0[k] * 1000:.2f} -> {w1[k] * 1000:.3f} mm'
                                     for k in ('radius_left', 'radius_right', 'separation')))
    if 'time_offset' in calib['estimated_from']:
        dw = w1['time_offset'] * 1000
        dl = calib['sensors']['lidar']['time_offset'] * 1000
        print(f'    시간    바퀴 {dw:+.2f} ms, LiDAR {dl:+.2f} ms (IMU 기준)')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(HEADER.format(date=datetime.date.today().isoformat()))
        yaml.dump(calib, f, Dumper=FlowListDumper, sort_keys=False, allow_unicode=True,
                  width=1000)
    print(f'saved {args.out}')

    if args.no_check:
        return
    print('[설계도 반영 확인]  새 YAML -> xacro -> URDF에서 다시 읽은 값 vs 넣은 값')
    rows = check_urdf(run_xacro(args.out, args.xacro), calib,
                      lidar_cam['matrix'] if lidar_cam is not None else None)
    worst = 0.0
    for name, rot, pos in rows:
        print(f'    {name:48s} 회전 {rot:.6f}°  위치 {pos:.6f} mm')
        worst = max(worst, rot * 100, pos)       # 0.01° ~ 1 mm 를 같은 무게로
    ok = worst < 1e-3
    verdict = '같음 (설계도에 그대로 반영됨)' if ok else '다름! 좌표계 이름이나 약속을 확인'
    print(f'    -> {verdict}')
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
