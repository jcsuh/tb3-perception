#!/usr/bin/env python3
"""
캘리브레이션 전 과정을 명령 하나로: 수집 -> 추정 -> 합치기·설계도 반영 확인 -> 채점 (세션 25, 시뮬레이션).

단계 (월드)
  1) 바퀴      umbmark           umbmark_run.py         -> umbmark_analyze.py
  2) IMU 장착  umbmark           imu_mount_collect.py   -> imu_mount_calib.py     (바퀴 결과 사용)
  3) LiDAR-카메라 lidar_cam      lidar_cam_collect.py   -> lidar_cam_calib.py     (빔 원뿔 반영)
  4) 시간 오프셋 turtlebot3_world time_sync_collect.py  -> time_offset_estimate.py (지연 0 주입 노드와 함께)
  5) 몸 기준 LiDAR·카메라 turtlebot3_world lidar_base_collect.py -> lidar_base_calib.py
  6) calib_merge.py: calibration.yaml 하나로 + 설계도(URDF) 반영 확인
  7) (시뮬레이션) tb3p_eval calib_error: 정답과 채점

진행 방식
  - 수집을 먼저 월드별로 묶어서 한다: 월드마다 Gazebo를 새로 켜고(--gui 없으면 화면 없이 서버만),
    준비(/scan, /joint_states 수신)를 기다린 뒤 수집하고, 프로세스 묶음째 끈다.
    켜기 전에 이전 Gazebo·브리지가 남아 있으면 멈춘다 (세션 10: 남은 프로세스가 데이터를 오염시킴).
  - 그다음 추정을 순서대로 (앞 결과가 뒤 단계 입력).
  - 결과는 --run-dir 한 폴더에 모인다 (단계별 기록 logs/). 이미 있는 결과는 건너뛰므로 중간에 끊겨도
    같은 --run-dir로 다시 실행하면 이어서 한다 (--force는 다시 함).

사용 예 (시뮬레이션은 끈 상태에서, 약 25분):
  python3 calibrate_all.py
  python3 calibrate_all.py --run-dir ~/calib_runs/20261024_1200        (끊긴 실행 이어서)
  python3 calibrate_all.py --only lidar_base merge verify --run-dir ...   (일부 단계만)
"""
import argparse
import datetime
import os
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..')
TRUTH = os.path.join(SRC, 'tb3p_gazebo', 'config', 'sim_truth.yaml')
LEFTOVER = 'gz sim|parameter_bridge|robot_state_publisher'
STAGES = ('wheels', 'imu', 'lidar_cam', 'time', 'lidar_base', 'merge', 'verify')


def files(run):
    """실행 폴더 안 파일 이름들."""
    p = {k: os.path.join(run, v) for k, v in {
        'umbmark': 'umbmark.npz', 'wheels': 'umbmark_wheels_ls.yaml',
        'imu_raw': 'imu_mount.npz', 'imu': 'imu_mount_imu.yaml',
        'lc_raw': 'lidar_cam.json', 'lidar_cam': 'lidar_cam_result.yaml',
        'time_raw': 'time_sync.npz', 'time': 'time_sync_time_offset.yaml',
        'lb_raw': 'lidar_base.npz', 'lidar_base': 'lidar_base_result.yaml',
        'calib': 'calibration_estimated.yaml', 'score': 'score.txt'}.items()}
    p['logs'] = os.path.join(run, 'logs')
    return p


def plan(f, truth, elev):
    """
    단계별 (이름, 월드, 수집 명령, 수집 결과, 도우미 명령, 추정 명령, 추정 결과).

    명령은 scripts 폴더의 스크립트 이름 + 인자. truth가 비면 채점 인자를 뺀다.
    """
    tr = ['--truth', truth] if truth else []
    inject = ['delay_injector.py', '--delays', 'imu=0', 'joint_states=0', 'scan=0']
    return [
        ('wheels', 'umbmark', ['umbmark_run.py', '--out', f['umbmark']], f['umbmark'], None,
         ['umbmark_analyze.py', f['umbmark']] + tr, f['wheels']),
        ('imu', 'umbmark', ['imu_mount_collect.py', '--out', f['imu_raw']], f['imu_raw'], None,
         ['imu_mount_calib.py', f['imu_raw'], '--wheels', f['wheels']] + tr, f['imu']),
        ('lidar_cam', 'lidar_cam', ['lidar_cam_collect.py', '--out', f['lc_raw']], f['lc_raw'],
         None, ['lidar_cam_calib.py', f['lc_raw'], '--beam-elev-deg', str(elev),
                '--out', f['lidar_cam']], f['lidar_cam']),
        ('time', 'turtlebot3_world', ['time_sync_collect.py', '--out', f['time_raw']],
         f['time_raw'], inject,
         ['time_offset_estimate.py', f['time_raw'], '--wheels', f['wheels']], f['time']),
        ('lidar_base', 'turtlebot3_world', ['lidar_base_collect.py', '--out', f['lb_raw']],
         f['lb_raw'], None,
         ['lidar_base_calib.py', f['lb_raw'], '--wheels', f['wheels'], '--lidar-cam',
          f['lidar_cam'], '--time', f['time'], '--out', f['lidar_base']] + tr, f['lidar_base']),
    ]


def say(msg):
    """진행 메시지 (시각과 함께)."""
    print(f'[calibrate_all {datetime.datetime.now():%H:%M:%S}] {msg}', flush=True)


def child_env():
    """자식 프로세스 환경: 그림은 창 없이 파일로만."""
    env = dict(os.environ)
    env['MPLBACKEND'] = 'Agg'
    return env


def run(cmd, log_path, label):
    """명령을 실행하며 출력을 화면과 기록 파일에 함께 남긴다. 실패하면 SystemExit."""
    shown = ' '.join(os.path.basename(c) if '/' in c else c for c in cmd)
    say(f'{label}: {shown}')
    with open(log_path, 'a', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                env=child_env(), cwd=HERE)
        for line in proc.stdout:
            sys.stdout.write(f'    {line}')
            log.write(line)
        code = proc.wait()
    if code != 0:
        raise SystemExit(f'{label} 실패 (종료 코드 {code}), 기록: {log_path}')


def script(cmd):
    """['x.py', 인자...] -> [python3, scripts/x.py, 인자...]."""
    return [sys.executable, os.path.join(HERE, cmd[0])] + cmd[1:]


def leftovers():
    """남아 있는 Gazebo·브리지 프로세스 목록 (문자열, 없으면 빈 문자열)."""
    r = subprocess.run(['pgrep', '-af', LEFTOVER], capture_output=True, text=True)
    return r.stdout.strip()


def wait_clean(timeout=30.0):
    """이전 시뮬레이션이 다 꺼질 때까지 기다린다. 끝까지 남으면 SystemExit."""
    t_end = time.time() + timeout
    while time.time() < t_end:
        if not leftovers():
            return
        time.sleep(1.0)
    how = '끄는 법: pkill -f parameter_bridge; pkill -f "gz sim"'
    raise SystemExit(f'Gazebo·브리지 프로세스가 남아 있음 (데이터가 섞일 수 있어 멈춤):\n'
                     f'{leftovers()}\n{how}')


def start(cmd, log_path):
    """프로세스 묶음(세션)으로 켠다 -> 나중에 묶음째 끌 수 있게."""
    log = open(log_path, 'a', encoding='utf-8')
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=child_env(),
                            cwd=HERE, start_new_session=True)


def stop(proc):
    """묶음째 끈다: Ctrl+C -> 15초 기다림 -> TERM -> 5초 -> KILL."""
    if proc is None or proc.poll() is not None:
        return
    for sig, wait in ((signal.SIGINT, 15.0), (signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def wait_topic(topic, msg_type, timeout):
    """
    토픽에서 메시지 하나를 받을 때까지 기다린다 (best effort). 받으면 True.

    ros2 topic echo는 토픽이 아직 없으면 형식을 몰라 바로 실패하므로 (세션 25 첫 실행),
    형식을 직접 주고, 실패하면 2초 뒤 다시 시도한다.
    """
    cmd = ['ros2', 'topic', 'echo', '--once', '--qos-reliability', 'best_effort', topic,
           msg_type]
    t_end = time.time() + timeout
    while time.time() < t_end:
        try:
            r = subprocess.run(cmd, capture_output=True, env=child_env(),
                               timeout=max(1.0, min(30.0, t_end - time.time())))
            if r.returncode == 0 and r.stdout:
                return True
        except subprocess.TimeoutExpired:
            pass
        time.sleep(2.0)
    return False


def start_sim(world, gui, log_path):
    """Gazebo + 로봇을 켜고 센서가 나올 때까지 기다린다."""
    wait_clean()
    screen = '켬' if gui else '끔'
    say(f'Gazebo 켜기: world:={world} (화면 {screen})')
    proc = start(['ros2', 'launch', 'tb3p_gazebo', 'sim.launch.py', f'world:={world}',
                  f'gui:={str(gui).lower()}'], log_path)
    for topic, msg_type in (('/scan', 'sensor_msgs/msg/LaserScan'),
                            ('/joint_states', 'sensor_msgs/msg/JointState')):
        if not wait_topic(topic, msg_type, 120.0):
            stop(proc)
            raise SystemExit(f'{topic}가 2분 안에 안 나옴 (Gazebo 기록: {log_path})')
    time.sleep(5.0)                                # 로봇이 바닥에 자리 잡을 시간
    say('센서 수신 확인')
    return proc


def collect(stages, steps, f, gui, force):
    """수집 단계들을 월드별로 묶어서 한다."""
    todo = [s for s in steps if s[0] in stages and (force or not os.path.exists(s[3]))]
    for s in steps:
        if s[0] in stages and s not in todo:
            say(f'{s[0]}: 수집 결과가 이미 있어 건너뜀 ({os.path.basename(s[3])})')
    world, sim = None, None
    try:
        for name, w, cmd, _, helper, _, _ in todo:
            if w != world:
                stop(sim)
                sim = start_sim(w, gui, os.path.join(f['logs'], f'gazebo_{w}.log'))
                world = w
            aid = None
            if helper:
                aid = start(script(helper), os.path.join(f['logs'], f'{name}_helper.log'))
                time.sleep(3.0)
            try:
                run(script(cmd), os.path.join(f['logs'], f'{name}_collect.log'),
                    f'{name} 수집')
            finally:
                stop(aid)
    finally:
        stop(sim)
        if sim is not None:
            wait_clean()


def estimate(stages, steps, f, force):
    """추정 단계들을 순서대로."""
    for name, _, _, raw, _, cmd, result in steps:
        if name not in stages:
            continue
        if os.path.exists(result) and not force:
            say(f'{name}: 추정 결과가 이미 있어 건너뜀 ({os.path.basename(result)})')
            continue
        if not os.path.exists(raw):
            raise SystemExit(f'{name}: 수집 결과가 없음 ({raw})')
        run(script(cmd), os.path.join(f['logs'], f'{name}_estimate.log'), f'{name} 추정')
        if not os.path.exists(result):
            raise SystemExit(f'{name}: 추정이 끝났는데 결과 파일이 없음 ({result})')


def main():
    """명령줄에서 실행할 때의 시작점."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    ap.add_argument('--run-dir', default=os.path.expanduser(f'~/calib_runs/{stamp}'),
                    help='결과 폴더 (있는 폴더를 주면 이어서)')
    ap.add_argument('--only', nargs='+', choices=STAGES, default=list(STAGES),
                    help='이 단계만 (기본: 전부)')
    ap.add_argument('--force', action='store_true', help='결과가 있어도 다시 함')
    ap.add_argument('--gui', action='store_true', help='Gazebo 화면을 켬 (기본: 서버만)')
    ap.add_argument('--beam-elev-deg', type=float, default=-0.444,
                    help='LiDAR 빔 높이각 (Gazebo GPU LiDAR: -0.444, 세션 13)')
    ap.add_argument('--truth', default=TRUTH, help='채점용 정답 (빈 문자열이면 채점 안 함)')
    args = ap.parse_args()

    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    f = files(run_dir)
    os.makedirs(f['logs'], exist_ok=True)
    steps = plan(f, args.truth, args.beam_elev_deg)
    stages = set(args.only)
    chosen = ' '.join(s for s in STAGES if s in stages)
    say(f'결과 폴더 {run_dir}, 단계 {chosen}')
    t0 = time.time()

    collect(stages, steps, f, args.gui, args.force)
    estimate(stages, steps, f, args.force)

    if 'merge' in stages:
        run(script(['calib_merge.py', '--wheels', f['wheels'], '--imu', f['imu'],
                    '--lidar-cam', f['lidar_cam'], '--time', f['time'],
                    '--lidar-base', f['lidar_base'], '--out', f['calib']]),
            os.path.join(f['logs'], 'merge.log'), '합치기 + 설계도 반영 확인')
    if 'verify' in stages and args.truth:
        if shutil.which('ros2') is None:
            raise SystemExit('ros2 명령이 없음 (source install/setup.bash 했나?)')
        open(f['score'], 'w').close()
        run(['ros2', 'run', 'tb3p_eval', 'calib_error', '--estimate', f['calib'],
             '--truth', args.truth], f['score'], '정답과 채점')

    calib, minutes = f['calib'], (time.time() - t0) / 60
    say(f'끝 ({minutes:.1f}분). 결과: {calib}')
    say(f'로봇이 이 값을 믿게 하려면: ros2 launch tb3p_gazebo sim.launch.py calib_file:={calib}')


if __name__ == '__main__':
    main()
