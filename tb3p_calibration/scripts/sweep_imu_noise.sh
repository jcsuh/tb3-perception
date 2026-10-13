#!/usr/bin/env bash
# IMU 노이즈 배율을 바꿔 가며 "녹화 → 변환 → Kalibr 카메라-IMU → 채점"을 반복한다.
#
# 사용 예:  sweep_imu_noise.sh 1 3 10 30
#
# 준비물 (세션 9에서 만든 것):
#   ~/kalibr_data/target.yaml          보드 설정 (주석 없는 것)
#   ~/kalibr_data/rig_01-camchain.yaml 카메라 내부 파라미터 (IMU 노이즈와 무관하므로 재사용)
#   tb3p_calibration/config/kalibr_imu_d435i_sim.yaml  IMU 노이즈 기준값 (배율을 곱해 씀)
#
# 결과: ~/kalibr_data/sweep_imu_noise/scale_<배율>/ 에 로그, Kalibr 결과, score.json
set -euo pipefail
trap 'echo "실패: ${BASH_SOURCE[0]}의 $LINENO번째 줄. 로그 폴더: ${out:-없음}"' ERR

SCALES=("$@")
[ ${#SCALES[@]} -eq 0 ] && SCALES=(1 3 10 30)

BASE=$HOME/kalibr_data
SWEEP=$BASE/sweep_imu_noise
REPO=$HOME/TB3P_ws/src/tb3-perception/tb3p_calibration
IMU_YAML=$REPO/config/kalibr_imu_d435i_sim.yaml
RECORD_SEC=${RECORD_SEC:-75}          # 실제 시계 기준 녹화 시간 (RTF 0.8이면 시뮬레이션 약 60초)
TOPICS=(/camera/camera/infra1/image_rect_raw /camera/camera/infra2/image_rect_raw
        /camera/camera/infra1/camera_info /camera/camera/infra2/camera_info
        /camera/camera/imu /rig/tf_static)

for f in "$BASE/target.yaml" "$BASE/rig_01-camchain.yaml" "$IMU_YAML"; do
  [ -f "$f" ] || { echo "없음: $f"; exit 1; }
done
mkdir -p "$SWEEP"
if pgrep -f "parameter_bridge|gz sim|robot_state_publisher|robot_state_publisher" > /dev/null; then
  echo "이미 실행 중인 Gazebo 또는 브리지가 있습니다. 정리 후 다시 실행하세요:"
  echo "  pkill -f parameter_bridge; pkill -f 'gz sim'"; exit 1
fi

for s in "${SCALES[@]}"; do
  out=$SWEEP/scale_$s
  if [ -f "$out/score.json" ]; then
    echo "[$s] 이미 완료됨, 건너뜀"; continue
  fi
  rm -rf "$out" && mkdir -p "$out"
  start=$SECONDS
  echo "[$s] $(date +%H:%M:%S) 시뮬레이션 시작"

  # 1) 시뮬레이션 (화면 없이), 리그가 흔들리기 시작할 때까지 기다림
  setsid ros2 launch tb3p_gazebo calib_rig.launch.py gui:=false imu_noise_scale:="$s" \
    > "$out/launch.log" 2>&1 &
  launch_pid=$!
  sleep 20

  # 2) 녹화
  echo "[$s] 녹화 ${RECORD_SEC}초"
  timeout --foreground -s INT "$RECORD_SEC" ros2 bag record --use-sim-time -o "$out/rig" "${TOPICS[@]}" \
    > "$out/record.log" 2>&1 || true
  # 시뮬레이션 프로세스 그룹 전체를 끈다 (음수 PID = 그룹)
  kill -TERM -- -"$launch_pid" 2>/dev/null || true
  for _ in $(seq 20); do kill -0 -- -"$launch_pid" 2>/dev/null || break; sleep 0.5; done
  kill -KILL -- -"$launch_pid" 2>/dev/null || true
  wait "$launch_pid" 2>/dev/null || true

  # 녹화 프로그램이 종료 중 죽으면 목차(metadata.yaml)가 없다 → 다시 만들어 본다
  if [ ! -f "$out/rig/metadata.yaml" ]; then
    echo "[$s] 녹화 종료 중 문제 발생, 목차 재생성 시도"
    ros2 bag reindex "$out/rig" -s mcap > "$out/reindex.log" 2>&1 || true
  fi
  if [ ! -f "$out/rig/metadata.yaml" ]; then
    echo "[$s] 녹화 실패 (로그: $out/record.log, $out/reindex.log)"; exit 1
  fi

  # 녹화가 충분히 길어야 한다 (짧으면 Kalibr 결과가 엉터리가 됨)
  dur=$(ros2 bag info "$out/rig" | awk '/^Duration:/ {gsub("s", "", $2); print int($2)}')
  echo "[$s] 녹화 길이 ${dur}초 (시뮬레이션 시계)"
  if [ "${dur:-0}" -lt "${MIN_SEC:-50}" ]; then
    echo "[$s] 녹화가 너무 짧음 (최소 ${MIN_SEC:-50}초). 로그: $out/record.log"; exit 1
  fi

  imu=$(ros2 bag info "$out/rig" | awk '/camera\/camera\/imu / {for (i = 1; i <= NF; i++) if ($i == "Count:") print $(i + 1)}')
  rate=$(( imu / dur ))
  echo "[$s] IMU ${imu}개, 평균 ${rate} Hz"
  if [ "$rate" -lt 180 ] || [ "$rate" -gt 220 ]; then
    echo "[$s] IMU 주기가 200 Hz가 아님 (중복 발행 의심). 로그: $out"; exit 1
  fi

  # 3) ROS 1 bag으로 변환 (정답 토픽은 빼고)
  rosbags-convert --src "$out/rig" --dst "$out/rig.bag" \
    --include-topic /camera/camera/infra1/image_rect_raw \
    --include-topic /camera/camera/infra2/image_rect_raw \
    --include-topic /camera/camera/imu > "$out/convert.log" 2>&1

  # 4) 배율을 곱한 IMU 노이즈 설정 (Kalibr에게도 실제 노이즈를 알려줌)
  python3 - "$IMU_YAML" "$out/imu.yaml" "$s" << 'PY'
import sys
src, dst, scale = sys.argv[1], sys.argv[2], float(sys.argv[3])
with open(src) as f, open(dst, 'w') as g:
    for line in f:
        key = line.split(':')[0]
        if key in ('accelerometer_noise_density', 'gyroscope_noise_density'):
            value = float(line.split(':')[1].split('#')[0])
            line = f'{key}: {value * scale:.6e}\n'
        g.write(line)
PY

  # 5) Kalibr 카메라-IMU (카메라 내부 파라미터는 세션 9 결과 재사용)
  echo "[$s] $(date +%H:%M:%S) Kalibr 실행"
  cp "$BASE/target.yaml" "$BASE/rig_01-camchain.yaml" "$out/"
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e MPLBACKEND=Agg \
    -v "$out":/data tb3p-kalibr bash -c "source /catkin_ws/devel/setup.bash && \
    rosrun kalibr kalibr_calibrate_imu_camera --bag rig.bag --target target.yaml \
    --cam rig_01-camchain.yaml --imu imu.yaml --dont-show-report" > "$out/kalibr.log" 2>&1

  # 6) 채점
  python3 "$REPO/scripts/kalibr_score.py" "$out/rig" "$out/rig-camchain-imucam.yaml" \
    --json "$out/score.json" | tee "$out/score.txt"
  rm -f "$out/rig.bag"                  # 변환본은 지움 (원본 rig/ 는 남김)
  echo "[$s] 완료 ($(( (SECONDS - start) / 60 ))분)"
done

python3 "$REPO/scripts/plot_sweep.py" "$SWEEP"
