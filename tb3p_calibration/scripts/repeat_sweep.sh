#!/usr/bin/env bash
# IMU 노이즈 스윕 반복 실행: 배율마다 여러 번 돌려 실행 간 퍼짐을 본다.
#   사용: repeat_sweep.sh <배율당 횟수> <배율...>      예) repeat_sweep.sh 3 1 3 10 30
#   결과: ~/kalibr_data/sweep_imu_noise/trials/scale_<배율>/t<번호>/score.json
#   - sweep_imu_noise.sh를 배율 하나씩 호출하고, 끝나면 결과 폴더를 trials/로 옮긴다.
#   - 이미 있는 회차는 다시 돌리지 않는다(중간에 끊겨도 같은 명령으로 이어서 실행).
#   - 녹화(rig/)는 회차마다 지워서 디스크를 아낀다. 점수(score.json)만 남는다.
set -euo pipefail
trap 'echo "[repeat] 오류로 중단 (줄 $LINENO): $BASH_COMMAND" >&2' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP="$HERE/sweep_imu_noise.sh"
ROOT="$HOME/kalibr_data/sweep_imu_noise"
MAX_FAIL=3
MAX_NORM_ACC=2.0   # Kalibr 정규화 가속도 잔차 평균 상한 (정상이면 약 1.1)

if [ $# -lt 2 ]; then
  echo "사용: $0 <배율당 횟수> <배율...>   예) $0 3 1 3 10 30" >&2
  exit 1
fi
N="$1"
shift
[ -x "$SWEEP" ] || { echo "실행 파일 없음: $SWEEP" >&2; exit 1; }
mkdir -p "$ROOT/trials"

# 한 번씩 돌린 기존 그래프는 sweep_imu_noise.sh가 덮어쓰므로 미리 보관
if [ -f "$ROOT/sweep_imu_noise.png" ] && [ ! -f "$ROOT/sweep_imu_noise_single.png" ]; then
  cp "$ROOT/sweep_imu_noise.png" "$ROOT/sweep_imu_noise_single.png"
fi

count_done() {  # 배율 $1의 완료된 회차 수
  { find "$ROOT/trials/scale_$1" -mindepth 2 -maxdepth 2 -name score.json 2>/dev/null || true; } \
    | wc -l
}

noise_ok() {  # Kalibr 잔차로 "녹화된 IMU 노이즈 = 요청한 노이즈"인지 확인
  local f v
  f=$(ls "$ROOT/scale_$1"/*results-imucam.txt 2>/dev/null | head -n 1)
  [ -n "$f" ] || { echo "[$1] Kalibr 결과 txt 없음" >&2; return 1; }
  v=$(awk '/^Accelerometer error \(imu0\):/ {gsub(",", "", $5); print $5; exit}' "$f")
  echo "[$1] 정규화 가속도 잔차 평균 = ${v:-?} (정상 약 1.1, 상한 $MAX_NORM_ACC)"
  awk -v v="${v:-999}" -v m="$MAX_NORM_ACC" 'BEGIN {exit !(v + 0 < m + 0)}'
}

save_trial() {  # scale_<s> → trials/scale_<s>/t<k> (녹화는 삭제)
  local s="$1" k=1 dst
  while [ -e "$ROOT/trials/scale_$s/t$k" ]; do k=$((k + 1)); done
  dst="$ROOT/trials/scale_$s/t$k"
  mkdir -p "$ROOT/trials/scale_$s"
  rm -rf "$ROOT/scale_$s/rig"
  mv "$ROOT/scale_$s" "$dst"
  echo "[$s] 저장: $dst"
}

wait_clean() {  # 이전 실행의 Gazebo·브리지가 완전히 꺼질 때까지 대기 (최대 30초)
  for _ in $(seq 30); do
    pgrep -f "parameter_bridge|gz sim|robot_state_publisher" >/dev/null || return 0
    sleep 1
  done
  echo "[repeat] 이전 시뮬레이션 프로세스가 남아 있음:" >&2
  pgrep -af "parameter_bridge|gz sim|robot_state_publisher" >&2 || true
  exit 1
}

# 1) 지난번 스윕 결과(scale_<s>/score.json)가 있으면 1회차로 편입
for s in "$@"; do
  if [ -f "$ROOT/scale_$s/score.json" ] && [ "$(count_done "$s")" -eq 0 ]; then
    echo "[$s] 지난 결과를 1회차로 편입"
    save_trial "$s"
  fi
done

# 2) 배율마다 N회가 될 때까지 실행
fail=0
start=$(date +%s)
for s in "$@"; do
  while [ "$(count_done "$s")" -lt "$N" ]; do
    k=$(( $(count_done "$s") + 1 ))
    echo "========== [$s] ${k}/${N} 회차  $(date +%H:%M:%S) =========="
    wait_clean
    rm -rf "${ROOT:?}/scale_$s"   # 실패하고 남은 폴더가 있으면 녹화가 시작되지 않음
    if "$SWEEP" "$s" && [ -f "$ROOT/scale_$s/score.json" ] && noise_ok "$s"; then
      save_trial "$s"
    else
      fail=$((fail + 1))
      bad="$ROOT/failed/scale_${s}_$(date +%H%M%S)"
      mkdir -p "$ROOT/failed"
      if [ -d "$ROOT/scale_$s" ]; then
        rm -rf "$ROOT/scale_$s/rig"
        mv "$ROOT/scale_$s" "$bad"
      fi
      echo "[$s] 실패 ${fail}/${MAX_FAIL} (남은 파일: $bad)" >&2
      if [ "$fail" -ge "$MAX_FAIL" ]; then
        echo "[repeat] 실패가 ${MAX_FAIL}번이라 중단합니다." >&2
        exit 1
      fi
    fi
  done
done

echo "[repeat] 전체 완료 ($(( ($(date +%s) - start) / 60 ))분)"
python3 "$HERE/plot_sweep_trials.py" "$ROOT/trials"
