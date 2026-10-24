# tb3-perception

TurtleBot3 Burger + Jetson Orin Nano Super + RealSense D435i 위에서 만드는
창고용 소형 자율 이동 로봇의 캘리브레이션(A) · 위치 추정(B) · 인식(C) 스택.
시뮬레이션(Gazebo Harmonic)과 실제 로봇에서 같은 코드로 동작합니다.

- ROS 2 Jazzy · Ubuntu 24.04 · Gazebo Harmonic
- 프로젝트 A: 시뮬레이션 단계 완료 (실제 로봇 단계는 하드웨어 구매 후)

## 프로젝트 A: 센서 캘리브레이션과 운행 중 드리프트 감시

정답을 아는 시뮬레이션에서 추정기를 먼저 검증합니다.
Gazebo 속 로봇은 공칭 설계값에서 일부러 조금씩 틀어 놓은 정답
([`sim_truth.yaml`](tb3p_gazebo/config/sim_truth.yaml))대로 조립되고,
로봇 소프트웨어와 모든 추정기는 공칭값([`calibration.yaml`](tb3p_description/config/calibration.yaml))과
센서 데이터만 봅니다. 정답은 채점 도구(`tb3p_eval`)만 읽습니다.

### 결과 (Gazebo, 정답 대비 오차)

| 항목 | 공칭 (설계값) | 추정 | 방법 |
|---|---|---|---|
| IMU → 몸 | 1.32° · 5.4 mm | **0.05°** · 2.0 mm ¹ | 정지 중 중력 + 원 돌기 옆 가속도 (TRIAD), 원심 가속도로 위치 |
| LiDAR → 몸 | 2.03° · 6.4 mm | **0.24°** · 4.0 mm ² | 평면 손-눈 (스캔 매칭 vs 바퀴 오도메트리) + 깊이 영상 바닥 평면 |
| 카메라 → 몸 | 2.68° · 7.0 mm | **0.13°** · 0.6 mm | 위 둘을 LiDAR-카메라 관계로 이음 |
| LiDAR ↔ 카메라 | 4.32° · 11.0 mm | **0.21°** · 3.9 mm | AprilGrid 보드 18자세 점-평면 최소제곱 (Python/SciPy, C++/Ceres 같은 답) |
| 카메라 ↔ 카메라 IMU | – | **0.019°** · 0.38 mm | Kalibr (손에 든 리그 흔들기) |
| 바퀴 (반지름 2개, 간격) | 간격 +1.25 % | **−0.01 %** 이내 | UMBmark 주행 + 전체 경로 맞추기 |
| 시간 오프셋 | 10–100 ms 주입 | 복원 오차 **≤ 0.63 ms** | 회전 구간 증가량 맞추기 (IMU·바퀴·LiDAR) |

합격 기준: 회전 0.5°, 위치 1 cm, 시간 ±5 ms — 모두 통과.
¹ IMU 높이는 추정하지 않음 (공칭값). ² 남은 오차는 LiDAR-카메라 관계의 오차가 그대로 전해진 것.

**드리프트 모니터** (`drift_monitor_node`, C++, `/diagnostics`):
카메라 2° 이상 틀어짐 감지율 100 % (107/107, 6–8초 안), 바퀴 배율 6 % 100 %,
시간 지연 20 ms 이상 100 %, 30분 정상 주행 ERROR 오경보 0번.
그래프: [`docs/results/drift_monitor`](docs/results/drift_monitor)

### 한 번에 실행

```bash
cd ~/TB3P_ws && source install/setup.bash
python3 src/tb3-perception/tb3p_calibration/scripts/calibrate_all.py   # 약 25분, Gazebo 자동 실행
```

월드를 바꿔 가며 수집(바퀴·IMU → LiDAR-카메라 → 시간 → 몸 기준 LiDAR·카메라)한 뒤
추정하고, `calibration.yaml` 하나로 합쳐 설계도(URDF)에 그대로 반영되는지 확인하고, 정답과 채점합니다.
결과를 로봇이 믿게 하려면 `ros2 launch tb3p_gazebo sim.launch.py calib_file:=<결과 YAML>`.

### 단계별 도구 (`tb3p_calibration/scripts`)

| 단계 | 수집 | 추정 |
|---|---|---|
| 바퀴 | `umbmark_run.py` | `umbmark_analyze.py` |
| IMU 장착 | `imu_mount_collect.py` | `imu_mount_calib.py` |
| LiDAR ↔ 카메라 | `lidar_cam_collect.py` | `lidar_cam_calib.py`, C++ `lidar_cam_calib_ceres` |
| 시간 오프셋 | `delay_injector.py` + `time_sync_collect.py` | `time_offset_estimate.py`, C++ `time_offset_node` |
| 몸 기준 LiDAR·카메라 | `lidar_base_collect.py` | `lidar_base_calib.py` |
| 합치기 | | `calib_merge.py` |
| 드리프트 감시 | `drift_experiment.py` | C++ `drift_monitor_node`, `drift_experiment_analyze.py` |

시험: `colcon test` — gtest (`test_time_offset`, `test_drift_core`), pytest
(`test_calib_merge`, `test_lidar_base_calib`, `test_estimators`), ROS 2 형식 검사. GitHub Actions에서 실행.

### 시뮬레이션에서 찾은 것 (실제 로봇 단계에서 확인할 것)

- Gazebo GPU LiDAR 빔이 모든 방향에서 약 0.44° 아래를 향함 (평면이 아니라 얕은 원뿔).
  카메라 앞쪽에서는 LiDAR가 숙여 붙은 것과 똑같이 보여 캘리브레이션만으로는 구분 불가.
- Gazebo IMU 시각이 약 1.3 ms 늦게 찍힘 (지연을 넣지 않은 센서를 정답 회전과 비교해 발견).
- 바퀴 충돌이 원통이면 회전 중심이 9 mm 옆으로 밀림 → 구 충돌로 수정.
- 가감속 때 바퀴 접촉 떨림으로 IMU 가속도가 움직임과 안 맞음 → 일정 속도 원 돌기로 우회.
- 몸 기준 LiDAR·카메라는 어느 캘리브레이션도 직접 주지 않음 → 손-눈 + 바닥 평면을 추가
  (통합 단계에서야 드러남).
