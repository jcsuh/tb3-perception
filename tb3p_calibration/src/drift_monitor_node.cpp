// drift_monitor_node
//
// 운행 중 캘리브레이션이 틀어지는지 감시하는 ROS 2 노드 (세션 21, W10–11).
//
// 검사 1 (카메라-LiDAR): LiDAR 점을 깊이 영상에 투영해 카메라가 마운트 축 둘레로 돈 각도를 추정
//   (drift_core.hpp, scripts/drift_check.py와 같은 방법). 영상 batch장마다 한 번 판단.
//   스캔과 영상 사이 로봇이 돈 만큼은 자이로로, 앞으로 간 만큼은 바퀴로 되돌린다.
//   로봇이 거의 서 있는 묶음은 건너뛴다 (한 장면만 보면 그 장면의 치우침이 결과가 됨).
// 검사 2 (회전 일관성): 자이로·바퀴·LiDAR 스캔 매칭이 잰 회전의 배율 (센서 회전 / IMU 회전)
// 검사 3 (시간 오프셋): 바퀴·LiDAR의 IMU 기준 시간 오프셋 (time_offset_core.hpp)
//   검사 2·3은 period_s초마다 최근 window_s초로 계산 (회전 변화가 부족하면 STALE)
//
// 판정: 처음 몇 번의 값으로 정상 기준(평균, 표준편차)을 배우고 (또는 파라미터로 기준을 주고)
//   |값 - 기준| >= error가 persist번 연속이면 ERROR, >= warn이면 WARN, 아니면 OK
// 결과: /diagnostics (diagnostic_msgs/DiagnosticArray, 1 Hz), 상태가 바뀔 때 로그
//
// 사용 예 (시뮬레이션):
//   ros2 run tb3p_calibration drift_monitor_node --ros-args -p use_sim_time:=true
//     -p calib_file:=$HOME/lidar_cam_data/session13_result.yaml

#include <yaml-cpp/yaml.h>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "drift_core.hpp"
#include "time_offset_core.hpp"

namespace dr = tb3p::drift;
namespace to = tb3p::time_offset;
using dr::Iso;
using dr::Level;

namespace
{

constexpr double kTwoPi = 6.28318530717958647692;
constexpr double kDeg = 180.0 / 3.14159265358979323846;

double to_sec(const builtin_interfaces::msg::Time & t)
{
  return static_cast<double>(t.sec) + static_cast<double>(t.nanosec) * 1e-9;
}

std::string fmt(double v, const char * f = "%.3f")
{
  char buf[64];
  std::snprintf(buf, sizeof(buf), f, v);
  return buf;
}

Level worst(Level a, Level b)
{
  // STALE(3)은 다른 값이 있으면 그쪽을 따른다
  if (a == Level::STALE) {
    return b;
  }
  if (b == Level::STALE) {
    return a;
  }
  return static_cast<int>(a) >= static_cast<int>(b) ? a : b;
}

}  // namespace

class DriftMonitorNode : public rclcpp::Node
{
public:
  DriftMonitorNode()
  : Node("drift_monitor_node"),
    j_yaw_(declare_parameter<double>("cam_warn_deg", 0.5),
      declare_parameter<double>("cam_error_deg", 1.0),
      declare_parameter<double>("k_sigma", 6.0),
      static_cast<int>(declare_parameter<int>("cam_learn_batches", 8)),
      static_cast<int>(declare_parameter<int>("persist", 2))),
    j_scale_w_(declare_parameter<double>("scale_warn", 0.02),
      declare_parameter<double>("scale_error", 0.05), dparam("k_sigma"),
      static_cast<int>(declare_parameter<int>("rot_learn_evals", 8)), iparam("persist")),
    j_scale_l_(dparam("scale_warn"), dparam("scale_error"), dparam("k_sigma"),
      iparam("rot_learn_evals"), iparam("persist")),
    j_off_w_(declare_parameter<double>("offset_warn_ms", 5.0),
      declare_parameter<double>("offset_error_ms", 10.0), dparam("k_sigma"),
      iparam("rot_learn_evals"), iparam("persist")),
    j_off_l_(dparam("offset_warn_ms"), dparam("offset_error_ms"), dparam("k_sigma"),
      iparam("rot_learn_evals"), iparam("persist"))
  {
    const auto calib_file = declare_parameter<std::string>("calib_file", "");
    depth_topic_ = declare_parameter<std::string>(
      "depth_topic", "/camera/camera/depth/image_rect_raw");
    const auto scan_topic = declare_parameter<std::string>("scan_topic", "/scan");
    const auto imu_topic = declare_parameter<std::string>("imu_topic", "/imu");
    const auto joint_topic = declare_parameter<std::string>("joint_topic", "/joint_states");
    radius_left_ = declare_parameter<double>("radius_left", 0.033);
    radius_right_ = declare_parameter<double>("radius_right", 0.033);
    separation_ = declare_parameter<double>("separation", 0.160);
    rate_ = declare_parameter<double>("cam_rate_hz", 2.0);
    batch_ = static_cast<int>(declare_parameter<int>("cam_batch", 6));
    window_ = declare_parameter<double>("window_s", 20.0);
    period_ = declare_parameter<double>("period_s", 5.0);
    min_rate_std_ = declare_parameter<double>("min_rate_std", 0.2);
    // 검사 1 묶음은 로봇이 이만큼 돌았거나 움직였을 때만 쓴다 (서 있으면 건너뜀)
    min_turn_ = declare_parameter<double>("cam_min_turn_deg", 5.0) / kDeg;
    min_move_ = declare_parameter<double>("cam_min_move_m", 0.10);
    // 검사 3이 LiDAR 시각이 기준에서 벗어났다고 추정하면 그만큼 검사 1의 스캔 시각을 고쳐 쓴다
    use_lidar_offset_ = declare_parameter<bool>("cam_use_lidar_offset", true);
    // 기준을 직접 줄 때 (NaN이면 처음 몇 번으로 배운다)
    const double base_yaw = declare_parameter<double>("cam_baseline_deg", std::nan(""));
    const double base_yaw_std = declare_parameter<double>("cam_baseline_std_deg", 0.0);
    const double exp_w = declare_parameter<double>("expected_wheel_offset_ms", std::nan(""));
    const double exp_l = declare_parameter<double>("expected_lidar_offset_ms", std::nan(""));
    if (std::isfinite(base_yaw)) {
      j_yaw_.set_baseline(base_yaw, base_yaw_std);
    }
    if (std::isfinite(exp_w)) {
      j_off_w_.set_baseline(exp_w, 0.0);
    }
    if (std::isfinite(exp_l)) {
      j_off_l_.set_baseline(exp_l, 0.0);
    }

    if (calib_file.empty()) {
      throw std::runtime_error("calib_file 파라미터가 필요함 (LiDAR-카메라 캘리브레이션 결과 YAML)");
    }
    const YAML::Node cal = YAML::LoadFile(calib_file);
    Eigen::Matrix4d m;
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        m(r, c) = cal["matrix"][r][c].as<double>();
      }
    }
    t_color_lidar_.matrix() = m;
    elev_deg_ = cal["beam_elev_deg"] ? cal["beam_elev_deg"].as<double>() : 0.0;

    const auto qos = rclcpp::QoS(rclcpp::KeepLast(2000)).best_effort();
    const auto latched = rclcpp::QoS(rclcpp::KeepLast(100)).reliable().transient_local();
    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
      depth_topic_, rclcpp::QoS(rclcpp::KeepLast(5)).best_effort(),
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {on_depth(*msg);});
    const std::string info_topic = depth_topic_.substr(0, depth_topic_.rfind('/')) +
      "/camera_info";
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      info_topic, qos, [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
        k_.fx = msg->k[0];
        k_.fy = msg->k[4];
        k_.cx = msg->k[2];
        k_.cy = msg->k[5];
        have_info_ = true;
      });
    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      scan_topic, qos, [this](sensor_msgs::msg::LaserScan::ConstSharedPtr msg) {on_scan(*msg);});
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_topic, qos, [this](sensor_msgs::msg::Imu::ConstSharedPtr msg) {on_imu(*msg);});
    joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      joint_topic, qos, [this](sensor_msgs::msg::JointState::ConstSharedPtr msg) {
        on_joint(*msg);
      });
    tf_sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
      "/tf_static", latched, [this](tf2_msgs::msg::TFMessage::ConstSharedPtr msg) {
        for (const auto & ts : msg->transforms) {
          const auto & tr = ts.transform.translation;
          const auto & q = ts.transform.rotation;
          Iso t = Iso::Identity();
          t.translate(Eigen::Vector3d(tr.x, tr.y, tr.z));
          t.rotate(Eigen::Quaterniond(q.w, q.x, q.y, q.z).normalized());
          edges_[ts.child_frame_id] = {ts.header.frame_id, t};
        }
      });
    diag_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/diagnostics", 10);
    timer_ = create_wall_timer(std::chrono::seconds(1), [this]() {publish_diagnostics();});
    RCLCPP_INFO(
      get_logger(), "드리프트 모니터 시작: 깊이 %s, 캘리브레이션 %s (빔 높이각 %+.3f°)",
      depth_topic_.c_str(), calib_file.c_str(), elev_deg_);
  }

private:
  double dparam(const std::string & name) const {return get_parameter(name).as_double();}
  int iparam(const std::string & name) const
  {
    return static_cast<int>(get_parameter(name).as_int());
  }

  // ===== TF (공칭, /tf_static) =====
  bool pose_of(const std::string & frame, Iso & out, int depth = 0) const
  {
    const auto it = edges_.find(frame);
    if (it == edges_.end()) {
      out = Iso::Identity();      // 뿌리 (base_footprint)
      return frame == "base_footprint";
    }
    Iso parent;
    if (depth > 30 || !pose_of(it->second.first, parent, depth + 1)) {
      return false;
    }
    out = parent * it->second.second;
    return true;
  }

  bool ready_geometry()
  {
    if (have_geo_) {
      return true;
    }
    Iso b_lidar, b_color, b_depth, b_mount;
    if (!have_info_ || !pose_of("base_scan", b_lidar) ||
      !pose_of("camera_color_optical_frame", b_color) ||
      !pose_of("camera_depth_optical_frame", b_depth) ||
      !pose_of("camera_bottom_screw_frame", b_mount))
    {
      return false;
    }
    t_base_lidar_ = b_lidar;
    const Iso t_color_depth = b_color.inverse() * b_depth;
    t_base_cam0_ = b_lidar * t_color_lidar_.inverse() * t_color_depth;
    mount_ = b_mount.translation();
    have_geo_ = true;
    RCLCPP_INFO(get_logger(), "TF와 카메라 정보 준비됨: 검사 1 시작");
    return true;
  }

  // ===== 센서 콜백 =====
  void on_imu(const sensor_msgs::msg::Imu & m)
  {
    const double t = to_sec(m.header.stamp);
    const double w = m.angular_velocity.z;
    if (have_imu_) {
      if (t <= last_imu_t_) {
        return;
      }
      imu_theta_ += 0.5 * (w + last_w_) * (t - last_imu_t_);
    }
    have_imu_ = true;
    last_imu_t_ = t;
    last_w_ = w;
    imu_.push(t, imu_theta_);
    rates_.emplace_back(t, w);
    process_pending();
    if (next_eval_ <= 0.0) {
      next_eval_ = t + window_;
    }
    if (t >= next_eval_) {
      evaluate_rotation(t);
      next_eval_ = t + period_;
    }
  }

  void on_joint(const sensor_msgs::msg::JointState & m)
  {
    const auto il = std::find(m.name.begin(), m.name.end(), "wheel_left_joint");
    const auto ir = std::find(m.name.begin(), m.name.end(), "wheel_right_joint");
    if (il == m.name.end() || ir == m.name.end()) {
      return;
    }
    const auto i = static_cast<std::size_t>(il - m.name.begin());
    const auto j = static_cast<std::size_t>(ir - m.name.begin());
    if (i >= m.position.size() || j >= m.position.size()) {
      return;
    }
    const double t = to_sec(m.header.stamp);
    if (have_joint_) {
      phi_l_ += std::remainder(m.position[i] - raw_l_, kTwoPi);
      phi_r_ += std::remainder(m.position[j] - raw_r_, kTwoPi);
    } else {
      phi_l_ = m.position[i];
      phi_r_ = m.position[j];
    }
    raw_l_ = m.position[i];
    raw_r_ = m.position[j];
    have_joint_ = true;
    dist_ = 0.5 * (radius_left_ * phi_l_ + radius_right_ * phi_r_);
    if (t - last_joint_t_ < 0.005) {
      return;
    }
    last_joint_t_ = t;
    wheel_.push(t, (radius_right_ * phi_r_ - radius_left_ * phi_l_) / separation_);
    dist_s_.push(t, dist_);
  }

  void on_scan(const sensor_msgs::msg::LaserScan & m)
  {
    const double t = to_sec(m.header.stamp);
    std::vector<float> ranges(m.ranges.begin(), m.ranges.end());
    angle_min_ = m.angle_min;
    angle_inc_ = m.angle_increment;
    if (!scans_.empty() && scans_.back().second.size() == ranges.size()) {
      const auto & prev = scans_.back().second;
      const double guess = to::shift_guess(prev, ranges, angle_inc_);
      const auto res = to::icp(
        to::scan_points(ranges, angle_min_, angle_inc_),
        to::scan_points(prev, angle_min_, angle_inc_), guess);
      lidar_theta_ += res.dth;
    }
    lidar_.push(t, lidar_theta_);
    scans_.emplace_back(t, std::move(ranges));
    while (scans_.size() > 5) {
      scans_.pop_front();
    }
  }

  void on_depth(const sensor_msgs::msg::Image & m)
  {
    const double t = to_sec(m.header.stamp);
    if (t - last_cap_ < 1.0 / rate_ || scans_.empty() || !ready_geometry()) {
      return;
    }
    dr::DepthImage img;
    img.width = static_cast<int>(m.width);
    img.height = static_cast<int>(m.height);
    img.data.assign(static_cast<std::size_t>(img.width) * img.height, std::nanf(""));
    const bool f32 = m.encoding == "32FC1";      // 시뮬레이션: m 단위 float
    const bool u16 = m.encoding == "16UC1";      // 실제 D435i: mm 단위 정수
    if (!f32 && !u16) {
      RCLCPP_WARN_ONCE(get_logger(), "모르는 깊이 형식 %s: 검사 1을 건너뜀", m.encoding.c_str());
      return;
    }
    for (int v = 0; v < img.height; ++v) {
      const uint8_t * row = m.data.data() + static_cast<std::size_t>(v) * m.step;
      float * out = img.data.data() + static_cast<std::size_t>(v) * img.width;
      for (int u = 0; u < img.width; ++u) {
        if (f32) {
          std::memcpy(out + u, row + 4 * u, 4);
        } else {
          uint16_t mm;
          std::memcpy(&mm, row + 2 * u, 2);
          out[u] = mm > 0 ? mm / 1000.0f : std::nanf("");
        }
      }
    }
    // 시각이 가장 가까운 스캔 (LiDAR 시각 보정 반영)
    const double c = lidar_corr_;
    const auto best = std::min_element(
      scans_.begin(), scans_.end(), [t, c](const auto & a, const auto & b) {
        return std::abs(a.first + c - t) < std::abs(b.first + c - t);
      });
    pending_.push_back({t, best->first + c, best->second, std::move(img)});
    last_cap_ = t;
    process_pending();
  }

  // 영상·스캔 시각까지 IMU가 들어온 프레임만 회전 보정해서 묶음에 넣는다
  void process_pending()
  {
    while (!pending_.empty() && imu_.size() > 1 && pending_.front().t_img <= imu_.t.back() &&
      pending_.front().t_scan >= imu_.t.front())
    {
      Pending p = std::move(pending_.front());
      pending_.pop_front();
      // 스캔 시각 -> 영상 시각 사이 로봇 움직임: 앞으로 ds 가고 dpsi 돌았다
      //   영상 시각의 base 좌표 = Rz(-dpsi) · (스캔 시각 base 좌표 - (ds, 0, 0))
      const double dpsi = to::interp(imu_, p.t_img) - to::interp(imu_, p.t_scan);
      const double ds = to::interp(dist_s_, p.t_img) - to::interp(dist_s_, p.t_scan);
      Iso comp = Iso::Identity();
      comp.rotate(Eigen::AngleAxisd(-dpsi, Eigen::Vector3d::UnitZ()));
      comp.translate(Eigen::Vector3d(-ds, 0.0, 0.0));
      dr::Frame f;
      f.pts_base = dr::transform(
        comp * t_base_lidar_, dr::lidar_points(p.ranges, angle_min_, angle_inc_, elev_deg_));
      f.depth = std::move(p.depth);
      frames_.push_back(std::move(f));
      batch_yaw_.push_back(to::interp(imu_, p.t_img));
      batch_dist_.push_back(dist_);
      if (static_cast<int>(frames_.size()) >= batch_) {
        if (dr::view_changed(batch_yaw_, batch_dist_, min_turn_, min_move_)) {
          evaluate_camera();
          still_ = false;
        } else {
          ++skipped_still_;
          if (!still_) {
            RCLCPP_INFO(get_logger(), "검사 1: 로봇이 거의 움직이지 않아 묶음을 건너뜀 (움직이면 다시 시작)");
          }
          still_ = true;
        }
        frames_.clear();
        batch_yaw_.clear();
        batch_dist_.clear();
      }
    }
    while (pending_.size() > 10) {
      pending_.pop_front();     // IMU가 안 오면 쌓이지 않게
    }
  }

  // ===== 검사 1 =====
  void evaluate_camera()
  {
    const auto est = dr::estimate_yaw(frames_, t_base_cam0_, mount_, k_);
    if (!est.ok) {
      RCLCPP_WARN(get_logger(), "검사 1: 깊이 영상과 비교할 LiDAR 점이 부족함");
      return;
    }
    yaw_deg_ = est.delta * kDeg;
    cost_zero_ = est.cost_zero;
    cost_best_ = est.cost_best;
    const Level l = j_yaw_.update(yaw_deg_);
    ++cam_batches_;
    RCLCPP_INFO(
      get_logger(), "검사 1 #%d: yaw %+.2f°, 기준 대비 %+.2f°, 잔차 %.1f -> %.1f mm [%s]",
      cam_batches_, yaw_deg_, j_yaw_.learned() ? j_yaw_.deviation() : 0.0, cost_zero_ * 1000,
      cost_best_ * 1000, dr::level_name(l));
    set_level(cam_level_, l, "검사 1 카메라-LiDAR",
      "yaw " + fmt(yaw_deg_, "%+.2f") + "°, 기준 대비 " +
      fmt(j_yaw_.learned() ? j_yaw_.deviation() : 0.0, "%+.2f") + "°");
    if (j_yaw_.learned() && j_yaw_.learned_count() == cam_batches_) {
      RCLCPP_INFO(
        get_logger(), "검사 1 기준 학습 완료: yaw %+.3f° ± %.3f° (WARN %.2f°, ERROR %.2f°)",
        j_yaw_.mean(), j_yaw_.std_dev(), j_yaw_.warn(), j_yaw_.error());
    }
  }

  // ===== 검사 2·3 =====
  void evaluate_rotation(double now)
  {
    imu_.trim_before(now - window_ - 0.6);
    wheel_.trim_before(now - window_ - 0.3);
    dist_s_.trim_before(now - window_ - 0.3);
    lidar_.trim_before(now - window_ - 0.3);
    while (!rates_.empty() && rates_.front().first < now - window_) {
      rates_.pop_front();
    }
    double mean = 0.0, sq = 0.0;
    for (const auto & r : rates_) {
      mean += r.second;
      sq += r.second * r.second;
    }
    const double n = std::max<double>(1.0, static_cast<double>(rates_.size()));
    rate_std_ = std::sqrt(std::max(0.0, sq / n - (mean / n) * (mean / n)));
    if (rate_std_ < min_rate_std_) {
      set_level(rot_level_, Level::STALE, "검사 2 회전 일관성", "회전 변화 부족");
      set_level(time_level_, Level::STALE, "검사 3 시간 오프셋", "회전 변화 부족");
      return;
    }
    const to::FitResult rw = to::fit_offset(wheel_, imu_, 0.1, 0.3);
    const to::FitResult rl = to::fit_offset(lidar_, imu_, 0.1, 0.3);
    if (!rw.ok || !rl.ok) {
      return;
    }
    scale_w_ = rw.scale;
    scale_l_ = rl.scale;
    off_w_ = rw.offset * 1000;
    off_l_ = rl.offset * 1000;
    ++rot_evals_;
    const Level l2 = worst(j_scale_w_.update(scale_w_), j_scale_l_.update(scale_l_));
    const Level l3 = worst(j_off_w_.update(off_w_), j_off_l_.update(off_l_));
    // 검사 3 약속: 오프셋 = 센서 시각에 더해야 맞는 값. 기준에서 벗어난 만큼만 스캔 시각에 더한다
    //   (예: LiDAR 시각이 50 ms 늦게 찍히면 오프셋이 기준보다 약 -50 ms -> 스캔 시각 - 50 ms)
    lidar_corr_ = use_lidar_offset_ && j_off_l_.learned() ?
      (off_l_ - j_off_l_.mean()) / 1000.0 : 0.0;
    set_level(rot_level_, l2, "검사 2 회전 일관성",
      "배율 바퀴 " + fmt(scale_w_, "%.4f") + ", LiDAR " + fmt(scale_l_, "%.4f"));
    set_level(time_level_, l3, "검사 3 시간 오프셋",
      "바퀴 " + fmt(off_w_, "%+.1f") + " ms, LiDAR " + fmt(off_l_, "%+.1f") + " ms");
  }

  void set_level(Level & cur, Level now, const char * name, const std::string & what)
  {
    if (now != cur) {
      const std::string text = std::string(name) + ": " + dr::level_name(cur) + " -> " +
        dr::level_name(now) + " (" + what + ")";
      if (now == Level::ERROR) {
        RCLCPP_ERROR(get_logger(), "%s", text.c_str());
      } else if (now == Level::WARN) {
        RCLCPP_WARN(get_logger(), "%s", text.c_str());
      } else {
        RCLCPP_INFO(get_logger(), "%s", text.c_str());
      }
    }
    cur = now;
  }

  // ===== /diagnostics =====
  diagnostic_msgs::msg::DiagnosticStatus status(
    const std::string & name, Level l, const std::string & message,
    const std::vector<std::pair<std::string, std::string>> & values) const
  {
    diagnostic_msgs::msg::DiagnosticStatus s;
    s.name = "tb3p_drift: " + name;
    s.hardware_id = "tb3p";
    s.level = static_cast<uint8_t>(l);
    s.message = message;
    for (const auto & kv : values) {
      diagnostic_msgs::msg::KeyValue v;
      v.key = kv.first;
      v.value = kv.second;
      s.values.push_back(v);
    }
    return s;
  }

  void publish_diagnostics()
  {
    diagnostic_msgs::msg::DiagnosticArray arr;
    arr.header.stamp = now();
    arr.status.push_back(status(
        "camera_lidar", cam_level_,
        j_yaw_.learned() ? "camera yaw " + fmt(yaw_deg_, "%+.2f") + " deg" : "learning baseline",
        {{"yaw_deg", fmt(yaw_deg_)}, {"baseline_deg", fmt(j_yaw_.mean())},
          {"baseline_std_deg", fmt(j_yaw_.std_dev())},
          {"deviation_deg", fmt(j_yaw_.learned() ? j_yaw_.deviation() : 0.0)},
          {"warn_deg", fmt(j_yaw_.warn())}, {"error_deg", fmt(j_yaw_.error())},
          {"residual_mm_at_zero", fmt(cost_zero_ * 1000, "%.1f")},
          {"residual_mm_at_best", fmt(cost_best_ * 1000, "%.1f")},
          {"batches", std::to_string(cam_batches_)},
          {"skipped_still", std::to_string(skipped_still_)},
          {"lidar_time_correction_ms", fmt(lidar_corr_ * 1000, "%.2f")}}));
    arr.status.push_back(status(
        "yaw_rate_consistency", rot_level_, "scale wheel " + fmt(scale_w_, "%.4f") +
        ", lidar " + fmt(scale_l_, "%.4f"),
        {{"wheel_scale", fmt(scale_w_, "%.5f")}, {"lidar_scale", fmt(scale_l_, "%.5f")},
          {"wheel_scale_baseline", fmt(j_scale_w_.mean(), "%.5f")},
          {"lidar_scale_baseline", fmt(j_scale_l_.mean(), "%.5f")},
          {"rate_std", fmt(rate_std_)}, {"evaluations", std::to_string(rot_evals_)}}));
    arr.status.push_back(status(
        "time_offset", time_level_, "wheel " + fmt(off_w_, "%+.1f") + " ms, lidar " +
        fmt(off_l_, "%+.1f") + " ms",
        {{"wheel_offset_ms", fmt(off_w_, "%.2f")}, {"lidar_offset_ms", fmt(off_l_, "%.2f")},
          {"wheel_offset_baseline_ms", fmt(j_off_w_.mean(), "%.2f")},
          {"lidar_offset_baseline_ms", fmt(j_off_l_.mean(), "%.2f")},
          {"evaluations", std::to_string(rot_evals_)}}));
    diag_pub_->publish(arr);
  }

  struct Pending
  {
    double t_img = 0.0;
    double t_scan = 0.0;
    std::vector<float> ranges;
    dr::DepthImage depth;
  };

  // 설정
  std::string depth_topic_;
  double radius_left_ = 0.033, radius_right_ = 0.033, separation_ = 0.160;
  double rate_ = 2.0, window_ = 20.0, period_ = 5.0, min_rate_std_ = 0.2;
  double min_turn_ = 0.0873, min_move_ = 0.10;
  bool use_lidar_offset_ = true;
  double lidar_corr_ = 0.0;     // 검사 1 스캔 시각에 더할 값 (s)
  int batch_ = 6;
  Iso t_color_lidar_ = Iso::Identity();
  double elev_deg_ = 0.0;

  // 판정
  dr::Judge j_yaw_, j_scale_w_, j_scale_l_, j_off_w_, j_off_l_;
  Level cam_level_ = Level::STALE, rot_level_ = Level::STALE, time_level_ = Level::STALE;

  // 기하
  std::map<std::string, std::pair<std::string, Iso>> edges_;
  bool have_geo_ = false, have_info_ = false;
  Iso t_base_lidar_ = Iso::Identity(), t_base_cam0_ = Iso::Identity();
  Eigen::Vector3d mount_ = Eigen::Vector3d::Zero();
  dr::Intrinsics k_;

  // 센서 기록
  to::Series imu_, wheel_, lidar_, dist_s_;     // dist_s_: 바퀴 이동 거리 (m)
  std::deque<std::pair<double, double>> rates_;
  std::deque<std::pair<double, std::vector<float>>> scans_;
  std::deque<Pending> pending_;
  std::vector<dr::Frame> frames_;
  std::vector<double> batch_yaw_, batch_dist_;     // 묶음 프레임들의 누적 yaw (rad), 이동 거리 (m)
  bool still_ = false;
  int skipped_still_ = 0;
  bool have_imu_ = false, have_joint_ = false;
  double last_imu_t_ = 0.0, last_w_ = 0.0, imu_theta_ = 0.0;
  double raw_l_ = 0.0, raw_r_ = 0.0, phi_l_ = 0.0, phi_r_ = 0.0, last_joint_t_ = -1.0, dist_ = 0.0;
  double angle_min_ = 0.0, angle_inc_ = 0.0, lidar_theta_ = 0.0;
  double last_cap_ = -1e9, next_eval_ = 0.0;

  // 최근 값
  double yaw_deg_ = 0.0, cost_zero_ = 0.0, cost_best_ = 0.0;
  double scale_w_ = 0.0, scale_l_ = 0.0, off_w_ = 0.0, off_l_ = 0.0, rate_std_ = 0.0;
  int cam_batches_ = 0, rot_evals_ = 0;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_sub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DriftMonitorNode>());
  rclcpp::shutdown();
  return 0;
}
