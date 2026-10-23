// time_offset_node
//
// IMU·바퀴·LiDAR의 시간 오프셋을 실시간으로 추정하는 ROS 2 노드 (세션 19).
// scripts/time_offset_estimate.py와 같은 방법 (time_offset_core.hpp)을 센서 토픽을 받으면서 돌린다.
//
// 메시지를 받을 때마다
//   - IMU: 자이로 z를 적분해 누적 회전 각도
//   - 바퀴: (오른쪽 반지름 × 오른쪽 각도 - 왼쪽 반지름 × 왼쪽 각도) / 간격
//   - LiDAR: 이전 스캔과 ICP로 맞춘 회전을 더함
// period_s초마다 최근 window_s초로 IMU 기준 오프셋을 다시 계산해 알린다
//   (회전이 충분히 변하지 않으면 건너뜀: 회전 속도 표준편차 < min_rate_std)
// 결과: 로그 + 토픽 time_offset/estimate (std_msgs/Float64MultiArray)
//   [바퀴 ms, LiDAR ms, 바퀴 배율, LiDAR 배율, 회전 속도 표준편차 rad/s]
//   ms 값 = 그 센서 시각에 더해야 할 보정값 (calibration.yaml의 time_offset, IMU 기준)
//
// 사용 예 (시뮬레이션, delay_injector.py와 time_sync_collect.py를 돌리는 동안):
//   ros2 run tb3p_calibration time_offset_node --ros-args -p use_sim_time:=true
//     -p radius_left:=0.0331 -p radius_right:=0.0330 -p separation:=0.162

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "time_offset_core.hpp"

namespace to = tb3p::time_offset;

namespace
{

constexpr double kTwoPi = 6.28318530717958647692;

double to_sec(const builtin_interfaces::msg::Time & t)
{
  return static_cast<double>(t.sec) + static_cast<double>(t.nanosec) * 1e-9;
}

double median(std::vector<double> v)
{
  if (v.empty()) {
    return 0.0;
  }
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}

}  // namespace

class TimeOffsetNode : public rclcpp::Node
{
public:
  TimeOffsetNode()
  : Node("time_offset_node")
  {
    const auto imu_topic = declare_parameter<std::string>("imu_topic", "/delayed/imu");
    const auto joint_topic = declare_parameter<std::string>(
      "joint_topic", "/delayed/joint_states");
    const auto scan_topic = declare_parameter<std::string>("scan_topic", "/delayed/scan");
    radius_left_ = declare_parameter<double>("radius_left", 0.033);
    radius_right_ = declare_parameter<double>("radius_right", 0.033);
    separation_ = declare_parameter<double>("separation", 0.160);
    window_ = declare_parameter<double>("window_s", 20.0);
    period_ = declare_parameter<double>("period_s", 5.0);
    max_lag_ = declare_parameter<double>("max_lag_s", 0.3);
    min_rate_std_ = declare_parameter<double>("min_rate_std", 0.2);

    // 계산하는 동안(약 0.5 s) 들어온 메시지를 버리지 않게 넉넉히 쌓아 둔다
    const auto qos = rclcpp::QoS(rclcpp::KeepLast(2000)).best_effort();
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_topic, qos, [this](sensor_msgs::msg::Imu::ConstSharedPtr m) {on_imu(*m);});
    joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      joint_topic, qos, [this](sensor_msgs::msg::JointState::ConstSharedPtr m) {on_joint(*m);});
    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      scan_topic, qos, [this](sensor_msgs::msg::LaserScan::ConstSharedPtr m) {on_scan(*m);});
    pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("time_offset/estimate", 10);
    RCLCPP_INFO(
      get_logger(), "구독: %s, %s, %s / 창 %.0f s, %.0f s마다 계산 (바퀴 %.4f %.4f %.4f m)",
      imu_topic.c_str(), joint_topic.c_str(), scan_topic.c_str(), window_, period_,
      radius_left_, radius_right_, separation_);
  }

private:
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
    if (next_eval_ <= 0.0) {
      next_eval_ = t + window_;
    }
    if (t >= next_eval_) {
      evaluate(t);
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
    // 각도가 ±π에서 넘어가도 되게 변화량을 감아서 쌓는다
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
    if (t - last_joint_t_ < 0.005) {       // 200 Hz 정도로 솎기
      return;
    }
    last_joint_t_ = t;
    wheel_.push(t, (radius_right_ * phi_r_ - radius_left_ * phi_l_) / separation_);
  }

  void on_scan(const sensor_msgs::msg::LaserScan & m)
  {
    const double t = to_sec(m.header.stamp);
    std::vector<float> ranges(m.ranges.begin(), m.ranges.end());
    if (!prev_scan_.empty() && prev_scan_.size() == ranges.size()) {
      const double guess = to::shift_guess(prev_scan_, ranges, m.angle_increment);
      const auto res = to::icp(
        to::scan_points(ranges, m.angle_min, m.angle_increment),
        to::scan_points(prev_scan_, m.angle_min, m.angle_increment), guess);
      lidar_theta_ += res.dth;
      icp_rms_ = res.rms;
    }
    prev_scan_ = std::move(ranges);
    lidar_.push(t, lidar_theta_);
  }

  void evaluate(double now)
  {
    imu_.trim_before(now - window_ - 2 * max_lag_);
    wheel_.trim_before(now - window_ - max_lag_);
    lidar_.trim_before(now - window_ - max_lag_);
    while (!rates_.empty() && rates_.front().first < now - window_) {
      rates_.pop_front();
    }
    double mean = 0.0, sq = 0.0;
    for (const auto & r : rates_) {
      mean += r.second;
      sq += r.second * r.second;
    }
    const double n = std::max<double>(1.0, static_cast<double>(rates_.size()));
    const double rate_std = std::sqrt(std::max(0.0, sq / n - (mean / n) * (mean / n)));
    if (rate_std < min_rate_std_) {
      RCLCPP_INFO(
        get_logger(), "회전 변화가 부족해 건너뜀 (회전 속도 표준편차 %.3f < %.3f rad/s)",
        rate_std, min_rate_std_);
      return;
    }
    const auto start = std::chrono::steady_clock::now();
    const to::FitResult rw = to::fit_offset(wheel_, imu_, 0.1, max_lag_);
    const to::FitResult rl = to::fit_offset(lidar_, imu_, 0.1, max_lag_);
    const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
    if (rw.ok) {
      hist_wheel_.push_back(rw.offset * 1000);
    }
    if (rl.ok) {
      hist_lidar_.push_back(rl.offset * 1000);
    }
    std_msgs::msg::Float64MultiArray out;
    out.layout.dim.resize(1);
    out.layout.dim[0].label = "wheel_ms,lidar_ms,wheel_scale,lidar_scale,rate_std";
    out.layout.dim[0].size = 5;
    out.layout.dim[0].stride = 5;
    out.data = {rw.ok ? rw.offset * 1000 : NAN, rl.ok ? rl.offset * 1000 : NAN,
      rw.scale, rl.scale, rate_std};
    pub_->publish(out);
    RCLCPP_INFO(
      get_logger(),
      "IMU 기준 오프셋: 바퀴 %+.2f ms (배율 %.4f), LiDAR %+.2f ms (배율 %.4f) | "
      "지금까지 %zu번 중앙값 바퀴 %+.2f, LiDAR %+.2f ms | 계산 %.0f ms, ICP 잔차 %.1f mm",
      rw.offset * 1000, rw.scale, rl.offset * 1000, rl.scale, hist_lidar_.size(),
      median(hist_wheel_), median(hist_lidar_), ms, icp_rms_ * 1000);
  }

  // 설정
  double radius_left_, radius_right_, separation_;
  double window_, period_, max_lag_, min_rate_std_;

  // 누적 각도 기록
  to::Series imu_, wheel_, lidar_;
  std::deque<std::pair<double, double>> rates_;   // (시각, 자이로 z)
  bool have_imu_ = false, have_joint_ = false;
  double last_imu_t_ = 0.0, last_w_ = 0.0, imu_theta_ = 0.0;
  double raw_l_ = 0.0, raw_r_ = 0.0, phi_l_ = 0.0, phi_r_ = 0.0, last_joint_t_ = -1.0;
  std::vector<float> prev_scan_;
  double lidar_theta_ = 0.0, icp_rms_ = 0.0;
  double next_eval_ = 0.0;
  std::vector<double> hist_wheel_, hist_lidar_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TimeOffsetNode>());
  rclcpp::shutdown();
  return 0;
}
