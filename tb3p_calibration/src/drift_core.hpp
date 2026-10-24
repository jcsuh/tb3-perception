// drift_core.hpp
//
// 드리프트 검사 계산 (세션 21, ROS 없는 순수 계산). scripts/drift_check.py를 C++로 옮긴 것.
//
// 검사 1: LiDAR 점을 깊이 영상에 투영해 같은 방향의 거리 차이(잔차)를 보고,
//   카메라가 마운트 나사의 수직축(base z) 둘레로 δ 돌았다고 놓았을 때 잔차가 가장 작은 δ를 찾는다.
// 판정: 정상 상태에서 배운 기준값과의 차이를 OK / WARN / ERROR로 (연속 조건 포함)

#ifndef TB3P_CALIBRATION__DRIFT_CORE_HPP_
#define TB3P_CALIBRATION__DRIFT_CORE_HPP_

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace tb3p
{
namespace drift
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kTrunc = 0.10;     // 잔차를 자르는 값 (m)
constexpr int kMinPoints = 30;

using Iso = Eigen::Isometry3d;
using Points = std::vector<Eigen::Vector3d>;

// 깊이 영상 (m, 값 없음 = NaN 또는 0 이하)
struct DepthImage
{
  int width = 0;
  int height = 0;
  std::vector<float> data;   // 행 우선
  float at(int u, int v) const {return data[static_cast<std::size_t>(v) * width + u];}
};

struct Intrinsics
{
  double fx = 0, fy = 0, cx = 0, cy = 0;
};

// 스캔 -> LiDAR 좌표계 3D 점 (빔 높이각 원뿔 반영)
inline Points lidar_points(
  const std::vector<float> & ranges, double angle_min, double angle_inc, double elev_deg,
  double rmin = 0.15, double rmax = 8.0)
{
  Points out;
  const double e = elev_deg * kPi / 180.0;
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    const double r = ranges[i];
    if (!std::isfinite(r) || r <= rmin || r >= rmax) {
      continue;
    }
    const double a = angle_min + angle_inc * static_cast<double>(i);
    out.emplace_back(r * std::cos(e) * std::cos(a), r * std::cos(e) * std::sin(a), r * std::sin(e));
  }
  return out;
}

inline Points transform(const Iso & t, const Points & pts)
{
  Points out;
  out.reserve(pts.size());
  for (const auto & p : pts) {
    out.push_back(t * p);
  }
  return out;
}

// 마운트 나사의 수직축(base z) 둘레로 δ (rad) 돈 카메라 자세 T_base_cam
inline Iso camera_pose_with_yaw(
  const Iso & t_base_cam0, const Eigen::Vector3d & mount,
  double delta)
{
  Iso r = Iso::Identity();
  r.translate(mount);
  r.rotate(Eigen::AngleAxisd(delta, Eigen::Vector3d::UnitZ()));
  r.translate(-mount);
  return r * t_base_cam0;
}

// base 좌표 점들을 깊이 영상에 투영한 잔차 (영상 깊이 - 점 깊이, m)
inline std::vector<double> residuals(
  const Points & pts_base, const Iso & t_base_cam, const Intrinsics & k, const DepthImage & img,
  int margin = 2)
{
  std::vector<double> out;
  const Iso inv = t_base_cam.inverse();
  for (const auto & pb : pts_base) {
    const Eigen::Vector3d pc = inv * pb;
    if (pc.z() <= 0.2) {
      continue;
    }
    const int u = static_cast<int>(std::lround(k.fx * pc.x() / pc.z() + k.cx));
    const int v = static_cast<int>(std::lround(k.fy * pc.y() / pc.z() + k.cy));
    if (u < margin || u >= img.width - margin || v < 0 || v >= img.height) {
      continue;
    }
    const float d = img.at(u, v);
    if (!std::isfinite(d) || d <= 0.1f) {
      continue;
    }
    out.push_back(static_cast<double>(d) - pc.z());
  }
  return out;
}

// 잘린 절댓값 평균 (m). 점이 너무 적으면 NaN
inline double robust_cost(const std::vector<double> & r)
{
  if (static_cast<int>(r.size()) < kMinPoints) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double s = 0.0;
  for (double x : r) {
    s += std::min(std::abs(x), kTrunc);
  }
  return s / static_cast<double>(r.size());
}

struct Frame
{
  Points pts_base;     // 움직임 보정한 base 좌표 점
  DepthImage depth;
};

struct YawEstimate
{
  bool ok = false;
  double delta = 0.0;        // rad
  double cost_zero = 0.0;    // δ = 0일 때 비용 (m)
  double cost_best = 0.0;
  int frames_used = 0;
};

// 여러 프레임을 함께 놓고 δ를 찾는다: ±max_deg를 step_deg 간격으로 훑고 포물선 보간
inline YawEstimate estimate_yaw(
  const std::vector<Frame> & frames, const Iso & t_base_cam0, const Eigen::Vector3d & mount,
  const Intrinsics & k, double max_deg = 8.0, double step_deg = 0.1)
{
  YawEstimate est;
  const int n = static_cast<int>(std::lround(2 * max_deg / step_deg)) + 1;
  std::vector<double> deltas(n), costs(n, std::numeric_limits<double>::quiet_NaN());
  int used = 0;
  for (int i = 0; i < n; ++i) {
    deltas[i] = (-max_deg + i * step_deg) * kPi / 180.0;
    const Iso tc = camera_pose_with_yaw(t_base_cam0, mount, deltas[i]);
    double sum = 0.0;
    int cnt = 0;
    for (const auto & f : frames) {
      const double c = robust_cost(residuals(f.pts_base, tc, k, f.depth));
      if (std::isfinite(c)) {
        sum += c;
        ++cnt;
      }
    }
    if (cnt > 0) {
      costs[i] = sum / cnt;
      used = std::max(used, cnt);
    }
  }
  int best = -1;
  for (int i = 0; i < n; ++i) {
    if (std::isfinite(costs[i]) && (best < 0 || costs[i] < costs[best])) {
      best = i;
    }
  }
  if (best < 0) {
    return est;
  }
  double d = deltas[best];
  if (best > 0 && best < n - 1 && std::isfinite(costs[best - 1]) &&
    std::isfinite(costs[best + 1]))
  {
    const double y0 = costs[best - 1], y1 = costs[best], y2 = costs[best + 1];
    const double den = y0 - 2 * y1 + y2;
    if (den > 0) {
      d += 0.5 * (y0 - y2) / den * (deltas[1] - deltas[0]);
    }
  }
  est.ok = true;
  est.delta = d;
  est.cost_best = costs[best];
  est.cost_zero = costs[n / 2];
  est.frames_used = used;
  return est;
}

// 묶음 안에서 로봇이 충분히 움직였나: 돈 각도의 범위 >= min_turn (rad) 또는 이동 거리의 범위 >= min_move (m).
// 제자리에 서 있으면 한 장면만 보게 되어 그 장면 특유의 치우침이 그대로 결과가 된다 (세션 21 결과)
inline bool view_changed(
  const std::vector<double> & yaw, const std::vector<double> & dist, double min_turn,
  double min_move)
{
  const auto span = [](const std::vector<double> & v) {
      if (v.empty()) {
        return 0.0;
      }
      const auto mm = std::minmax_element(v.begin(), v.end());
      return *mm.second - *mm.first;
    };
  return span(yaw) >= min_turn || span(dist) >= min_move;
}

// ===== 판정 =====

enum class Level {OK = 0, WARN = 1, ERROR = 2, STALE = 3};

inline const char * level_name(Level l)
{
  switch (l) {
    case Level::OK: return "OK";
    case Level::WARN: return "WARN";
    case Level::ERROR: return "ERROR";
    default: return "STALE";
  }
}

// 가운데 값 (중앙값)
inline double median(std::vector<double> v)
{
  if (v.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const std::size_t n = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + n, v.end());
  const double hi = v[n];
  if (v.size() % 2 == 1) {
    return hi;
  }
  return 0.5 * (hi + *std::max_element(v.begin(), v.begin() + n));
}

// 기준값 학습 + 임계값 판정.
// 처음 learn_n개 값으로 기준을 배운다: 중심 = 중앙값, 퍼짐 = 1.4826 × 중앙값 절대 편차 (MAD).
//   평균·표준편차 대신 쓰는 이유: 학습 중 튀는 값 하나가 기준을 끌고 가지 않게 (세션 22 결과)
//   1.4826을 곱하면 정규분포에서 표준편차와 같은 크기가 된다.
// 그 뒤에는 |값 - 기준| 으로 판정:
//   >= error가 persist번 연속이면 ERROR, >= warn이면 WARN (ERROR 연속 조건을 아직 못 채운 경우 포함)
// error·warn은 "최소 고정값"과 "기준 표준편차 × k" 중 큰 쪽 (정상 상태가 시끄러우면 기준이 넓어진다)
class Judge
{
public:
  Judge(double warn, double error, double k_sigma, int learn_n, int persist)
  : warn_min_(warn), error_min_(error), k_(k_sigma), learn_n_(learn_n), persist_(persist) {}

  void set_baseline(double mean, double std)
  {
    mean_ = mean;
    std_ = std;
    learned_ = true;
  }

  bool learned() const {return learned_;}
  double mean() const {return mean_;}
  double std_dev() const {return std_;}
  int learned_count() const {return static_cast<int>(samples_.size());}
  double warn() const {return std::max(warn_min_, k_ * std_ * 0.5);}
  double error() const {return std::max(error_min_, k_ * std_);}

  Level update(double value)
  {
    last_ = value;
    if (!learned_) {
      samples_.push_back(value);
      if (static_cast<int>(samples_.size()) >= learn_n_) {
        const double m = median(samples_);
        std::vector<double> dev;
        for (double x : samples_) {
          dev.push_back(std::abs(x - m));
        }
        set_baseline(m, 1.4826 * median(dev));
      }
      return Level::STALE;    // 배우는 중
    }
    const double dev = std::abs(value - mean_);
    over_ = dev >= error() ? over_ + 1 : 0;
    if (over_ >= persist_) {
      return Level::ERROR;
    }
    return dev >= warn() ? Level::WARN : Level::OK;
  }

  double deviation() const {return last_ - mean_;}

private:
  double warn_min_, error_min_, k_;
  int learn_n_, persist_;
  std::vector<double> samples_;
  bool learned_ = false;
  double mean_ = 0.0, std_ = 0.0, last_ = 0.0;
  int over_ = 0;
};

}  // namespace drift
}  // namespace tb3p

#endif  // TB3P_CALIBRATION__DRIFT_CORE_HPP_
