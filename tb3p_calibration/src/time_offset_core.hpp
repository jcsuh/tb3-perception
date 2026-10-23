// time_offset_core.hpp
//
// 센서 사이 시간 오프셋 추정과 2D 스캔 매칭 (세션 19, ROS 없는 순수 계산).
// scripts/time_offset_estimate.py, scripts/scan_match.py와 같은 방법을 C++로 옮긴 것.
//
// 시간 오프셋 (구간 증가량 맞추기)
//   센서 S의 누적 회전 각도 θ_S(t_k)와 기준(IMU)의 누적 각도 θ_R(t)가 있을 때
//   증가량 θ_S(t_k+m) - θ_S(t_k)  vs  θ_R(t_k+m + δ) - θ_R(t_k + δ)
//   를 배율·바이어스와 함께 최소제곱으로 맞춘 잔차가 가장 작은 δ를 찾는다.
//   δ = 그 센서 시각에 더해야 할 보정값 (calibration.yaml의 time_offset 약속)
//
// 스캔 매칭 (점-선 ICP)
//   제자리 회전 첫 추정(거리 배열이 몇 칸 밀렸나) -> 가까운 점과 그 표면 법선으로 회전·이동 보정 반복

#ifndef TB3P_CALIBRATION__TIME_OFFSET_CORE_HPP_
#define TB3P_CALIBRATION__TIME_OFFSET_CORE_HPP_

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <utility>
#include <vector>

namespace tb3p
{
namespace time_offset
{

// 시각 순서대로 쌓이는 (시각, 누적 각도) 기록. 오래된 것은 앞에서 지운다.
struct Series
{
  std::deque<double> t;
  std::deque<double> theta;

  void push(double time, double value)
  {
    if (!t.empty() && time <= t.back()) {
      return;  // 시각이 거꾸로 가거나 같은 메시지는 버린다
    }
    t.push_back(time);
    theta.push_back(value);
  }

  void trim_before(double time)
  {
    while (!t.empty() && t.front() < time) {
      t.pop_front();
      theta.pop_front();
    }
  }

  std::size_t size() const {return t.size();}
};

// 선형 보간 (범위 밖은 끝 값)
inline double interp(const Series & s, double time)
{
  if (s.t.empty()) {
    return 0.0;
  }
  if (time <= s.t.front()) {
    return s.theta.front();
  }
  if (time >= s.t.back()) {
    return s.theta.back();
  }
  const auto it = std::upper_bound(s.t.begin(), s.t.end(), time);
  const std::size_t i = static_cast<std::size_t>(it - s.t.begin());
  const double a = (time - s.t[i - 1]) / (s.t[i] - s.t[i - 1]);
  return s.theta[i - 1] + a * (s.theta[i] - s.theta[i - 1]);
}

struct FitResult
{
  bool ok = false;
  double offset = 0.0;   // s
  double scale = 1.0;    // 센서 증가량 / 기준 증가량
  double cost = 0.0;     // 잔차 제곱합 (rad²)
  int samples = 0;
};

// 구간 증가량 비교 비용: δ에서의 잔차 제곱합과 배율
inline FitResult increment_cost(
  const std::vector<double> & ts, const std::vector<double> & ths, const Series & ref,
  double delta, std::size_t step)
{
  FitResult r;
  // 정규방정식: d_s = a d_r + b dt
  double srr = 0, srt = 0, stt = 0, ssr = 0, sst = 0, sss = 0;
  int n = 0;
  for (std::size_t k = 0; k + step < ts.size(); ++k) {
    const double ds = ths[k + step] - ths[k];
    const double dt = ts[k + step] - ts[k];
    const double dr = interp(ref, ts[k + step] + delta) - interp(ref, ts[k] + delta);
    srr += dr * dr;
    srt += dr * dt;
    stt += dt * dt;
    ssr += ds * dr;
    sst += ds * dt;
    sss += ds * ds;
    ++n;
  }
  const double det = srr * stt - srt * srt;
  if (n < 10 || std::abs(det) < 1e-18) {
    return r;
  }
  const double a = (ssr * stt - sst * srt) / det;
  const double b = (sst * srr - ssr * srt) / det;
  // 잔차 제곱합 = Σ(ds - a dr - b dt)²
  r.cost = sss - 2 * a * ssr - 2 * b * sst + a * a * srr + 2 * a * b * srt + b * b * stt;
  r.scale = a;
  r.samples = n;
  r.ok = true;
  return r;
}

// 센서를 기준에 맞추는 δ: 2 ms 간격으로 훑은 뒤 가장 작은 곳 근처에서 황금분할 탐색
inline FitResult fit_offset(
  const Series & sensor, const Series & ref, double window = 0.1, double max_lag = 0.3)
{
  FitResult best;
  if (sensor.size() < 20 || ref.size() < 20) {
    return best;
  }
  std::vector<double> ts, ths;
  for (std::size_t k = 0; k < sensor.size(); ++k) {
    if (sensor.t[k] > ref.t.front() + max_lag && sensor.t[k] < ref.t.back() - max_lag) {
      ts.push_back(sensor.t[k]);
      ths.push_back(sensor.theta[k]);
    }
  }
  if (ts.size() < 20) {
    return best;
  }
  std::vector<double> dts;
  for (std::size_t k = 1; k < ts.size(); ++k) {
    dts.push_back(ts[k] - ts[k - 1]);
  }
  std::nth_element(dts.begin(), dts.begin() + dts.size() / 2, dts.end());
  const double med = dts[dts.size() / 2];
  const std::size_t step = std::max<std::size_t>(1, static_cast<std::size_t>(
        std::lround(window / med)));

  double best_d = 0.0, best_c = std::numeric_limits<double>::infinity();
  for (double d = -max_lag; d <= max_lag + 1e-9; d += 0.002) {
    const FitResult r = increment_cost(ts, ths, ref, d, step);
    if (r.ok && r.cost < best_c) {
      best_c = r.cost;
      best_d = d;
    }
  }
  if (!std::isfinite(best_c)) {
    return best;
  }
  // 황금분할 탐색 (±4 ms)
  const double g = (std::sqrt(5.0) - 1.0) / 2.0;
  double lo = best_d - 0.004, hi = best_d + 0.004;
  double x1 = hi - g * (hi - lo), x2 = lo + g * (hi - lo);
  double f1 = increment_cost(ts, ths, ref, x1, step).cost;
  double f2 = increment_cost(ts, ths, ref, x2, step).cost;
  while (hi - lo > 1e-5) {
    if (f1 < f2) {
      hi = x2;
      x2 = x1;
      f2 = f1;
      x1 = hi - g * (hi - lo);
      f1 = increment_cost(ts, ths, ref, x1, step).cost;
    } else {
      lo = x1;
      x1 = x2;
      f1 = f2;
      x2 = lo + g * (hi - lo);
      f2 = increment_cost(ts, ths, ref, x2, step).cost;
    }
  }
  best = increment_cost(ts, ths, ref, (lo + hi) / 2, step);
  best.offset = (lo + hi) / 2;
  return best;
}

// ===== 2D 스캔 매칭 =====

using Points = std::vector<Eigen::Vector2d>;

inline Points scan_points(
  const std::vector<float> & ranges, double angle_min, double angle_inc, double rmin = 0.12,
  double rmax = 8.0)
{
  Points out;
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    const double r = ranges[i];
    if (std::isfinite(r) && r > rmin && r < rmax) {
      const double a = angle_min + angle_inc * static_cast<double>(i);
      out.emplace_back(r * std::cos(a), r * std::sin(a));
    }
  }
  return out;
}

// 제자리 회전 첫 추정: r_new[i] ≈ r_prev[i + s]가 가장 잘 맞는 s -> 회전 s × 간격 (rad)
inline double shift_guess(
  const std::vector<float> & r_prev, const std::vector<float> & r_new, double angle_inc,
  int max_shift = 40)
{
  const int n = static_cast<int>(r_prev.size());
  if (n == 0 || static_cast<int>(r_new.size()) != n) {
    return 0.0;
  }
  double best = std::numeric_limits<double>::infinity();
  int best_s = 0;
  std::vector<double> d;
  for (int s = -max_shift; s <= max_shift; ++s) {
    d.clear();
    for (int i = 0; i < n; ++i) {
      const float a = r_prev[((i + s) % n + n) % n];
      const float b = r_new[i];
      if (std::isfinite(a) && std::isfinite(b)) {
        d.push_back(std::abs(static_cast<double>(b) - a));
      }
    }
    if (d.size() < static_cast<std::size_t>(0.3 * n)) {
      continue;
    }
    std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
    if (d[d.size() / 2] < best) {
      best = d[d.size() / 2];
      best_s = s;
    }
  }
  return best_s * angle_inc;
}

// 점마다 이웃 k개로 법선과 곧은 정도 (0~1). 점이 적어 단순 전수 탐색
inline void normals(
  const Points & pts, std::vector<Eigen::Vector2d> & nrm, std::vector<double> & flat,
  int k = 6)
{
  const std::size_t n = pts.size();
  nrm.assign(n, Eigen::Vector2d::Zero());
  flat.assign(n, 0.0);
  std::vector<std::pair<double, std::size_t>> dist(n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      dist[j] = {(pts[j] - pts[i]).squaredNorm(), j};
    }
    const std::size_t kk = std::min<std::size_t>(static_cast<std::size_t>(k), n);
    std::partial_sort(dist.begin(), dist.begin() + kk, dist.end());
    Eigen::Vector2d mean = Eigen::Vector2d::Zero();
    for (std::size_t m = 0; m < kk; ++m) {
      mean += pts[dist[m].second];
    }
    mean /= static_cast<double>(kk);
    Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
    for (std::size_t m = 0; m < kk; ++m) {
      const Eigen::Vector2d c = pts[dist[m].second] - mean;
      cov += c * c.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(cov);
    nrm[i] = eig.eigenvectors().col(0);
    flat[i] = 1.0 - eig.eigenvalues()(0) / std::max(eig.eigenvalues()(1), 1e-12);
  }
}

struct IcpResult
{
  double dx = 0.0, dy = 0.0, dth = 0.0;
  int used = 0;
  double rms = 0.0;
};

// 점-선 ICP: src(새 스캔)를 dst(이전 스캔)에 맞추는 변환. dth = 그 사이 LiDAR가 돈 각도
inline IcpResult icp(
  const Points & src, const Points & dst, double init_th, int iters = 30,
  double max_dist = 0.3)
{
  IcpResult res;
  res.dth = init_th;
  if (src.size() < 20 || dst.size() < 20) {
    return res;
  }
  std::vector<Eigen::Vector2d> nrm;
  std::vector<double> flat;
  normals(dst, nrm, flat);
  double lim = max_dist;
  for (int it = 0; it < iters; ++it) {
    const double c = std::cos(res.dth), s = std::sin(res.dth);
    Eigen::Matrix3d ata = Eigen::Matrix3d::Zero();
    Eigen::Vector3d atb = Eigen::Vector3d::Zero();
    double sse = 0.0;
    int used = 0;
    for (const auto & q0 : src) {
      const Eigen::Vector2d p(c * q0.x() - s * q0.y() + res.dx, s * q0.x() + c * q0.y() + res.dy);
      double best = std::numeric_limits<double>::infinity();
      std::size_t j = 0;
      for (std::size_t m = 0; m < dst.size(); ++m) {
        const double d2 = (dst[m] - p).squaredNorm();
        if (d2 < best) {
          best = d2;
          j = m;
        }
      }
      if (std::sqrt(best) > lim || flat[j] < 0.9) {
        continue;
      }
      const Eigen::Vector2d & n = nrm[j];
      const double r = n.dot(p - dst[j]);
      const Eigen::Vector3d jac(n.x(), n.y(), -n.x() * p.y() + n.y() * p.x());
      const double w = 1.0 / std::max(1.0, std::abs(r) / 0.02);
      ata += w * w * jac * jac.transpose();
      atb += -w * w * r * jac;
      sse += r * r;
      ++used;
    }
    if (used < 20) {
      break;
    }
    const Eigen::Vector3d step = ata.ldlt().solve(atb);
    res.dx += step(0);
    res.dy += step(1);
    res.dth += step(2);
    res.used = used;
    res.rms = std::sqrt(sse / used);
    lim = std::max(0.05, lim * 0.8);
    if (step.cwiseAbs().maxCoeff() < 1e-7) {
      break;
    }
  }
  return res;
}

}  // namespace time_offset
}  // namespace tb3p

#endif  // TB3P_CALIBRATION__TIME_OFFSET_CORE_HPP_
