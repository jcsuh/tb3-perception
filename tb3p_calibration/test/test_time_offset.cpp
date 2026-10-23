// time_offset_core.hpp 시험 (세션 19)
//  1) 여러 사인파의 합으로 돈 회전을 기준 200 Hz, 센서 50 Hz로 재고 센서 시각을 37 ms 늦췄을 때
//     δ = -37 ms를 0.5 ms 안으로 찾는지
//  2) 상자 방 안에서 제자리 회전한 두 스캔을 만들어 ICP가 회전을 0.1° 안으로 찾는지

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "time_offset_core.hpp"

namespace to = tb3p::time_offset;

namespace
{

constexpr double kPi = 3.14159265358979323846;

double yaw_rate(double t)
{
  return 0.6 * std::sin(2 * kPi * 0.23 * t + 0.3) + 0.4 * std::sin(2 * kPi * 0.79 * t + 1.1) +
         0.3 * std::sin(2 * kPi * 1.07 * t + 2.0);
}

// 누적 각도를 촘촘한 사다리꼴 적분으로
double yaw_at(double t)
{
  const int n = 2000;
  double s = 0.0, h = t / n;
  for (int i = 0; i < n; ++i) {
    s += 0.5 * h * (yaw_rate(i * h) + yaw_rate((i + 1) * h));
  }
  return s;
}

// 축에 나란한 직사각형들(방 벽 + 상자)에 광선 쏘기
double raycast(double px, double py, double a)
{
  struct Box {double x0, y0, x1, y1;};
  const Box boxes[] = {{-3.0, -2.5, 3.5, 3.0}, {1.0, 0.8, 1.6, 1.5}, {-2.0, -1.8, -1.2, -1.0}};
  const double dx = std::cos(a), dy = std::sin(a);
  double best = std::numeric_limits<double>::infinity();
  for (const auto & b : boxes) {
    const double xs[2] = {b.x0, b.x1}, ys[2] = {b.y0, b.y1};
    for (double x : xs) {
      if (std::abs(dx) > 1e-12) {
        const double t = (x - px) / dx;
        const double y = py + t * dy;
        if (t > 0 && y >= b.y0 && y <= b.y1) {
          best = std::min(best, t);
        }
      }
    }
    for (double y : ys) {
      if (std::abs(dy) > 1e-12) {
        const double t = (y - py) / dy;
        const double x = px + t * dx;
        if (t > 0 && x >= b.x0 && x <= b.x1) {
          best = std::min(best, t);
        }
      }
    }
  }
  return best;
}

std::vector<float> make_scan(double heading)
{
  std::vector<float> r(400);
  for (int i = 0; i < 400; ++i) {
    r[i] = static_cast<float>(raycast(0.2, 0.1, heading + i * 2 * kPi / 400));
  }
  return r;
}

}  // namespace

TEST(TimeOffset, RecoversDelayOfSlowSensor)
{
  to::Series ref, sensor;
  for (double t = 0.0; t < 30.0; t += 0.005) {
    ref.push(t, yaw_at(t));
  }
  const double delay = 0.037;  // 센서 시각이 37 ms 늦게 찍힘
  for (double t = 0.0; t < 30.0; t += 0.02) {
    sensor.push(t + delay, yaw_at(t));
  }
  const to::FitResult r = to::fit_offset(sensor, ref);
  ASSERT_TRUE(r.ok);
  EXPECT_NEAR(r.offset, -delay, 0.0005);
  EXPECT_NEAR(r.scale, 1.0, 0.001);
}

TEST(TimeOffset, InterpolatesLinearly)
{
  to::Series s;
  s.push(0.0, 0.0);
  s.push(1.0, 2.0);
  s.push(1.0, 5.0);  // 같은 시각은 버림
  EXPECT_DOUBLE_EQ(to::interp(s, 0.25), 0.5);
  EXPECT_DOUBLE_EQ(to::interp(s, 5.0), 2.0);
  EXPECT_EQ(s.size(), 2u);
}

TEST(ScanMatch, RecoversInPlaceRotation)
{
  const double inc = 2 * kPi / 400;
  const auto prev = make_scan(0.0);
  const double turn = 0.1;  // rad, 로봇이 왼쪽으로 돈 각도
  const auto next = make_scan(turn);
  const double guess = to::shift_guess(prev, next, inc);
  EXPECT_NEAR(guess, turn, inc);
  const auto res = to::icp(to::scan_points(next, 0.0, inc), to::scan_points(prev, 0.0, inc),
      guess);
  EXPECT_GT(res.used, 100);
  EXPECT_NEAR(res.dth, turn, 0.1 * kPi / 180);
}
