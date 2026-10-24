// test_drift_core.cpp
//
// drift_core.hpp 시험 (세션 21): 가짜 방(벽 + 상자 2개)을 만들어
//   진짜 카메라를 마운트 축 둘레로 δ 돌린 깊이 영상과 LiDAR 스캔을 광선 추적으로 만들고,
//   estimate_yaw가 δ를 되찾는지 본다. 판정(Judge)의 학습·WARN·ERROR·연속 조건도 본다.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "drift_core.hpp"

namespace dr = tb3p::drift;
using Eigen::Vector3d;

namespace
{

constexpr double kDegToRad = dr::kPi / 180.0;

struct Box
{
  Vector3d lo, hi;
};

// 광선 p + t·d 가 상자와 만나는 [t_in, t_out] (못 만나면 false)
bool slab(const Box & b, const Vector3d & p, const Vector3d & d, double & t_in, double & t_out)
{
  t_in = -std::numeric_limits<double>::infinity();
  t_out = std::numeric_limits<double>::infinity();
  for (int i = 0; i < 3; ++i) {
    if (std::abs(d[i]) < 1e-12) {
      if (p[i] < b.lo[i] || p[i] > b.hi[i]) {
        return false;
      }
      continue;
    }
    double t1 = (b.lo[i] - p[i]) / d[i];
    double t2 = (b.hi[i] - p[i]) / d[i];
    if (t1 > t2) {
      std::swap(t1, t2);
    }
    t_in = std::max(t_in, t1);
    t_out = std::min(t_out, t2);
  }
  return t_in <= t_out;
}

// 방 안에서 광선이 처음 닿는 곳까지의 t (방 벽 = 안에서 나가는 곳, 상자 = 들어가는 곳)
double cast(const Vector3d & p, const Vector3d & d)
{
  const Box room{Vector3d(-2.5, -1.8, -0.1), Vector3d(3.0, 2.2, 2.5)};
  const std::vector<Box> obstacles = {
    {Vector3d(1.2, -0.9, -0.1), Vector3d(1.6, -0.3, 0.8)},
    {Vector3d(1.9, 0.4, -0.1), Vector3d(2.3, 1.1, 0.6)},
  };
  double t_in, t_out;
  slab(room, p, d, t_in, t_out);
  double best = t_out;
  for (const auto & b : obstacles) {
    if (slab(b, p, d, t_in, t_out) && t_in > 1e-6 && t_in < best) {
      best = t_in;
    }
  }
  return best;
}

// 공칭 카메라 자세: 로봇 앞쪽을 본다 (광학 좌표: z 앞, x 오른쪽, y 아래)
dr::Iso nominal_camera()
{
  dr::Iso t = dr::Iso::Identity();
  t.translate(Vector3d(0.07, 0.01, 0.25));
  Eigen::Matrix3d r;
  r << 0, 0, 1,
    -1, 0, 0,
    0, -1, 0;
  t.rotate(r);
  return t;
}

dr::Intrinsics intrinsics()
{
  dr::Intrinsics k;
  k.fx = k.fy = 220.0;
  k.cx = 160.0;
  k.cy = 120.0;
  return k;
}

dr::DepthImage render(const dr::Iso & t_base_cam, const dr::Intrinsics & k)
{
  dr::DepthImage img;
  img.width = 320;
  img.height = 240;
  img.data.resize(320 * 240);
  for (int v = 0; v < img.height; ++v) {
    for (int u = 0; u < img.width; ++u) {
      // 광학 좌표에서 z = 1인 방향 -> 광선 길이 t가 곧 깊이(z)
      const Vector3d dc((u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0);
      const Vector3d d = t_base_cam.linear() * dc;
      img.data[v * img.width + u] = static_cast<float>(cast(t_base_cam.translation(), d));
    }
  }
  return img;
}

dr::Iso lidar_pose()
{
  dr::Iso t = dr::Iso::Identity();
  t.translate(Vector3d(-0.03, 0.0, 0.17));
  return t;
}

std::vector<float> scan(int n, double angle_min, double angle_inc)
{
  const dr::Iso tl = lidar_pose();
  std::vector<float> r(n);
  for (int i = 0; i < n; ++i) {
    const double a = angle_min + angle_inc * i;
    r[i] = static_cast<float>(cast(tl.translation(), Vector3d(std::cos(a), std::sin(a), 0.0)));
  }
  return r;
}

double recover_deg(double true_deg, int n_frames = 1)
{
  const dr::Iso t0 = nominal_camera();
  const Vector3d mount(0.05, 0.0, 0.22);
  const dr::Intrinsics k = intrinsics();
  const dr::Iso truth = dr::camera_pose_with_yaw(t0, mount, true_deg * kDegToRad);
  const int n = 360;
  const double amin = -dr::kPi, ainc = 2 * dr::kPi / n;
  dr::Frame f;
  f.pts_base = dr::transform(lidar_pose(), dr::lidar_points(scan(n, amin, ainc), amin, ainc, 0.0));
  f.depth = render(truth, k);
  const std::vector<dr::Frame> frames(n_frames, f);
  const auto est = dr::estimate_yaw(frames, t0, mount, k);
  EXPECT_TRUE(est.ok);
  EXPECT_EQ(est.frames_used, n_frames);
  if (std::abs(true_deg) > 0.5) {
    EXPECT_GT(est.cost_zero, est.cost_best);    // 돌았으면 δ=0일 때 잔차가 더 크다
  }
  return est.delta / kDegToRad;
}

}  // namespace

TEST(DriftCore, LidarPointsSkipInvalidAndApplyElevation)
{
  const std::vector<float> r = {1.0f, std::nanf(""), 0.1f, 9.0f, 2.0f};
  const auto p = dr::lidar_points(r, 0.0, dr::kPi / 2, 1.0);
  ASSERT_EQ(p.size(), 2u);
  EXPECT_NEAR(p[0].x(), std::cos(kDegToRad), 1e-12);
  EXPECT_NEAR(p[0].z(), std::sin(kDegToRad), 1e-12);
  // 5번째 빔: 각도 2π -> +x 방향
  EXPECT_NEAR(p[1].x(), 2.0 * std::cos(kDegToRad), 1e-9);
  EXPECT_NEAR(p[1].y(), 0.0, 1e-9);
}

TEST(DriftCore, CameraYawKeepsMountFixed)
{
  const dr::Iso t0 = nominal_camera();
  const Vector3d mount(0.05, 0.0, 0.22);
  const dr::Iso t = dr::camera_pose_with_yaw(t0, mount, 3.0 * kDegToRad);
  // 마운트 점은 그대로, 광축은 base z 둘레로 3° 돈다
  const Vector3d m_cam = t0.inverse() * mount;
  EXPECT_NEAR((t * m_cam - mount).norm(), 0.0, 1e-12);
  const Vector3d axis0 = t0.linear().col(2), axis = t.linear().col(2);
  EXPECT_NEAR(std::atan2(axis.y(), axis.x()) - std::atan2(axis0.y(), axis0.x()),
    3.0 * kDegToRad, 1e-12);
}

TEST(DriftCore, PerfectCalibrationGivesSmallResidual)
{
  const dr::Iso t0 = nominal_camera();
  const dr::Intrinsics k = intrinsics();
  const int n = 360;
  const double amin = -dr::kPi, ainc = 2 * dr::kPi / n;
  const auto pts = dr::transform(lidar_pose(), dr::lidar_points(scan(n, amin, ainc), amin, ainc,
      0.0));
  const auto res = dr::residuals(pts, t0, k, render(t0, k));
  ASSERT_GE(static_cast<int>(res.size()), dr::kMinPoints);
  EXPECT_LT(dr::robust_cost(res), 0.01);
  EXPECT_TRUE(std::isnan(dr::robust_cost(std::vector<double>(5, 0.0))));
}

TEST(DriftCore, RecoversCameraYaw)
{
  // 픽셀 반올림 때문에 ±0.1° 정도는 남는다 (실제 모니터는 배운 기준과의 차이를 보므로 괜찮다)
  EXPECT_NEAR(recover_deg(0.0), 0.0, 0.15);
  EXPECT_NEAR(recover_deg(1.0), 1.0, 0.15);
  EXPECT_NEAR(recover_deg(2.0), 2.0, 0.15);
  EXPECT_NEAR(recover_deg(-2.0), -2.0, 0.15);
  EXPECT_NEAR(recover_deg(5.0, 3), 5.0, 0.15);
}

TEST(DriftCore, NoPointsGivesNotOk)
{
  dr::Frame f;
  f.depth.width = 10;
  f.depth.height = 10;
  f.depth.data.assign(100, 1.0f);
  const auto est = dr::estimate_yaw({f}, nominal_camera(), Vector3d::Zero(), intrinsics());
  EXPECT_FALSE(est.ok);
}

TEST(DriftCore, ViewChangedNeedsTurnOrMove)
{
  const double turn = 5.0 * kDegToRad, move = 0.10;
  // 서 있음: 자이로 잡음 정도만
  EXPECT_FALSE(dr::view_changed({0.0, 0.001, -0.001, 0.0}, {1.0, 1.0, 1.0, 1.0}, turn, move));
  // 제자리 회전 0.3 rad/s × 2.5 s
  EXPECT_TRUE(dr::view_changed({0.0, 0.15, 0.3, 0.45, 0.6, 0.75}, {1.0, 1.0, 1.0}, turn, move));
  // 왼쪽으로 돌다 오른쪽으로 (범위로 보므로 처음과 끝이 같아도 움직인 것)
  EXPECT_TRUE(dr::view_changed({0.0, 0.2, 0.0}, {0.0, 0.0, 0.0}, turn, move));
  // 직진 0.1 m/s × 2.5 s
  EXPECT_TRUE(dr::view_changed({0.0, 0.0}, {2.00, 2.05, 2.10, 2.15, 2.20, 2.25}, turn, move));
  EXPECT_FALSE(dr::view_changed({}, {}, turn, move));
}

TEST(Judge, LearnsBaselineThenJudges)
{
  dr::Judge j(0.5, 1.0, 6.0, 3, 2);
  EXPECT_EQ(j.update(0.10), dr::Level::STALE);
  EXPECT_EQ(j.update(0.12), dr::Level::STALE);
  EXPECT_FALSE(j.learned());
  EXPECT_EQ(j.update(0.08), dr::Level::STALE);
  ASSERT_TRUE(j.learned());
  EXPECT_NEAR(j.mean(), 0.10, 1e-12);
  EXPECT_NEAR(j.std_dev(), 1.4826 * 0.02, 1e-12);   // 중앙값 절대 편차 0.02
  EXPECT_DOUBLE_EQ(j.warn(), 0.5);     // 6 × 0.03 × 0.5 = 0.09 < 0.5
  EXPECT_DOUBLE_EQ(j.error(), 1.0);

  EXPECT_EQ(j.update(0.30), dr::Level::OK);
  EXPECT_EQ(j.update(0.70), dr::Level::WARN);
  EXPECT_NEAR(j.deviation(), 0.60, 1e-12);
  // ERROR 크기라도 1번만이면 WARN, 2번 연속이면 ERROR
  EXPECT_EQ(j.update(1.20), dr::Level::WARN);
  EXPECT_EQ(j.update(-1.00), dr::Level::ERROR);
  EXPECT_EQ(j.update(1.50), dr::Level::ERROR);
  // 한 번 정상으로 돌아오면 연속 횟수는 처음부터
  EXPECT_EQ(j.update(0.10), dr::Level::OK);
  EXPECT_EQ(j.update(1.20), dr::Level::WARN);
}

TEST(Judge, MedianIgnoresOneOutlierWhileLearning)
{
  EXPECT_DOUBLE_EQ(dr::median({3.0, 1.0, 2.0}), 2.0);
  EXPECT_DOUBLE_EQ(dr::median({4.0, 1.0, 3.0, 2.0}), 2.5);
  // 학습 중 하나가 튀어도 (단계가 바뀌는 묶음 등) 기준과 퍼짐은 거의 그대로
  dr::Judge j(0.5, 1.0, 6.0, 5, 2);
  for (double v : {-0.10, -0.12, 0.90, -0.08, -0.11}) {
    j.update(v);
  }
  ASSERT_TRUE(j.learned());
  EXPECT_NEAR(j.mean(), -0.10, 1e-12);
  EXPECT_LT(j.std_dev(), 0.05);
  EXPECT_DOUBLE_EQ(j.error(), 1.0);
}

TEST(Judge, NoisyBaselineWidensThresholds)
{
  dr::Judge j(0.5, 1.0, 6.0, 1, 1);
  j.set_baseline(-0.1, 0.3);
  EXPECT_NEAR(j.warn(), 0.9, 1e-12);
  EXPECT_NEAR(j.error(), 1.8, 1e-12);
  EXPECT_EQ(j.update(0.6), dr::Level::OK);
  EXPECT_EQ(j.update(1.0), dr::Level::WARN);
  EXPECT_EQ(j.update(1.8), dr::Level::ERROR);    // persist 1
}
