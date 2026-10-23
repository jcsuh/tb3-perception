// lidar_cam_calib_ceres
//
// 점-평면 잔차를 최소화해 LiDAR-카메라 관계(카메라 <- LiDAR)를 추정한다 (세션 14).
// lidar_cam_calib.py(Python, scipy)와 똑같은 문제를 C++ Ceres Solver로 풀고 두 답을 비교한다.
//
// 문제 (lidar_cam_calib.py와 같음)
//   자세 i의 보드 평면(카메라 좌표계): n_i · x = d_i
//   자세 i의 보드 위 LiDAR 점 p_ij (LiDAR 좌표계, 빔 높이각은 이미 반영됨)
//   파라미터 x(6개) = 공칭 관계 T0 = [R0 | t0]에 더하는 작은 보정, 카메라 좌표계 기준
//     R = Rot(x[0..2]) R0,  t = t0 + x[3..5]      (Rot: 회전 벡터 -> 회전 행렬)
//   잔차 r_ij = w_ij (n_i · (R p_ij + t) - d_i)      (w = 가중치, 없으면 1)
//
// Python과 다른 점
//   - 야코비안: scipy는 수치 미분(조금씩 바꿔 보기), Ceres는 자동 미분(AutoDiff, 정확한 값)
//   - 불확실성: ceres::Covariance로 (JᵀJ)⁻¹를 구하고 잔차 크기 s²를 곱한다
//
// 입력: lidar_cam_calib.py --export-problem 이 만든 YAML (점, 평면, 공칭·정답, Python 답)
// 출력: 결과와 Python 답과의 차이를 화면에, 결과 YAML을 입력 옆 *_ceres.yaml에
//
// 사용 예:
//   ros2 run tb3p_calibration lidar_cam_calib_ceres ~/lidar_cam_data/session13_problem.yaml

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace
{

constexpr double kPi = 3.14159265358979323846;

double deg(double rad) {return rad * 180.0 / kPi;}
double rad(double deg) {return deg * kPi / 180.0;}

struct Pose
{
  std::string name;
  Eigen::Vector3d n;
  double d = 0.0;
  std::vector<Eigen::Vector3d> points;
  std::vector<double> weights;
};

struct CalibProblem
{
  std::string cam_frame;
  std::string lidar_frame;
  double beam_elev_deg = 0.0;
  double pass_rot_deg = 0.5;
  double pass_pos_mm = 10.0;
  Eigen::Matrix4d t_nominal = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d t_truth = Eigen::Matrix4d::Identity();
  bool has_truth = false;
  std::vector<Pose> poses;
  YAML::Node reference;   // Python 답 (비교용, 풀이에는 안 씀)
};

Eigen::Matrix4d read_matrix4(const YAML::Node & node)
{
  Eigen::Matrix4d m;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      m(r, c) = node[r][c].as<double>();
    }
  }
  return m;
}

Eigen::Vector3d read_vector3(const YAML::Node & node)
{
  return Eigen::Vector3d(node[0].as<double>(), node[1].as<double>(), node[2].as<double>());
}

CalibProblem load_problem(const std::string & path)
{
  const YAML::Node root = YAML::LoadFile(path);
  CalibProblem prob;
  prob.cam_frame = root["cam_frame"].as<std::string>();
  prob.lidar_frame = root["lidar_frame"].as<std::string>();
  prob.beam_elev_deg = root["beam_elev_deg"].as<double>(0.0);
  prob.pass_rot_deg = root["pass_rot_deg"].as<double>(0.5);
  prob.pass_pos_mm = root["pass_pos_mm"].as<double>(10.0);
  prob.t_nominal = read_matrix4(root["T_nominal"]);
  if (root["T_truth"]) {
    prob.t_truth = read_matrix4(root["T_truth"]);
    prob.has_truth = true;
  }
  for (const auto & item : root["poses"]) {
    Pose pose;
    pose.name = item["name"].as<std::string>();
    pose.n = read_vector3(item["n"]);
    pose.d = item["d"].as<double>();
    for (const auto & p : item["points"]) {
      pose.points.push_back(read_vector3(p));
    }
    if (item["weights"]) {
      pose.weights = item["weights"].as<std::vector<double>>();
    } else {
      pose.weights.assign(pose.points.size(), 1.0);
    }
    if (pose.weights.size() != pose.points.size()) {
      throw std::runtime_error("자세 " + pose.name + ": 가중치 수와 점 수가 다름");
    }
    prob.poses.push_back(std::move(pose));
  }
  prob.reference = root["reference_python"];
  return prob;
}

// 점 하나의 점-평면 잔차. Ceres가 이 템플릿 함수로 자동 미분(야코비안)까지 계산한다.
// 공칭 회전 R0는 미리 곱해 둔다: q = R0 p  ->  R p + t = Rot(x[0..2]) q + t0 + x[3..5]
struct PointToPlaneResidual
{
  PointToPlaneResidual(
    const Eigen::Vector3d & q, const Eigen::Vector3d & t0, const Eigen::Vector3d & n,
    double d, double w)
  : q_(q), t0_(t0), n_(n), d_(d), w_(w)
  {
  }

  template<typename T>
  bool operator()(const T * const x, T * residual) const
  {
    const T q[3] = {T(q_.x()), T(q_.y()), T(q_.z())};
    T rq[3];
    ceres::AngleAxisRotatePoint(x, q, rq);
    T dist = T(-d_);
    for (int k = 0; k < 3; ++k) {
      dist += T(n_[k]) * (rq[k] + T(t0_[k]) + x[3 + k]);
    }
    residual[0] = T(w_) * dist;
    return true;
  }

  static ceres::CostFunction * create(
    const Eigen::Vector3d & q, const Eigen::Vector3d & t0, const Eigen::Vector3d & n,
    double d, double w)
  {
    return new ceres::AutoDiffCostFunction<PointToPlaneResidual, 1, 6>(
      new PointToPlaneResidual(q, t0, n, d, w));
  }

  Eigen::Vector3d q_;
  Eigen::Vector3d t0_;
  Eigen::Vector3d n_;
  double d_;
  double w_;
};

// 보정 x(6개)를 공칭 관계 t0에 적용한 4x4 변환 (Python의 apply()와 같음)
Eigen::Matrix4d apply_correction(const Eigen::Matrix4d & t0, const double * x)
{
  Eigen::Matrix3d rot;
  ceres::AngleAxisToRotationMatrix(x, ceres::ColumnMajorAdapter3x3(rot.data()));
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t.topLeftCorner<3, 3>() = rot * t0.topLeftCorner<3, 3>();
  t.topRightCorner<3, 1>() = t0.topRightCorner<3, 1>() + Eigen::Vector3d(x[3], x[4], x[5]);
  return t;
}

// 회전 행렬이 나타내는 회전의 크기 (°). acos(trace)는 아주 작은 각도에서 부정확해서
// 쿼터니언을 거치는 Eigen::AngleAxisd를 쓴다 (Python과 Ceres 답의 미세한 차이를 보려고)
double rotation_angle_deg(const Eigen::Matrix3d & r)
{
  return deg(Eigen::AngleAxisd(r).angle());
}

// 두 변환의 차이: 회전 (°), 위치 (mm)
std::pair<double, double> difference(const Eigen::Matrix4d & a, const Eigen::Matrix4d & b)
{
  const Eigen::Matrix3d r = a.topLeftCorner<3, 3>().transpose() * b.topLeftCorner<3, 3>();
  const double pos = (a.topRightCorner<3, 1>() - b.topRightCorner<3, 1>()).norm() * 1000.0;
  return {rotation_angle_deg(r), pos};
}

// 가중치 없는 점-평면 거리의 RMS (mm): 자세 하나 또는 전체
double residual_rms_mm(const CalibProblem & prob, const Eigen::Matrix4d & t, int only = -1)
{
  double sum = 0.0;
  int count = 0;
  for (int i = 0; i < static_cast<int>(prob.poses.size()); ++i) {
    if (only >= 0 && i != only) {
      continue;
    }
    const Pose & pose = prob.poses[i];
    for (const auto & p : pose.points) {
      const Eigen::Vector3d q = t.topLeftCorner<3, 3>() * p + t.topRightCorner<3, 1>();
      const double r = pose.n.dot(q) - pose.d;
      sum += r * r;
      ++count;
    }
  }
  return count ? std::sqrt(sum / count) * 1000.0 : 0.0;
}

std::vector<double> yaml_vector(const YAML::Node & node)
{
  return node ? node.as<std::vector<double>>() : std::vector<double>();
}

void emit_vector(YAML::Emitter & out, const std::string & key, const std::vector<double> & v)
{
  out << YAML::Key << key << YAML::Value << YAML::Flow << YAML::BeginSeq;
  for (double x : v) {
    out << x;
  }
  out << YAML::EndSeq;
}

std::string output_path(const std::string & input)
{
  const auto dot = input.find_last_of('.');
  const auto slash = input.find_last_of('/');
  const bool has_ext = dot != std::string::npos && (slash == std::string::npos || dot > slash);
  return (has_ext ? input.substr(0, dot) : input) + "_ceres.yaml";
}

int run(const std::string & in_path, const std::string & out_path)
{
  const CalibProblem prob = load_problem(in_path);
  const Eigen::Matrix3d r0 = prob.t_nominal.topLeftCorner<3, 3>();
  const Eigen::Vector3d t0 = prob.t_nominal.topRightCorner<3, 1>();

  // 1) 문제 만들기: 점 하나 = 잔차 블록 하나, 모두 같은 파라미터 블록 x(6개)를 공유
  double x[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  ceres::Problem problem;
  int n_points = 0;
  for (const Pose & pose : prob.poses) {
    for (size_t j = 0; j < pose.points.size(); ++j) {
      problem.AddResidualBlock(
        PointToPlaneResidual::create(r0 * pose.points[j], t0, pose.n, pose.d, pose.weights[j]),
        nullptr, x);
      ++n_points;
    }
  }
  if (n_points <= 6) {
    std::fprintf(stderr, "점이 %d개뿐: 6개보다 많아야 함\n", n_points);
    return 1;
  }

  // 2) 풀기: Python과 같은 Levenberg-Marquardt, 작은 밀집 문제라 DENSE_QR
  ceres::Solver::Options options;
  options.minimizer_type = ceres::TRUST_REGION;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.linear_solver_type = ceres::DENSE_QR;
  options.function_tolerance = 1e-12;
  options.gradient_tolerance = 1e-12;
  options.parameter_tolerance = 1e-12;
  options.max_num_iterations = 100;
  options.logging_type = ceres::SILENT;
  ceres::Solver::Summary summary;
  const auto start = std::chrono::steady_clock::now();
  ceres::Solve(options, &problem, &summary);
  const double solve_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - start).count();
  const Eigen::Matrix4d t_est = apply_correction(prob.t_nominal, x);

  // 3) 불확실성: Σ = s² (JᵀJ)⁻¹,  s² = 잔차 제곱합 / (점 수 - 6)
  //    ceres::Covariance는 (JᵀJ)⁻¹만 주므로 s²는 직접 곱한다 (Ceres cost = ½ Σ r²)
  const double s2 = 2.0 * summary.final_cost / (n_points - 6);
  ceres::Covariance::Options cov_options;
  cov_options.algorithm_type = ceres::DENSE_SVD;
  ceres::Covariance covariance(cov_options);
  std::vector<std::pair<const double *, const double *>> blocks = {{x, x}};
  Eigen::Matrix<double, 6, 6, Eigen::RowMajor> cov = Eigen::Matrix<double, 6, 6>::Zero();
  const bool cov_ok = covariance.Compute(blocks, &problem) &&
    covariance.GetCovarianceBlock(x, x, cov.data());
  cov *= s2;

  // 가장 약한 방향: 합격 기준 단위(0.5°, 10 mm = 1)로 맞춘 JᵀJ의 가장 작은 고유값 방향
  ceres::CRSMatrix jac;
  problem.Evaluate(ceres::Problem::EvaluateOptions(), nullptr, nullptr, nullptr, &jac);
  Eigen::MatrixXd j = Eigen::MatrixXd::Zero(jac.num_rows, jac.num_cols);
  for (int r = 0; r < jac.num_rows; ++r) {
    for (int k = jac.rows[r]; k < jac.rows[r + 1]; ++k) {
      j(r, jac.cols[k]) = jac.values[k];
    }
  }
  Eigen::Matrix<double, 6, 1> unit;
  unit << Eigen::Vector3d::Constant(rad(prob.pass_rot_deg)),
    Eigen::Vector3d::Constant(prob.pass_pos_mm / 1000.0);
  const Eigen::MatrixXd js = j * unit.asDiagonal();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eig(js.transpose() * js);
  const Eigen::Matrix<double, 6, 1> weakest = eig.eigenvectors().col(0);
  const double cond = eig.eigenvalues()(5) / std::max(eig.eigenvalues()(0), 1e-30);

  std::vector<double> std_rot(3), std_pos(3);
  for (int k = 0; k < 3; ++k) {
    std_rot[k] = deg(std::sqrt(std::max(cov(k, k), 0.0)));
    std_pos[k] = std::sqrt(std::max(cov(3 + k, 3 + k), 0.0)) * 1000.0;
  }
  const double rms_after = residual_rms_mm(prob, t_est);

  // 4) 출력
  std::printf(
    "자세 %zu개, 점 %d개, 빔 높이각 %+.3f°\n", prob.poses.size(), n_points,
    prob.beam_elev_deg);
  std::printf(
    "[Ceres] %s, 반복 %d회, %.2f ms, cost %.6e -> %.6e\n",
    summary.termination_type == ceres::CONVERGENCE ? "수렴" : "수렴 못함",
    static_cast<int>(summary.iterations.size()) - 1, solve_ms, summary.initial_cost,
    summary.final_cost);
  std::printf("[자세별 잔차 RMS, mm]  추정 전(공칭) -> 추정 후\n");
  for (int i = 0; i < static_cast<int>(prob.poses.size()); ++i) {
    std::printf(
      "    %-4s %6.1f -> %5.1f\n", prob.poses[i].name.c_str(),
      residual_rms_mm(prob, prob.t_nominal, i), residual_rms_mm(prob, t_est, i));
  }
  std::printf(
    "    전체 %6.1f -> %5.1f\n", residual_rms_mm(prob, prob.t_nominal), rms_after);

  double err_rot = 0.0, err_pos = 0.0;
  if (prob.has_truth) {
    const auto before = difference(prob.t_nominal, prob.t_truth);
    std::tie(err_rot, err_pos) = difference(t_est, prob.t_truth);
    const bool ok = err_rot <= prob.pass_rot_deg && err_pos <= prob.pass_pos_mm;
    std::printf("[정답 대비 오차]\n");
    std::printf("    추정 전 (공칭)  회전 %6.3f°  위치 %6.1f mm\n", before.first, before.second);
    std::printf(
      "    추정 후         회전 %6.3f°  위치 %6.1f mm  -> %s\n", err_rot, err_pos,
      ok ? "합격" : "불합격");
  }
  std::printf("[추정 불확실성 1σ]%s\n", cov_ok ? "" : "  (공분산 계산 실패)");
  std::printf(
    "    회전 (°)   x축 %.3f  y축 %.3f  z축 %.3f\n", std_rot[0], std_rot[1], std_rot[2]);
  std::printf("    위치 (mm)  x %.1f  y %.1f  z %.1f\n", std_pos[0], std_pos[1], std_pos[2]);
  std::printf(
    "    가장 약한 방향 (회전 x y z, 위치 x y z): %+.2f %+.2f %+.2f %+.2f %+.2f %+.2f"
    "  (조건수 %.1e)\n", weakest[0], weakest[1], weakest[2], weakest[3], weakest[4],
    weakest[5], cond);

  // Python 답과 비교
  double diff_rot = -1.0, diff_pos = -1.0;
  if (prob.reference && prob.reference["matrix"]) {
    const Eigen::Matrix4d t_py = read_matrix4(prob.reference["matrix"]);
    std::tie(diff_rot, diff_pos) = difference(t_est, t_py);
    const auto py_rot = yaml_vector(prob.reference["std_rot_deg"]);
    const auto py_pos = yaml_vector(prob.reference["std_pos_mm"]);
    std::printf("[Python(scipy) 답과 비교]\n");
    std::printf("    추정값 차이   회전 %.2e°  위치 %.2e mm\n", diff_rot, diff_pos);
    std::printf(
      "    잔차 RMS      Python %.4f mm  Ceres %.4f mm\n",
      prob.reference["residual_rms_mm"].as<double>(0.0), rms_after);
    if (py_rot.size() == 3 && py_pos.size() == 3) {
      std::printf(
        "    1σ 회전 (°)   Python %.4f %.4f %.4f  Ceres %.4f %.4f %.4f\n",
        py_rot[0], py_rot[1], py_rot[2], std_rot[0], std_rot[1], std_rot[2]);
      std::printf(
        "    1σ 위치 (mm)  Python %.3f %.3f %.3f  Ceres %.3f %.3f %.3f\n",
        py_pos[0], py_pos[1], py_pos[2], std_pos[0], std_pos[1], std_pos[2]);
    }
    std::printf(
      "    풀이 시간     Python %.1f ms (잔차 계산 %d회)  Ceres %.2f ms\n",
      prob.reference["solve_ms"].as<double>(0.0), prob.reference["nfev"].as<int>(0), solve_ms);
    const bool same = diff_rot < 1e-3 && diff_pos < 1e-2;
    std::printf(
      "    판정: %s (기준: 회전 0.001° 미만, 위치 0.01 mm 미만)\n",
      same ? "같은 답" : "다른 답");
  }

  // 결과 YAML (Python의 *_result.yaml과 같은 키 + 풀이 정보)
  Eigen::AngleAxisd aa(Eigen::Matrix3d(t_est.topLeftCorner<3, 3>()));
  const Eigen::Vector3d rvec = aa.angle() * aa.axis();
  YAML::Emitter out;
  out.SetDoublePrecision(15);
  out << YAML::BeginMap;
  out << YAML::Key << "parent" << YAML::Value << prob.cam_frame;
  out << YAML::Key << "child" << YAML::Value << prob.lidar_frame;
  emit_vector(out, "xyz", {t_est(0, 3), t_est(1, 3), t_est(2, 3)});
  emit_vector(out, "rotation_vector", {rvec.x(), rvec.y(), rvec.z()});
  out << YAML::Key << "matrix" << YAML::Value << YAML::BeginSeq;
  for (int r = 0; r < 4; ++r) {
    out << YAML::Flow << YAML::BeginSeq;
    for (int c = 0; c < 4; ++c) {
      out << t_est(r, c);
    }
    out << YAML::EndSeq;
  }
  out << YAML::EndSeq;
  out << YAML::Key << "poses_used" << YAML::Value << YAML::Flow << YAML::BeginSeq;
  for (const Pose & pose : prob.poses) {
    out << pose.name;
  }
  out << YAML::EndSeq;
  out << YAML::Key << "beam_elev_deg" << YAML::Value << prob.beam_elev_deg;
  out << YAML::Key << "residual_rms_mm" << YAML::Value << rms_after;
  emit_vector(out, "std_rot_deg", std_rot);
  emit_vector(out, "std_pos_mm", std_pos);
  if (prob.has_truth) {
    out << YAML::Key << "error_vs_truth" << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "rot_deg" << YAML::Value << err_rot;
    out << YAML::Key << "pos_mm" << YAML::Value << err_pos;
    out << YAML::EndMap;
  }
  out << YAML::Key << "solver" << YAML::Value << YAML::BeginMap;
  out << YAML::Key << "name" << YAML::Value << "ceres (LM, DENSE_QR, autodiff)";
  out << YAML::Key << "iterations" << YAML::Value <<
    static_cast<int>(summary.iterations.size()) - 1;
  out << YAML::Key << "solve_ms" << YAML::Value << solve_ms;
  out << YAML::Key << "converged" << YAML::Value <<
    (summary.termination_type == ceres::CONVERGENCE);
  out << YAML::EndMap;
  if (diff_rot >= 0.0) {
    out << YAML::Key << "diff_vs_python" << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "rot_deg" << YAML::Value << diff_rot;
    out << YAML::Key << "pos_mm" << YAML::Value << diff_pos;
    out << YAML::EndMap;
  }
  out << YAML::EndMap;
  std::ofstream file(out_path);
  file << out.c_str() << "\n";
  if (!file) {
    std::fprintf(stderr, "저장 실패: %s\n", out_path.c_str());
    return 1;
  }
  std::printf("saved %s\n", out_path.c_str());
  return summary.IsSolutionUsable() ? 0 : 1;
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc < 2 || argc > 3 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
    std::printf(
      "사용법: lidar_cam_calib_ceres <problem.yaml> [result.yaml]\n"
      "  problem.yaml: lidar_cam_calib.py --export-problem 이 만든 파일\n"
      "  result.yaml : 결과 (기본: 입력 옆 *_ceres.yaml)\n");
    return argc == 2 ? 0 : 1;
  }
  const std::string in_path = argv[1];
  const std::string out_path = argc == 3 ? argv[2] : output_path(in_path);
  try {
    return run(in_path, out_path);
  } catch (const std::exception & e) {
    std::fprintf(stderr, "오류: %s\n", e.what());
    return 1;
  }
}
