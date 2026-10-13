// tb3p::Excitation
//
// 손에 든 캘리브레이션 리그를 정해진 흔들기 궤적으로 움직이는 Gazebo 시스템 플러그인.
//
// - 궤적: 중심 자세 + (축마다 다른 주파수의 사인파) x (처음 ramp 초 동안 0에서 1로 커지는 배율)
// - 매 물리 스텝마다 "궤적을 따라가는 데 필요한 힘과 토크"를 계산해 리그에 가한다
//   힘   = m (a_목표 + Kp 위치오차 + Kd 속도오차 - g)      (중력은 여기서 상쇄)
//   토크 = I (α_목표 + Kp 자세오차 + Kd 각속도오차) + ω x (I ω)
// - 속도를 직접 덮어쓰지 않고 힘으로 움직이므로, 물리 엔진이 계산한 가속도가 실제와 같고
//   IMU 센서가 손으로 흔들 때와 같은 값을 낸다
//
// SDF 파라미터 (괄호 안은 기본값)
//   center             리그 중심 자세 x y z roll pitch yaw (0 0 0.5 0 0 0)
//   amplitude_xyz      위치 진폭 m (0.10 0.10 0.06)
//   amplitude_rpy_deg  자세 진폭 deg (20 15 20)
//   freq_xyz           위치 주파수 Hz (0.37 0.53 0.71)
//   freq_rpy           자세 주파수 Hz (0.61 0.43 0.29)
//   start_delay        시작 전 정지 시간 s (5)
//   ramp               진폭이 0에서 최대로 커지는 시간 s (3)

#include <gz/math/Matrix3.hh>
#include <gz/math/Pose3.hh>
#include <gz/math/Quaternion.hh>
#include <gz/math/Vector3.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/World.hh>
#include <sdf/Element.hh>

#include <chrono>
#include <cmath>
#include <memory>

namespace tb3p
{

class Excitation
  : public gz::sim::System,
  public gz::sim::ISystemConfigure,
  public gz::sim::ISystemPreUpdate
{
public:
  void Configure(
    const gz::sim::Entity & entity,
    const std::shared_ptr<const sdf::Element> & sdf,
    gz::sim::EntityComponentManager & ecm,
    gz::sim::EventManager & /*event_mgr*/) override
  {
    gz::sim::Model model(entity);
    link_ = gz::sim::Link(model.CanonicalLink(ecm));
    link_.EnableVelocityChecks(ecm, true);

    center_ = sdf->Get<gz::math::Pose3d>("center", center_).first;
    amp_xyz_ = sdf->Get<gz::math::Vector3d>("amplitude_xyz", amp_xyz_).first;
    amp_rpy_ = sdf->Get<gz::math::Vector3d>("amplitude_rpy_deg", amp_rpy_).first * (GZ_PI / 180.0);
    freq_xyz_ = sdf->Get<gz::math::Vector3d>("freq_xyz", freq_xyz_).first;
    freq_rpy_ = sdf->Get<gz::math::Vector3d>("freq_rpy", freq_rpy_).first;
    start_delay_ = sdf->Get<double>("start_delay", start_delay_).first;
    ramp_ = sdf->Get<double>("ramp", ramp_).first;

    gz::sim::World world(gz::sim::worldEntity(ecm));
    auto gravity = world.Gravity(ecm);
    if (gravity) {
      gravity_ = *gravity;
    }
  }

  void PreUpdate(
    const gz::sim::UpdateInfo & info,
    gz::sim::EntityComponentManager & ecm) override
  {
    if (info.paused) {
      return;
    }
    const double now = std::chrono::duration<double>(info.simTime).count();

    auto pose = link_.WorldPose(ecm);
    auto com_pose = link_.WorldInertialPose(ecm);
    auto inertial = link_.WorldInertial(ecm);
    auto inertia = link_.WorldInertiaMatrix(ecm);
    auto omega = link_.WorldAngularVelocity(ecm);
    if (!pose || !com_pose || !inertial || !inertia || !omega) {
      return;  // 물리 엔진이 아직 값을 채우지 않은 첫 스텝
    }
    if (!initialized_) {
      // 링크 원점에서 본 무게중심 위치 (링크 좌표계, 고정값)
      com_offset_ = pose->Rot().RotateVectorReverse(com_pose->Pos() - pose->Pos());
      t0_ = now;
      initialized_ = true;
    }
    auto vel = link_.WorldLinearVelocity(ecm, com_offset_);
    if (!vel) {
      return;
    }

    const double t = now - t0_ - start_delay_;
    const double h = 1e-3;

    // 무게중심의 목표 위치, 속도, 가속도 (수치 미분)
    const gz::math::Vector3d x_m = DesiredCom(t - h);
    const gz::math::Vector3d x_0 = DesiredCom(t);
    const gz::math::Vector3d x_p = DesiredCom(t + h);
    const gz::math::Vector3d v_d = (x_p - x_m) / (2.0 * h);
    const gz::math::Vector3d a_d = (x_p - x_0 * 2.0 + x_m) / (h * h);

    // 목표 자세, 각속도, 각가속도 (월드 좌표계)
    const gz::math::Quaterniond q_d = Desired(t).Rot();
    const gz::math::Vector3d w_d = DesiredOmega(t);
    const gz::math::Vector3d alpha_d = (DesiredOmega(t + h) - DesiredOmega(t - h)) / (2.0 * h);

    const double mass = inertial->MassMatrix().Mass();
    const gz::math::Vector3d pos_err = x_0 - com_pose->Pos();
    const gz::math::Vector3d vel_err = v_d - *vel;
    const gz::math::Vector3d rot_err = RotationVector(q_d * pose->Rot().Inverse());
    const gz::math::Vector3d omega_err = w_d - *omega;

    const gz::math::Vector3d force =
      (a_d + pos_err * kp_ + vel_err * kd_ - gravity_) * mass;
    const gz::math::Vector3d gyro = omega->Cross((*inertia) * (*omega));
    const gz::math::Vector3d torque =
      (*inertia) * (alpha_d + rot_err * kp_ + omega_err * kd_) + gyro;

    link_.AddWorldWrench(ecm, force, torque, com_offset_);
  }

private:
  // 0 → 1로 부드럽게 커지는 배율 (시작할 때 속도와 가속도가 갑자기 튀지 않게)
  double Envelope(double t) const
  {
    if (t <= 0.0) {
      return 0.0;
    }
    if (t >= ramp_) {
      return 1.0;
    }
    const double u = t / ramp_;
    return u * u * (3.0 - 2.0 * u);
  }

  // 시각 t에서 리그 링크 원점의 목표 자세 (월드 좌표계)
  gz::math::Pose3d Desired(double t) const
  {
    const double s = Envelope(t);
    const double two_pi = 2.0 * GZ_PI;
    const gz::math::Vector3d dp(
      amp_xyz_.X() * std::sin(two_pi * freq_xyz_.X() * t),
      amp_xyz_.Y() * std::sin(two_pi * freq_xyz_.Y() * t + 1.1),
      amp_xyz_.Z() * std::sin(two_pi * freq_xyz_.Z() * t + 2.3));
    const gz::math::Vector3d drpy(
      amp_rpy_.X() * std::sin(two_pi * freq_rpy_.X() * t + 0.7),
      amp_rpy_.Y() * std::sin(two_pi * freq_rpy_.Y() * t + 1.9),
      amp_rpy_.Z() * std::sin(two_pi * freq_rpy_.Z() * t + 2.9));
    const gz::math::Quaterniond q =
      center_.Rot() * gz::math::Quaterniond(drpy.X() * s, drpy.Y() * s, drpy.Z() * s);
    return gz::math::Pose3d(center_.Pos() + dp * s, q);
  }

  gz::math::Vector3d DesiredCom(double t) const
  {
    const gz::math::Pose3d p = Desired(t);
    return p.Pos() + p.Rot().RotateVector(com_offset_);
  }

  // 목표 각속도 (월드 좌표계): 아주 짧은 시간의 자세 변화 / 시간
  gz::math::Vector3d DesiredOmega(double t) const
  {
    const double h = 1e-4;
    const gz::math::Quaterniond q_m = Desired(t - h).Rot();
    const gz::math::Quaterniond q_p = Desired(t + h).Rot();
    return RotationVector(q_p * q_m.Inverse()) / (2.0 * h);
  }

  // 회전을 "축 x 각도" 벡터로 (짧은 쪽 회전)
  static gz::math::Vector3d RotationVector(gz::math::Quaterniond q)
  {
    q.Normalize();
    if (q.W() < 0.0) {
      q = gz::math::Quaterniond(-q.W(), -q.X(), -q.Y(), -q.Z());
    }
    gz::math::Vector3d axis;
    double angle = 0.0;
    q.AxisAngle(axis, angle);
    return axis * angle;
  }

  gz::sim::Link link_;
  gz::math::Pose3d center_{0.0, 0.0, 0.5, 0.0, 0.0, 0.0};
  gz::math::Vector3d amp_xyz_{0.10, 0.10, 0.06};
  gz::math::Vector3d amp_rpy_{20.0, 15.0, 20.0};
  gz::math::Vector3d freq_xyz_{0.37, 0.53, 0.71};
  gz::math::Vector3d freq_rpy_{0.61, 0.43, 0.29};
  gz::math::Vector3d gravity_{0.0, 0.0, -9.8};
  gz::math::Vector3d com_offset_{0.0, 0.0, 0.0};
  double start_delay_{5.0};
  double ramp_{3.0};
  double t0_{0.0};
  bool initialized_{false};
  const double kp_{100.0};  // 자연 진동수 10 rad/s
  const double kd_{20.0};   // 임계 감쇠
};

}  // namespace tb3p

GZ_ADD_PLUGIN(
  tb3p::Excitation,
  gz::sim::System,
  tb3p::Excitation::ISystemConfigure,
  tb3p::Excitation::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(tb3p::Excitation, "tb3p::Excitation")
