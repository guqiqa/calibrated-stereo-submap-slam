// Four-layer convention test for the metric-path pose chain.
//
// Layer 1: solvePnP round trip in pure OpenCV coordinates.
// Layer 2: compose_world_pose_from_current_T_previous direction semantics.
// Layer 3: optical pose -> Gazebo pose (mount transform and ground yaw).
// Layer 4: point-cloud axis reflection S = diag(1,-1,1) conjugation.
//
// Each layer asserts what it can prove and prints the numbers it cannot, so a
// failing layer names the place where the convention breaks instead of leaving
// a sign to be guessed from a simulation run.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "visual_slam/geometry.hpp"

namespace
{

int g_failures = 0;
int g_checks = 0;

void expect_near(const std::string & name, double actual, double expected, double tolerance)
{
  ++g_checks;
  const double error = std::fabs(actual - expected);
  if (error > tolerance) {
    ++g_failures;
    std::printf("  FAIL %-58s actual=%+.6f expected=%+.6f tol=%.1e\n",
      name.c_str(), actual, expected, tolerance);
  }
}

void expect_true(const std::string & name, bool condition)
{
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::printf("  FAIL %s\n", name.c_str());
  }
}

void report(const std::string & name, double value)
{
  std::printf("  info %-58s %+.6f\n", name.c_str(), value);
}

double yaw_deg(const cv::Mat & transform)
{
  return std::atan2(transform.at<double>(0, 2), transform.at<double>(2, 2)) * 180.0 / CV_PI;
}

cv::Mat make_pose(double yaw_degrees, double tx, double ty, double tz)
{
  const double radians = yaw_degrees * CV_PI / 180.0;
  const double cosine = std::cos(radians);
  const double sine = std::sin(radians);
  cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
  pose.at<double>(0, 0) = cosine;
  pose.at<double>(0, 2) = sine;
  pose.at<double>(2, 0) = -sine;
  pose.at<double>(2, 2) = cosine;
  pose.at<double>(0, 3) = tx;
  pose.at<double>(1, 3) = ty;
  pose.at<double>(2, 3) = tz;
  return pose;
}

// Layer 1 ---------------------------------------------------------------
void layer_pnp_round_trip()
{
  std::printf("\n[L1] solvePnP round trip in pure OpenCV coordinates\n");
  const int width = 640;
  const int height = 480;
  const double fx = 400.0;
  const double fy = 400.0;
  const double cx = width / 2.0;
  const double cy = height / 2.0;

  const double delta_deg = 5.0;
  const cv::Mat current_T_reference = make_pose(delta_deg, 0.10, -0.02, 0.30);

  std::vector<cv::Point3f> object_points;
  for (int row = -3; row <= 3; ++row) {
    for (int column = -3; column <= 3; ++column) {
      object_points.emplace_back(static_cast<float>(column * 0.35),
        static_cast<float>(row * 0.25), static_cast<float>(2.0 + (row + 3) * 0.45));
    }
  }

  std::vector<cv::Point2f> image_points;
  for (const auto & point : object_points) {
    const cv::Mat vector = (cv::Mat_<double>(4, 1) << point.x, point.y, point.z, 1.0);
    const cv::Mat camera = current_T_reference * vector;
    image_points.emplace_back(static_cast<float>(fx * camera.at<double>(0) / camera.at<double>(2) + cx),
      static_cast<float>(fy * camera.at<double>(1) / camera.at<double>(2) + cy));
  }

  const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
  cv::Mat rvec;
  cv::Mat tvec;
  const bool ok = cv::solvePnPRansac(object_points, image_points, camera_matrix, cv::Mat(),
      rvec, tvec, false, 200, 3.0, 0.99, cv::noArray(), cv::SOLVEPNP_ITERATIVE);
  expect_true("solvePnPRansac converged", ok);
  if (!ok) {
    return;
  }
  cv::Mat rotation;
  cv::Rodrigues(rvec, rotation);
  cv::Mat recovered = cv::Mat::eye(4, 4, CV_64F);
  rotation.copyTo(recovered(cv::Rect(0, 0, 3, 3)));
  tvec.copyTo(recovered(cv::Rect(3, 0, 1, 3)));

  expect_near("L1 recovered yaw == declared current_T_reference yaw",
    yaw_deg(recovered), delta_deg, 0.05);
  expect_near("L1 recovered tx", recovered.at<double>(0, 3), 0.10, 5e-3);
  expect_near("L1 recovered tz", recovered.at<double>(2, 3), 0.30, 5e-3);
  report("L1 yaw(current_T_reference)", yaw_deg(current_T_reference));
}

// Layer 2 ---------------------------------------------------------------
void layer_compose_direction()
{
  std::printf("\n[L2] compose direction semantics\n");
  const cv::Mat world_T_reference = make_pose(10.0, 1.0, 0.0, 2.0);
  const cv::Mat current_T_reference = make_pose(5.0, 0.0, 0.0, -0.5);
  const cv::Mat reference_T_current = current_T_reference.inv();

  const cv::Mat from_current = visual_slam::compose_world_pose_from_current_T_previous(
    world_T_reference, current_T_reference);
  const cv::Mat from_reference = visual_slam::compose_world_pose_from_current_T_previous(
    world_T_reference, reference_T_current);

  // compose(A, B) == A * B^-1, so the same physical motion expressed in the two
  // opposite directions must give yaw of opposite sign.
  expect_near("L2 compose(A, current_T_reference) yaw == 10 - 5",
    yaw_deg(from_current), 5.0, 1e-6);
  expect_near("L2 compose(A, reference_T_current) yaw == 10 + 5",
    yaw_deg(from_reference), 15.0, 1e-6);
  const cv::Mat by_product = world_T_reference * current_T_reference.inv();
  expect_near("L2 compose(A,B) == A * B^-1 (rotation)",
    cv::norm(from_current(cv::Rect(0, 0, 3, 3)) - by_product(cv::Rect(0, 0, 3, 3)), cv::NORM_INF),
    0.0, 1e-9);
  expect_near("L2 compose(A,B) == A * B^-1 (translation)",
    cv::norm(from_current(cv::Rect(3, 0, 1, 3)) - by_product(cv::Rect(3, 0, 1, 3)), cv::NORM_INF),
    0.0, 1e-9);
  report("L2 world tx via current_T_reference", from_current.at<double>(0, 3));
  report("L2 world tz via current_T_reference", from_current.at<double>(2, 3));
  report("L2 world tx via reference_T_current", from_reference.at<double>(0, 3));
  report("L2 world tz via reference_T_current", from_reference.at<double>(2, 3));

  // The two inputs are not interchangeable: selecting the wrong direction
  // changes the sign of the yaw increment.
  const double delta_from_current = yaw_deg(from_current) - yaw_deg(world_T_reference);
  const double delta_from_reference = yaw_deg(from_reference) - yaw_deg(world_T_reference);
  expect_true("L2 the two directions differ in sign",
    delta_from_current * delta_from_reference < 0.0);
  report("L2 delta yaw via current_T_reference", delta_from_current);
  report("L2 delta yaw via reference_T_current", delta_from_reference);
}

// Layer 3 ---------------------------------------------------------------
void layer_optical_to_gazebo()
{
  std::printf("\n[L3] optical pose -> Gazebo pose\n");
  const cv::Mat gazebo_T_map = visual_slam::make_gazebo_T_initial_optical(0.08, 0.035, 0.32, 0.0);

  // Pure forward motion in the optical/map frame: +z.
  const cv::Mat forward_pose = make_pose(0.0, 0.0, 0.0, 1.0);
  const cv::Mat gazebo_forward = gazebo_T_map * forward_pose;
  expect_near("L3 optical +z forward maps to Gazebo +x",
    gazebo_forward.at<double>(0, 3), 0.08 + 1.0, 1e-6);

  // Heading: how does an optical yaw appear as a Gazebo yaw?
  const double delta_deg = 5.0;
  const cv::Mat yawed_pose = make_pose(delta_deg, 0.0, 0.0, 0.0);
  const cv::Mat gazebo_yawed = gazebo_T_map * yawed_pose;

  const cv::Mat optical_forward = yawed_pose(cv::Rect(0, 0, 3, 3)) * (cv::Mat_<double>(3, 1) <<
      0.0, 0.0, 1.0);
  const cv::Mat gazebo_axis = gazebo_T_map(cv::Rect(0, 0, 3, 3)) * optical_forward;
  const double gazebo_yaw = std::atan2(
    gazebo_axis.at<double>(1), gazebo_axis.at<double>(0)) * 180.0 / CV_PI;

  report("L3 optical yaw +5.00 deg", yaw_deg(yawed_pose));
  report("L3 same rotation as Gazebo yaw", gazebo_yaw);
  expect_near("L3 Gazebo yaw of an optical +5 deg yaw", gazebo_yaw, -delta_deg, 1e-6);
  expect_true("L3 optical yaw and Gazebo yaw carry opposite signs",
    yaw_deg(yawed_pose) * gazebo_yaw < 0.0);
  (void)gazebo_yawed;
}

// Layer 4 ---------------------------------------------------------------
void layer_axis_reflection()
{
  std::printf("\n[L4] point-cloud axis reflection S = diag(1,-1,1)\n");
  const double delta_deg = 5.0;
  const cv::Mat rotation = make_pose(delta_deg, 0.0, 0.0, 0.0)(cv::Rect(0, 0, 3, 3));

  cv::Mat reflection = cv::Mat::eye(3, 3, CV_64F);
  reflection.at<double>(1, 1) = -1.0;
  const cv::Mat conjugated = reflection * rotation * reflection;

  const cv::Mat identity = cv::Mat::eye(3, 3, CV_64F);
  const double orthogonality = cv::norm(conjugated * conjugated.t() - identity, cv::NORM_INF);
  const double determinant = cv::determinant(conjugated);
  cv::Mat conjugated_pose = cv::Mat::eye(4, 4, CV_64F);
  conjugated.copyTo(conjugated_pose(cv::Rect(0, 0, 3, 3)));

  expect_near("L4 conjugated rotation stays orthonormal", orthogonality, 0.0, 1e-9);
  expect_near("L4 conjugated rotation has det +1 (still SO(3))", determinant, 1.0, 1e-9);
  cv::Mat original_pose = cv::Mat::eye(4, 4, CV_64F);
  rotation.copyTo(original_pose(cv::Rect(0, 0, 3, 3)));
  report("L4 yaw(R)", yaw_deg(original_pose));
  report("L4 yaw(S R S)", yaw_deg(conjugated_pose));

  // The reflection flips the yaw only because it also flips the y axis of the
  // rotation. A y-only change of the point cloud without the matching reflection
  // of the rotation would not be a similarity and must not be assumed to flip.
  expect_near("L4 S R S keeps the yaw magnitude", std::fabs(yaw_deg(conjugated_pose)),
    delta_deg, 1e-6);
}

// Layer 5 ---------------------------------------------------------------
// Static injection through the real publish chain: pose -> quaternion (the same
// helper the odometry publisher uses) -> yaw extraction, plus the mount
// conversion used by the export. No ROS, no camera, nothing can overwrite it.
void layer_static_injection()
{
  std::printf("\n[L5] static injection through the publish chain\n");
  const cv::Mat gazebo_T_map = visual_slam::make_gazebo_T_initial_optical(0.08, 0.035, 0.32, 0.0);

  struct Case
  {
    const char * name;
    double yaw_deg;
    double tx;
    double ty;
    double tz;
  };
  const Case cases[] = {
    {"identity", 0.0, 0.0, 0.0, 0.0},
    {"+30 yaw", 30.0, 0.0, 0.0, 0.0},
    {"-30 yaw", -30.0, 0.0, 0.0, 0.0},
    {"+90 yaw", 90.0, 0.0, 0.0, 0.0},
    {"pure translation", 0.0, 0.5, 0.0, 1.0},
  };

  std::printf("  %-16s %9s %9s %9s %9s\n",
    "case", "optical", "quat2y", "quatZ", "gazebo");
  for (const auto & item : cases) {
    const cv::Mat pose = make_pose(item.yaw_deg, item.tx, item.ty, item.tz);
    const cv::Mat rotation = pose(cv::Rect(0, 0, 3, 3));
    const cv::Vec4d quaternion = visual_slam::rotation_to_quaternion_xyzw(rotation);
    const double optical_yaw = yaw_deg(pose);
    const double quat_y_yaw = 2.0 * std::atan2(quaternion[1], quaternion[3]) * 180.0 / CV_PI;
    const double quat_z_yaw = std::atan2(
      2.0 * (quaternion[3] * quaternion[2] + quaternion[0] * quaternion[1]),
      1.0 - 2.0 * (quaternion[1] * quaternion[1] + quaternion[2] * quaternion[2])) * 180.0 / CV_PI;
    const cv::Mat forward = rotation * (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0);
    const cv::Mat gazebo_forward = gazebo_T_map(cv::Rect(0, 0, 3, 3)) * forward;
    const double gazebo_yaw = std::atan2(
      gazebo_forward.at<double>(1), gazebo_forward.at<double>(0)) * 180.0 / CV_PI;
    std::printf("  %-16s %+9.3f %+9.3f %+9.3f %+9.3f\n",
      item.name, optical_yaw, quat_y_yaw, quat_z_yaw, gazebo_yaw);

    if (std::fabs(item.tx) + std::fabs(item.ty) + std::fabs(item.tz) < 1e-9 || item.yaw_deg != 0.0) {
      expect_near(std::string("L5 2*atan2(qy,qw) reproduces the optical yaw for ") + item.name,
        quat_y_yaw, optical_yaw, 1e-6);
      expect_near(std::string("L5 quaternion carries no z-yaw for ") + item.name,
        quat_z_yaw, 0.0, 1e-6);
      expect_near(std::string("L5 mount maps optical yaw to the opposite Gazebo yaw for ") +
        item.name, gazebo_yaw, -optical_yaw, 1e-6);
    }
  }

  // The export file itself has no orientation column, so an "export yaw" does
  // not exist and must not be compared with anything.
  std::printf("  note: trajectory_visual.csv exports positions only (no yaw column)\n");
}

// Layer 6 ---------------------------------------------------------------
// Rigid-transform contract with explicit 4x4 matrices.
//
// The reference camera sits at world_T_reference. The current camera is at
// reference_T_current (its pose expressed in the reference frame), so by
// definition world_T_current = world_T_reference * reference_T_current.
// solvePnP over reference points and current pixels returns current_T_reference,
// and compose(world_T_reference, current_T_reference) must reproduce the defined
// world pose exactly, rotation and translation together.
void layer_rigid_contract()
{
  std::printf("\n[L6] rigid transform contract (explicit 4x4)\n");
  const double fx = 420.0;
  const double fy = 420.0;
  const double cx = 320.0;
  const double cy = 240.0;
  const cv::Mat camera_matrix =
    (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
  const cv::Mat world_T_reference = make_pose(12.0, 1.0, 0.0, 2.0);

  struct Case
  {
    const char * name;
    double yaw_deg;
    double forward_m;
  };
  const Case cases[] = {
    {"pure forward", 0.0, 0.5},
    {"pure rotation", 5.0, 0.0},
    {"forward + yaw", 5.0, 0.5},
    {"forward - yaw", -5.0, 0.5},
  };

  std::vector<cv::Point3f> object_points;
  for (int row = -3; row <= 3; ++row) {
    for (int column = -3; column <= 3; ++column) {
      object_points.emplace_back(static_cast<float>(column * 0.35),
        static_cast<float>(row * 0.25), static_cast<float>(2.0 + (row + 3) * 0.45));
    }
  }

  for (const auto & item : cases) {
    // Explicit reference_T_current: the current camera as seen from the reference.
    const cv::Mat reference_T_current = make_pose(item.yaw_deg, 0.0, 0.0, item.forward_m);
    const cv::Mat current_T_reference = reference_T_current.inv();
    const cv::Mat expected_world = world_T_reference * reference_T_current;

    std::vector<cv::Point2f> image_points;
    for (const auto & point : object_points) {
      const cv::Mat vector = (cv::Mat_<double>(4, 1) << point.x, point.y, point.z, 1.0);
      const cv::Mat camera = current_T_reference * vector;
      image_points.emplace_back(
        static_cast<float>(fx * camera.at<double>(0) / camera.at<double>(2) + cx),
        static_cast<float>(fy * camera.at<double>(1) / camera.at<double>(2) + cy));
    }

    cv::Mat rvec;
    cv::Mat tvec;
    const bool ok = cv::solvePnPRansac(object_points, image_points, camera_matrix, cv::Mat(),
        rvec, tvec, false, 200, 3.0, 0.99, cv::noArray(), cv::SOLVEPNP_ITERATIVE);
    expect_true(std::string("L6 solvePnP converged for ") + item.name, ok);
    if (!ok) {
      continue;
    }
    cv::Mat rotation;
    cv::Rodrigues(rvec, rotation);
    cv::Mat recovered = cv::Mat::eye(4, 4, CV_64F);
    rotation.copyTo(recovered(cv::Rect(0, 0, 3, 3)));
    tvec.copyTo(recovered(cv::Rect(3, 0, 1, 3)));

    const cv::Mat composed = visual_slam::compose_world_pose_from_current_T_previous(
      world_T_reference, recovered);
    const cv::Mat composed_rotation = composed(cv::Rect(0, 0, 3, 3));

    expect_near(std::string("L6 rotation of ") + item.name,
      cv::norm(composed_rotation - expected_world(cv::Rect(0, 0, 3, 3)), cv::NORM_INF), 0.0, 1e-6);
    expect_near(std::string("L6 translation of ") + item.name,
      cv::norm(composed(cv::Rect(3, 0, 1, 3)) - expected_world(cv::Rect(3, 0, 1, 3)),
      cv::NORM_INF), 0.0, 1e-6);
    expect_near(std::string("L6 determinant of ") + item.name,
      cv::determinant(composed_rotation), 1.0, 1e-9);
    expect_near(std::string("L6 orthonormality of ") + item.name,
      cv::norm(composed_rotation * composed_rotation.t() - cv::Mat::eye(3, 3, CV_64F),
      cv::NORM_INF), 0.0, 1e-9);

    std::printf(
      "  %-14s yaw in %+7.3f  recovered %+7.3f  composed %+7.3f  expected %+7.3f  "
      "forward t %+6.3f\n",
      item.name, item.yaw_deg, yaw_deg(recovered), yaw_deg(composed), yaw_deg(expected_world),
      composed.at<double>(2, 3) - world_T_reference.at<double>(2, 3));
  }

  // The rotation-only path hands compose a reference_T_current shaped matrix
  // (apply_temporal_rotation inverts it internally). That must raise the world
  // heading by exactly its yaw, with no translation change.
  const cv::Mat temporal_rotation = make_pose(5.0, 0.0, 0.0, 0.0);
  const cv::Mat applied = visual_slam::compose_world_pose_from_current_T_previous(
    world_T_reference, temporal_rotation.inv());
  expect_near("L6 rotation-only input raises the world yaw by +5",
    yaw_deg(applied) - yaw_deg(world_T_reference), 5.0, 1e-6);
  expect_near("L6 rotation-only input leaves the translation unchanged",
    cv::norm(applied(cv::Rect(3, 0, 1, 3)) - world_T_reference(cv::Rect(3, 0, 1, 3)),
    cv::NORM_INF), 0.0, 1e-9);
}

}  // namespace

// Layer 7 ---------------------------------------------------------------
// The map frame that a database was built in must be able to differ from the pose a
// later run starts at. world_T_current_ is the start pose expressed in that frame.
void layer_world_frame_anchor()
{
  std::printf("\n[L7] map frame anchoring for query runs\n");
  const double camera_offset_x = 0.08;
  const double camera_offset_y = 0.035;

  // Baseline: the map frame equals the start pose, so the anchor is identity.
  const cv::Mat baseline_map = visual_slam::make_gazebo_T_initial_optical(
    camera_offset_x, camera_offset_y, 0.32, 0.0);
  const cv::Mat baseline_start = visual_slam::make_gazebo_T_initial_optical(
    camera_offset_x, camera_offset_y, 0.32, 0.0);
  const cv::Mat baseline_anchor = baseline_map.inv() * baseline_start;
  expect_near("baseline anchor translation x", baseline_anchor.at<double>(0, 3), 0.0, 1e-9);
  expect_near("baseline anchor translation y", baseline_anchor.at<double>(1, 3), 0.0, 1e-9);
  expect_near("baseline anchor translation z", baseline_anchor.at<double>(2, 3), 0.0, 1e-9);
  expect_near("baseline anchor heading", yaw_deg(baseline_anchor), 0.0, 1e-9);

  // Query case: database built at the origin, run starts at (-0.5, -17.35) facing pi.
  const double base_x = -0.5;
  const double base_y = -17.35;
  const double base_yaw = CV_PI;
  const double camera_x = base_x + std::cos(base_yaw) * camera_offset_x -
    std::sin(base_yaw) * camera_offset_y;
  const double camera_y = base_y + std::sin(base_yaw) * camera_offset_x +
    std::cos(base_yaw) * camera_offset_y;
  const cv::Mat query_start = visual_slam::make_gazebo_T_initial_optical(
    camera_x, camera_y, 0.32, base_yaw);
  const cv::Mat anchor = baseline_map.inv() * query_start;

  // The anchor must reproduce the start pose when converted back to Gazebo.
  const cv::Mat round_trip = baseline_map * anchor;
  expect_near("round trip camera x", round_trip.at<double>(0, 3), camera_x, 1e-9);
  expect_near("round trip camera y", round_trip.at<double>(1, 3), camera_y, 1e-9);
  expect_near("round trip yaw", yaw_deg(round_trip), yaw_deg(query_start), 1e-9);

  // The anchor must carry a real offset, not an identity.
  expect_true("anchor differs from identity",
              std::abs(anchor.at<double>(0, 3)) + std::abs(anchor.at<double>(1, 3)) > 1.0);
  // Heading must stay finite and match the start yaw in [-180, 180].
  double anchor_heading = yaw_deg(anchor);
  while (anchor_heading > 180.0) {
    anchor_heading -= 360.0;
  }
  while (anchor_heading < -180.0) {
    anchor_heading += 360.0;
  }
  expect_near("anchor heading magnitude", std::abs(anchor_heading), 180.0, 1e-6);
  report("anchor heading deg", anchor_heading);
  report("anchor translation x", anchor.at<double>(0, 3));
  report("anchor translation y", anchor.at<double>(1, 3));
}

int main()
{
  std::printf("pose convention test (seven layers)\n");
  layer_pnp_round_trip();
  layer_compose_direction();
  layer_optical_to_gazebo();
  layer_axis_reflection();
  layer_static_injection();
  layer_rigid_contract();
  layer_world_frame_anchor();
  std::printf("\nchecks=%d failures=%d\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
