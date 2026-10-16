#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>

#include <opencv2/core.hpp>

#include "visual_slam/geometry.hpp"

int main()
{
  const cv::Mat transform = visual_slam::make_gazebo_T_initial_optical(0.08, 0.035, 0.32);
  const cv::Point3f origin = visual_slam::transform_point(transform, cv::Point3f(0.0F, 0.0F, 0.0F));
  assert(std::abs(origin.x - 0.08F) < 1e-6F);
  assert(std::abs(origin.y - 0.035F) < 1e-6F);
  assert(std::abs(origin.z - 0.32F) < 1e-6F);

  const cv::Point3f forward = visual_slam::transform_point(transform, cv::Point3f(0.0F, 0.0F, 2.0F));
  assert(std::abs(forward.x - 2.08F) < 1e-6F);
  assert(std::abs(forward.y - 0.035F) < 1e-6F);
  assert(std::abs(forward.z - 0.32F) < 1e-6F);

  const cv::Point3f right = visual_slam::transform_point(transform, cv::Point3f(1.0F, 0.0F, 0.0F));
  const cv::Point3f down = visual_slam::transform_point(transform, cv::Point3f(0.0F, 1.0F, 0.0F));
  assert(std::abs(right.y + 0.965F) < 1e-6F);
  assert(std::abs(down.z + 0.68F) < 1e-6F);

  const cv::Mat rotated = visual_slam::make_gazebo_T_initial_optical(
    -0.08, -0.035, 0.32, std::acos(-1.0));
  const cv::Point3f rotated_forward = visual_slam::transform_point(
    rotated, cv::Point3f(0.0F, 0.0F, 2.0F));
  assert(std::abs(rotated_forward.x + 2.08F) < 1e-6F);
  assert(std::abs(rotated_forward.y + 0.035F) < 1e-6F);

  cv::Mat drifting_pose = cv::Mat::eye(4, 4, CV_64F);
  drifting_pose.at<double>(0, 0) = 0.96;
  drifting_pose.at<double>(0, 1) = 0.08;
  drifting_pose.at<double>(0, 2) = 0.25;
  drifting_pose.at<double>(1, 0) = -0.10;
  drifting_pose.at<double>(1, 1) = 0.98;
  drifting_pose.at<double>(1, 2) = 0.15;
  drifting_pose.at<double>(2, 0) = -0.24;
  drifting_pose.at<double>(2, 1) = -0.17;
  drifting_pose.at<double>(2, 2) = 0.94;
  drifting_pose.at<double>(0, 3) = 0.4;
  drifting_pose.at<double>(1, 3) = 1.2;
  drifting_pose.at<double>(2, 3) = 3.5;
  const cv::Mat planar = visual_slam::constrain_optical_pose_to_ground_plane(drifting_pose);
  assert(std::abs(planar.at<double>(1, 3)) < 1e-12);
  assert(std::abs(planar.at<double>(0, 1)) < 1e-12);
  assert(std::abs(planar.at<double>(1, 0)) < 1e-12);
  assert(std::abs(planar.at<double>(1, 1) - 1.0) < 1e-12);
  assert(std::abs(planar.at<double>(1, 2)) < 1e-12);
  assert(std::abs(planar.at<double>(2, 1)) < 1e-12);
  assert(std::abs(planar.at<double>(0, 3) - 0.4) < 1e-12);
  assert(std::abs(planar.at<double>(2, 3) - 3.5) < 1e-12);

  cv::Mat drifting_motion = cv::Mat::eye(4, 4, CV_64F);
  drifting_motion.at<double>(0, 0) = 0.99;
  drifting_motion.at<double>(0, 1) = 0.04;
  drifting_motion.at<double>(0, 2) = 0.12;
  drifting_motion.at<double>(1, 0) = -0.03;
  drifting_motion.at<double>(1, 1) = 0.99;
  drifting_motion.at<double>(1, 2) = 0.05;
  drifting_motion.at<double>(2, 0) = -0.11;
  drifting_motion.at<double>(2, 1) = -0.06;
  drifting_motion.at<double>(2, 2) = 0.98;
  drifting_motion.at<double>(0, 3) = 0.02;
  drifting_motion.at<double>(1, 3) = -0.01;
  drifting_motion.at<double>(2, 3) = -0.03;
  const cv::Mat straight = visual_slam::constrain_optical_relative_motion_to_straight_ground(
    drifting_motion);
  assert(std::abs(straight.at<double>(0, 3)) < 1e-12);
  assert(std::abs(straight.at<double>(1, 3)) < 1e-12);
  assert(std::abs(straight.at<double>(2, 3) + 0.03) < 1e-12);
  assert(std::abs(straight.at<double>(0, 1)) < 1e-12);
  assert(std::abs(straight.at<double>(1, 0)) < 1e-12);
  assert(std::abs(straight.at<double>(1, 2)) < 1e-12);
  assert(std::abs(straight.at<double>(2, 1)) < 1e-12);
  assert(visual_slam::has_significant_planar_yaw(drifting_motion, 0.05));
  cv::Mat near_straight_motion = cv::Mat::eye(4, 4, CV_64F);
  const double near_straight_yaw = 0.1 * std::acos(-1.0) / 180.0;
  near_straight_motion.at<double>(0, 0) = std::cos(near_straight_yaw);
  near_straight_motion.at<double>(0, 2) = std::sin(near_straight_yaw);
  near_straight_motion.at<double>(2, 0) = -std::sin(near_straight_yaw);
  near_straight_motion.at<double>(2, 2) = std::cos(near_straight_yaw);
  assert(!visual_slam::has_significant_planar_yaw(near_straight_motion, 0.35 *
    std::acos(-1.0) / 180.0));

  const double expected_yaw = 0.37;
  const double yaw_cosine = std::cos(expected_yaw);
  const double yaw_sine = std::sin(expected_yaw);
  const cv::Mat expected_rotation = (cv::Mat_<double>(3, 3) <<
    yaw_cosine, 0.0, yaw_sine,
    0.0, 1.0, 0.0,
    -yaw_sine, 0.0, yaw_cosine);
  const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
    420.0, 0.0, 320.0,
    0.0, 420.0, 240.0,
    0.0, 0.0, 1.0);
  const cv::Mat homography = camera_matrix * expected_rotation * camera_matrix.inv() * 2.7;
  const cv::Mat recovered_rotation =
    visual_slam::calibrated_homography_to_planar_rotation(homography, camera_matrix);
  assert(!recovered_rotation.empty());
  const double recovered_yaw = std::atan2(
    recovered_rotation.at<double>(0, 2), recovered_rotation.at<double>(2, 2));
  assert(std::abs(recovered_yaw - expected_yaw) < 1e-9);

  std::vector<cv::Point2f> previous_pixels;
  std::vector<cv::Point2f> current_pixels;
  for (int row = 0; row < 5; ++row) {
    for (int column = 0; column < 7; ++column) {
      const cv::Vec3d previous_ray(
        (160.0 + 45.0 * column - 320.0) / 420.0,
        (120.0 + 45.0 * row - 240.0) / 420.0, 1.0);
      const cv::Mat current_ray = expected_rotation.t() * cv::Mat(previous_ray);
      previous_pixels.emplace_back(160.0F + 45.0F * column, 120.0F + 45.0F * row);
      current_pixels.emplace_back(
        static_cast<float>(420.0 * current_ray.at<double>(0) / current_ray.at<double>(2) + 320.0),
        static_cast<float>(420.0 * current_ray.at<double>(1) / current_ray.at<double>(2) + 240.0));
    }
  }
  int bearing_inliers = 0;
  double bearing_rmse = 0.0;
  const cv::Mat bearing_rotation = visual_slam::bearing_correspondences_to_planar_rotation(
    previous_pixels, current_pixels, camera_matrix, 1.0, 15,
    bearing_inliers, bearing_rmse);
  assert(!bearing_rotation.empty());
  const double bearing_yaw = std::atan2(
    bearing_rotation.at<double>(0, 2), bearing_rotation.at<double>(2, 2));
  assert(std::abs(bearing_yaw - expected_yaw) < 1e-6);
  assert(bearing_inliers == 35);
  assert(bearing_rmse < 1e-4);

  cv::Mat accumulated_pose = cv::Mat::eye(4, 4, CV_64F);
  double expected_accumulated_yaw = 0.0;
  const std::vector<double> turn_increments = {
    175.0 * CV_PI / 180.0,
    -86.0 * CV_PI / 180.0,
    -24.0 * CV_PI / 180.0,
    91.0 * CV_PI / 180.0};
  for (const double increment : turn_increments) {
    const double cosine = std::cos(increment);
    const double sine = std::sin(increment);
    cv::Mat current_T_previous = cv::Mat::eye(4, 4, CV_64F);
    current_T_previous.at<double>(0, 0) = cosine;
    current_T_previous.at<double>(0, 2) = -sine;
    current_T_previous.at<double>(2, 0) = sine;
    current_T_previous.at<double>(2, 2) = cosine;
    accumulated_pose = visual_slam::compose_world_pose_from_current_T_previous(
      accumulated_pose, current_T_previous);
    expected_accumulated_yaw += increment;
    const double accumulated_yaw = std::atan2(
      accumulated_pose.at<double>(0, 2), accumulated_pose.at<double>(2, 2));
    const double yaw_error = std::atan2(
      std::sin(accumulated_yaw - expected_accumulated_yaw),
      std::cos(accumulated_yaw - expected_accumulated_yaw));
    assert(std::abs(yaw_error) < 1e-9);
  }
  const double final_turn_yaw = std::atan2(
    accumulated_pose.at<double>(0, 2), accumulated_pose.at<double>(2, 2));
  const double final_expected_yaw = 156.0 * CV_PI / 180.0;
  const double final_yaw_error = std::atan2(
    std::sin(final_turn_yaw - final_expected_yaw),
    std::cos(final_turn_yaw - final_expected_yaw));
  assert(visual_slam::has_significant_planar_yaw(accumulated_pose, 0.5));
  assert(std::abs(final_yaw_error) < 1e-9);

  return 0;
}
