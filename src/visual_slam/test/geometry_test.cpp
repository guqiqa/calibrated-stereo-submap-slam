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
  return 0;
}
