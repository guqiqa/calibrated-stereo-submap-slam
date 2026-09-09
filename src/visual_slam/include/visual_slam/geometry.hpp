#pragma once

#include <cmath>

#include <opencv2/core.hpp>

namespace visual_slam
{

inline cv::Mat make_gazebo_T_initial_optical(double x, double y, double z)
{
  cv::Mat transform = cv::Mat::eye(4, 4, CV_64F);
  // Gazebo uses x-forward/y-left/z-up; ROS optical uses z-forward/x-right/y-down.
  transform.at<double>(0, 0) = 0.0;
  transform.at<double>(0, 1) = 0.0;
  transform.at<double>(0, 2) = 1.0;
  transform.at<double>(1, 0) = -1.0;
  transform.at<double>(1, 1) = 0.0;
  transform.at<double>(1, 2) = 0.0;
  transform.at<double>(2, 0) = 0.0;
  transform.at<double>(2, 1) = -1.0;
  transform.at<double>(2, 2) = 0.0;
  transform.at<double>(0, 3) = x;
  transform.at<double>(1, 3) = y;
  transform.at<double>(2, 3) = z;
  return transform;
}

inline cv::Point3f transform_point(const cv::Mat & transform, const cv::Point3f & point)
{
  return cv::Point3f(
    static_cast<float>(transform.at<double>(0, 0) * point.x +
      transform.at<double>(0, 1) * point.y + transform.at<double>(0, 2) * point.z +
      transform.at<double>(0, 3)),
    static_cast<float>(transform.at<double>(1, 0) * point.x +
      transform.at<double>(1, 1) * point.y + transform.at<double>(1, 2) * point.z +
      transform.at<double>(1, 3)),
    static_cast<float>(transform.at<double>(2, 0) * point.x +
      transform.at<double>(2, 1) * point.y + transform.at<double>(2, 2) * point.z +
      transform.at<double>(2, 3)));
}

inline cv::Mat constrain_optical_pose_to_ground_plane(const cv::Mat & pose)
{
  cv::Mat constrained = pose.clone();
  // Optical y is vertical (down). Ground motion retains translation in x/z and yaw about y.
  const double yaw = std::atan2(pose.at<double>(0, 2), pose.at<double>(2, 2));
  const double cosine = std::cos(yaw);
  const double sine = std::sin(yaw);
  constrained.at<double>(0, 0) = cosine;
  constrained.at<double>(0, 1) = 0.0;
  constrained.at<double>(0, 2) = sine;
  constrained.at<double>(1, 0) = 0.0;
  constrained.at<double>(1, 1) = 1.0;
  constrained.at<double>(1, 2) = 0.0;
  constrained.at<double>(2, 0) = -sine;
  constrained.at<double>(2, 1) = 0.0;
  constrained.at<double>(2, 2) = cosine;
  constrained.at<double>(1, 3) = 0.0;
  return constrained;
}

inline cv::Mat constrain_optical_relative_motion_to_straight_ground(const cv::Mat & motion)
{
  cv::Mat constrained = motion.clone();
  // For the straight corridor profile, optical z is the travel axis and optical x/y
  // are constrained to zero. The robot keeps its heading, so relative yaw is zero.
  constrained.at<double>(0, 0) = 1.0;
  constrained.at<double>(0, 1) = 0.0;
  constrained.at<double>(0, 2) = 0.0;
  constrained.at<double>(1, 0) = 0.0;
  constrained.at<double>(1, 1) = 1.0;
  constrained.at<double>(1, 2) = 0.0;
  constrained.at<double>(2, 0) = 0.0;
  constrained.at<double>(2, 1) = 0.0;
  constrained.at<double>(2, 2) = 1.0;
  constrained.at<double>(0, 3) = 0.0;
  constrained.at<double>(1, 3) = 0.0;
  return constrained;
}

}  // namespace visual_slam
