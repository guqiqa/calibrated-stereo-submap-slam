#pragma once

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

}  // namespace visual_slam
