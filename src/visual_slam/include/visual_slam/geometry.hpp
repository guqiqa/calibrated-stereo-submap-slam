#pragma once

#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include <opencv2/core.hpp>

namespace visual_slam
{

inline cv::Mat make_gazebo_T_initial_optical(double x, double y, double z, double yaw = 0.0)
{
  cv::Mat transform = cv::Mat::eye(4, 4, CV_64F);
  // Gazebo uses x-forward/y-left/z-up; ROS optical uses z-forward/x-right/y-down.
  const double cosine = std::cos(yaw);
  const double sine = std::sin(yaw);
  transform.at<double>(0, 0) = sine;
  transform.at<double>(0, 1) = 0.0;
  transform.at<double>(0, 2) = cosine;
  transform.at<double>(1, 0) = -cosine;
  transform.at<double>(1, 1) = 0.0;
  transform.at<double>(1, 2) = sine;
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

inline cv::Mat compose_world_pose_from_current_T_previous(
  const cv::Mat & world_T_previous, const cv::Mat & current_T_previous)
{
  if (world_T_previous.rows != 4 || world_T_previous.cols != 4 ||
    current_T_previous.rows != 4 || current_T_previous.cols != 4)
  {
    return {};
  }
  return world_T_previous * current_T_previous.inv();
}

inline bool has_significant_planar_yaw(const cv::Mat & motion, double threshold_rad)
{
  if (motion.rows != 4 || motion.cols != 4 || !std::isfinite(threshold_rad) ||
    threshold_rad < 0.0)
  {
    return false;
  }
  const double yaw = std::atan2(motion.at<double>(0, 2), motion.at<double>(2, 2));
  return std::isfinite(yaw) && std::abs(yaw) >= threshold_rad;
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

inline cv::Mat calibrated_homography_to_planar_rotation(
  const cv::Mat & homography, const cv::Mat & camera_matrix)
{
  if (homography.empty() || homography.rows != 3 || homography.cols != 3 ||
    camera_matrix.empty() || camera_matrix.rows != 3 || camera_matrix.cols != 3)
  {
    return {};
  }
  cv::Mat normalized = camera_matrix.inv() * homography * camera_matrix;
  normalized.convertTo(normalized, CV_64F);
  const double determinant = cv::determinant(normalized);
  const double scale = std::cbrt(std::abs(determinant));
  if (!std::isfinite(scale) || scale < 1e-9) {
    return {};
  }
  normalized /= scale;
  cv::SVD svd(normalized, cv::SVD::FULL_UV);
  cv::Mat rotation = svd.u * svd.vt;
  if (cv::determinant(rotation) < 0.0) {
    cv::Mat corrected_u = svd.u.clone();
    corrected_u.col(2) *= -1.0;
    rotation = corrected_u * svd.vt;
  }
  const double yaw = std::atan2(rotation.at<double>(0, 2), rotation.at<double>(2, 2));
  const double cosine = std::cos(yaw);
  const double sine = std::sin(yaw);
  return (cv::Mat_<double>(3, 3) <<
    cosine, 0.0, sine,
    0.0, 1.0, 0.0,
    -sine, 0.0, cosine);
}

inline cv::Mat bearing_correspondences_to_planar_rotation(
  const std::vector<cv::Point2f> & previous_pixels,
  const std::vector<cv::Point2f> & current_pixels,
  const cv::Mat & camera_matrix, double max_residual_px,
  int minimum_inliers, int & inlier_count, double & residual_rmse_px)
{
  inlier_count = 0;
  residual_rmse_px = std::numeric_limits<double>::infinity();
  if (previous_pixels.size() != current_pixels.size() ||
    previous_pixels.size() < static_cast<std::size_t>(minimum_inliers) ||
    camera_matrix.empty() || max_residual_px <= 0.0)
  {
    return {};
  }
  const double fx = camera_matrix.at<double>(0, 0);
  const double fy = camera_matrix.at<double>(1, 1);
  const double cx = camera_matrix.at<double>(0, 2);
  const double cy = camera_matrix.at<double>(1, 2);
  std::vector<cv::Vec3d> previous_rays, current_rays;
  for (std::size_t index = 0; index < previous_pixels.size(); ++index) {
    cv::Vec3d previous_ray(
      (previous_pixels[index].x - cx) / fx,
      (previous_pixels[index].y - cy) / fy, 1.0);
    cv::Vec3d current_ray(
      (current_pixels[index].x - cx) / fx,
      (current_pixels[index].y - cy) / fy, 1.0);
    previous_rays.push_back(previous_ray / cv::norm(previous_ray));
    current_rays.push_back(current_ray / cv::norm(current_ray));
  }
  std::vector<int> active(previous_pixels.size());
  std::iota(active.begin(), active.end(), 0);
  cv::Mat ray_rotation;
  for (int iteration = 0; iteration < 4; ++iteration) {
    cv::Mat covariance = cv::Mat::zeros(3, 3, CV_64F);
    for (const int index : active) {
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          covariance.at<double>(row, column) +=
            current_rays[index][row] * previous_rays[index][column];
        }
      }
    }
    cv::SVD svd(covariance, cv::SVD::FULL_UV);
    ray_rotation = svd.u * svd.vt;
    if (cv::determinant(ray_rotation) < 0.0) {
      cv::Mat corrected_u = svd.u.clone();
      corrected_u.col(2) *= -1.0;
      ray_rotation = corrected_u * svd.vt;
    }
    std::vector<int> next_active;
    for (std::size_t index = 0; index < previous_rays.size(); ++index) {
      const cv::Mat predicted = ray_rotation * cv::Mat(previous_rays[index]);
      if (predicted.at<double>(2) <= 1e-9) {
        continue;
      }
      const cv::Point2d projected(
        fx * predicted.at<double>(0) / predicted.at<double>(2) + cx,
        fy * predicted.at<double>(1) / predicted.at<double>(2) + cy);
      if (cv::norm(projected - cv::Point2d(current_pixels[index])) <= max_residual_px) {
        next_active.push_back(static_cast<int>(index));
      }
    }
    if (next_active.size() < static_cast<std::size_t>(minimum_inliers)) {
      return {};
    }
    if (next_active == active) {
      break;
    }
    active = std::move(next_active);
  }
  double squared_error = 0.0;
  for (const int index : active) {
    const cv::Mat predicted = ray_rotation * cv::Mat(previous_rays[index]);
    const cv::Point2d projected(
      fx * predicted.at<double>(0) / predicted.at<double>(2) + cx,
      fy * predicted.at<double>(1) / predicted.at<double>(2) + cy);
    const double residual = cv::norm(projected - cv::Point2d(current_pixels[index]));
    squared_error += residual * residual;
  }
  inlier_count = static_cast<int>(active.size());
  residual_rmse_px = std::sqrt(squared_error / static_cast<double>(active.size()));
  const cv::Mat vo_rotation = ray_rotation.t();
  const double yaw = std::atan2(
    vo_rotation.at<double>(0, 2), vo_rotation.at<double>(2, 2));
  const double cosine = std::cos(yaw);
  const double sine = std::sin(yaw);
  return (cv::Mat_<double>(3, 3) <<
    cosine, 0.0, sine,
    0.0, 1.0, 0.0,
    -sine, 0.0, cosine);
}

// Quaternion (x, y, z, w) of a rotation matrix.
//
// Shared by the odometry publisher and the convention tests so that both read
// the same numbers; the node only wraps the result in its message type.
inline cv::Vec4d rotation_to_quaternion_xyzw(const cv::Mat & rotation)
{
  const double r00 = rotation.at<double>(0, 0);
  const double r01 = rotation.at<double>(0, 1);
  const double r02 = rotation.at<double>(0, 2);
  const double r10 = rotation.at<double>(1, 0);
  const double r11 = rotation.at<double>(1, 1);
  const double r12 = rotation.at<double>(1, 2);
  const double r20 = rotation.at<double>(2, 0);
  const double r21 = rotation.at<double>(2, 1);
  const double r22 = rotation.at<double>(2, 2);
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double w = 1.0;
  const double trace = r00 + r11 + r22;
  if (trace > 0.0) {
    const double s = 2.0 * std::sqrt(trace + 1.0);
    w = 0.25 * s;
    x = (r21 - r12) / s;
    y = (r02 - r20) / s;
    z = (r10 - r01) / s;
  } else if (r00 > r11 && r00 > r22) {
    const double s = 2.0 * std::sqrt(1.0 + r00 - r11 - r22);
    w = (r21 - r12) / s;
    x = 0.25 * s;
    y = (r01 + r10) / s;
    z = (r02 + r20) / s;
  } else if (r11 > r22) {
    const double s = 2.0 * std::sqrt(1.0 + r11 - r00 - r22);
    w = (r02 - r20) / s;
    x = (r01 + r10) / s;
    y = 0.25 * s;
    z = (r12 + r21) / s;
  } else {
    const double s = 2.0 * std::sqrt(1.0 + r22 - r00 - r11);
    w = (r10 - r01) / s;
    x = (r02 + r20) / s;
    y = (r12 + r21) / s;
    z = 0.25 * s;
  }
  return cv::Vec4d(x, y, z, w);
}

}  // namespace visual_slam
