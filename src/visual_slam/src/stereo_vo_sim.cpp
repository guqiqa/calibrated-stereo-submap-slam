#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <limits>
#include <memory>
#include <map>
#include <filesystem>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "visual_slam/geometry.hpp"

#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>

namespace
{
int64_t stamp_ns(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec;
}

geometry_msgs::msg::Quaternion rotation_to_quaternion(const cv::Mat & rotation)
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
  geometry_msgs::msg::Quaternion q;
  const double trace = r00 + r11 + r22;
  if (trace > 0.0) {
    const double s = 2.0 * std::sqrt(trace + 1.0);
    q.w = 0.25 * s;
    q.x = (r21 - r12) / s;
    q.y = (r02 - r20) / s;
    q.z = (r10 - r01) / s;
  } else if (r00 > r11 && r00 > r22) {
    const double s = 2.0 * std::sqrt(1.0 + r00 - r11 - r22);
    q.w = (r21 - r12) / s;
    q.x = 0.25 * s;
    q.y = (r01 + r10) / s;
    q.z = (r02 + r20) / s;
  } else if (r11 > r22) {
    const double s = 2.0 * std::sqrt(1.0 + r11 - r00 - r22);
    q.w = (r02 - r20) / s;
    q.x = (r01 + r10) / s;
    q.y = 0.25 * s;
    q.z = (r12 + r21) / s;
  } else {
    const double s = 2.0 * std::sqrt(1.0 + r22 - r00 - r11);
    q.w = (r10 - r01) / s;
    q.x = (r02 + r20) / s;
    q.y = (r12 + r21) / s;
    q.z = 0.25 * s;
  }
  return q;
}
}  // namespace

class StereoVoSim final : public rclcpp::Node
{
public:
  StereoVoSim()
  : Node("stereo_vo_sim"), world_T_current_(cv::Mat::eye(4, 4, CV_64F))
  {
    global_map_.id = -1;
    global_map_.center_pose = cv::Mat::eye(4, 4, CV_64F);
    left_topic_ = declare_parameter<std::string>("left_topic", "/sim/left/image_raw");
    right_topic_ = declare_parameter<std::string>("right_topic", "/sim/right/image_raw");
    left_info_topic_ = declare_parameter<std::string>("left_info_topic", "/sim/left/camera_info");
    right_info_topic_ = declare_parameter<std::string>("right_info_topic", "/sim/right/camera_info");
    odometry_topic_ = declare_parameter<std::string>("odometry_topic", "/visual_slam/odometry_sim");
    status_topic_ = declare_parameter<std::string>("status_topic", "/visual_slam/status_sim");
    max_pair_delta_ms_ = declare_parameter<double>("max_pair_delta_ms", 10.0);
    processing_scale_ = declare_parameter<double>("processing_scale", 0.75);
    max_features_ = declare_parameter<int>("max_features", 1200);
    max_disparity_ = declare_parameter<double>("max_disparity", 160.0);
    min_disparity_ = declare_parameter<double>("min_disparity", 2.0);
    stereo_epipolar_tolerance_px_ = declare_parameter<double>("stereo_epipolar_tolerance_px", 2.0);
    stereo_reprojection_error_px_ = declare_parameter<double>("stereo_reprojection_error_px", 1.5);
    temporal_match_distance_ = declare_parameter<int>("temporal_match_distance", 52);
    min_stereo_points_ = declare_parameter<int>("min_stereo_points", 35);
    min_pnp_inliers_ = declare_parameter<int>("min_pnp_inliers", 18);
    max_depth_m_ = declare_parameter<double>("max_depth_m", 25.0);
    min_3d3d_inliers_ = declare_parameter<int>("min_3d3d_inliers", 20);
    max_3d3d_residual_m_ = declare_parameter<double>("max_3d3d_residual_m", 0.08);
    motion_translation_tolerance_m_ = declare_parameter<double>("motion_translation_tolerance_m", 0.20);
    motion_rotation_tolerance_deg_ = declare_parameter<double>("motion_rotation_tolerance_deg", 8.0);
    stable_map_min_observations_ = declare_parameter<int>("stable_map_min_observations", 2);
    motion_reference_frames_ = declare_parameter<int>("motion_reference_frames", 5);
    motion_reference_translation_m_ = declare_parameter<double>("motion_reference_translation_m", 0.04);
    planar_motion_constraint_ = declare_parameter<bool>("planar_motion_constraint", true);
    loop_closure_enabled_ = declare_parameter<bool>("loop_closure_enabled", true);
    loop_closure_apply_correction_ = declare_parameter<bool>("loop_closure_apply_correction", false);
    loop_check_period_frames_ = declare_parameter<int>("loop_check_period_frames", 15);
    loop_min_keyframe_gap_ = declare_parameter<int>("loop_min_keyframe_gap", 8);
    loop_min_matches_ = declare_parameter<int>("loop_min_matches", 35);
    loop_min_inliers_ = declare_parameter<int>("loop_min_inliers", 25);
    loop_max_reprojection_px_ = declare_parameter<double>("loop_max_reprojection_px", 2.5);
    initial_gazebo_x_ = declare_parameter<double>("initial_gazebo_x", 0.08);
    initial_gazebo_y_ = declare_parameter<double>("initial_gazebo_y", 0.035);
    initial_gazebo_z_ = declare_parameter<double>("initial_gazebo_z", 0.32);
    odom_frame_ = declare_parameter<std::string>("odom_frame", "visual_odom_sim");
    camera_frame_ = declare_parameter<std::string>("camera_frame", "stereo_left_optical_frame");
    submap_mode_ = declare_parameter<std::string>("submap_mode", "joint");
    submap_length_m_ = declare_parameter<double>("submap_length_m", 5.0);
    submap_overlap_m_ = declare_parameter<double>("submap_overlap_m", 1.5);
    candidate_count_ = declare_parameter<int>("candidate_count", 3);
    representative_keyframes_ = declare_parameter<int>("representative_keyframes", 3);
    local_window_keyframes_ = declare_parameter<int>("local_window_keyframes", 15);
    keyframe_distance_m_ = declare_parameter<double>("keyframe_distance_m", 0.3);
    keyframe_angle_deg_ = declare_parameter<double>("keyframe_angle_deg", 5.0);
    switch_confirm_frames_ = declare_parameter<int>("switch_confirm_frames", 3);
    switch_min_matches_ = declare_parameter<int>("switch_min_matches", 30);
    switch_min_inliers_ = declare_parameter<int>("switch_min_inliers", 20);
    switch_max_reprojection_px_ = declare_parameter<double>("switch_max_reprojection_px", 3.0);
    smooth_transition_frames_ = declare_parameter<int>("smooth_transition_frames", 5);
    map_fusion_radius_m_ = declare_parameter<double>("map_fusion_radius_m", 0.10);
    map_export_path_ = declare_parameter<std::string>("map_export_path", "");
    gazebo_T_map_ = visual_slam::make_gazebo_T_initial_optical(
      initial_gazebo_x_, initial_gazebo_y_, initial_gazebo_z_);

    if (processing_scale_ < 0.25 || processing_scale_ > 1.0) {
      throw std::runtime_error("processing_scale must be within [0.25, 1.0]");
    }
    if (submap_mode_ != "global" && submap_mode_ != "fixed" && submap_mode_ != "joint") {
      throw std::runtime_error("submap_mode must be global, fixed or joint");
    }
    if (submap_length_m_ <= 0.0 || submap_overlap_m_ < 0.0 || submap_overlap_m_ >= submap_length_m_) {
      throw std::runtime_error("submap_length_m and submap_overlap_m are invalid");
    }
    if (candidate_count_ < 1 || representative_keyframes_ < 1 || local_window_keyframes_ < 1 ||
      switch_confirm_frames_ < 1 || switch_min_matches_ < 6 || switch_min_inliers_ < 6 ||
      smooth_transition_frames_ < 1)
    {
      throw std::runtime_error("invalid submap manager parameters");
    }
    if (map_fusion_radius_m_ <= 0.0) {
      throw std::runtime_error("map_fusion_radius_m must be positive");
    }
    if (min_disparity_ <= 0.0 || min_3d3d_inliers_ < 3 || max_3d3d_residual_m_ <= 0.0 ||
      motion_translation_tolerance_m_ <= 0.0 || motion_rotation_tolerance_deg_ <= 0.0 ||
      stable_map_min_observations_ < 1 || motion_reference_frames_ < 2 ||
      motion_reference_translation_m_ <= 0.0 || loop_check_period_frames_ < 1 ||
      loop_min_keyframe_gap_ < 1 || loop_min_matches_ < 6 || loop_min_inliers_ < 6 ||
      loop_max_reprojection_px_ <= 0.0)
    {
      throw std::runtime_error("invalid stereo geometry quality parameters");
    }
    orb_ = cv::ORB::create(max_features_);
    matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, false);
    odometry_publisher_ = create_publisher<nav_msgs::msg::Odometry>(odometry_topic_, 10);
    status_publisher_ = create_publisher<std_msgs::msg::String>(status_topic_, 10);
    const auto qos = rclcpp::SensorDataQoS().keep_last(5);
    left_subscription_ = create_subscription<sensor_msgs::msg::Image>(
      left_topic_, qos, [this](sensor_msgs::msg::Image::ConstSharedPtr message) {
        left_queue_.push_back(std::move(message));
        trim_queue(left_queue_);
        try_process_queued_pair();
      });
    right_subscription_ = create_subscription<sensor_msgs::msg::Image>(
      right_topic_, qos, [this](sensor_msgs::msg::Image::ConstSharedPtr message) {
        right_queue_.push_back(std::move(message));
        trim_queue(right_queue_);
        try_process_queued_pair();
      });
    left_info_subscription_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      left_info_topic_, qos, [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr message) {
        left_info_ = *message;
        try_initialize_calibration();
      });
    right_info_subscription_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      right_info_topic_, qos, [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr message) {
        right_info_ = *message;
        try_initialize_calibration();
      });
    publish_status("waiting_for_camera_info", 0, 0, 0, 0.0, 0.0);
  }

  ~StereoVoSim() override
  {
    export_map();
  }

private:
  struct Frame
  {
    cv::Mat left_image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<cv::Point3f> points3d;
    double median_disparity{0.0};
    double min_depth{0.0};
    double median_depth{0.0};
    double stereo_reprojection_rmse{0.0};
  };

  struct TrajectorySample
  {
    double stamp{0.0};
    cv::Mat pose;
  };

  struct LoopClosure
  {
    bool accepted{false};
    int keyframe_id{-1};
    int matches{0};
    int inliers{0};
    double reprojection_rmse{std::numeric_limits<double>::infinity()};
    cv::Mat corrected_pose;
  };

  struct Keyframe
  {
    int id{};
    double distance{};
    cv::Mat world_pose;
    Frame frame;
  };

  struct Submap
  {
    int id{};
    double start_distance{};
    double end_distance{};
    cv::Mat center_pose;
    std::vector<int> keyframe_ids;
    std::vector<cv::Point3f> map_points;
    cv::Mat map_descriptors;
    cv::Mat representative_descriptors;
    std::deque<int> map_point_counts;
    std::vector<cv::Point3f> fused_map_points;
    std::vector<int> fused_map_observations;
    std::map<std::array<int, 3>, int> fused_voxel_indices;
    bool loaded{true};
  };

  struct MapMatch
  {
    int submap_id{-1};
    int matches{};
    int inliers{};
    double reprojection_rmse{std::numeric_limits<double>::infinity()};
    double visual_score{};
    double pose_score{};
    double topology_score{};
    cv::Mat world_pose;
  };

  void trim_queue(std::deque<sensor_msgs::msg::Image::ConstSharedPtr> & queue)
  {
    while (queue.size() > 6U) {
      queue.pop_front();
      ++rejected_pairs_;
    }
  }

  void try_initialize_calibration()
  {
    if (calibration_ready_) {
      return;
    }
    if (!left_info_ || !right_info_ || left_info_->width == 0 || right_info_->width == 0) {
      return;
    }
    const int width = static_cast<int>(left_info_->width);
    const int height = static_cast<int>(left_info_->height);
    if (width != static_cast<int>(right_info_->width) || height != static_cast<int>(right_info_->height) ||
      width < 32 || height < 32 || left_info_->k[0] <= 0.0 || right_info_->k[0] <= 0.0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "invalid or mismatched stereo CameraInfo");
      return;
    }
    const double fx = left_info_->p[0] > 0.0 ? left_info_->p[0] : left_info_->k[0];
    double baseline = left_info_->p[3] != 0.0 ? std::abs(left_info_->p[3] / fx) : 0.0;
    if (baseline < 0.001 && right_info_->p[3] != 0.0) {
      baseline = std::abs(right_info_->p[3] / fx);
    }
    baseline = baseline > 0.001 ? baseline : declare_or_get_baseline();
    if (baseline < 0.001 || baseline > 0.5) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 3000, "invalid stereo baseline %.4f m", baseline);
      return;
    }

    k1_ = cv::Mat(3, 3, CV_64F, const_cast<double *>(left_info_->k.data())).clone();
    k2_ = cv::Mat(3, 3, CV_64F, const_cast<double *>(right_info_->k.data())).clone();
    d1_ = cv::Mat(1, static_cast<int>(left_info_->d.size()), CV_64F,
      const_cast<double *>(left_info_->d.data())).clone();
    d2_ = cv::Mat(1, static_cast<int>(right_info_->d.size()), CV_64F,
      const_cast<double *>(right_info_->d.data())).clone();
    const cv::Mat r = cv::Mat::eye(3, 3, CV_64F);
    const cv::Mat t = (cv::Mat_<double>(3, 1) << -baseline, 0.0, 0.0);
    native_size_ = cv::Size(width, height);
    processing_size_ = cv::Size(
      std::max(1, static_cast<int>(std::lround(width * processing_scale_))),
      std::max(1, static_cast<int>(std::lround(height * processing_scale_))));
    cv::Mat full_p1;
    cv::Mat full_p2;
    cv::stereoRectify(k1_, d1_, k2_, d2_, native_size_, r, t, r1_, r2_, full_p1, full_p2, q_,
      cv::CALIB_ZERO_DISPARITY, 0.0, native_size_);
    p1_ = full_p1.clone();
    p2_ = full_p2.clone();
    p1_.row(0) *= processing_scale_;
    p1_.row(1) *= processing_scale_;
    p2_.row(0) *= processing_scale_;
    p2_.row(1) *= processing_scale_;
    cv::initUndistortRectifyMap(k1_, d1_, r1_, p1_, processing_size_, CV_32FC1, left_map_x_, left_map_y_);
    cv::initUndistortRectifyMap(k2_, d2_, r2_, p2_, processing_size_, CV_32FC1, right_map_x_, right_map_y_);
    rectified_k_ = p1_(cv::Rect(0, 0, 3, 3)).clone();
    calibration_ready_ = true;
    previous_.reset();
    world_T_current_ = cv::Mat::eye(4, 4, CV_64F);
    reference_world_pose_ = world_T_current_.clone();
    reference_age_ = 0;
    committed_distance_ = 0.0;
    RCLCPP_INFO(get_logger(), "camera info ready: %dx%d -> %dx%d, fx=%.2f px, baseline=%.3f m",
      width, height, processing_size_.width, processing_size_.height, rectified_k_.at<double>(0, 0), baseline);
    publish_status("camera_info_ready", 0, 0, 0, 0.0, 0.0);
  }

  double declare_or_get_baseline()
  {
    if (!baseline_parameter_declared_) {
      fallback_baseline_m_ = declare_parameter<double>("fallback_baseline_m", 0.07);
      baseline_parameter_declared_ = true;
    }
    return fallback_baseline_m_;
  }

  static double pose_translation(const cv::Mat & pose)
  {
    const double x = pose.at<double>(0, 3);
    const double y = pose.at<double>(1, 3);
    const double z = pose.at<double>(2, 3);
    return std::sqrt(x * x + y * y + z * z);
  }

  static double pose_distance(const cv::Mat & lhs, const cv::Mat & rhs)
  {
    const double dx = lhs.at<double>(0, 3) - rhs.at<double>(0, 3);
    const double dy = lhs.at<double>(1, 3) - rhs.at<double>(1, 3);
    const double dz = lhs.at<double>(2, 3) - rhs.at<double>(2, 3);
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  static cv::Point3f transform_point(const cv::Mat & pose, const cv::Point3f & point)
  {
    return cv::Point3f(
      static_cast<float>(pose.at<double>(0, 0) * point.x + pose.at<double>(0, 1) * point.y +
      pose.at<double>(0, 2) * point.z + pose.at<double>(0, 3)),
      static_cast<float>(pose.at<double>(1, 0) * point.x + pose.at<double>(1, 1) * point.y +
      pose.at<double>(1, 2) * point.z + pose.at<double>(1, 3)),
      static_cast<float>(pose.at<double>(2, 0) * point.x + pose.at<double>(2, 1) * point.y +
      pose.at<double>(2, 2) * point.z + pose.at<double>(2, 3)));
  }

  static cv::Mat pose_from_camera_transform(const cv::Mat & rotation, const cv::Mat & translation)
  {
    cv::Mat camera_T_world = cv::Mat::eye(4, 4, CV_64F);
    rotation.copyTo(camera_T_world(cv::Rect(0, 0, 3, 3)));
    translation.copyTo(camera_T_world(cv::Rect(3, 0, 1, 3)));
    return camera_T_world.inv();
  }

  static double rotation_difference_deg(const cv::Mat & lhs, const cv::Mat & rhs)
  {
    const cv::Mat lhs_rotation = lhs(cv::Rect(0, 0, 3, 3));
    const cv::Mat rhs_rotation = rhs(cv::Rect(0, 0, 3, 3));
    cv::Mat vector;
    cv::Rodrigues(lhs_rotation * rhs_rotation.t(), vector);
    return cv::norm(vector) * 180.0 / CV_PI;
  }

  static double json_number(double value)
  {
    return std::isfinite(value) ? value : -1.0;
  }

  static bool rigid_transform(const std::vector<cv::Point3f> & source,
    const std::vector<cv::Point3f> & target, const std::vector<int> & indices, cv::Mat & transform)
  {
    if (indices.size() < 3U) {
      return false;
    }
    cv::Vec3d source_center(0.0, 0.0, 0.0);
    cv::Vec3d target_center(0.0, 0.0, 0.0);
    for (const int index : indices) {
      source_center += cv::Vec3d(source[index].x, source[index].y, source[index].z);
      target_center += cv::Vec3d(target[index].x, target[index].y, target[index].z);
    }
    source_center *= 1.0 / static_cast<double>(indices.size());
    target_center *= 1.0 / static_cast<double>(indices.size());
    cv::Mat covariance = cv::Mat::zeros(3, 3, CV_64F);
    for (const int index : indices) {
      const cv::Vec3d src(source[index].x - source_center[0], source[index].y - source_center[1],
        source[index].z - source_center[2]);
      const cv::Vec3d dst(target[index].x - target_center[0], target[index].y - target_center[1],
        target[index].z - target_center[2]);
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
          covariance.at<double>(row, col) += dst[row] * src[col];
        }
      }
    }
    cv::Mat u, singular_values, vt;
    cv::SVD::compute(covariance, singular_values, u, vt);
    cv::Mat rotation = u * vt;
    if (cv::determinant(rotation) < 0.0) {
      u.col(2) *= -1.0;
      rotation = u * vt;
    }
    if (!std::isfinite(cv::determinant(rotation))) {
      return false;
    }
    cv::Mat source_center_mat = (cv::Mat_<double>(3, 1) << source_center[0], source_center[1], source_center[2]);
    cv::Mat target_center_mat = (cv::Mat_<double>(3, 1) << target_center[0], target_center[1], target_center[2]);
    cv::Mat translation = target_center_mat - rotation * source_center_mat;
    transform = cv::Mat::eye(4, 4, CV_64F);
    rotation.copyTo(transform(cv::Rect(0, 0, 3, 3)));
    translation.copyTo(transform(cv::Rect(3, 0, 1, 3)));
    return true;
  }

  bool estimate_3d3d_ransac(const std::vector<cv::Point3f> & source,
    const std::vector<cv::Point3f> & target, cv::Mat & transform, int & inliers, double & rmse) const
  {
    inliers = 0;
    rmse = std::numeric_limits<double>::infinity();
    if (source.size() != target.size() || source.size() < 6U) {
      return false;
    }
    std::mt19937 generator(1337U + static_cast<unsigned int>(processed_frames_));
    std::uniform_int_distribution<int> distribution(0, static_cast<int>(source.size()) - 1);
    std::vector<int> best_indices;
    cv::Mat best_transform;
    for (int iteration = 0; iteration < 100; ++iteration) {
      std::set<int> sample_set;
      while (sample_set.size() < 3U) {
        sample_set.insert(distribution(generator));
      }
      std::vector<int> sample(sample_set.begin(), sample_set.end());
      cv::Mat candidate;
      if (!rigid_transform(source, target, sample, candidate)) {
        continue;
      }
      std::vector<int> current_indices;
      double squared_error = 0.0;
      for (std::size_t i = 0; i < source.size(); ++i) {
        const cv::Point3f predicted = transform_point(candidate, source[i]);
        const double error = std::sqrt(
          std::pow(predicted.x - target[i].x, 2.0) +
          std::pow(predicted.y - target[i].y, 2.0) +
          std::pow(predicted.z - target[i].z, 2.0));
        if (error <= max_3d3d_residual_m_) {
          current_indices.push_back(static_cast<int>(i));
          squared_error += error * error;
        }
      }
      if (current_indices.size() > best_indices.size() ||
        (current_indices.size() == best_indices.size() && squared_error < rmse * rmse * current_indices.size()))
      {
        best_indices = std::move(current_indices);
        best_transform = candidate;
        rmse = best_indices.empty() ? std::numeric_limits<double>::infinity() :
          std::sqrt(squared_error / static_cast<double>(best_indices.size()));
      }
    }
    if (best_indices.size() < static_cast<std::size_t>(min_3d3d_inliers_)) {
      return false;
    }
    if (!rigid_transform(source, target, best_indices, transform)) {
      transform = best_transform;
    }
    double squared_error = 0.0;
    for (const int index : best_indices) {
      const cv::Point3f predicted = transform_point(transform, source[index]);
      const double error = std::sqrt(
        std::pow(predicted.x - target[index].x, 2.0) +
        std::pow(predicted.y - target[index].y, 2.0) +
        std::pow(predicted.z - target[index].z, 2.0));
      squared_error += error * error;
    }
    inliers = static_cast<int>(best_indices.size());
    rmse = std::sqrt(squared_error / static_cast<double>(best_indices.size()));
    return true;
  }

  LoopClosure detect_loop_closure(const Frame & frame) const
  {
    LoopClosure best;
    if (!loop_closure_enabled_ || frame.descriptors.empty() || keyframes_.empty() ||
      next_keyframe_id_ < loop_min_keyframe_gap_)
    {
      return best;
    }
    for (const auto & entry : keyframes_) {
      const Keyframe & keyframe = entry.second;
      if (keyframe.id + loop_min_keyframe_gap_ >= next_keyframe_id_ ||
        keyframe.frame.descriptors.empty())
      {
        continue;
      }
      std::vector<std::vector<cv::DMatch>> candidates;
      matcher_->knnMatch(keyframe.frame.descriptors, frame.descriptors, candidates, 2);
      std::vector<cv::Point3f> object_points;
      std::vector<cv::Point2f> image_points;
      std::set<int> used_current_indices;
      for (const auto & candidate : candidates) {
        if (candidate.size() != 2 || candidate[0].distance >= 0.76F * candidate[1].distance ||
          candidate[0].distance > temporal_match_distance_ ||
          !used_current_indices.insert(candidate[0].trainIdx).second)
        {
          continue;
        }
        object_points.push_back(transform_point(keyframe.world_pose,
          keyframe.frame.points3d[candidate[0].queryIdx]));
        image_points.push_back(frame.keypoints[candidate[0].trainIdx].pt);
      }
      if (static_cast<int>(object_points.size()) < loop_min_matches_) {
        continue;
      }
      cv::Mat rotation_vector;
      cv::Mat translation_vector;
      cv::Mat inliers;
      const bool ok = cv::solvePnPRansac(object_points, image_points, rectified_k_, cv::Mat(),
        rotation_vector, translation_vector, false, 150, loop_max_reprojection_px_, 0.99,
        inliers, cv::SOLVEPNP_ITERATIVE);
      if (!ok || inliers.rows < loop_min_inliers_ || inliers.rows <= best.inliers) {
        continue;
      }
      cv::Mat rotation;
      cv::Rodrigues(rotation_vector, rotation);
      cv::Mat corrected_pose = pose_from_camera_transform(rotation, translation_vector);
      std::vector<cv::Point2f> projected;
      cv::projectPoints(object_points, rotation_vector, translation_vector, rectified_k_, cv::Mat(), projected);
      double squared_error = 0.0;
      for (int i = 0; i < inliers.rows; ++i) {
        const int index = inliers.at<int>(i, 0);
        squared_error += cv::norm(projected[index] - image_points[index]) *
          cv::norm(projected[index] - image_points[index]);
      }
      best.accepted = true;
      best.keyframe_id = keyframe.id;
      best.matches = static_cast<int>(object_points.size());
      best.inliers = inliers.rows;
      best.reprojection_rmse = std::sqrt(squared_error / static_cast<double>(inliers.rows));
      best.corrected_pose = corrected_pose;
    }
    if (best.accepted && best.reprojection_rmse > loop_max_reprojection_px_) {
      best.accepted = false;
    }
    return best;
  }

  void ensure_submaps_for_distance(double distance)
  {
    const double stride = submap_length_m_ - submap_overlap_m_;
    const int last_id = std::max(0, static_cast<int>(std::floor(distance / stride)));
    for (int id = static_cast<int>(submaps_.size()); id <= last_id; ++id) {
      Submap submap;
      submap.id = id;
      submap.start_distance = id * stride;
      submap.end_distance = submap.start_distance + submap_length_m_;
      submaps_.push_back(std::move(submap));
    }
  }

  void append_frame_to_map(Submap & submap, const Keyframe & keyframe)
  {
    const bool representative = static_cast<int>(submap.keyframe_ids.size()) < representative_keyframes_;
    submap.keyframe_ids.push_back(keyframe.id);
    if (submap.center_pose.empty()) {
      submap.center_pose = keyframe.world_pose.clone();
    }
    for (std::size_t i = 0; i < keyframe.frame.points3d.size(); ++i) {
      const cv::Point3f world_point = transform_point(keyframe.world_pose, keyframe.frame.points3d[i]);
      submap.map_points.push_back(world_point);
      submap.map_descriptors.push_back(keyframe.frame.descriptors.row(static_cast<int>(i)));
      fuse_map_point(submap, world_point);
    }
    submap.map_point_counts.push_back(static_cast<int>(keyframe.frame.points3d.size()));
    if (representative && !keyframe.frame.descriptors.empty()) {
      submap.representative_descriptors.push_back(keyframe.frame.descriptors);
    }
    if (submap.id >= 0) {
      trim_submap_window(submap);
    }
  }

  void fuse_map_point(Submap & submap, const cv::Point3f & point)
  {
    const double radius = map_fusion_radius_m_;
    const std::array<int, 3> voxel = {
      static_cast<int>(std::floor(point.x / radius)),
      static_cast<int>(std::floor(point.y / radius)),
      static_cast<int>(std::floor(point.z / radius))};
    const auto found = submap.fused_voxel_indices.find(voxel);
    if (found == submap.fused_voxel_indices.end()) {
      const int index = static_cast<int>(submap.fused_map_points.size());
      submap.fused_voxel_indices.emplace(voxel, index);
      submap.fused_map_points.push_back(point);
      submap.fused_map_observations.push_back(1);
      return;
    }
    const int index = found->second;
    const int observations = submap.fused_map_observations[index];
    const float weight = static_cast<float>(observations);
    submap.fused_map_points[index] = cv::Point3f(
      (submap.fused_map_points[index].x * weight + point.x) / (weight + 1.0F),
      (submap.fused_map_points[index].y * weight + point.y) / (weight + 1.0F),
      (submap.fused_map_points[index].z * weight + point.z) / (weight + 1.0F));
    submap.fused_map_observations[index] = observations + 1;
  }

  void trim_submap_window(Submap & submap)
  {
    while (static_cast<int>(submap.keyframe_ids.size()) > local_window_keyframes_) {
      const int removed_points = submap.map_point_counts.front();
      submap.keyframe_ids.erase(submap.keyframe_ids.begin());
      submap.map_point_counts.pop_front();
      if (removed_points <= 0) {
        continue;
      }
      submap.map_points.erase(submap.map_points.begin(), submap.map_points.begin() + removed_points);
      submap.map_descriptors = submap.map_descriptors.rowRange(removed_points, submap.map_descriptors.rows).clone();
    }
  }

  void insert_keyframe(const Frame & frame, const cv::Mat & world_pose, double distance)
  {
    ensure_submaps_for_distance(distance);
    Keyframe keyframe;
    keyframe.id = next_keyframe_id_++;
    keyframe.distance = distance;
    keyframe.world_pose = world_pose.clone();
    keyframe.frame = frame;
    keyframe.frame.left_image.release();
    keyframes_[keyframe.id] = keyframe;
    append_frame_to_map(global_map_, keyframe);
    for (auto & submap : submaps_) {
      if (distance + 1e-6 >= submap.start_distance && distance <= submap.end_distance + 1e-6) {
        append_frame_to_map(submap, keyframe);
      }
    }
    last_keyframe_pose_ = world_pose.clone();
    last_keyframe_distance_ = distance;
    ++keyframe_count_;
  }

  bool should_insert_keyframe(const cv::Mat & world_pose, int inliers) const
  {
    if (!last_keyframe_pose_) {
      return true;
    }
    const double distance = pose_distance(*last_keyframe_pose_, world_pose);
    cv::Mat delta = world_pose(cv::Rect(0, 0, 3, 3)) * (*last_keyframe_pose_)(cv::Rect(0, 0, 3, 3)).t();
    cv::Mat rotation_vector;
    cv::Rodrigues(delta, rotation_vector);
    const double angle_deg = cv::norm(rotation_vector) * 180.0 / CV_PI;
    return distance >= keyframe_distance_m_ || angle_deg >= keyframe_angle_deg_ || inliers < min_pnp_inliers_ * 2;
  }

  int distance_submap_id(double distance) const
  {
    const double stride = submap_length_m_ - submap_overlap_m_;
    return std::max(0, static_cast<int>(std::floor(distance / stride)));
  }

  void update_loaded_submaps()
  {
    if (submaps_.empty()) {
      return;
    }
    ensure_submaps_for_distance(trajectory_distance_);
    const int expected = distance_submap_id(trajectory_distance_);
    const int anchor = std::max(active_submap_id_, expected);
    for (auto & submap : submaps_) {
      submap.loaded = std::abs(submap.id - anchor) <= 1;
    }
    if (trajectory_distance_ >= submaps_[anchor].end_distance - 1.0) {
      const int next = anchor + 1;
      if (next < static_cast<int>(submaps_.size())) {
        submaps_[next].loaded = true;
      }
    }
  }

  double descriptor_score(const Frame & frame, const Submap & submap) const
  {
    if (frame.descriptors.empty() || submap.representative_descriptors.empty()) {
      return 0.0;
    }
    std::vector<std::vector<cv::DMatch>> candidates;
    matcher_->knnMatch(frame.descriptors, submap.representative_descriptors, candidates, 2);
    int good = 0;
    for (const auto & candidate : candidates) {
      if (candidate.size() == 2 && candidate[0].distance < 0.80F * candidate[1].distance) {
        ++good;
      }
    }
    const int normalizer = std::max(1, std::min(switch_min_matches_, frame.descriptors.rows));
    return std::min(1.0, static_cast<double>(good) / normalizer);
  }

  MapMatch validate_map(const Frame & frame, const Submap & submap, double visual_score, double topology_score)
  {
    MapMatch result;
    result.submap_id = submap.id;
    result.visual_score = visual_score;
    result.topology_score = topology_score;
    if (frame.descriptors.empty() || submap.map_descriptors.empty()) {
      return result;
    }
    std::vector<std::vector<cv::DMatch>> candidates;
    matcher_->knnMatch(frame.descriptors, submap.map_descriptors, candidates, 2);
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    for (const auto & candidate : candidates) {
      if (candidate.size() != 2 || candidate[0].distance >= 0.78F * candidate[1].distance ||
        candidate[0].distance > temporal_match_distance_)
      {
        continue;
      }
      object_points.push_back(submap.map_points[candidate[0].trainIdx]);
      image_points.push_back(frame.keypoints[candidate[0].queryIdx].pt);
    }
    result.matches = static_cast<int>(object_points.size());
    if (result.matches < switch_min_matches_) {
      return result;
    }
    cv::Mat rotation_vector;
    cv::Mat translation_vector;
    cv::Mat inliers;
    const bool ok = cv::solvePnPRansac(
      object_points, image_points, rectified_k_, cv::Mat(), rotation_vector, translation_vector, false,
      100, switch_max_reprojection_px_, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
    result.inliers = ok ? inliers.rows : 0;
    if (!ok || result.inliers < switch_min_inliers_) {
      return result;
    }
    cv::Mat rotation;
    cv::Rodrigues(rotation_vector, rotation);
    result.world_pose = pose_from_camera_transform(rotation, translation_vector);
    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_points, rotation_vector, translation_vector, rectified_k_, cv::Mat(), projected);
    double squared_error = 0.0;
    int used = 0;
    for (int i = 0; i < inliers.rows; ++i) {
      const int index = inliers.at<int>(i, 0);
      const double dx = projected[index].x - image_points[index].x;
      const double dy = projected[index].y - image_points[index].y;
      squared_error += dx * dx + dy * dy;
      ++used;
    }
    result.reprojection_rmse = used > 0 ? std::sqrt(squared_error / used) : std::numeric_limits<double>::infinity();
    result.pose_score = std::exp(-pose_distance(result.world_pose, world_T_current_) *
      pose_distance(result.world_pose, world_T_current_) / 18.0);
    return result;
  }

  static bool better_match(const MapMatch & lhs, const MapMatch & rhs)
  {
    if (lhs.inliers != rhs.inliers) {
      return lhs.inliers > rhs.inliers;
    }
    return lhs.reprojection_rmse < rhs.reprojection_rmse;
  }

  void smooth_to_pose(const cv::Mat & target_pose)
  {
    if (smooth_transition_frames_ <= 0) {
      world_T_current_ = target_pose.clone();
      return;
    }
    const double alpha = 1.0 / static_cast<double>(smooth_frames_remaining_);
    const cv::Mat current_rotation = world_T_current_(cv::Rect(0, 0, 3, 3)).clone();
    const cv::Mat target_rotation = target_pose(cv::Rect(0, 0, 3, 3)).clone();
    cv::Mat delta_rotation = target_rotation * current_rotation.t();
    cv::Mat delta_vector;
    cv::Rodrigues(delta_rotation, delta_vector);
    cv::Mat step_rotation;
    cv::Rodrigues(delta_vector * alpha, step_rotation);
    cv::Mat blended = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat blended_rotation = step_rotation * current_rotation;
    blended_rotation.copyTo(blended(cv::Rect(0, 0, 3, 3)));
    for (int axis = 0; axis < 3; ++axis) {
      blended.at<double>(axis, 3) = world_T_current_.at<double>(axis, 3) +
        alpha * (target_pose.at<double>(axis, 3) - world_T_current_.at<double>(axis, 3));
    }
    world_T_current_ = blended;
    if (smooth_frames_remaining_ > 0) {
      --smooth_frames_remaining_;
    }
    if (smooth_frames_remaining_ == 0) {
      world_T_current_ = target_pose.clone();
      smooth_target_pose_.release();
    }
  }

  double topology_score(int submap_id) const
  {
    if (submap_id == active_submap_id_) {
      return 1.0;
    }
    if (std::abs(submap_id - active_submap_id_) == 1) {
      return 0.8;
    }
    return 0.2;
  }

  std::vector<int> fixed_candidate_ids() const
  {
    std::vector<int> ids;
    for (int offset = -1; offset <= 1; ++offset) {
      const int id = active_submap_id_ + offset;
      if (id >= 0 && id < static_cast<int>(submaps_.size())) {
        ids.push_back(id);
      }
    }
    return ids;
  }

  void evaluate_submaps(const Frame & frame)
  {
    last_candidates_.clear();
    last_matching_ms_ = 0.0;
    if (submaps_.empty() || frame.descriptors.empty()) {
      return;
    }
    const auto started = std::chrono::steady_clock::now();
    std::vector<int> ids;
    if (submap_mode_ == "global") {
      MapMatch match = validate_map(frame, global_map_, 1.0, 1.0);
      match.submap_id = -1;
      last_candidates_.push_back(match);
      last_matching_ms_ = elapsed_ms(started);
      return;
    }
    if (submap_mode_ == "fixed") {
      ids = fixed_candidate_ids();
    } else {
      std::vector<std::pair<double, int>> ranked;
      ranked.reserve(submaps_.size());
      for (const auto & submap : submaps_) {
        const double visual = descriptor_score(frame, submap);
        const double pose = submap.center_pose.empty() ? 0.0 :
          std::exp(-pose_distance(submap.center_pose, world_T_current_) *
          pose_distance(submap.center_pose, world_T_current_) / 18.0);
        const double combined = 0.50 * visual + 0.30 * pose + 0.20 * topology_score(submap.id);
        ranked.emplace_back(combined, submap.id);
      }
      std::sort(ranked.begin(), ranked.end(), [](const auto & lhs, const auto & rhs) {
        return lhs.first > rhs.first;
      });
      for (int i = 0; i < std::min(candidate_count_, static_cast<int>(ranked.size())); ++i) {
        ids.push_back(ranked[i].second);
      }
    }
    for (const int id : ids) {
      auto & submap = submaps_[id];
      const double visual = descriptor_score(frame, submap);
      MapMatch match = validate_map(frame, submap, visual, topology_score(id));
      match.pose_score = submap.center_pose.empty() ? 0.0 :
        std::exp(-pose_distance(submap.center_pose, world_T_current_) *
        pose_distance(submap.center_pose, world_T_current_) / 18.0);
      last_candidates_.push_back(std::move(match));
    }
    std::sort(last_candidates_.begin(), last_candidates_.end(), [](const auto & lhs, const auto & rhs) {
      const double lhs_score = 0.50 * lhs.visual_score + 0.30 * lhs.pose_score + 0.20 * lhs.topology_score;
      const double rhs_score = 0.50 * rhs.visual_score + 0.30 * rhs.pose_score + 0.20 * rhs.topology_score;
      return lhs_score > rhs_score;
    });
    last_matching_ms_ = elapsed_ms(started);
    if (last_candidates_.empty()) {
      return;
    }
    const auto best = std::find_if(last_candidates_.begin(), last_candidates_.end(), [this](const MapMatch & match) {
      return match.inliers >= switch_min_inliers_ && match.reprojection_rmse <= switch_max_reprojection_px_;
    });
    if (best == last_candidates_.end() || best->submap_id == active_submap_id_) {
      pending_submap_id_ = active_submap_id_;
      pending_switch_frames_ = 0;
      return;
    }
    if (pending_submap_id_ == best->submap_id) {
      ++pending_switch_frames_;
    } else {
      pending_submap_id_ = best->submap_id;
      pending_switch_frames_ = 1;
    }
    if (pending_switch_frames_ < switch_confirm_frames_) {
      return;
    }
    active_submap_id_ = best->submap_id;
    ++switch_count_;
    pending_switch_frames_ = 0;
    if (!best->world_pose.empty()) {
      smooth_frames_remaining_ = smooth_transition_frames_;
      smooth_target_pose_ = best->world_pose.clone();
    }
  }

  std::string candidate_ids_json() const
  {
    std::ostringstream data;
    data << '[';
    for (std::size_t i = 0; i < last_candidates_.size(); ++i) {
      if (i > 0) {
        data << ',';
      }
      data << last_candidates_[i].submap_id;
    }
    data << ']';
    return data.str();
  }

  int active_keyframe_count() const
  {
    if (submap_mode_ == "global") {
      return static_cast<int>(global_map_.keyframe_ids.size());
    }
    int count = 0;
    for (const auto & submap : submaps_) {
      if (submap.loaded) {
        count += static_cast<int>(submap.keyframe_ids.size());
      }
    }
    return count;
  }

  int active_map_point_count() const
  {
    if (submap_mode_ == "global") {
      return static_cast<int>(global_map_.map_points.size());
    }
    int count = 0;
    for (const auto & submap : submaps_) {
      if (submap.loaded) {
        count += static_cast<int>(submap.map_points.size());
      }
    }
    return count;
  }

  int stable_map_point_count() const
  {
    int count = 0;
    for (std::size_t i = 0; i < global_map_.fused_map_points.size(); ++i) {
      if (global_map_.fused_map_observations[i] >= stable_map_min_observations_) {
        ++count;
      }
    }
    return count;
  }

  static cv::Mat image_to_mono8(const sensor_msgs::msg::Image & image)
  {
    int channels = 1;
    int type = CV_8UC1;
    if (image.encoding == "rgb8" || image.encoding == "bgr8") {
      channels = 3;
      type = CV_8UC3;
    } else if (image.encoding != "mono8" && image.encoding != "8UC1") {
      throw std::runtime_error("unsupported image encoding: " + image.encoding);
    }
    const std::size_t required = static_cast<std::size_t>(image.step) * image.height;
    if (image.step < image.width * channels || image.data.size() < required) {
      throw std::runtime_error("invalid image buffer or step");
    }
    cv::Mat source(static_cast<int>(image.height), static_cast<int>(image.width), type,
      const_cast<unsigned char *>(image.data.data()), image.step);
    if (channels == 1) {
      return source.clone();
    }
    cv::Mat gray;
    cv::cvtColor(source, gray, image.encoding == "rgb8" ? cv::COLOR_RGB2GRAY : cv::COLOR_BGR2GRAY);
    return gray;
  }

  void try_process_queued_pair()
  {
    if (!calibration_ready_ || left_queue_.empty() || right_queue_.empty()) {
      return;
    }
    auto best_left = left_queue_.end();
    auto best_right = right_queue_.end();
    int64_t smallest_delta_ns = std::numeric_limits<int64_t>::max();
    for (auto left = left_queue_.begin(); left != left_queue_.end(); ++left) {
      for (auto right = right_queue_.begin(); right != right_queue_.end(); ++right) {
        const int64_t delta = std::llabs(stamp_ns((*left)->header.stamp) - stamp_ns((*right)->header.stamp));
        if (delta < smallest_delta_ns) {
          smallest_delta_ns = delta;
          best_left = left;
          best_right = right;
        }
      }
    }
    const double pair_delta_ms = smallest_delta_ns / 1e6;
    if (pair_delta_ms > max_pair_delta_ms_) {
      if (stamp_ns(left_queue_.front()->header.stamp) < stamp_ns(right_queue_.front()->header.stamp)) {
        left_queue_.pop_front();
      } else {
        right_queue_.pop_front();
      }
      ++rejected_pairs_;
      return;
    }
    const auto left = *best_left;
    const auto right = *best_right;
    left_queue_.erase(best_left);
    right_queue_.erase(best_right);
    if (left->width != static_cast<uint32_t>(native_size_.width) || left->height != static_cast<uint32_t>(native_size_.height) ||
      right->width != static_cast<uint32_t>(native_size_.width) || right->height != static_cast<uint32_t>(native_size_.height)) {
      publish_status("rejected_resolution", 0, 0, 0, pair_delta_ms, 0.0);
      return;
    }
    try {
      process_pair(left->header.stamp, image_to_mono8(*left), image_to_mono8(*right), pair_delta_ms);
    } catch (const std::exception & error) {
      ++tracking_failures_;
      publish_status("rejected_image", 0, 0, 0, pair_delta_ms, 0.0);
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "image conversion failed: %s", error.what());
    }
  }

  Frame make_stereo_frame(const cv::Mat & left_gray, const cv::Mat & right_gray)
  {
    cv::Mat left_rectified;
    cv::Mat right_rectified;
    cv::remap(left_gray, left_rectified, left_map_x_, left_map_y_, cv::INTER_LINEAR);
    cv::remap(right_gray, right_rectified, right_map_x_, right_map_y_, cv::INTER_LINEAR);
    std::vector<cv::KeyPoint> left_keypoints;
    std::vector<cv::KeyPoint> right_keypoints;
    cv::Mat left_descriptors;
    cv::Mat right_descriptors;
    orb_->detectAndCompute(left_rectified, cv::noArray(), left_keypoints, left_descriptors);
    orb_->detectAndCompute(right_rectified, cv::noArray(), right_keypoints, right_descriptors);
    Frame frame;
    frame.left_image = left_rectified;
    if (left_descriptors.empty() || right_descriptors.empty()) {
      return frame;
    }
    std::vector<std::vector<cv::DMatch>> candidates;
    matcher_->knnMatch(left_descriptors, right_descriptors, candidates, 2);
    std::vector<cv::Point2f> matched_left;
    std::vector<cv::Point2f> matched_right;
    std::vector<int> source_indices;
    std::set<int> used_right_indices;
    for (const auto & candidate : candidates) {
      if (candidate.size() != 2 || candidate[0].distance >= 0.76F * candidate[1].distance) {
        continue;
      }
      const auto & left_keypoint = left_keypoints[candidate[0].queryIdx];
      const auto & right_keypoint = right_keypoints[candidate[0].trainIdx];
      const double disparity = left_keypoint.pt.x - right_keypoint.pt.x;
      if (std::abs(left_keypoint.pt.y - right_keypoint.pt.y) > stereo_epipolar_tolerance_px_ ||
        disparity < min_disparity_ || disparity > max_disparity_) {
        continue;
      }
      if (!used_right_indices.insert(candidate[0].trainIdx).second) {
        continue;
      }
      matched_left.push_back(left_keypoint.pt);
      matched_right.push_back(right_keypoint.pt);
      source_indices.push_back(candidate[0].queryIdx);
    }
    if (matched_left.size() < 6U) {
      return frame;
    }
    cv::Mat homogeneous;
    cv::triangulatePoints(p1_, p2_, matched_left, matched_right, homogeneous);
    cv::Mat homogeneous64;
    homogeneous.convertTo(homogeneous64, CV_64F);
    std::vector<double> accepted_disparities;
    std::vector<double> accepted_depths;
    double squared_reprojection_error = 0.0;
    for (int column = 0; column < homogeneous64.cols; ++column) {
      const double w = homogeneous64.at<double>(3, column);
      if (std::abs(w) < 1e-9) {
        continue;
      }
      const cv::Point3f point(
        static_cast<float>(homogeneous64.at<double>(0, column) / w),
        static_cast<float>(homogeneous64.at<double>(1, column) / w),
        static_cast<float>(homogeneous64.at<double>(2, column) / w));
      if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z) ||
        point.z <= 0.05F || point.z > static_cast<float>(max_depth_m_)) {
        continue;
      }
      const auto reprojection_error = [](const cv::Mat & projection, const cv::Point3f & value,
        const cv::Point2f & measured) {
          const double x = static_cast<double>(value.x);
          const double y = static_cast<double>(value.y);
          const double z = static_cast<double>(value.z);
          const double u = projection.at<double>(0, 0) * x + projection.at<double>(0, 1) * y +
            projection.at<double>(0, 2) * z + projection.at<double>(0, 3);
          const double v = projection.at<double>(1, 0) * x + projection.at<double>(1, 1) * y +
            projection.at<double>(1, 2) * z + projection.at<double>(1, 3);
          const double w = projection.at<double>(2, 0) * x + projection.at<double>(2, 1) * y +
            projection.at<double>(2, 2) * z + projection.at<double>(2, 3);
          if (std::abs(w) < 1e-9) {
            return std::numeric_limits<double>::infinity();
          }
          return std::hypot(u / w - measured.x, v / w - measured.y);
        };
      const cv::Point2f measured_left = matched_left[column];
      const cv::Point2f measured_right = matched_right[column];
      const double left_error = reprojection_error(p1_, point, measured_left);
      const double right_error = reprojection_error(p2_, point, measured_right);
      if (left_error > stereo_reprojection_error_px_ || right_error > stereo_reprojection_error_px_) {
        continue;
      }
      frame.keypoints.push_back(left_keypoints[source_indices[column]]);
      frame.points3d.push_back(point);
      frame.descriptors.push_back(left_descriptors.row(source_indices[column]));
      accepted_disparities.push_back(static_cast<double>(measured_left.x - measured_right.x));
      accepted_depths.push_back(static_cast<double>(point.z));
      squared_reprojection_error += left_error * left_error + right_error * right_error;
    }
    if (!accepted_depths.empty()) {
      std::sort(accepted_disparities.begin(), accepted_disparities.end());
      std::sort(accepted_depths.begin(), accepted_depths.end());
      const std::size_t middle = accepted_depths.size() / 2U;
      frame.median_disparity = accepted_disparities[middle];
      frame.min_depth = accepted_depths.front();
      frame.median_depth = accepted_depths[middle];
      frame.stereo_reprojection_rmse = std::sqrt(
        squared_reprojection_error / (2.0 * static_cast<double>(accepted_depths.size())));
    }
    return frame;
  }

  void process_pair(const builtin_interfaces::msg::Time & stamp, const cv::Mat & left, const cv::Mat & right,
    double pair_delta_ms)
  {
    const auto started = std::chrono::steady_clock::now();
    ++processed_frames_;
    Frame current = make_stereo_frame(left, right);
    const int stereo_points = static_cast<int>(current.points3d.size());
    last_frame_median_disparity_ = current.median_disparity;
    last_frame_min_depth_ = current.min_depth;
    last_frame_median_depth_ = current.median_depth;
    last_frame_reprojection_rmse_ = current.stereo_reprojection_rmse;
    if (stereo_points < min_stereo_points_) {
      ++tracking_failures_;
      publish_status("insufficient_stereo_features", stereo_points, 0, 0, pair_delta_ms, elapsed_ms(started));
      return;
    }
    if (!previous_) {
      previous_ = std::move(current);
      reference_world_pose_ = world_T_current_.clone();
      reference_age_ = 0;
      insert_keyframe(*previous_, world_T_current_, trajectory_distance_);
      trajectory_samples_.push_back({static_cast<double>(stamp_ns(stamp)) / 1e9, world_T_current_.clone()});
      update_loaded_submaps();
      publish_status("initialized", stereo_points, 0, 0, pair_delta_ms, elapsed_ms(started));
      return;
    }
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    std::vector<cv::Point3f> current_points3d;
    std::set<int> used_current_indices;
    std::vector<cv::Point2f> previous_pixels;
    previous_pixels.reserve(previous_->keypoints.size());
    for (const auto & keypoint : previous_->keypoints) {
      previous_pixels.push_back(keypoint.pt);
    }
    std::vector<cv::Point2f> tracked_pixels;
    std::vector<cv::Point2f> backward_pixels;
    std::vector<unsigned char> forward_status;
    std::vector<unsigned char> backward_status;
    std::vector<float> forward_error;
    std::vector<float> backward_error;
    cv::calcOpticalFlowPyrLK(previous_->left_image, current.left_image, previous_pixels,
      tracked_pixels, forward_status, forward_error, cv::Size(21, 21), 3,
      cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01));
    cv::calcOpticalFlowPyrLK(current.left_image, previous_->left_image, tracked_pixels,
      backward_pixels, backward_status, backward_error, cv::Size(21, 21), 3,
      cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01));
    for (std::size_t index = 0; index < previous_pixels.size(); ++index) {
      if (!forward_status[index] || !backward_status[index] ||
        cv::norm(previous_pixels[index] - backward_pixels[index]) > 1.0 ||
        tracked_pixels[index].x < 1.0F || tracked_pixels[index].y < 1.0F ||
        tracked_pixels[index].x >= current.left_image.cols - 1.0F ||
        tracked_pixels[index].y >= current.left_image.rows - 1.0F)
      {
        continue;
      }
      int nearest_index = -1;
      double nearest_distance = 2.0;
      for (std::size_t current_index = 0; current_index < current.keypoints.size(); ++current_index) {
        const double distance = cv::norm(current.keypoints[current_index].pt - tracked_pixels[index]);
        if (distance < nearest_distance) {
          nearest_distance = distance;
          nearest_index = static_cast<int>(current_index);
        }
      }
      if (nearest_index < 0 || !used_current_indices.insert(nearest_index).second) {
        continue;
      }
      object_points.push_back(previous_->points3d[index]);
      image_points.push_back(tracked_pixels[index]);
      current_points3d.push_back(current.points3d[nearest_index]);
    }
    cv::Mat rotation_vector;
    cv::Mat translation_vector;
    cv::Mat inliers;
    const bool pnp_ok = object_points.size() >= 6U && cv::solvePnPRansac(
      object_points, image_points, rectified_k_, cv::Mat(), rotation_vector, translation_vector, false,
      100, 2.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
    const int inlier_count = pnp_ok ? inliers.rows : 0;
    cv::Mat pnp_transform;
    if (pnp_ok) {
      cv::Mat rotation_matrix;
      cv::Rodrigues(rotation_vector, rotation_matrix);
      pnp_transform = cv::Mat::eye(4, 4, CV_64F);
      rotation_matrix.copyTo(pnp_transform(cv::Rect(0, 0, 3, 3)));
      translation_vector.copyTo(pnp_transform(cv::Rect(3, 0, 1, 3)));
    }
    cv::Mat rigid_transform_estimate;
    int rigid_inliers = 0;
    double rigid_rmse = std::numeric_limits<double>::infinity();
    const bool rigid_ok = estimate_3d3d_ransac(
      object_points, current_points3d, rigid_transform_estimate, rigid_inliers, rigid_rmse);
    double motion_translation_difference = std::numeric_limits<double>::infinity();
    double motion_rotation_difference = std::numeric_limits<double>::infinity();
    bool motion_consistent = true;
    if (pnp_ok && rigid_ok) {
      motion_translation_difference = cv::norm(
        pnp_transform(cv::Rect(3, 0, 1, 3)) - rigid_transform_estimate(cv::Rect(3, 0, 1, 3)));
      motion_rotation_difference = rotation_difference_deg(pnp_transform, rigid_transform_estimate);
      motion_consistent = motion_translation_difference <= motion_translation_tolerance_m_ &&
        motion_rotation_difference <= motion_rotation_tolerance_deg_;
    }
    last_rigid_inliers_ = rigid_inliers;
    last_rigid_rmse_ = rigid_rmse;
    last_motion_translation_difference_ = motion_translation_difference;
    last_motion_rotation_difference_ = motion_rotation_difference;
    last_motion_consistency_ = motion_consistent ? 1.0 : 0.0;
    cv::Mat selected_transform;
    const double rigid_translation = rigid_ok ? pose_translation(rigid_transform_estimate) : 0.0;
    const double pnp_translation = pnp_ok ? pose_translation(pnp_transform) : 0.0;
    const bool rigid_quantized = rigid_ok && pnp_ok && pnp_translation > 0.003 &&
      rigid_translation < 0.25 * pnp_translation;
    if (pnp_ok && inlier_count >= min_pnp_inliers_ && motion_consistent) {
      // LK-PnP retains sub-pixel image motion in distant corridor views.
      // Stereo 3D-3D remains an independent metric consistency check.
      selected_transform = pnp_transform;
      motion_source_ = rigid_quantized ? "pnp_quantized_3d3d" : "pnp_stereo_consistent";
    } else if (rigid_ok && !pnp_ok) {
      selected_transform = rigid_transform_estimate;
      motion_source_ = "stereo_3d3d_fallback";
    } else if (pnp_ok && inlier_count >= min_pnp_inliers_ && (!rigid_ok || rigid_quantized)) {
      selected_transform = pnp_transform;
      motion_source_ = rigid_quantized ? "pnp_quantized_3d3d" : "pnp_fallback";
    } else if (rigid_ok && pnp_ok && !motion_consistent) {
      motion_source_ = "inconsistent_motion";
    }
    if (selected_transform.empty()) {
      ++tracking_failures_;
      publish_status("tracking_lost", stereo_points, static_cast<int>(object_points.size()), inlier_count,
        pair_delta_ms, elapsed_ms(started));
      return;
    }
    last_relative_translation_[0] = selected_transform.at<double>(0, 3);
    last_relative_translation_[1] = selected_transform.at<double>(1, 3);
    last_relative_translation_[2] = selected_transform.at<double>(2, 3);
    // previous_ is held for a short reference window, so selected_transform
    // already represents current_T_reference and must not be multiplied by
    // estimates from earlier frames in the same window.
    world_T_current_ = reference_world_pose_ * selected_transform.inv();
    if (planar_motion_constraint_) {
      world_T_current_ = visual_slam::constrain_optical_pose_to_ground_plane(world_T_current_);
    }
    trajectory_distance_ = committed_distance_ + pose_translation(selected_transform);
    update_loaded_submaps();
    evaluate_submaps(current);
    last_loop_closure_ = LoopClosure{};
    if (loop_closure_enabled_ && processed_frames_ % loop_check_period_frames_ == 0) {
      last_loop_closure_ = detect_loop_closure(current);
      if (loop_closure_apply_correction_ && last_loop_closure_.accepted &&
        pose_distance(last_loop_closure_.corrected_pose, world_T_current_) > 0.5)
      {
        smooth_frames_remaining_ = smooth_transition_frames_;
        smooth_target_pose_ = last_loop_closure_.corrected_pose.clone();
        ++loop_closure_count_;
      }
    }
    if (smooth_frames_remaining_ > 0 && !smooth_target_pose_.empty()) {
      smooth_to_pose(smooth_target_pose_);
      if (planar_motion_constraint_) {
        world_T_current_ = visual_slam::constrain_optical_pose_to_ground_plane(world_T_current_);
      }
    }
    if (last_keyframe_pose_ && should_insert_keyframe(world_T_current_,
      std::max(inlier_count, rigid_inliers))) {
      insert_keyframe(current, world_T_current_, trajectory_distance_);
      update_loaded_submaps();
    }
    ++reference_age_;
    if (reference_age_ >= motion_reference_frames_ ||
      pose_translation(selected_transform) >= motion_reference_translation_m_)
    {
      committed_distance_ = trajectory_distance_;
      reference_world_pose_ = world_T_current_.clone();
      previous_ = std::move(current);
      reference_age_ = 0;
    }
    ++tracked_pairs_;
    trajectory_samples_.push_back({static_cast<double>(stamp_ns(stamp)) / 1e9, world_T_current_.clone()});
    publish_odometry(stamp, inlier_count);
    publish_status("tracking", stereo_points, static_cast<int>(object_points.size()), inlier_count,
      pair_delta_ms, elapsed_ms(started));
  }

  static double elapsed_ms(const std::chrono::steady_clock::time_point & started)
  {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  }

  void publish_odometry(const builtin_interfaces::msg::Time & stamp, int inliers)
  {
    nav_msgs::msg::Odometry odometry;
    odometry.header.stamp = stamp;
    odometry.header.frame_id = odom_frame_;
    odometry.child_frame_id = camera_frame_;
    odometry.pose.pose.position.x = world_T_current_.at<double>(0, 3);
    odometry.pose.pose.position.y = world_T_current_.at<double>(1, 3);
    odometry.pose.pose.position.z = world_T_current_.at<double>(2, 3);
    odometry.pose.pose.orientation = rotation_to_quaternion(world_T_current_(cv::Rect(0, 0, 3, 3)));
    const double variance = 1.0 / static_cast<double>(std::max(1, inliers));
    odometry.pose.covariance[0] = variance;
    odometry.pose.covariance[7] = variance;
    odometry.pose.covariance[14] = variance;
    odometry.pose.covariance[21] = variance * 4.0;
    odometry.pose.covariance[28] = variance * 4.0;
    odometry.pose.covariance[35] = variance * 4.0;
    odometry_publisher_->publish(odometry);
  }

  void publish_status(const std::string & state, int stereo_points, int temporal_matches, int pnp_inliers,
    double pair_delta_ms, double processing_ms)
  {
    std_msgs::msg::String status;
    std::ostringstream data;
    const std::string switch_state = pending_switch_frames_ > 0 ? "confirming" : "stable";
    data << "{\"state\":\"" << state << "\",\"stereo_points\":" << stereo_points
         << ",\"temporal_matches\":" << temporal_matches << ",\"pnp_inliers\":" << pnp_inliers
         << ",\"pair_delta_ms\":" << pair_delta_ms << ",\"processing_ms\":" << processing_ms
         << ",\"tracked_pairs\":" << tracked_pairs_ << ",\"rejected_pairs\":" << rejected_pairs_
         << ",\"tracking_failures\":" << tracking_failures_
         << ",\"submap_mode\":\"" << submap_mode_ << "\""
         << ",\"active_submap\":" << active_submap_id_
         << ",\"candidate_submaps\":" << candidate_ids_json()
         << ",\"switch_state\":\"" << switch_state << "\""
         << ",\"visual_score\":" << (last_candidates_.empty() ? 0.0 : last_candidates_.front().visual_score)
         << ",\"pose_score\":" << (last_candidates_.empty() ? 0.0 : last_candidates_.front().pose_score)
         << ",\"topology_score\":" << (last_candidates_.empty() ? 0.0 : last_candidates_.front().topology_score)
         << ",\"active_keyframes\":" << active_keyframe_count()
         << ",\"active_map_points\":" << active_map_point_count()
         << ",\"matching_ms\":" << last_matching_ms_
         << ",\"keyframes\":" << keyframe_count_
         << ",\"submaps\":" << submaps_.size()
         << ",\"switch_count\":" << switch_count_
         << ",\"wrong_switch_count\":" << wrong_switch_count_
         << ",\"motion_source\":\"" << motion_source_ << "\""
         << ",\"valid_matches\":" << temporal_matches
         << ",\"ransac_inliers\":" << last_rigid_inliers_
         << ",\"median_disparity_px\":" << last_frame_median_disparity_
         << ",\"depth_min_m\":" << last_frame_min_depth_
         << ",\"depth_median_m\":" << last_frame_median_depth_
         << ",\"stereo_reprojection_px\":" << last_frame_reprojection_rmse_
         << ",\"motion_consistency\":" << last_motion_consistency_
         << ",\"visual_distance_m\":" << trajectory_distance_
         << ",\"map_points_raw\":" << static_cast<int>(global_map_.map_points.size())
         << ",\"map_points_stable\":" << stable_map_point_count()
         << ",\"rigid_rmse_m\":" << json_number(last_rigid_rmse_)
         << ",\"motion_translation_difference_m\":" << json_number(last_motion_translation_difference_)
         << ",\"motion_rotation_difference_deg\":" << json_number(last_motion_rotation_difference_)
         << ",\"relative_tx_m\":" << last_relative_translation_[0]
         << ",\"relative_ty_m\":" << last_relative_translation_[1]
         << ",\"relative_tz_m\":" << last_relative_translation_[2]
         << ",\"loop_closure_accepted\":" << (last_loop_closure_.accepted ? 1 : 0)
         << ",\"loop_closure_keyframe\":" << last_loop_closure_.keyframe_id
         << ",\"loop_closure_matches\":" << last_loop_closure_.matches
         << ",\"loop_closure_inliers\":" << last_loop_closure_.inliers
         << ",\"loop_closure_rmse_px\":" << json_number(last_loop_closure_.reprojection_rmse)
         << ",\"loop_closure_apply_correction\":" << (loop_closure_apply_correction_ ? "true" : "false")
         << ",\"loop_closure_count\":" << loop_closure_count_
         << "}";
    status.data = data.str();
    status_publisher_->publish(status);
  }

  void export_map() const
  {
    if (map_export_path_.empty()) {
      return;
    }
    const std::filesystem::path requested_path(map_export_path_);
    const std::filesystem::path output_dir = requested_path.parent_path().empty() ?
      std::filesystem::current_path() : requested_path.parent_path();
    std::error_code error;
    std::filesystem::create_directories(output_dir, error);
    std::ofstream combined(requested_path);
    std::ofstream raw(output_dir / "raw_map_points.csv");
    std::ofstream stable(output_dir / "stable_map_points.csv");
    std::ofstream keyframes(output_dir / "keyframes.csv");
    std::ofstream trajectory(output_dir / "trajectory_visual.csv");
    if (!combined || !raw || !stable || !keyframes || !trajectory) {
      RCLCPP_WARN(const_cast<StereoVoSim *>(this)->get_logger(),
        "cannot open map export files in: %s", output_dir.string().c_str());
      return;
    }
    const char * header =
      "scope,submap_id,keyframe_id,point_id,x_optical,y_optical,z_optical,x_gazebo,y_gazebo,z_gazebo,observations,stable\n";
    combined << header;
    raw << header;
    stable << header;
    const auto write_point = [this](std::ostream & output, const char * scope, int submap_id,
      int keyframe_id, int point_id, const cv::Point3f & point, int observations, bool is_stable) {
      const cv::Point3f gazebo_point = visual_slam::transform_point(gazebo_T_map_, point);
      output << scope << ',' << submap_id << ',' << keyframe_id << ',' << point_id << ','
             << point.x << ',' << point.y << ',' << point.z << ','
             << gazebo_point.x << ',' << gazebo_point.y << ',' << gazebo_point.z << ','
             << observations << ',' << (is_stable ? 1 : 0) << '\n';
    };
    const auto write_submap = [&write_point](std::ostream & output, const Submap & submap, const char * scope) {
      std::size_t point_index = 0;
      for (std::size_t keyframe_index = 0; keyframe_index < submap.keyframe_ids.size(); ++keyframe_index) {
        const int keyframe_id = submap.keyframe_ids[keyframe_index];
        const int point_count = keyframe_index < submap.map_point_counts.size() ?
          submap.map_point_counts[keyframe_index] : 0;
        for (int i = 0; i < point_count && point_index < submap.map_points.size(); ++i) {
          const auto & point = submap.map_points[point_index++];
          write_point(output, scope, submap.id, keyframe_id, static_cast<int>(point_index - 1), point, 1, false);
        }
      }
    };
    const auto write_fused = [&write_point](std::ostream & output, const Submap & submap, const char * scope,
      int minimum_observations) {
      for (std::size_t i = 0; i < submap.fused_map_points.size(); ++i) {
        if (submap.fused_map_observations[i] < minimum_observations) {
          continue;
        }
        const auto & point = submap.fused_map_points[i];
        write_point(output, scope, submap.id, -1, static_cast<int>(i), point,
          submap.fused_map_observations[i], minimum_observations > 1);
      }
    };
    write_submap(combined, global_map_, "global");
    write_fused(combined, global_map_, "global_fused", 1);
    write_fused(combined, global_map_, "global_stable", stable_map_min_observations_);
    write_submap(raw, global_map_, "global");
    write_fused(stable, global_map_, "global_stable", stable_map_min_observations_);
    for (const auto & submap : submaps_) {
      write_submap(combined, submap, "submap");
      write_fused(combined, submap, "submap_fused", 1);
      write_fused(combined, submap, "submap_stable", stable_map_min_observations_);
      write_submap(raw, submap, "submap");
      write_fused(stable, submap, "submap_stable", stable_map_min_observations_);
    }
    keyframes << "id,distance,x_optical,y_optical,z_optical,x_gazebo,y_gazebo,z_gazebo\n";
    for (const auto & entry : keyframes_) {
      const Keyframe & keyframe = entry.second;
      const cv::Point3f optical(
        static_cast<float>(keyframe.world_pose.at<double>(0, 3)),
        static_cast<float>(keyframe.world_pose.at<double>(1, 3)),
        static_cast<float>(keyframe.world_pose.at<double>(2, 3)));
      const cv::Point3f gazebo = visual_slam::transform_point(gazebo_T_map_, optical);
      keyframes << keyframe.id << ',' << keyframe.distance << ',' << optical.x << ',' << optical.y << ',' << optical.z
                << ',' << gazebo.x << ',' << gazebo.y << ',' << gazebo.z << '\n';
    }
    trajectory << "stamp,x_optical,y_optical,z_optical,x_gazebo,y_gazebo,z_gazebo\n";
    for (const auto & sample : trajectory_samples_) {
      const cv::Point3f optical(
        static_cast<float>(sample.pose.at<double>(0, 3)),
        static_cast<float>(sample.pose.at<double>(1, 3)),
        static_cast<float>(sample.pose.at<double>(2, 3)));
      const cv::Point3f gazebo = visual_slam::transform_point(gazebo_T_map_, optical);
      trajectory << sample.stamp << ',' << optical.x << ',' << optical.y << ',' << optical.z
                 << ',' << gazebo.x << ',' << gazebo.y << ',' << gazebo.z << '\n';
    }
  }

  std::string left_topic_;
  std::string right_topic_;
  std::string left_info_topic_;
  std::string right_info_topic_;
  std::string odometry_topic_;
  std::string status_topic_;
  std::string odom_frame_;
  std::string camera_frame_;
  double max_pair_delta_ms_{};
  double processing_scale_{};
  int max_features_{};
  double max_disparity_{};
  double stereo_epipolar_tolerance_px_{};
  double stereo_reprojection_error_px_{};
  int temporal_match_distance_{};
  int min_stereo_points_{};
  int min_pnp_inliers_{};
  double max_depth_m_{};
  double min_disparity_{};
  int min_3d3d_inliers_{};
  double max_3d3d_residual_m_{};
  double motion_translation_tolerance_m_{};
  double motion_rotation_tolerance_deg_{};
  int stable_map_min_observations_{};
  int motion_reference_frames_{};
  double motion_reference_translation_m_{};
  bool planar_motion_constraint_{true};
  bool loop_closure_enabled_{true};
  bool loop_closure_apply_correction_{false};
  int loop_check_period_frames_{};
  int loop_min_keyframe_gap_{};
  int loop_min_matches_{};
  int loop_min_inliers_{};
  double loop_max_reprojection_px_{};
  double initial_gazebo_x_{};
  double initial_gazebo_y_{};
  double initial_gazebo_z_{};
  double fallback_baseline_m_{0.07};
  std::string submap_mode_;
  double submap_length_m_{};
  double submap_overlap_m_{};
  int candidate_count_{};
  int representative_keyframes_{};
  int local_window_keyframes_{};
  double keyframe_distance_m_{};
  double keyframe_angle_deg_{};
  int switch_confirm_frames_{};
  int switch_min_matches_{};
  int switch_min_inliers_{};
  double switch_max_reprojection_px_{};
  int smooth_transition_frames_{};
  double map_fusion_radius_m_{};
  std::string map_export_path_;
  bool baseline_parameter_declared_{false};
  bool calibration_ready_{false};
  int rejected_pairs_{0};
  int tracking_failures_{0};
  int tracked_pairs_{0};
  int processed_frames_{0};
  int last_rigid_inliers_{0};
  double last_rigid_rmse_{std::numeric_limits<double>::infinity()};
  double last_motion_translation_difference_{std::numeric_limits<double>::infinity()};
  double last_motion_rotation_difference_{std::numeric_limits<double>::infinity()};
  double last_motion_consistency_{0.0};
  double last_frame_median_disparity_{0.0};
  double last_frame_min_depth_{0.0};
  double last_frame_median_depth_{0.0};
  double last_frame_reprojection_rmse_{0.0};
  std::array<double, 3> last_relative_translation_{0.0, 0.0, 0.0};
  std::string motion_source_{"none"};
  std::optional<sensor_msgs::msg::CameraInfo> left_info_;
  std::optional<sensor_msgs::msg::CameraInfo> right_info_;
  cv::Mat k1_, k2_, d1_, d2_, r1_, r2_, p1_, p2_, q_, rectified_k_;
  cv::Mat left_map_x_, left_map_y_, right_map_x_, right_map_y_;
  cv::Size native_size_;
  cv::Size processing_size_;
  cv::Mat world_T_current_;
  cv::Mat gazebo_T_map_;
  cv::Mat reference_world_pose_;
  double trajectory_distance_{0.0};
  double committed_distance_{0.0};
  int reference_age_{0};
  int loop_closure_count_{0};
  int next_keyframe_id_{0};
  int keyframe_count_{0};
  int active_submap_id_{0};
  int pending_submap_id_{0};
  int pending_switch_frames_{0};
  int switch_count_{0};
  int wrong_switch_count_{0};
  int smooth_frames_remaining_{0};
  double last_keyframe_distance_{0.0};
  double last_matching_ms_{0.0};
  cv::Mat smooth_target_pose_;
  std::optional<cv::Mat> last_keyframe_pose_;
  Submap global_map_;
  std::map<int, Keyframe> keyframes_;
  std::vector<Submap> submaps_;
  std::vector<MapMatch> last_candidates_;
  std::vector<TrajectorySample> trajectory_samples_;
  LoopClosure last_loop_closure_;
  cv::Ptr<cv::ORB> orb_;
  cv::Ptr<cv::BFMatcher> matcher_;
  std::optional<Frame> previous_;
  std::deque<sensor_msgs::msg::Image::ConstSharedPtr> left_queue_;
  std::deque<sensor_msgs::msg::Image::ConstSharedPtr> right_queue_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr right_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr right_info_subscription_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_publisher_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StereoVoSim>());
  rclcpp::shutdown();
  return 0;
}
