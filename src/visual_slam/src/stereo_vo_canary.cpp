#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <rclcpp/rclcpp.hpp>
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
  geometry_msgs::msg::Quaternion quaternion;
  const double trace = r00 + r11 + r22;
  if (trace > 0.0) {
    const double s = 2.0 * std::sqrt(trace + 1.0);
    quaternion.w = 0.25 * s;
    quaternion.x = (r21 - r12) / s;
    quaternion.y = (r02 - r20) / s;
    quaternion.z = (r10 - r01) / s;
  } else if (r00 > r11 && r00 > r22) {
    const double s = 2.0 * std::sqrt(1.0 + r00 - r11 - r22);
    quaternion.w = (r21 - r12) / s;
    quaternion.x = 0.25 * s;
    quaternion.y = (r01 + r10) / s;
    quaternion.z = (r02 + r20) / s;
  } else if (r11 > r22) {
    const double s = 2.0 * std::sqrt(1.0 + r11 - r00 - r22);
    quaternion.w = (r02 - r20) / s;
    quaternion.x = (r01 + r10) / s;
    quaternion.y = 0.25 * s;
    quaternion.z = (r12 + r21) / s;
  } else {
    const double s = 2.0 * std::sqrt(1.0 + r22 - r00 - r11);
    quaternion.w = (r10 - r01) / s;
    quaternion.x = (r02 + r20) / s;
    quaternion.y = (r12 + r21) / s;
    quaternion.z = 0.25 * s;
  }
  return quaternion;
}
}  // namespace

class StereoVoCanary final : public rclcpp::Node
{
public:
  StereoVoCanary()
  : Node("stereo_vo_canary"), world_T_current_(cv::Mat::eye(4, 4, CV_64F))
  {
    left_topic_ = declare_parameter<std::string>("left_topic", "/camera_native/left/image_raw");
    right_topic_ = declare_parameter<std::string>("right_topic", "/camera_native/right/image_raw");
    calibration_file_ = declare_parameter<std::string>("calibration_file", "");
    max_pair_delta_ms_ = declare_parameter<double>("max_pair_delta_ms", 5.0);
    expected_width_ = declare_parameter<int>("expected_width", 1088);
    expected_height_ = declare_parameter<int>("expected_height", 1280);
    processing_scale_ = declare_parameter<double>("processing_scale", 0.4);
    max_features_ = declare_parameter<int>("max_features", 1200);
    stereo_epipolar_tolerance_px_ = declare_parameter<double>("stereo_epipolar_tolerance_px", 2.0);
    temporal_match_distance_ = declare_parameter<int>("temporal_match_distance", 48);
    min_stereo_points_ = declare_parameter<int>("min_stereo_points", 40);
    min_pnp_inliers_ = declare_parameter<int>("min_pnp_inliers", 20);
    odom_frame_ = declare_parameter<std::string>("odom_frame", "visual_odom");
    camera_frame_ = declare_parameter<std::string>("camera_frame", "camera_native_left_optical_frame");

    if (!load_calibration()) {
      throw std::runtime_error("stereo_vo_canary requires a valid calibration_file");
    }
    orb_ = cv::ORB::create(max_features_);
    matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, false);
    odometry_publisher_ = create_publisher<nav_msgs::msg::Odometry>("visual_slam/odometry", 10);
    status_publisher_ = create_publisher<std_msgs::msg::String>("visual_slam/status", 10);
    const auto qos = rclcpp::SensorDataQoS().keep_last(5);
    left_subscription_ = create_subscription<sensor_msgs::msg::Image>(
      left_topic_, qos, [this](sensor_msgs::msg::Image::ConstSharedPtr image) { on_left_image(std::move(image)); });
    right_subscription_ = create_subscription<sensor_msgs::msg::Image>(
      right_topic_, qos, [this](sensor_msgs::msg::Image::ConstSharedPtr image) { on_right_image(std::move(image)); });
    publish_status("waiting_for_stereo", 0, 0, 0.0);
  }

private:
  struct Frame
  {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<cv::Point3f> points3d;
  };

  bool load_calibration()
  {
    cv::FileStorage storage(calibration_file_, cv::FileStorage::READ);
    if (!storage.isOpened()) {
      RCLCPP_ERROR(get_logger(), "Cannot open calibration_file: %s", calibration_file_.c_str());
      return false;
    }
    storage["cameraMatrix1"] >> k1_;
    storage["distCoeffs1"] >> d1_;
    storage["cameraMatrix2"] >> k2_;
    storage["distCoeffs2"] >> d2_;
    storage["R"] >> rotation_;
    storage["T"] >> translation_;
    if (k1_.empty() || d1_.empty() || k2_.empty() || d2_.empty() || rotation_.empty() || translation_.empty()) {
      RCLCPP_ERROR(get_logger(), "Calibration is missing K/D/R/T matrices: %s", calibration_file_.c_str());
      return false;
    }
    if (processing_scale_ < 0.25 || processing_scale_ > 1.0) {
      RCLCPP_ERROR(get_logger(), "processing_scale must be within [0.25, 1.0]");
      return false;
    }
    const cv::Size native_size(expected_width_, expected_height_);
    processing_size_ = cv::Size(
      std::max(1, static_cast<int>(std::lround(expected_width_ * processing_scale_))),
      std::max(1, static_cast<int>(std::lround(expected_height_ * processing_scale_))));
    cv::Mat full_p1;
    cv::Mat full_p2;
    cv::stereoRectify(k1_, d1_, k2_, d2_, native_size, rotation_, translation_,
      r1_, r2_, full_p1, full_p2, q_, cv::CALIB_ZERO_DISPARITY, 0.0, native_size);
    p1_ = full_p1.clone();
    p2_ = full_p2.clone();
    p1_.row(0) *= processing_scale_;
    p1_.row(1) *= processing_scale_;
    p2_.row(0) *= processing_scale_;
    p2_.row(1) *= processing_scale_;
    cv::initUndistortRectifyMap(k1_, d1_, r1_, p1_, processing_size_, CV_32FC1, left_map_x_, left_map_y_);
    cv::initUndistortRectifyMap(k2_, d2_, r2_, p2_, processing_size_, CV_32FC1, right_map_x_, right_map_y_);
    rectified_k_ = p1_(cv::Rect(0, 0, 3, 3)).clone();
    const double baseline_m = cv::norm(translation_);
    RCLCPP_INFO(get_logger(), "Loaded calibration baseline=%.3f mm, native=%dx%d, processing=%dx%d", baseline_m * 1000.0,
      expected_width_, expected_height_, processing_size_.width, processing_size_.height);
    return baseline_m > 0.01 && baseline_m < 0.2;
  }

  static bool image_to_mono8(const sensor_msgs::msg::Image::ConstSharedPtr & message, cv::Mat * output)
  {
    if (message->encoding != "mono8" && message->encoding != "8UC1") {
      return false;
    }
    if (message->step < message->width || message->data.size() < message->step * message->height) {
      return false;
    }
    const cv::Mat view(static_cast<int>(message->height), static_cast<int>(message->width), CV_8UC1,
      const_cast<uint8_t *>(message->data.data()), message->step);
    *output = view.clone();
    return true;
  }

  void on_left_image(sensor_msgs::msg::Image::ConstSharedPtr left)
  {
    left_queue_.push_back(std::move(left));
    trim_queue(left_queue_);
    try_process_queued_pair();
  }

  void on_right_image(sensor_msgs::msg::Image::ConstSharedPtr right)
  {
    right_queue_.push_back(std::move(right));
    trim_queue(right_queue_);
    try_process_queued_pair();
  }

  void trim_queue(std::deque<sensor_msgs::msg::Image::ConstSharedPtr> & queue)
  {
    while (queue.size() > 5U) {
      queue.pop_front();
      ++rejected_pairs_;
    }
  }

  void try_process_queued_pair()
  {
    if (left_queue_.empty() || right_queue_.empty()) {
      return;
    }
    auto best_left = left_queue_.end();
    auto best_right = right_queue_.end();
    int64_t smallest_delta_ns = std::numeric_limits<int64_t>::max();
    for (auto left = left_queue_.begin(); left != left_queue_.end(); ++left) {
      for (auto right = right_queue_.begin(); right != right_queue_.end(); ++right) {
        const int64_t delta_ns = std::llabs(stamp_ns((*left)->header.stamp) - stamp_ns((*right)->header.stamp));
        if (delta_ns < smallest_delta_ns) {
          smallest_delta_ns = delta_ns;
          best_left = left;
          best_right = right;
        }
      }
    }
    const double pair_delta_ms = smallest_delta_ns / 1e6;
    if (pair_delta_ms > max_pair_delta_ms_) {
      // The earlier front frame cannot match a later frame that has not arrived.
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
    if (left->width != static_cast<uint32_t>(expected_width_) || left->height != static_cast<uint32_t>(expected_height_) ||
      right->width != static_cast<uint32_t>(expected_width_) || right->height != static_cast<uint32_t>(expected_height_)) {
      publish_status("rejected_resolution", 0, 0, pair_delta_ms);
      return;
    }
    cv::Mat left_gray;
    cv::Mat right_gray;
    if (!image_to_mono8(left, &left_gray) || !image_to_mono8(right, &right_gray)) {
      publish_status("rejected_encoding", 0, 0, pair_delta_ms);
      return;
    }
    process_pair(left->header.stamp, left_gray, right_gray, pair_delta_ms);
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
    if (left_descriptors.empty() || right_descriptors.empty()) {
      return frame;
    }
    std::vector<std::vector<cv::DMatch>> candidates;
    matcher_->knnMatch(left_descriptors, right_descriptors, candidates, 2);
    std::vector<cv::Point2f> matched_left;
    std::vector<cv::Point2f> matched_right;
    std::vector<int> source_indices;
    for (const auto & candidate : candidates) {
      if (candidate.size() != 2 || candidate[0].distance >= 0.75F * candidate[1].distance) {
        continue;
      }
      const auto & left_keypoint = left_keypoints[candidate[0].queryIdx];
      const auto & right_keypoint = right_keypoints[candidate[0].trainIdx];
      // The accepted calibration has a predominantly vertical baseline; rectification
      // therefore aligns the x coordinate and carries disparity along y.
      if (std::abs(left_keypoint.pt.x - right_keypoint.pt.x) > stereo_epipolar_tolerance_px_) {
        continue;
      }
      matched_left.push_back(left_keypoint.pt);
      matched_right.push_back(right_keypoint.pt);
      source_indices.push_back(candidate[0].queryIdx);
    }
    if (matched_left.empty()) {
      return frame;
    }
    cv::Mat homogeneous;
    cv::triangulatePoints(p1_, p2_, matched_left, matched_right, homogeneous);
    for (int column = 0; column < homogeneous.cols; ++column) {
      const float w = homogeneous.at<float>(3, column);
      if (std::abs(w) < 1e-6F) {
        continue;
      }
      const cv::Point3f point(homogeneous.at<float>(0, column) / w,
        homogeneous.at<float>(1, column) / w, homogeneous.at<float>(2, column) / w);
      if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z) || point.z <= 0.05F || point.z > 15.0F) {
        continue;
      }
      frame.keypoints.push_back(left_keypoints[source_indices[column]]);
      frame.points3d.push_back(point);
      frame.descriptors.push_back(left_descriptors.row(source_indices[column]));
    }
    return frame;
  }

  void process_pair(const builtin_interfaces::msg::Time & stamp, const cv::Mat & left, const cv::Mat & right,
    double pair_delta_ms)
  {
    Frame current = make_stereo_frame(left, right);
    if (static_cast<int>(current.points3d.size()) < min_stereo_points_) {
      ++tracking_failures_;
      publish_status("insufficient_stereo_features", static_cast<int>(current.points3d.size()), 0, pair_delta_ms);
      return;
    }
    if (!previous_) {
      previous_ = std::move(current);
      publish_status("initialized", static_cast<int>(previous_->points3d.size()), 0, pair_delta_ms);
      return;
    }
    std::vector<cv::DMatch> matches;
    matcher_->match(previous_->descriptors, current.descriptors, matches);
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    for (const auto & match : matches) {
      if (match.distance > temporal_match_distance_) {
        continue;
      }
      object_points.push_back(previous_->points3d[match.queryIdx]);
      image_points.push_back(current.keypoints[match.trainIdx].pt);
    }
    cv::Mat rotation_vector;
    cv::Mat translation_vector;
    cv::Mat inliers;
    const bool pnp_ok = object_points.size() >= 6 && cv::solvePnPRansac(object_points, image_points, rectified_k_,
      cv::Mat(), rotation_vector, translation_vector, false, 100, 2.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
    const int inlier_count = pnp_ok ? inliers.rows : 0;
    if (!pnp_ok || inlier_count < min_pnp_inliers_) {
      ++tracking_failures_;
      previous_ = std::move(current);
      publish_status("tracking_lost", static_cast<int>(previous_->points3d.size()), inlier_count, pair_delta_ms);
      return;
    }
    cv::Mat rotation_matrix;
    cv::Rodrigues(rotation_vector, rotation_matrix);
    cv::Mat current_T_previous = cv::Mat::eye(4, 4, CV_64F);
    rotation_matrix.copyTo(current_T_previous(cv::Rect(0, 0, 3, 3)));
    translation_vector.copyTo(current_T_previous(cv::Rect(3, 0, 1, 3)));
    world_T_current_ = world_T_current_ * current_T_previous.inv();
    previous_ = std::move(current);
    publish_odometry(stamp, inlier_count);
    publish_status("tracking", static_cast<int>(previous_->points3d.size()), inlier_count, pair_delta_ms);
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

  void publish_status(const std::string & state, int stereo_points, int pnp_inliers, double pair_delta_ms)
  {
    std_msgs::msg::String status;
    std::ostringstream data;
    data << "{\"state\":\"" << state << "\",\"stereo_points\":" << stereo_points
         << ",\"pnp_inliers\":" << pnp_inliers << ",\"pair_delta_ms\":" << pair_delta_ms
         << ",\"rejected_pairs\":" << rejected_pairs_ << ",\"tracking_failures\":" << tracking_failures_ << "}";
    status.data = data.str();
    status_publisher_->publish(status);
  }

  std::string left_topic_;
  std::string right_topic_;
  std::string calibration_file_;
  std::string odom_frame_;
  std::string camera_frame_;
  double max_pair_delta_ms_{};
  int expected_width_{};
  int expected_height_{};
  double processing_scale_{};
  int max_features_{};
  double stereo_epipolar_tolerance_px_{};
  int temporal_match_distance_{};
  int min_stereo_points_{};
  int min_pnp_inliers_{};
  int rejected_pairs_{};
  int tracking_failures_{};
  cv::Mat k1_, d1_, k2_, d2_, rotation_, translation_, r1_, r2_, p1_, p2_, q_, rectified_k_;
  cv::Mat left_map_x_, left_map_y_, right_map_x_, right_map_y_;
  cv::Size processing_size_;
  cv::Mat world_T_current_;
  cv::Ptr<cv::ORB> orb_;
  cv::Ptr<cv::BFMatcher> matcher_;
  std::optional<Frame> previous_;
  std::deque<sensor_msgs::msg::Image::ConstSharedPtr> left_queue_;
  std::deque<sensor_msgs::msg::Image::ConstSharedPtr> right_queue_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr right_subscription_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_publisher_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StereoVoCanary>());
  rclcpp::shutdown();
  return 0;
}
