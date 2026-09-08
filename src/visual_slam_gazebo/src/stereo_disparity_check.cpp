#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <deque>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

class StereoDisparityCheck : public rclcpp::Node
{
public:
  StereoDisparityCheck()
  : Node("stereo_disparity_check")
  {
    left_topic_ = declare_parameter<std::string>("left_topic", "/sim/left/image_raw");
    right_topic_ = declare_parameter<std::string>("right_topic", "/sim/right/image_raw");
    max_pair_delta_ms_ = declare_parameter<double>("max_pair_delta_ms", 10.0);
    baseline_m_ = declare_parameter<double>("baseline_m", 0.07);
    focal_length_px_ = declare_parameter<double>("focal_length_px", 554.3827);
    max_disparity_ = declare_parameter<int>("max_disparity", 96);

    auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    left_sub_ = create_subscription<sensor_msgs::msg::Image>(
      left_topic_, qos, std::bind(&StereoDisparityCheck::left_callback, this, std::placeholders::_1));
    right_sub_ = create_subscription<sensor_msgs::msg::Image>(
      right_topic_, qos, std::bind(&StereoDisparityCheck::right_callback, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(), "checking stereo topics left=%s right=%s baseline=%.3f m fx=%.2f px",
      left_topic_.c_str(), right_topic_.c_str(), baseline_m_, focal_length_px_);
  }

private:
  struct StampedImage
  {
    sensor_msgs::msg::Image::ConstSharedPtr message;
    rclcpp::Time stamp;
  };

  void left_callback(const sensor_msgs::msg::Image::ConstSharedPtr message)
  {
    std::vector<std::pair<StampedImage, StampedImage>> pairs;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      left_queue_.push_back({message, message->header.stamp});
      ++left_count_;
      pair_queues(pairs);
    }
    for (const auto & pair : pairs) {
      process_pair(pair.first, pair.second);
    }
  }

  void right_callback(const sensor_msgs::msg::Image::ConstSharedPtr message)
  {
    std::vector<std::pair<StampedImage, StampedImage>> pairs;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      right_queue_.push_back({message, message->header.stamp});
      ++right_count_;
      pair_queues(pairs);
    }
    for (const auto & pair : pairs) {
      process_pair(pair.first, pair.second);
    }
  }

  void pair_queues(std::vector<std::pair<StampedImage, StampedImage>> & pairs)
  {
    while (!left_queue_.empty() && !right_queue_.empty()) {
      const auto delta = left_queue_.front().stamp - right_queue_.front().stamp;
      const double delta_ms = std::abs(delta.nanoseconds()) / 1.0e6;
      if (delta_ms <= max_pair_delta_ms_) {
        auto left = left_queue_.front();
        auto right = right_queue_.front();
        left_queue_.pop_front();
        right_queue_.pop_front();
        ++paired_count_;
        pairs.emplace_back(std::move(left), std::move(right));
      } else if (left_queue_.front().stamp < right_queue_.front().stamp) {
        left_queue_.pop_front();
        ++unsynchronized_count_;
      } else {
        right_queue_.pop_front();
        ++unsynchronized_count_;
      }
    }
  }

  cv::Mat to_gray(const sensor_msgs::msg::Image & image) const
  {
    int channels = 1;
    int type = CV_8UC1;
    if (image.encoding == "rgb8" || image.encoding == "bgr8") {
      channels = 3;
      type = CV_8UC3;
    } else if (image.encoding != "mono8" && image.encoding != "8UC1") {
      throw std::runtime_error("unsupported image encoding: " + image.encoding);
    }
    const std::size_t expected = static_cast<std::size_t>(image.width) * image.height * channels;
    if (image.data.size() < expected || image.step < image.width * channels) {
      throw std::runtime_error("invalid image buffer or step");
    }
    cv::Mat source(
      static_cast<int>(image.height), static_cast<int>(image.width), type,
      const_cast<unsigned char *>(image.data.data()), image.step);
    cv::Mat gray;
    if (channels == 3) {
      const int conversion = image.encoding == "rgb8" ? cv::COLOR_RGB2GRAY : cv::COLOR_BGR2GRAY;
      cv::cvtColor(source, gray, conversion);
    } else {
      gray = source.clone();
    }
    return gray;
  }

  void process_pair(const StampedImage & left, const StampedImage & right)
  {
    try {
      const double delta_ms = std::abs((left.stamp - right.stamp).nanoseconds()) / 1.0e6;
      if (first_pair_wall_.time_since_epoch().count() == 0) {
        first_pair_wall_ = std::chrono::steady_clock::now();
      }
      const cv::Mat left_gray = to_gray(*left.message);
      const cv::Mat right_gray = to_gray(*right.message);
      if (left_gray.size() != right_gray.size()) {
        ++invalid_count_;
        return;
      }

      const int disparities = std::max(16, (max_disparity_ / 16) * 16);
      auto matcher = cv::StereoSGBM::create(
        0, disparities, 5, 8 * 1 * 5 * 5, 32 * 1 * 5 * 5, 1, 63, 10, 100, 32,
        cv::StereoSGBM::MODE_SGBM_3WAY);
      cv::Mat disparity_16;
      matcher->compute(left_gray, right_gray, disparity_16);

      std::vector<float> valid_disparities;
      valid_disparities.reserve(static_cast<std::size_t>(disparity_16.total() / 8));
      for (int y = 5; y < disparity_16.rows - 5; y += 2) {
        for (int x = 5; x < disparity_16.cols - 5; x += 2) {
          const float disparity = disparity_16.at<short>(y, x) / 16.0F;
          if (disparity > 1.0F && disparity < static_cast<float>(disparities)) {
            valid_disparities.push_back(disparity);
          }
        }
      }

      const double sample_count = static_cast<double>(
        ((disparity_16.rows - 10 + 1) / 2) * ((disparity_16.cols - 10 + 1) / 2));
      const double valid_ratio = sample_count > 0.0 ? valid_disparities.size() / sample_count : 0.0;
      float median_disparity = 0.0F;
      double median_depth = 0.0;
      if (!valid_disparities.empty()) {
        const auto middle = valid_disparities.begin() + valid_disparities.size() / 2;
        std::nth_element(valid_disparities.begin(), middle, valid_disparities.end());
        median_disparity = *middle;
        median_depth = focal_length_px_ * baseline_m_ / median_disparity;
      }

      const auto now = std::chrono::steady_clock::now();
      if (last_report_.time_since_epoch().count() == 0 || now - last_report_ >= std::chrono::seconds(2)) {
        const double pair_rate = elapsed_seconds() > 0.0 ? paired_count_ / elapsed_seconds() : 0.0;
        RCLCPP_INFO(
          get_logger(),
          "pairs=%zu left=%zu right=%zu unsync=%zu invalid=%zu pair_rate=%.2f Hz dt=%.2f ms "
          "valid_disparity=%.1f%% median_disp=%.2f px median_depth=%.2f m",
          paired_count_, left_count_, right_count_, unsynchronized_count_, invalid_count_, pair_rate,
          delta_ms, valid_ratio * 100.0, median_disparity, median_depth);
        last_report_ = now;
      }
    } catch (const std::exception & error) {
      ++invalid_count_;
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "failed to process stereo pair: %s", error.what());
    }
  }

  double elapsed_seconds() const
  {
    if (first_pair_wall_.time_since_epoch().count() == 0) {
      return 0.0;
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - first_pair_wall_).count();
  }

  std::string left_topic_;
  std::string right_topic_;
  double max_pair_delta_ms_{};
  double baseline_m_{};
  double focal_length_px_{};
  int max_disparity_{};
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr right_sub_;
  std::deque<StampedImage> left_queue_;
  std::deque<StampedImage> right_queue_;
  std::mutex mutex_;
  std::size_t left_count_{0};
  std::size_t right_count_{0};
  std::size_t paired_count_{0};
  std::size_t unsynchronized_count_{0};
  std::size_t invalid_count_{0};
  std::chrono::steady_clock::time_point first_pair_wall_{};
  std::chrono::steady_clock::time_point last_report_{};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StereoDisparityCheck>());
  rclcpp::shutdown();
  return 0;
}
