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
  return 0;
}
