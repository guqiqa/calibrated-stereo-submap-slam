cmake_minimum_required(VERSION 3.8)
project(visual_slam_gazebo)

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

install(
  DIRECTORY launch worlds models urdf
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/check_stereo_topics.sh scripts/run_submap_experiment.py
           scripts/analyze_route_segments.py scripts/run_experiment_profile.py
           scripts/calibrate_sim_speed.py scripts/plot_map_alignment.py
           scripts/start_campaign_detached.sh scripts/submap_retrieval_protocol.py
           scripts/calibration_metrics.py scripts/evaluate_retrieval_protocol.py
           scripts/test_retrieval_protocol.py scripts/run_protocol_fixture.py
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(stereo_disparity_check src/stereo_disparity_check.cpp)
target_compile_features(stereo_disparity_check PUBLIC cxx_std_17)
target_include_directories(stereo_disparity_check PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(stereo_disparity_check ${OpenCV_LIBS})
ament_target_dependencies(stereo_disparity_check rclcpp sensor_msgs)
install(TARGETS stereo_disparity_check DESTINATION lib/${PROJECT_NAME})

ament_package()
