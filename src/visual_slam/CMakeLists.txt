cmake_minimum_required(VERSION 3.8)
project(visual_slam)

option(VISUAL_SLAM_STANDALONE
  "Build the manual canary when the board image lacks ament's Python package metadata dependency" OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
include(CTest)

set(VISUAL_SLAM_OPENCV_ROOT "" CACHE PATH
  "Private OpenCV root with include/opencv4 and lib (used when no system OpenCV CMake package exists)")
if(VISUAL_SLAM_OPENCV_ROOT)
  set(OpenCV_INCLUDE_DIRS "${VISUAL_SLAM_OPENCV_ROOT}/include/opencv4")
  set(OpenCV_LIBS)
  foreach(component core imgproc calib3d features2d video)
    file(GLOB opencv_component_candidates
      "${VISUAL_SLAM_OPENCV_ROOT}/lib/libopencv_${component}.so.[0-9]*")
    list(SORT opencv_component_candidates)
    list(GET opencv_component_candidates 0 opencv_component_library)
    list(APPEND OpenCV_LIBS "${opencv_component_library}")
  endforeach()
  foreach(required_path ${OpenCV_INCLUDE_DIRS} ${OpenCV_LIBS})
    if(NOT EXISTS "${required_path}")
      message(FATAL_ERROR "VISUAL_SLAM_OPENCV_ROOT is incomplete: ${required_path}")
    endif()
  endforeach()
else()
  find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d features2d video)
endif()

add_executable(stereo_vo_canary src/stereo_vo_canary.cpp)
target_compile_features(stereo_vo_canary PUBLIC cxx_std_17)
target_include_directories(stereo_vo_canary PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(stereo_vo_canary PRIVATE
  ${OpenCV_LIBS}
  rclcpp::rclcpp
  sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
  nav_msgs::nav_msgs__rosidl_typesupport_cpp
  geometry_msgs::geometry_msgs__rosidl_typesupport_cpp
  std_msgs::std_msgs__rosidl_typesupport_cpp)
if(VISUAL_SLAM_STANDALONE AND EXISTS "/usr/lib/aarch64-linux-gnu/libpython3.10.so.1")
  set(VISUAL_SLAM_COMPAT_LIB_DIR "${CMAKE_BINARY_DIR}/compat/lib")
  file(MAKE_DIRECTORY "${VISUAL_SLAM_COMPAT_LIB_DIR}")
  execute_process(COMMAND "${CMAKE_COMMAND}" -E create_symlink
    "/usr/lib/aarch64-linux-gnu/libpython3.10.so.1"
    "${VISUAL_SLAM_COMPAT_LIB_DIR}/libpython3.10.so")
  target_link_directories(stereo_vo_canary BEFORE PRIVATE "${VISUAL_SLAM_COMPAT_LIB_DIR}")
endif()
if(VISUAL_SLAM_OPENCV_ROOT)
  set_target_properties(stereo_vo_canary PROPERTIES
    BUILD_RPATH "${VISUAL_SLAM_OPENCV_ROOT}/lib"
    INSTALL_RPATH "${VISUAL_SLAM_OPENCV_ROOT}/lib")
endif()

add_executable(stereo_vo_sim src/stereo_vo_sim.cpp)
target_compile_features(stereo_vo_sim PUBLIC cxx_std_17)
target_include_directories(stereo_vo_sim PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(stereo_vo_sim PRIVATE
  ${OpenCV_LIBS}
  rclcpp::rclcpp
  sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
  nav_msgs::nav_msgs__rosidl_typesupport_cpp
  geometry_msgs::geometry_msgs__rosidl_typesupport_cpp
  std_msgs::std_msgs__rosidl_typesupport_cpp
  yaml-cpp)
if(VISUAL_SLAM_OPENCV_ROOT)
  set_target_properties(stereo_vo_sim PROPERTIES
    BUILD_RPATH "${VISUAL_SLAM_OPENCV_ROOT}/lib"
    INSTALL_RPATH "${VISUAL_SLAM_OPENCV_ROOT}/lib")
endif()

install(TARGETS stereo_vo_canary stereo_vo_sim DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
if(BUILD_TESTING)
  add_executable(geometry_test test/geometry_test.cpp)
  target_compile_features(geometry_test PRIVATE cxx_std_17)
  target_include_directories(geometry_test PRIVATE include ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(geometry_test PRIVATE ${OpenCV_LIBS})
  add_test(NAME geometry_test COMMAND geometry_test)

  add_executable(observability_test test/observability_test.cpp)
  target_compile_features(observability_test PRIVATE cxx_std_17)
  target_include_directories(observability_test PRIVATE include)
  add_test(NAME observability_test COMMAND observability_test)
endif()
if(NOT VISUAL_SLAM_STANDALONE)
  ament_package()
endif()
