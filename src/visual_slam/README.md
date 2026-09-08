# visual_slam canary

`stereo_vo_canary` is an isolated calibrated stereo visual-odometry probe. It
subscribes to the native left/right mono8 streams, rejects pairs over the
timestamp tolerance, rectifies with the accepted calibration, triangulates ORB
features, and estimates temporal motion with PnP RANSAC.

It always subscribes to the calibrated native 1088x1280 streams. The default
`processing_scale: 0.25` creates 272x320 rectification maps and scales both
projection matrices before triangulation, reducing CPU load without using an
inconsistent camera model.

It publishes only:

- `visual_slam/odometry` (`nav_msgs/Odometry`, frame `visual_odom`)
- `visual_slam/status` (`std_msgs/String`)

For the Gazebo experiment, use the separate `stereo_vo_sim` executable and
`stereo_vo_sim.launch.py`. It subscribes to `/sim/left/image_raw`,
`/sim/right/image_raw` and both `/sim/*/camera_info` topics, so it does not
reuse the device calibration file or the device node defaults. It publishes
`/visual_slam/odometry_sim` and `/visual_slam/status_sim`, and uses a horizontal
70 mm stereo baseline with ORB, epipolar filtering, triangulation and PnP.

## Submap experiment

`stereo_vo_sim` supports the three fixed experimental modes used by the paper:

- `global`: all keyframes and map points are matched as one global map (G);
- `fixed`: the active submap and its neighbors are matched (S);
- `joint`: a lightweight visual, VO-pose and topology index selects candidates,
  followed by local PnP/RANSAC validation (P).

The default experiment uses 5 m submaps, 1.5 m overlap, three candidates, and
five-frame pose transition smoothing. The parameters are in
`config/stereo_vo_sim.yaml`; freeze them after development-sequence tuning.

Build and run in WSL:

```bash
source /opt/ros/humble/setup.bash
cd /mnt/d/AIview/ROS2-clearRob
colcon build --symlink-install --packages-select visual_slam_gazebo visual_slam
source install/setup.bash
ros2 launch visual_slam stereo_vo_sim.launch.py submap_mode:=joint
```

Use `submap_mode:=global`, `fixed`, or `joint` for G, S, or P. The status JSON
on `/visual_slam/status_sim` includes the active/candidate submaps, matching
time, active keyframes/map points, transition state, and switch counters.
`/sim/odom` is simulation truth for evaluation only and is never consumed by
the VO or submap manager.

It does not broadcast TF, publish a map, issue motion commands, alter S90, or
install a service. `visual_odom -> base_link` remains reserved for the future
EKF, and `map -> odom` for the future mapping backend.

The package requires a device-side OpenCV development configuration. Verify
that before building; do not install dependencies as part of this canary. On
the current device, use the already deployed private copy instead of a system
install:

```bash
colcon build --packages-select visual_slam \
  --cmake-args -DVISUAL_SLAM_OPENCV_ROOT=/root/stereonet_test_20260828_device/thirdparty/opencv
```

This is a future manual canary build command, not an instruction to deploy the
package as a service.

The current ROS image omits `spdlogConfig.cmake` even though its headers and
runtime library exist. The package carries a build-local compatibility target
under `cmake/compat`; it must be supplied through `spdlog_DIR` together with
`CMAKE_LIBRARY_PATH=/opt/ros/humble/lib` when building directly with CMake.

It also lacks Python `catkin_pkg`, so the first board test uses
`-DVISUAL_SLAM_STANDALONE=ON`. That omits only `ament_package()` metadata
generation; it does not omit ROS subscriptions, parameters, or messages. Run
the installed binary directly for this canary rather than registering it as a
ROS package or a service.

The CMake target links only the C++ ROS typesupport libraries. This is
intentional: the board's message-package exports reference a missing unversioned
Python development library even though this C++ node has no Python dependency.

The board's private OpenCV directory has invalid unversioned `.so` placeholder
files. The CMake logic deliberately resolves the versioned `libopencv_*.so.*`
files, which are the actual shared libraries used by the existing StereoNet run.

For the standalone board build only, CMake creates a compatibility link under
its own build directory from the installed `libpython3.10.so.1` to the name
required by the ROS export. It never writes to `/usr/lib`.
