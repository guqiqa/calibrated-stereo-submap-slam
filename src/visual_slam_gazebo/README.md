# visual_slam_gazebo

独立的 ROS 2 Humble + Gazebo Classic 基础双目视觉 SLAM 仿真包。

## 构建

```bash
cd /mnt/d/AIview/ROS2-clearRob
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select visual_slam_gazebo
source install/setup.bash
```

运行前提：WSL 中已安装 ROS 2 Humble、Gazebo Classic 11 和 `gazebo_ros`。启动文件会自动设置 Gazebo ROS 插件路径。双目机器人由 world 直接加载，基础模式关闭不必要的 `/spawn_entity` 动态生成插件。

## 启动

```bash
ros2 launch visual_slam_gazebo stereo_terminal.launch.py
```

无 GUI 启动：

```bash
ros2 launch visual_slam_gazebo stereo_terminal.launch.py gui:=false
```

## 主要话题

- `/sim/left/image_raw`
- `/sim/right/image_raw`
- `/sim/left/camera_info`
- `/sim/right/camera_info`
- `/sim/cmd_vel`
- `/sim/odom`
- `/clock`

场景现在是约 `136 m` 的最大室内走廊：`0～120 m` 可作为最长有效路线，`120～132 m` 为末端缓冲区，端墙位于约 `131.8 m`。当前论文主实验截取 `0～45 m`，其余 `60/90/120 m` 路线在同一场景中按位置截取，避免重复修改和重新加载不同几何场景。左右墙使用两套可重复平铺的 PPM 纹理，并叠加非周期彩色面板、门框、变尺寸立柱和侧向行李车，覆盖整个120 m有效路线，避免主路线末端直接看到端墙。直行 SLAM 基线中行李车仅保留可见模型，用作视觉遮挡，不加入物理碰撞，避免无避障控制的机器人在中途停止。左右相机基线固定为 `70 mm`；右目插件通过 `<hack_baseline>0.07</hack_baseline>` 将该基线写入右目 `CameraInfo.P[3]`，用于验证双目三角测量接口。机器人使用 `gazebo_ros_planar_move` 进行平面运动，默认不发布 odom TF，避免与后续 EKF 冲突。

## 控制机器人

前进 2 秒：

```bash
timeout 2s ros2 topic pub -r 20 /sim/cmd_vel geometry_msgs/msg/Twist \
  '{linear: {x: 0.3}, angular: {z: 0.0}}'
```

逆时针旋转 2 秒：

```bash
timeout 2s ros2 topic pub -r 20 /sim/cmd_vel geometry_msgs/msg/Twist \
  '{linear: {x: 0.0}, angular: {z: 0.4}}'
```

查看仿真里程计：

```bash
ros2 topic echo /sim/odom
```

停止运动：

```bash
ros2 topic pub --once /sim/cmd_vel geometry_msgs/msg/Twist \
  '{linear: {x: 0.0}, angular: {z: 0.0}}'
```

## G/S/P 自动实验

保持 Gazebo GUI 运行，在另一个终端执行：

```bash
source /opt/ros/humble/setup.bash
source /mnt/d/AIview/ROS2-clearRob/install/setup.bash
ros2 run visual_slam_gazebo run_submap_experiment.py \
  --modes global fixed joint --runs 10 --duration 20 \
  --speed 0.2 --angular 0.0 --output /mnt/d/AIview/ROS2-clearRob/experiments/submap
```

45 m主实验建议使用：

```bash
ros2 run visual_slam_gazebo run_submap_experiment.py \
  --modes global fixed joint --runs 10 --duration 320 \
  --speed 0.2 --angular 0.0 \
  --output /mnt/d/AIview/ROS2-clearRob/experiments/main_45m_gsp
```

主性能统计按真实位置 `0～45 m` 截取：

```bash
ros2 run visual_slam_gazebo analyze_route_segments.py \
  /mnt/d/AIview/ROS2-clearRob/experiments/main_45m_gsp/joint/run_01 \
  --segment-size 5.0 --max-position 45.0
```

也可以使用统一的规模实验入口切换路线。该入口会检查 `gzserver` 进程和
`/reset_world`、`/unpause_physics` 服务，需要时自动启动 Gazebo，并将 Gazebo
进程放在独立进程组中，避免当前终端接管 GUI 键盘事件：

```bash
ros2 run visual_slam_gazebo run_experiment_profile.py --profile main_45m
ros2 run visual_slam_gazebo run_experiment_profile.py --profile highload_60m
ros2 run visual_slam_gazebo run_experiment_profile.py --profile highload_90m
ros2 run visual_slam_gazebo run_experiment_profile.py --profile highload_120m
```

也可以通过独立后台脚本运行，日志和 PID 会保存到实验目录，不占用当前终端：

```bash
ros2 run visual_slam_gazebo start_campaign_detached.sh \
  /mnt/d/科研slam/experiments/main_45m_gsp main_45m 10
```

可用下面的命令查看所有实验选择：

```bash
ros2 run visual_slam_gazebo run_experiment_profile.py --list-profiles
```

入口会调用现有 G/S/P 记录器，并在实验完成后按配置的有效路线长度自动生成 5 m
分段报告。每次运行会在输出目录写入 `profile.json` 和 `campaign_state.json`，每个
`mode/run_XX` 目录保存原始状态、双目视觉里程计、仿真真值、VO 日志、摘要、参数
和分段指标。可用 `--runs`、`--modes` 和 `--output-root` 覆盖重复次数、实验组和
输出位置。

入口使用 `/tmp/visual_slam_gazebo_experiment.lock` 防止两个 campaign 并发启动。
如果已存在一个 `gzserver` 但 ROS 服务尚未发现，脚本会等待该实例，不会
启动第二个。如果启动前或启动过程中发现多个 `gzserver`，实验会立即中止，
保留日志并等待人工确认残留进程，不会自动误杀其他实验。

常用控制选项：

```bash
# 使用已经打开的 Gazebo，不自动启动新实例
ros2 run visual_slam_gazebo run_experiment_profile.py \
  --profile main_45m --no-launch-gazebo

# 无 GUI 运行，适合无 WSLg 环境
ros2 run visual_slam_gazebo run_experiment_profile.py \
  --profile main_45m --no-gui

# 实验完成后同时关闭本次脚本启动的 Gazebo
ros2 run visual_slam_gazebo run_experiment_profile.py \
  --profile main_45m --shutdown-gazebo
```

联调时可以临时缩短单次时长；正式实验不传该参数，使用 profile 中的固定时长：

```bash
ros2 run visual_slam_gazebo run_experiment_profile.py \
  --profile main_45m --modes global --runs 1 --duration 10 \
  --output-root /mnt/d/科研slam/experiments/script_smoke
```

每次实验会先发送零速度、重置 `/reset_world`、解暂停并确认机器人回到原点，再启动
对应 `submap_mode` 的 VO，发布固定速度，并保存 `status.jsonl`、视觉里程计、
`/sim/odom` 真值、VO 日志、摘要和实际生效的 `resolved_parameters.yaml`。
`metadata.json` 会保留 70 mm 基线、5 m 子地图、1.5 m 重叠、3 个候选、
15 帧局部窗口、3 帧切换确认和 5 帧平滑等关键实验参数。

## 分段评测

新的实验记录会在视觉里程计和仿真真值 CSV 中保存 ROS 接收时间。该时间仅用于将
状态 JSON 离线对齐到 `/sim/odom`，不输入视觉算法。按 5 m 分段统计有效路线：

```bash
ros2 run visual_slam_gazebo analyze_route_segments.py \
  /mnt/d/AIview/ROS2-clearRob/experiments/valid_route_gsp_v2/joint/run_01 \
  --segment-size 5.0 --max-position 40.0
```

脚本在 run 目录生成 `segment_metrics.csv` 和 `segment_report.json`，其中包含各区间的
tracking/失败比例、活动地图点、匹配耗时、处理耗时和切换数。

以 `0.2 m/s` 指令速度进行 45 m 直行时，Gazebo 实际速度约为 `0.145 m/s`，建议
将 `--duration` 设为 `320` 秒，并在离线评测中截取 `0～45 m` 主实验路线；最大场景的后续路段可继续用于60/90/120 m高负载验证。短测可使用 `60` 秒验证首个子地图切换。正式主实验不应把端墙退化段纳入性能统计，端墙区域单独作为低纹理/低视差退化工况。

## 场景规模扩展

当前136 m最大场景统一服务于45/60/90/120 m路线。后续扩展只改变实验配置中的 `route_m`、运行时长和评测截断位置，不再修改场景几何；每条有效路线末端至少保留约10～12 m缓冲区。扩展目标是验证地图规模增长后，子地图卸载和联合检索是否出现算力收益；不同路线长度必须分组统计，不能混合成一个平均值。

## 基础检查

```bash
ros2 run visual_slam_gazebo check_stereo_topics.sh
ros2 topic hz /sim/left/image_raw
ros2 topic hz /sim/right/image_raw
ros2 topic echo /sim/left/camera_info --once
ros2 topic echo /tf --once
```

## 连续图像和视差检查

启动仿真后运行：

```bash
ros2 run visual_slam_gazebo stereo_disparity_check
```

该节点按时间戳配对左右图像，使用 OpenCV SGBM 计算水平视差，仅输出检查统计，不发布 SLAM 位姿。默认参数为 `B=0.07 m`、`fx=554.3827 px`，深度估算使用 `Z=fx*B/disparity`。

验证结果：左右图像连续配对约 `15 Hz`，时间差约 `0 ms`；静止时中位视差约 `6 px`、估算深度约 `6.47 m`。机器人以 `0.3 m/s` 前进后，中位视差曾达到 `18、24、34.75 px`，估算深度约为 `2.16、1.62、1.12 m`，说明运动产生了可测视差。接近场景边界后有效视差会下降，后续需要增加纹理和延长可运动路线。

无显示环境优先使用 `gui:=false`；WSLg 可用时再使用默认 GUI 模式。

## 仿真视觉里程计

### 基础建图真实性检查

当前基础建图分支使用连续帧 3D-3D 刚体 RANSAC 作为主要运动估计，PnP 仅在 3D-3D 匹配不足时回退。状态 JSON 增加视差、深度、双目重投影、RANSAC 内点和两种运动估计一致性字段。地图导出使用统一的初始左目光学坐标到 Gazebo world 坐标变换，避免轨迹和地图采用不同的显示坐标。

每个运行目录可获得：

- `raw_map_points.csv`：逐关键帧原始三维点；
- `stable_map_points.csv`：经过体素融合且达到观测次数门限的点；
- `keyframes.csv`：关键帧位姿；
- `trajectory_visual.csv`：视觉轨迹及其 Gazebo 对齐坐标。

基础 45 m 冒烟测试：

```bash
ros2 run visual_slam_gazebo run_experiment_profile.py \
  --profile main_45m --modes global --runs 1 --duration 20 \
  --output-root /mnt/d/科研slam/experiments/basic_map_smoke \
  --gui --record
```

`--record` 会为每个运行保存双目图像、CameraInfo、`/sim/odom`、视觉里程计和状态的 rosbag；`--gui` 显式保持 Gazebo 可视化启动，`--no-gui` 可用于无 WSLg 环境。

往返实验的 `summary.json` 同时保存 `truth_path_distance_m`、`visual_path_distance_m`、`visual_truth_distance_ratio` 和 `visual_closure_error_m`。路径比例用于尺度评估，闭合误差用于评估累计漂移，不能用起终点跨度代替往返路径长度。

视觉回环检查默认每 15 帧运行一次，要求至少 35 个匹配、25 个 PnP 内点且重投影误差不超过 2.5 px；通过后使用 5 帧平滑校正。相关字段包括 `loop_closure_accepted`、`loop_closure_keyframe`、`loop_closure_inliers` 和 `loop_closure_count`。

另开终端启动独立仿真 VO：

```bash
source /opt/ros/humble/setup.bash
source /mnt/d/AIview/ROS2-clearRob/install/setup.bash
ros2 launch visual_slam stereo_vo_sim.launch.py
```

`stereo_vo_sim` 从 `/sim/left/camera_info` 和 `/sim/right/camera_info` 自动读取内参、分辨率和水平基线，处理 `rgb8` 双目图像，执行 ORB 特征、左右极线约束、三角化和 PnP RANSAC。输出为：

- `/visual_slam/odometry_sim` (`nav_msgs/Odometry`)
- `/visual_slam/status_sim` (`std_msgs/String`)

仿真真值 `/sim/odom` 只用于实验对比，不作为视觉算法输入；节点不发布 TF、不控制机器人。状态 JSON 包含双目三维点数、时序匹配数、PnP 内点数、图像时间差和处理耗时。

建议首先使用 `0.2~0.3 m/s` 直行验证连续跟踪，再加入小角速度形成平滑弧线。纯原地大角度旋转会让视场主要落在单侧墙面，双目有效点可能下降，应作为纹理退化和鲁棒性专项工况记录。

### 重复 campaign 汇总

对已完成的 G/S/P 重复实验，可使用离线汇总脚本：

```bash
python3 /mnt/d/科研slam/src/visual_slam_gazebo/scripts/aggregate_campaign.py \
  /mnt/d/科研slam/experiments/formal_gsp_45m_10x_20261008
```

脚本不修改原始 `run_XX` 数据，在 `analysis/` 生成 `campaign_runs.csv`、
`campaign_summary.json`、`campaign_metrics.png` 和 `campaign_repeatability.png`。默认通过条件为路径比例在 `0.98~1.02`、无跟踪失败和错误子地图切换。

60/90/120 m 多路线 campaign 可使用：

```bash
python3 /mnt/d/科研slam/src/visual_slam_gazebo/scripts/aggregate_highload_campaign.py \
  /mnt/d/科研slam/experiments/formal_highload_60_90_120m_20261009
```

该脚本在总目录 `analysis/` 生成跨路线 `highload_summary.csv`，
`highload_summary.json` 和 `highload_scaling.png`。单程实验的起终点距离不用作闭合误差；
120 m 必须另外报告末段失锁率，不应只看平均路径比例。
