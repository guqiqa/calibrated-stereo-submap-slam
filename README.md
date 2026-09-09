# 民航视觉 SLAM 子地图算力优化研究代码

本目录是从 `D:\AIview\ROS2-clearRob` 复制出的研究代码备份，当前版本包含：

- `src/visual_slam`：双目视觉里程计、ORB、三角化、PnP/RANSAC、G/S/P 子地图模式；
- `src/visual_slam_gazebo`：48 m 纹理走廊、双目机器人、视差检查和自动实验器；
- `experiments`：已完成的 G/S/P 短测和路线验证数据；
- `参数可视化图表.html`：固定参数、处理链路、实验分组和短测结果图表；
- `民航视觉SLAM主线小论文实验流程-子地图修订版.md`：实验计划。

关键固定参数：640 x 480、15 Hz、70 mm 双目基线、处理缩放 0.75、子地图长度 5.0 m、重叠 1.5 m、Top-3 候选、局部窗口 15 帧。

后续仿真仍在 `D:\AIview\ROS2-clearRob` 的 ROS 2 工作区编译和运行，`D:\科研slam` 用作独立代码备份与研究材料归档。

## 当前基线

- Git 分支：`main`
- 基线标签：`baseline-pure-stereo`
- 图像：`640 x 480`，约 `15 Hz`
- 双目水平基线：`70 mm`
- 子地图长度：`5.0 m`，重叠：`1.5 m`
- 子地图模式：`global`、`fixed`、`joint`

当前版本保持纯双目视觉方案，不接入 IMU、惯导或其他传感器。`/sim/odom` 只用于仿真真值记录和离线评测，不输入视觉算法，也不参与子地图切换决策。

## 目录说明

```text
D:\科研slam
├── src/                                  # ROS 2 源码、Gazebo 场景和脚本
├── experiments/                          # 实验摘要、参数和统计结果
├── 参数可视化图表.html                    # 参数与实验分组可视化
└── 民航视觉SLAM主线小论文实验流程-子地图修订版.md
```

`build/`、`install/`、`log/` 是 ROS 2/colcon 生成目录，已由 `.gitignore` 排除。逐帧状态 JSONL、视觉/真值轨迹 CSV 和运行日志体积较大，也默认只保存在本机；实验摘要、参数快照和分段指标会纳入版本库。

## 环境与构建

推荐在 WSL2 中使用 Ubuntu 22.04、ROS 2 Humble、Gazebo Classic 11、`gazebo_ros`、OpenCV 和 C++17。Windows 路径 `D:\科研slam` 在 WSL 中对应 `/mnt/d/科研slam`。

```bash
cd /mnt/d/科研slam
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select visual_slam visual_slam_gazebo
source install/setup.bash
```

如需在原 ROS 2 工作区运行，可将上述路径替换为 `/mnt/d/AIview/ROS2-clearRob`。

## 启动仿真和 VO

启动最大走廊场景：

```bash
source /opt/ros/humble/setup.bash
source /mnt/d/科研slam/install/setup.bash
ros2 launch visual_slam_gazebo stereo_terminal.launch.py
```

无 GUI 模式：

```bash
ros2 launch visual_slam_gazebo stereo_terminal.launch.py gui:=false
```

另开终端启动纯双目 VO：

```bash
source /opt/ros/humble/setup.bash
source /mnt/d/科研slam/install/setup.bash
ros2 launch visual_slam stereo_vo_sim.launch.py
```

输出话题为 `/visual_slam/odometry_sim` 和 `/visual_slam/status_sim`。节点不会发布 TF，也不会控制机器人。Gazebo 场景的图像话题包括 `/sim/left/image_raw`、`/sim/right/image_raw`、`/sim/left/camera_info` 和 `/sim/right/camera_info`。

## 自动实验

统一入口负责仿真重置、速度发布、实验选择、结果保存和分段统计：

```bash
ros2 run visual_slam_gazebo run_experiment_profile.py --list-profiles
ros2 run visual_slam_gazebo run_experiment_profile.py --profile main_45m
```

高负载路线：

```bash
ros2 run visual_slam_gazebo run_experiment_profile.py --profile highload_60m
ros2 run visual_slam_gazebo run_experiment_profile.py --profile highload_90m
ros2 run visual_slam_gazebo run_experiment_profile.py --profile highload_120m
```

调试时可缩短实验：

```bash
ros2 run visual_slam_gazebo run_experiment_profile.py \
  --profile main_45m --modes global --runs 1 --duration 10 \
  --output-root /mnt/d/科研slam/experiments/script_smoke
```

每次运行会先发送零速度、重置仿真、解除暂停，再启动对应 VO 模式并记录结果。每个实验目录保存 `profile.json`、`campaign_state.json`、`metadata.json`、`resolved_parameters.yaml`、`summary.json` 和分段指标。

分段评测：

```bash
ros2 run visual_slam_gazebo analyze_route_segments.py \
  /mnt/d/科研slam/experiments/main_45m_gsp/joint/run_01 \
  --segment-size 5.0 --max-position 45.0
```

## 视差检查

```bash
ros2 run visual_slam_gazebo stereo_disparity_check
```

该节点只配对左右图像并统计 SGBM 视差，不发布位姿。深度估算关系为 `Z = fx * B / disparity`，其中 `B = 0.07 m`。

## Git 工作流

```bash
cd /mnt/d/科研slam
git status
git log --oneline --decorate -5
```

完成一个可独立验证的改动后提交：

```bash
git add src/visual_slam src/visual_slam_gazebo README.md
git commit -m "fix: describe the change"
```

实验结果建议只提交轻量文件：

```bash
git add experiments/<experiment-name>/profile.json \
  experiments/<experiment-name>/campaign_state.json \
  experiments/<experiment-name>/**/metadata.json \
  experiments/<experiment-name>/**/resolved_parameters.yaml \
  experiments/<experiment-name>/**/summary.json \
  experiments/<experiment-name>/**/segment_metrics.csv \
  experiments/<experiment-name>/**/segment_report.json
git commit -m "实验: record <experiment-name> results"
```

建议后续按功能建立分支：

```bash
git switch -c fix/vo-scale-and-pose
```

每次正式实验前冻结代码提交和参数，并在实验记录中保留对应提交哈希。

## 基础建图真实性修复

当前 `fix/metric-stereo-map` 分支先完善纯双目建图，再开展子地图检索实验。VO 内部使用初始左目光学坐标系，地图导出同时提供 Gazebo world 对齐坐标：

```text
Gazebo X = optical Z + 0.08
Gazebo Y = -optical X + 0.035
Gazebo Z = -optical Y + 0.32
```

视觉运动估计以连续帧 3D-3D 刚体 RANSAC 为主，PnP 作为无可靠 3D-3D 对应时的回退；两者不一致时拒绝更新当前位姿。新增视差、深度、重投影、RANSAC 内点和运动一致性诊断，`/sim/odom` 仍只用于离线评测。

```bash
ros2 run visual_slam_gazebo run_experiment_profile.py \
  --profile main_45m --modes global --runs 1 --duration 20 \
  --output-root /mnt/d/科研slam/experiments/basic_map_smoke \
  --gui --record
```

启用地图导出后，每个运行目录会生成 `map_points.csv`、`raw_map_points.csv`、`stable_map_points.csv`、`keyframes.csv` 和 `trajectory_visual.csv`。后三类坐标同时包含初始光学坐标和 Gazebo 对齐坐标；`stable_map_points.csv` 只保留达到最小观测次数的融合点。往返实验摘要同时报告路径长度比例和闭合误差，避免用路线跨度错误评价往返路程。

当前版本增加轻量视觉回环检查：定期将当前帧与早期关键帧进行描述子匹配和 PnP 几何验证，只有达到匹配数、内点数和重投影误差门限才允许平滑位姿修正。回环候选与校正次数会写入状态 JSON；该模块仍只使用双目图像。

## 当前限制与下一步

现有 30 m 结果表明视觉轨迹长度明显小于仿真真值，说明 VO 仍存在尺度偏差和累计位姿误差，地图点也可能出现离群。当前不进入 IMU 或其他传感器融合；下一步优先修正双目三角化尺度、PnP 位姿方向和 VO 累计更新，再重新评估 45 m 及更长路线下 G/S/P 的算力收益。
