# 民航视觉 SLAM 子地图算力优化研究代码

本目录是从 `D:\AIview\ROS2-clearRob` 复制出的研究代码备份，当前版本包含：

- `src/visual_slam`：双目视觉里程计、ORB、三角化、PnP/RANSAC、C0/C1/C4/P 子地图模式；
- `src/visual_slam_gazebo`：154 m 纹理走廊、参数化大厅—双环地图、双目机器人和自动实验器；
- `experiments`：已完成的 G/S/P 短测和路线验证数据；
- `参数可视化图表.html`：固定参数、处理链路、实验分组和短测结果图表；
- `民航视觉SLAM主线小论文实验流程-子地图修订版.md`：实验计划。

关键固定参数：640 x 480、15 Hz、70 mm 双目基线、处理缩放 0.75、子地图长度 5.0 m、重叠 1.5 m、视觉/位姿/拓扑候选预算 4/3/2、总预算 10、局部窗口 15 帧。

后续仿真仍在 `D:\AIview\ROS2-clearRob` 的 ROS 2 工作区编译和运行，`D:\科研slam` 用作独立代码备份与研究材料归档。

## 当前基线

- Git 分支：`main`
- 基线标签：`baseline-pure-stereo`
- 图像：`640 x 480`，约 `15 Hz`
- 双目水平基线：`70 mm`
- 子地图长度：`5.0 m`，重叠：`1.5 m`
- 正式方法：`c0`、`c1`、`c4`、`p`；兼容旧模式：`global`、`fixed`、`joint`

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

## 参数化拓扑场景

生成大厅—双环—径向通道场景：

```bash
python3 src/visual_slam_gazebo/scripts/generate_topology_scene.py \
  --family perturbed_ring --seed 2103 \
  --output generated_scenes/perturbed_ring_2103 --validate --preview
```

当前 schema v2 使用通道 5 m、1.5 m 重叠，大厅 7 m 网格和 3 m 路口子地图。
先生成 1 个场景完成 smoke；通过后，前置验证才生成 6 个基础布局及 12 个障碍变体：

```bash
python3 src/visual_slam_gazebo/scripts/generate_topology_scene.py \
  --generate-dataset --output generated_scenes/topology_dataset_v1
```

正式实验需要显式生成 30 个基础布局及 60 个障碍变体：

```bash
python3 src/visual_slam_gazebo/scripts/generate_topology_scene.py \
  --generate-dataset --dataset-profile full \
  --output generated_scenes/topology_dataset_full_v1
```

详细格式、实验接入和验证命令见 `docs/topology_scene_generator.md`。
分类 query 共八条，分别覆盖内外环同向、内外环反向、跨环、跨大厅、冷启动和
none-of-the-above；空间与视角条件按任务类型和检索机会帧检查。

## 自动实验

统一项目入口负责项目切换、仿真重置、单实例保护、数据采集、基础分析和轻量归档：

```bash
python3 src/visual_slam_gazebo/scripts/experiment_manager.py --list-projects
python3 src/visual_slam_gazebo/scripts/experiment_manager.py \
  --project c1_c4_p_basic --methods c0 c1 c4 p \
  --runs 3 --gui --record --analyze --archive
```

旧 profile 入口继续兼容：

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

详细的冻结实验流程见 `docs/calibrated_submap_experiment_plan.md`。当前
`ood_rejection` 和 `multi_layout_calibration` 只登记配置且禁止启动，直到独立
mapping/query 数据生成与扰动注入实现；旧日志不会被补写伪造的覆盖率或校准指标。

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

可将 Gazebo 几何、稳定地图点、视觉轨迹和真值轨迹绘制到同一坐标系：

```bash
ros2 run visual_slam_gazebo plot_map_alignment.py \
  /mnt/d/科研slam/experiments/<campaign>/<mode>/run_01 \
  --world /mnt/d/科研slam/src/visual_slam_gazebo/worlds/terminal_basic.world \
  --route-length 45
```

输出文件为运行目录内的 `map_alignment.png`，上半部分检查俯视结构对齐，下半部分检查高度漂移和垂直离群点。

当前版本增加轻量视觉回环检查：定期将当前帧与早期关键帧进行描述子匹配和 PnP 几何验证，回环候选与校正次数会写入状态 JSON；该模块仍只使用双目图像。由于当前实现尚未包含位姿图优化，单独修改当前位姿会使历史关键帧和地图点产生断层，因此 `loop_closure_apply_correction` 默认关闭，回环仅作为诊断。完成地图级一致性优化前不得在正式基础建图实验中开启校正。

## 当前限制与下一步

早期 30 m 结果暴露了尺度偏差和累计位姿误差；当前纯双目修复版本的 45 m 往返 smoke test 已将路径比例改善到 `0.951～0.977`，且三组均无跟踪失败。环境几何叠加图仍显示明显的高度和俯仰累计漂移，因此基础地图尚未通过真实性验收。当前地面机器人仿真通过可关闭的 `planar_motion_constraint` 使用平面运动先验，不引入 IMU、真值或其他传感器；非平面运动数据必须关闭该参数。

## 45 m 子地图 Smoke Test 记录

实验目录：`experiments/submap_45m_smoke`。本次使用 `basic_map_45m_roundtrip`，路线为 45 m 往返，目标实际速度为 `0.2 m/s`；根据速度标定，控制指令为 `0.254522061 m/s`。三组均运行 1 次，跟踪失败数为 0。

| 模式 | 视觉/真值路径比例 | 闭合误差 (m) | 末帧活动地图点 | 平均匹配 (ms) | 平均处理 (ms) | 切换次数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G global | 0.951 | 0.109 | 30,951 | 1.84 | 9.88 | 0 |
| S fixed | 0.966 | 0.140 | 2,570 | 1.91 | 9.92 | 32 |
| P joint | 0.977 | 0.028 | 2,638 | 3.02 | 11.16 | 31 |

本次结果支持两个阶段性判断：固定子地图明显降低了活动地图点数量，且在当前 45 m 场景中平均处理时间基本没有变化；联合检索的平均匹配时间和处理时间反而上升，说明当前场景规模或检索实现尚未足以摊薄联合检索成本。三组均未发生连续跟踪丢失，子地图切换没有记录错误调用。

该结果是单次 smoke test，不作为最终统计结论。下一步先检查并优化联合检索的候选评分和局部匹配开销，再使用相同校准速度开展多次 45 m 实验；之后再扩展到 60/90/120 m，验证子地图收益是否随地图规模增长而显现。实验中的 `truth_distance_m` 约 45 m 是端点跨度，往返评价应使用 `truth_path_distance_m` 约 90 m。

### 平面约束复核

提交 `0f2a656` 增加了纯视觉地面机器人平面运动约束。`experiments/basic_map_planar_45m/global/run_01` 的 45 m 单向结果为：真值路径 `45.081 m`，视觉路径 `41.682 m`，路径比例 `0.925`，跟踪失败数为 0。稳定地图共 2,471 点，仅 2 点位于 `-0.1～3.1 m` 场景高度范围之外，越界率为 `0.08%`；视觉轨迹高度保持为相机初始高度 `0.32 m`，此前随里程增加的地下漂移已消除。

该结果说明坐标方向和高度累计问题已修复，但 PnP 平移仍存在约 `7.5%` 的尺度低估，横向也有小幅累计漂移。逐帧诊断中 `87.7%` 的运动来源为 `pnp_quantized_3d3d`，说明当前 70 mm 基线下远距离 3D-3D 深度量化较明显，主要运动量仍由 PnP 提供。后续先进行 45 m 往返验证，不使用 `/sim/odom` 在线修正；若尺度偏差可重复，再仅在开发序列上确定并冻结尺度校准参数。

开启平面约束但关闭回环位姿校正后的 12 m 往返测试位于 `experiments/basic_map_diagnostic_loop_smoke`：真值路径 `24.042 m`，视觉路径 `22.560 m`，比例 `0.938`，三维闭合误差 `0.345 m`，跟踪失败数为 0。共记录 35 个回环候选，但校正次数为 0，叠加图中的视觉轨迹保持连续。下一轮完整 45 m 往返基础验证沿用该配置。

完整无校正 45 m 往返结果位于 `experiments/basic_map_planar_45m_no_loop_correction`：真值路径 `90.139 m`，视觉路径 `83.516 m`，比例 `0.927`，三维闭合误差 `2.361 m`，跟踪失败数为 0。6,026 个稳定点中有 29 个超出场景高度范围，越界率 `0.48%`；轨迹最大帧间位移 `0.128 m`，未出现旧版回环校正造成的米级断层。该结果未通过最终闭合精度标准，主要剩余问题为 PnP 尺度低估和平面 yaw 累计漂移。

在进入 G/S/P 正式对比前，下一实现项为纯视觉 SE(2) 关键帧位姿图优化：将里程计相对位姿作为顺序边，将通过几何验证的回环作为约束边，联合优化关键帧平面位姿，并用关键帧修正量同步更新地图点。不得恢复仅修正当前帧的旧回环逻辑，也不得使用 `/sim/odom`、控制指令或其他传感器参与在线优化。

当前开发修正增加两个可关闭参数：`straight_motion_constraint` 对直线走廊的相对位姿清除横向/竖直增量，仅保留偏航和行进方向；`stereo_translation_scale_factor=1.08` 是根据前一轮开发序列的约 `0.927` 路径比例冻结的初始尺度校准值。该系数只适用于同一相机标定、图像缩放和前端配置，必须在独立开发序列上复核后才能用于论文正式结果，不能视为通用双目尺度修正。

尺度和直线偏移短测结果：`experiments/basic_map_scale_straight_forward` 的 12 m 前进实验，视觉/真值路径比例为 `1.002`；`experiments/basic_map_scale_straight_roundtrip` 的 12 m 往返实验，比例为 `1.004`，三维闭合误差为 `0.091 m`，两组均无跟踪失败。短测中视觉轨迹的 Gazebo 坐标范围为 `Y=0.035 m`、`Z=0.32 m`，说明前进阶段也存在的横向偏移趋势已被抑制。

该修正只适用于当前“保持朝向、沿走廊直线前进或倒退”的仿真工况。对于弧线、调头或真实非平面运动，必须关闭 `straight_motion_constraint`；对于不同双目基线、分辨率或标定参数，必须重新做开发序列尺度标定，不能直接沿用 `1.08`。

### 最新 45 m 完整往返复核（2026-10-08）

使用提交 `4735b1e` 重跑 `global` 基础实验，结果位于 `experiments/basic_map_scale_straight_45m_roundtrip_20261008`。真值路径为 `90.147 m`，视觉路径为 `89.596 m`，路径比例 `0.994`；三维闭合误差 `0.148 m`，跟踪失败数为 0，回环候选 139 帧且位姿校正次数为 0。视觉轨迹横向坐标保持 `Y=0.035 m`，高度保持 `Z=0.32 m`，最大帧间位移为 `0.113 m`，未发生轨迹跳变。

本次导出 7,700 个稳定地图点，其中 28 个超出 `-0.1～3.1 m` 场景高度范围，越界率 `0.36%`。VO 有效处理频率为 `9.64 Hz`，平均处理时间 `9.726 ms`，P95 为 `11.440 ms`；平均匹配时间 `1.678 ms`，P95 为 `2.616 ms`。该结果达到当前直线往返基础建图的尺度、连续性和闭合误差目标，可作为后续 G/S/P 子地图实验的新基础版本。

### 子地图主线预实验

`experiments/submap_45m_roundtrip_calibrated_smoke_20261008` 的单次 G/S/P 结果显示，固定子地图和联合检索将平均活动地图点分别降低 `85.9%` 和 `86.6%`。但是切换时直接应用局部 PnP 位姿，使 `fixed/joint` 最大帧间位移达到 `0.34/0.21 m`，并将路径比例虚增至 `1.028/1.018`；`global` 最大帧间位移仅 `0.11 m`。因此该批次只作为问题定位记录，不进入论文正式统计。

新增 `submap_apply_pose_correction=false`：子地图切换继续执行候选评分、PnP 几何验证、活动地图切换和卸载，但默认不直接修改连续 VO 位姿。与回环一致，只有完成关键帧位姿图和地图点一致性更新后，才能启用子地图重定位的位姿校正。

连续性修复后的完整 45 m G/S/P 复核位于 `experiments/submap_45m_roundtrip_continuous_20261008`。G/S/P 路径比例分别为 `0.995/0.994/0.994`，闭合误差为 `0.035/0.142/0.023 m`，三组均无跟踪失败和错误切换。S/P 平均活动地图点较 G 降低 `85.4%/86.3%`；S 的平均匹配耗时下降 `2.3%`，平均处理时间增加 `1.2%`；P 的平均匹配和处理时间仍分别增加 `79.0%/13.6%`。

为降低 P 组开销，联合检索改用单个全局代表描述子索引：当前帧只执行一次代表描述子匹配，并按子地图投票获得候选，再对 Top-3 执行局部 PnP，避免逐子地图调用 BFMatcher 以及 Top-3 视觉得分重复计算。12 m 往返短测 `experiments/joint_global_index_smoke_20261008` 中，平均匹配耗时由同长度旧实现约 `1.96 ms` 降至约 `1.68 ms`，路径比例 `1.003`，闭合误差 `0.065 m`，10 次切换后返程正确检索到子地图 0，错误切换为 0。
