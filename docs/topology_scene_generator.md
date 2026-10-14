# 大厅—双环—径向通道地图生成器

生成器从同一个 NetworkX/Shapely 场景对象导出 Gazebo Classic world、Open-RMF
building map、拓扑图、异构子地图、合法标签、mapping/query 路线、占据栅格和预览图。
Open-RMF 转换器是可选后端；当前机器未安装系统级
`rmf_building_map_tools` 时，`building.yaml` 仍会保存，Gazebo world 由同一场景对象直接导出。

## 单场景

```bash
python3 src/visual_slam_gazebo/scripts/generate_topology_scene.py \
  --family perturbed_ring \
  --seed 2103 \
  --obstacle-mode none \
  --output generated_scenes/perturbed_ring_2103 \
  --validate --preview
```

布局族为 `regular_baseline`、`perturbed_ring` 和 `asymmetric_extension`。
障碍模式为 `none`、`visual_only` 和 `topology_change`。同一配置和种子必须产生相同的
拓扑、标签、路线及 Gazebo world。

## 冻结数据集

```bash
python3 src/visual_slam_gazebo/scripts/generate_topology_scene.py \
  --generate-dataset \
  --output generated_scenes/topology_dataset_v1
```

默认生成 pilot 数据集：6 个基础布局，每类 2 个，按 1/1/0 划分为 development、
calibration、test，并为每个基础布局生成 `visual_only` 和 `topology_change` 变体。
pilot 只用于接口、场景、路线和校准闭环验证，不用于论文最终结论。

正式实验显式使用 full profile：

```bash
python3 src/visual_slam_gazebo/scripts/generate_topology_scene.py \
  --generate-dataset --dataset-profile full \
  --output generated_scenes/topology_dataset_full_v1
```

full profile 生成 30 个基础布局，每类 10 个，按 4/3/3 划分为 development、
calibration、test；每个基础布局同时生成两类障碍变体。种子在
`config/topology_scene_defaults.yaml` 中冻结，正式测试不得改写 test 种子或场景文件。

## Gazebo 与实验入口

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 src/visual_slam_gazebo/scripts/experiment_manager.py \
  --project topology_scene_smoke \
  --scene-manifest generated_scenes/perturbed_ring_2103/scene_manifest.json \
  --waypoint-route generated_scenes/perturbed_ring_2103/query_routes/query_same_view_inner.csv \
  --methods c1 c4 p --runs 1 --gui --record --analyze --archive --shutdown-gazebo
```

`/sim/odom` 仅供 waypoint 控制器执行预定路线以及离线评价；视觉 SLAM、候选评分和子地图
切换不订阅该真值输入。实验管理器会验证 `scene.world` 的 SHA-256，并将场景 ID、地图版本和
manifest 哈希写入 campaign manifest。

waypoint 路线启动时，控制器使用 CSV 中冻结的首个位姿初始化机器人。该首姿态也用于构造
初始左目光学坐标系到 Gazebo world 的固定变换；这是实验配置，不是运行时真值反馈。
waypoint 模式会自动关闭仅适用于直线走廊的 `straight_motion_constraint`，保留平面转向估计。
相机安装偏移仍为前向 `0.08 m`、左向 `0.035 m`、高度 `0.32 m`。

## 验证

```bash
source /opt/ros/humble/setup.bash
python3 src/visual_slam_gazebo/scripts/test_topology_scene_generator.py -v
export GAZEBO_MODEL_PATH=$PWD/src/visual_slam_gazebo/models:/usr/share/gazebo-11/models
gz sdf -k generated_scenes/perturbed_ring_2103/scene.world
```

通道子地图保持 `5 m/1.5 m`，大厅使用 `7 m/1.5 m` 网格，路口使用半径 `3 m` 的
junction 子地图。邻接由拓扑连接和最小重叠面积共同确定。自动验证覆盖图连通性、环路、
径向连接数、声明死端、三类子地图、合法集合数量、none-of-the-above、障碍变量隔离和确定性。
`candidate_coverage_miss` 必须在运行候选检索后由
“合法集合与实际候选集合”比较得出，生成器不会预先伪造该结果。

按冻结划分批量运行可先执行 dry-run：

```bash
python3 src/visual_slam_gazebo/scripts/run_topology_dataset_campaign.py \
  generated_scenes/topology_dataset_v1/dataset_manifest.json \
  --split development --output-root experiments/topology_development \
  --methods c1 c4 p --runs 1 --dry-run
```

去掉 `--dry-run` 后逐布局运行；增加 `--include-variants` 才会同时运行两类障碍变体。

## 当前接口边界

在线节点已加载 `submaps.json` 并维护显式的 `runtime_id -> scene_id` 映射。在线状态不使用
仿真真值；实验记录器仅在旁路用 world 真值生成 `legal_submaps`、coverage 和 query 帧角色。
旧场景、旧距离型子地图数据库和旧 query 标签不兼容新 schema，必须重新生成，且不得作为
正式 calibration/test 证据。
