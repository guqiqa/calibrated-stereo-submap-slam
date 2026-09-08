# 民航视觉 SLAM 子地图算力优化研究代码

本目录是从 `D:\AIview\ROS2-clearRob` 复制出的研究代码备份，当前版本包含：

- `src/visual_slam`：双目视觉里程计、ORB、三角化、PnP/RANSAC、G/S/P 子地图模式；
- `src/visual_slam_gazebo`：48 m 纹理走廊、双目机器人、视差检查和自动实验器；
- `experiments`：已完成的 G/S/P 短测和路线验证数据；
- `参数可视化图表.html`：固定参数、处理链路、实验分组和短测结果图表；
- `民航视觉SLAM主线小论文实验流程-子地图修订版.md`：实验计划。

关键固定参数：640 x 480、15 Hz、70 mm 双目基线、处理缩放 0.75、子地图长度 5.0 m、重叠 1.5 m、Top-3 候选、局部窗口 15 帧。

后续仿真仍在 `D:\AIview\ROS2-clearRob` 的 ROS 2 工作区编译和运行，`D:\科研slam` 用作独立代码备份与研究材料归档。
