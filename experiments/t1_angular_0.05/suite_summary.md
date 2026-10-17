# 可观测性对照套件汇总

每组重复 1 次，表中为 均值 [最小~最大]。

| 指标 | b2_dual |
| --- | ---: |
| frames | 2758 |
| duration_s | 403.0 |
| truth_yaw_path_deg | 83.08 |
| truth_yaw_max_deg | 82.41 |
| visual_yaw_path_deg | 91.38 |
| visual_yaw_max_deg | 59.20 |
| blind_window_s | 20.8 |
| blind_truth_yaw_deg | 6.81 |
| blind_applied_yaw_deg | 7.29 |
| dropped_truth_yaw_deg | 65.85 |
| applied_intervals | 85 |
| applied_yaw_deg | 19.26 |
| UNOBSERVED 区间 | 598帧/29°/未闭合 |

b0_plain = 旧行为；b1_flag = 单参考 + 失效标记；b2_dual = 双参考 + 显式 unobserved。
