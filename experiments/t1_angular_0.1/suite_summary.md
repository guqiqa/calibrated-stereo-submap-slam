# 可观测性对照套件汇总

每组重复 1 次，表中为 均值 [最小~最大]。

| 指标 | b2_dual |
| --- | ---: |
| frames | 2135 |
| duration_s | 297.6 |
| truth_yaw_path_deg | 176.39 |
| truth_yaw_max_deg | 176.07 |
| visual_yaw_path_deg | 151.42 |
| visual_yaw_max_deg | 142.25 |
| blind_window_s | 31.6 |
| blind_truth_yaw_deg | -43.05 |
| blind_applied_yaw_deg | -9.46 |
| dropped_truth_yaw_deg | -0.00 |
| applied_intervals | 549 |
| applied_yaw_deg | -142.87 |
| UNOBSERVED 区间 | 259帧/-51°/-1.0 |

b0_plain = 旧行为；b1_flag = 单参考 + 失效标记；b2_dual = 双参考 + 显式 unobserved。
