# 可观测性对照套件汇总

每组重复 1 次，表中为 均值 [最小~最大]。

| 指标 | b2_dual |
| --- | ---: |
| frames | 1510 |
| duration_s | 183.4 |
| truth_yaw_path_deg | 176.84 |
| truth_yaw_max_deg | 176.08 |
| visual_yaw_path_deg | 174.34 |
| visual_yaw_max_deg | 156.00 |
| blind_window_s | 1.6 |
| blind_truth_yaw_deg | 26.29 |
| blind_applied_yaw_deg | 8.13 |
| dropped_truth_yaw_deg | 0.06 |
| applied_intervals | 77 |
| applied_yaw_deg | 160.76 |
| UNOBSERVED 区间 | 19帧/36°/-1.0 |

b0_plain = 旧行为；b1_flag = 单参考 + 失效标记；b2_dual = 双参考 + 显式 unobserved。
