# 可观测性对照套件汇总

每组重复 1 次，表中为 均值 [最小~最大]。

| 指标 | b2_dual |
| --- | ---: |
| frames | 1373 |
| duration_s | 200.5 |
| truth_yaw_path_deg | 176.42 |
| truth_yaw_max_deg | 176.10 |
| visual_yaw_path_deg | 148.00 |
| visual_yaw_max_deg | 138.88 |
| blind_window_s | 7.5 |
| blind_truth_yaw_deg | -41.42 |
| blind_applied_yaw_deg | -4.66 |
| dropped_truth_yaw_deg | -0.00 |
| applied_intervals | 187 |
| applied_yaw_deg | -139.76 |
| UNOBSERVED 区间 | 56帧/-51°/-1.0 |

b0_plain = 旧行为；b1_flag = 单参考 + 失效标记；b2_dual = 双参考 + 显式 unobserved。
