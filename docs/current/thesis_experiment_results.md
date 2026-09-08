# 实验结果事实源

> 本文件是项目实验结果的唯一事实源。论文、汇报、计划和归档说明中的数值、实验设置与结论均应链接到本文件，不再复制维护结果表。

## 1. 统计口径与数据源

最终主比较覆盖 3 个训练随机种子、5 个场景、3 种方法、3 个独立评估随机种子，每个评估 seed 运行 100 episodes，共 13,500 episodes：

```text
3 train seeds x 5 scenarios x 3 methods x 3 eval seeds x 100 episodes
```

正式评估使用 PyBullet 固定基座 UR5、单个动态球形障碍物和静态目标位置到达任务。checkpoint 由 `validation_seed=2001`、每 checkpoint 20 episodes 的验证结果选择；正式评估使用 held-out eval seeds `1004`、`1005`、`1006`。每个场景-方法-train seed 先合并 3 个 eval seed 的 300 episodes，再计算 3 个 train seed 均值之间的 `mean +/- sample std`（`n=3`）。

原始汇总文件：

```text
outputs/rechecks/heldout_1004_1006/final_3methods/all_eval_episodes.csv
outputs/rechecks/heldout_1004_1006/final_3methods/eval_summary_by_train_seed.csv
outputs/rechecks/heldout_1004_1006/final_3methods/eval_summary_across_train_seeds.csv
outputs/rechecks/heldout_1004_1006/final_3methods/eval_summary_macro_across_train_seeds.csv
```

主比较方法：

| 方法 | 含义 |
| --- | --- |
| `ee_fixed` | 末端风险 + 固定风险惩罚 SAC + 固定平滑 |
| `link_fixed_penalty1` | 连杆级动态风险 + 固定风险惩罚 SAC + 固定平滑，`w_R=1.0` |
| `ldrc_fixed` | 连杆级动态风险 + 约束 SAC + 固定平滑 |

`w_R=1.0` 由 train seeds `101`、`202` 的小规模筛选提出，随后在相同三个 train seeds 上重训；因此最终 eval seeds 独立，但完整的超参数选择和训练随机性并非完全独立。

## 2. Random Crossing 主对比

每种方法共 900 个正式评估 episodes。

| 方法 | success rate | collision rate | non-end-link collision | final position error | min distance | safety violation rate | RMS jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ee_fixed` | 17.0% +/- 2.0% | 13.44% +/- 5.19% | 13.11% +/- 5.17% | 0.369 +/- 0.017 m | 0.177 +/- 0.012 m | 2.38% +/- 1.44% | 25.0 +/- 2.0 |
| `link_fixed_penalty1` | 64.2% +/- 12.5% | 6.89% +/- 2.22% | 6.89% +/- 2.22% | 0.125 +/- 0.029 m | 0.193 +/- 0.020 m | 1.27% +/- 0.92% | 32.3 +/- 2.7 |
| `ldrc_fixed` | 56.3% +/- 18.0% | 5.56% +/- 2.91% | 5.56% +/- 2.91% | 0.172 +/- 0.054 m | 0.167 +/- 0.008 m | 3.12% +/- 1.27% | 32.7 +/- 0.2 |

`link_fixed_penalty1` 的成功率最高，最终位置误差、安全距离违反率和最小距离优于 `ldrc_fixed`；`ldrc_fixed` 的碰撞率略低，因此存在局部安全取舍。`ee_fixed` 在任务完成和最终位置误差上明显落后。

## 3. 非末端区域压力测试

每种“区域-方法”包含 900 episodes。

| 区域 | 方法 | success rate | collision rate | final position error | RMS jerk |
| --- | --- | ---: | ---: | ---: | ---: |
| upper arm | `ee_fixed` | 13.3% +/- 0.6% | 3.00% +/- 3.61% | 0.414 +/- 0.034 m | 25.6 +/- 1.4 |
| upper arm | `link_fixed_penalty1` | 56.7% +/- 17.1% | 11.67% +/- 7.51% | 0.154 +/- 0.063 m | 34.8 +/- 3.9 |
| upper arm | `ldrc_fixed` | 54.2% +/- 19.3% | 9.33% +/- 0.58% | 0.180 +/- 0.083 m | 35.5 +/- 1.5 |
| elbow | `ee_fixed` | 16.7% +/- 1.2% | 5.67% +/- 3.21% | 0.381 +/- 0.008 m | 25.2 +/- 1.7 |
| elbow | `link_fixed_penalty1` | 63.8% +/- 16.4% | 8.67% +/- 5.03% | 0.135 +/- 0.049 m | 33.0 +/- 3.0 |
| elbow | `ldrc_fixed` | 56.6% +/- 24.1% | 7.33% +/- 2.31% | 0.169 +/- 0.079 m | 35.3 +/- 1.7 |
| forearm | `ee_fixed` | 19.4% +/- 5.5% | 5.33% +/- 3.21% | 0.390 +/- 0.029 m | 23.9 +/- 1.5 |
| forearm | `link_fixed_penalty1` | 72.8% +/- 16.2% | 3.00% +/- 2.65% | 0.109 +/- 0.037 m | 33.1 +/- 3.9 |
| forearm | `ldrc_fixed` | 67.0% +/- 18.2% | 4.33% +/- 1.15% | 0.131 +/- 0.052 m | 34.8 +/- 0.7 |
| wrist | `ee_fixed` | 15.8% +/- 3.2% | 11.00% +/- 3.46% | 0.429 +/- 0.038 m | 23.8 +/- 2.1 |
| wrist | `link_fixed_penalty1` | 76.1% +/- 14.7% | 2.33% +/- 2.52% | 0.099 +/- 0.031 m | 32.0 +/- 2.5 |
| wrist | `ldrc_fixed` | 62.3% +/- 25.4% | 3.00% +/- 1.00% | 0.147 +/- 0.071 m | 33.8 +/- 1.3 |

`link_fixed_penalty1` 在五个场景均获得最高平均成功率；在 upper arm 和 elbow 场景，`ldrc_fixed` 的碰撞率较低，不能宣称固定惩罚方法在所有安全指标上严格占优。

## 4. 五场景宏平均

| 方法 | success rate | collision rate | final position error | min distance | safety violation rate | RMS jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ee_fixed` | 16.4% +/- 1.4% | 7.69% +/- 2.68% | 0.396 +/- 0.014 m | 0.189 +/- 0.013 m | 1.69% +/- 0.61% | 24.7 +/- 1.5 |
| `link_fixed_penalty1` | 66.7% +/- 15.3% | 6.51% +/- 3.63% | 0.125 +/- 0.041 m | 0.202 +/- 0.011 m | 1.25% +/- 0.63% | 33.0 +/- 3.1 |
| `ldrc_fixed` | 59.3% +/- 20.6% | 5.91% +/- 1.27% | 0.160 +/- 0.067 m | 0.169 +/- 0.012 m | 3.58% +/- 0.98% | 34.4 +/- 0.4 |

## 5. 结论边界

可以支持：

- 连杆级动态风险配合经标定的固定风险惩罚 SAC，相较末端风险 baseline 提高目标到达表现并降低最终误差。
- `link_fixed_penalty1` 在五场景中平均成功率最高，并在多数安全与平滑指标上优于 `ldrc_fixed`。
- `ldrc_fixed` 在部分高风险区域和宏平均碰撞率上略低，说明存在指标取舍。

不能支持：

- 约束 SAC 已整体优于固定惩罚 SAC；
- 自适应平滑或风险自适应 jerk QP 已被多 seed、多场景验证；
- 三个 train seed 已证明统计显著性、广泛泛化或真实机器人安全保证。

原四方法矩阵、`ldrc_adaptive` 反事实、自适应平滑诊断和旧 `outputs/formal/` 结果仅作历史/失败消融，不参与最终排序。正式结果由旧固定一阶 EMA 执行器（`fixed_beta=0.35`）产生，不得与尚未完成重训评估的 RTB 执行器结果混合。
