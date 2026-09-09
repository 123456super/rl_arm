# Held-out 评估结果目录

本目录保存 `link_fixed_penalty1` 重训后的 held-out 评估结果。论文正文、汇报和后续分析只能使用 `final_3methods/` 下的文件。

## 目录说明

| 路径 | 状态 | 用途 |
| --- | --- | --- |
| `eval/` | 原始数据 | 最终三方法的 135 个 eval-seed CSV；每个文件 100 episodes。 |
| `final_3methods/` | 最终可信 | `ee_fixed`、`link_fixed_penalty1`、`ldrc_fixed` 的 held-out 三方法汇总，共 13,500 episodes。 |
| `outputs/archive_old/heldout_two_methods_initial/` | 已替代 | 早于 `ee_fixed` 补评估生成的两方法中间汇总，共 9,000 episodes；不得用于论文主表或最终排序。 |

最终评估设置：train seeds 为 101、202、303；eval seeds 为 1004、1005、1006；场景为 `random_crossing`、`upper_arm_crossing`、`elbow_crossing`、`forearm_crossing`、`wrist_crossing`。

`outputs/archive_old/formal_legacy/summary/` 是另一套旧四方法实验，使用不同的 eval seeds 和固定惩罚配置，只能作历史或敏感性分析。

所有 CSV 应按表头字段名读取，不应按固定列号解析：旧 `formal` 数据与本目录的列顺序、字段集合不同。
