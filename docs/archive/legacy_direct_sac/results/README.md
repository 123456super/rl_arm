# Legacy direct-SAC 结果归档

本目录保存适合版本管理的核心摘要和诊断报告副本。原始 CSV、checkpoint、训练日志、replay 和 trace 不移动，详见 [资产索引](../evidence/asset_index.md)；副本来源和 SHA-256 见 [结果来源说明](../evidence/result_provenance.md)。

## 核心结果

| 阶段 | 固定评估结果 | 归档文件 | 口径 |
| --- | ---: | --- | --- |
| frozen S0 actor 静态迁移 | `428/600=71.33%`，156 timeout，16 collision | [s1_static_obstacle_summary.json](s1_static_obstacle_summary.json) | final manifest `9001--9200`，3 train seeds |
| S1 fine-tune | `501/600=83.50%`，99 timeout，0 collision | [s1_static_obstacle_finetune_summary.json](s1_static_obstacle_finetune_summary.json) | 同一 final manifest |
| extend +100k | `501/600=83.50%`，99 timeout，0 collision | [s1_static_extend100k_summary.json](s1_static_extend100k_summary.json) | 同一 final manifest |
| candidate terminal refine 当前复跑 | `511/600=85.17%`，89 timeout，0 collision | [s1_static_candidate_terminal_refine_current_rerun_summary.json](s1_static_candidate_terminal_refine_current_rerun_summary.json) | 2026-08-19 归档时磁盘副本 |
| R3 candidate repair blind | `505/600=84.17%`，95 timeout，0 collision | [s1_r3_blind_summary.json](s1_r3_blind_summary.json) | blind manifest `9401--9600` |
| R4 hard-case repair blind | `505/600=84.17%`，95 timeout，0 collision | [s1_r4_blind_summary.json](s1_r4_blind_summary.json) | blind manifest `9401--9600` |

## 诊断材料

- [s1_static_failure_diagnostics_summary.json](s1_static_failure_diagnostics_summary.json)：针对 501/600 fine-tune 批次的失败分类。
- [s1_terminal_refine_diagnostics_summary.json](s1_terminal_refine_diagnostics_summary.json)：针对当前 511/600 复跑的失败分类。
- [s1_terminal_refine_detailed_failure_report.txt](s1_terminal_refine_detailed_failure_report.txt)：当前复跑逐失败 episode 的离线诊断。

## 历史最好结果的特殊说明

旧协议和旧研究状态记录 candidate terminal refine 曾达到 `521/600=86.83%`、79 timeout、0 collision、mean final error `0.0699 m`。归档时，同名 `outputs` 路径已经被后续复跑覆盖为 `511/600=85.17%`，因此这里不伪造一份不存在的 521/600 JSON。历史最好只能引用[旧协议快照](../documents/s1_static_obstacle_protocol_2026-08-18.md)并注明“历史文档记录”；机器可读的归档副本明确命名为 `current_rerun`。
