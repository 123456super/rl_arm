# 旧结果与无效结果归档

本目录集中保存依据 `docs/current/` 判定为旧、已替代、失败诊断、smoke 或尚未完成正式验证的输出。
这些文件仅用于追溯、复现实验过程和失败分析，不得作为当前论文主表、最终排序或实机部署候选。

## 分类

| 目录 | 内容 | 使用边界 |
| --- | --- | --- |
| `formal_legacy/` | 旧四方法 formal 矩阵、训练、评估、图表和诊断 | 历史/失败消融；不参与最终排序 |
| `heldout_two_methods_initial/` | 仅含 `link_fixed_penalty1` 与 `ldrc_fixed` 的 9000-episode 中间汇总 | 已被 `outputs/current_results/heldout_1004_1006/final_3methods/` 替代 |
| `link_fixed_penalty1_legacy/` | 使用旧 eval seeds `1001/1002/1003` 的 link 基线重训结果 | 非当前 held-out 主结果 |
| `minimal_qp_direct_failed/` | 最小 QP 直接修正动作的初版结果 | 失败诊断；physics RMS jerk 明显升高 |
| `minimal_qp_smooth_superseded/` | smooth-QP 的早期运行、smoke 和轨迹 | 已由 `minimal_qp_smooth_recheck` 复核版本替代，且仍仅为单场景 |
| `predictive_risk_smoke/` | 预测风险模块 smoke 输出 | 仅功能检查 |
| `deployment_preflight_legacy/` | 绑定旧 `link_fixed_penalty1` 的离线部署预检 | 不能作为最终 `Proposed` 部署报告 |
| `predictive_link_incomplete/` | 预测风险策略单 seed、小规模训练和 20-episode 评估 | P2 未完成探索；不能证明优于 Instant-Link-SAC |
| `tuning_sweeps/` | `c_safe`、`event_cost`、`fixed_risk_penalty` 参数扫描 | 调参/敏感性分析，不是主比较 |

当前应优先使用的结果仍位于：

```text
outputs/current_results/heldout_1004_1006/final_3methods/
outputs/current_results/heldout_1004_1006/eval/
outputs/current_results/paper_final_materials/
outputs/current_results/predictive_risk_offline_eval/corrected_summary_by_scenario.csv
outputs/current_results/minimal_qp_smooth_recheck/
```

`outputs/current_results/minimal_qp_smooth_recheck/` 目前仍是单场景复核结果，不是最终多场景主结果；它保留在当前结果区是因为这是当前最新的 QP 复核，而不是已完成的论文主结论。

归档采用移动而非删除，原始文件仍可追溯。
