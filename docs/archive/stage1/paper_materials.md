# 阶段一论文材料与图表清单

> **重构说明（2026-07-28）**：本文件中的产物是原主题的阶段一可复现材料。它们可用于新论文的预研究、基线或附录，但不再自动构成新论文的正文主表。新论文所需的材料矩阵见 [research_direction.md](../../design/research_direction.md)。

本文件将阶段一论文材料、附录与可复现产物对应起来。阶段一结论只使用 `outputs/rechecks/heldout_1004_1006/final_3methods/` 的 held-out 三方法数据；不要以旧四方法矩阵或小规模参数筛选替代阶段一主表。

## 阶段一正文/预研究材料

| 论文位置 | 内容 | 可复现产物 |
| --- | --- | --- |
| Table 1 | `random_crossing` 三方法主对比 | `outputs/paper/final_materials/table_1_random_crossing.csv` |
| Table 2 | 四个非末端区域压力测试 | `outputs/paper/final_materials/table_2_non_end_link_stress.csv` |
| Figure 1 | 五场景宏平均的任务与安全指标 | `outputs/paper/final_materials/figure_1_heldout_macro_comparison.png` |
| Figure 2 | 五个场景的成功率与碰撞率 | `outputs/paper/final_materials/figure_2_heldout_by_scenario.png` |

表 1 和表 2 的每个单元格均先合并同一 train seed 下 3 个 eval seeds 的 300 episodes，再对 3 个 train seeds 报告 `mean +/- sample std`（`n=3`）。每种“场景-方法”共 900 held-out episodes。checkpoint 使用 `validation_seed=2001` 选择；正式评估使用 eval seeds 1004、1005、1006。表中的 `collision rate` 是阶段一环境的综合 collision 事件指标，不等同于 P3 诊断中的 `capsule_overlap` 或 `pybullet_contact`。

## 附录材料

| 附录位置 | 内容 | 可复现产物或数据源 | 使用限制 |
| --- | --- | --- | --- |
| Table A1 | 每个 train seed 的完整 held-out 结果 | `outputs/paper/final_materials/appendix_table_a1_by_train_seed.csv` | 可用于展示跨 seed 波动，不作显著性检验。 |
| Figure A1 | 固定风险惩罚权重筛选 | `outputs/paper/final_materials/figure_a1_fixed_penalty_sensitivity.png` | 两个 train seeds 的探索性筛选；解释 `w_R=1.0` 的来源，不可替代主表。 |
| Figure A2 | `ldrc_fixed` 训练诊断 | `outputs/paper/final_materials/figure_a2_ldrc_training_diagnostics.png` | 说明 lambda 与风险代价的训练行为，不作为约束 SAC 整体更优的证据。 |
| Table A2 | adaptive actor / execution 反事实诊断 | `docs/archive/p3_pre_20260803/experiment_progress.md` 第 5 节 | 失败消融；不得纳入主方法排序。 |
| Figure A3 | adaptive 典型轨迹的风险与运动曲线 | `outputs/formal/figures/wrist_ldrc_adaptive_seed1001/` | 仅说明实现的 beta 响应和失败现象，不证明机制有效。 |

## 一键重建

在项目根目录运行：

```bash
conda run -n rl python scripts/prepare_paper_materials.py
```

脚本从 held-out 汇总 CSV、固定惩罚筛选汇总和历史 `ldrc_fixed` 训练日志重新生成上述表格与图片。生成结果是论文材料的派生产物，不修改原始评估数据。

## 阶段一结论边界

- 核心结论：`link_fixed_penalty1`（连杆级动态风险 + 固定风险惩罚 SAC，`w_R=1.0`）在五个 held-out 场景拥有最高的三-seed 平均成功率，并在最终误差、最小距离、安全距离违反率和 jerk 等多数指标上优于 `ldrc_fixed`。
- 必须同时报告的取舍：`ldrc_fixed` 的宏平均碰撞率略低（5.91% 对 6.51%），并在 upper arm、elbow 压力场景碰撞率更低。
- 不可作出的主张：约束 SAC 已整体优于经标定的固定惩罚 SAC；自适应平滑已提升安全性或平滑性；三颗 train seeds 已证明统计显著性或广泛泛化。
