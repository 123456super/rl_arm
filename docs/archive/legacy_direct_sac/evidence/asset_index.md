# Legacy direct-SAC 资产索引

本页索引保留在原位置的大型或仍被复现链引用的资料。它们没有被移动；当前 canonical 实验只能使用 `configs/experiments/hierarchical/`，以下配置均为 legacy。

## 配置和任务清单

| 范围 | 原路径 | 内容 |
| --- | --- | --- |
| S0 reaching recovery | `configs/experiments/reaching_recovery/` | v1/v2 基础配置、3 个 train seeds、validation/final manifests |
| S1 direct-SAC 全部实验 | `configs/experiments/reaching_incremental/` | frozen actor 迁移、fine-tune、extend、terminal refine、R3/R4 和诊断配置 |
| S0 actor 冻结映射 | `configs/experiments/reaching_incremental/frozen_reaching_v2_actors.json` | 三个 seed 的冻结 checkpoint 路径 |
| 原 final manifest | `configs/experiments/reaching_recovery/manifests/v1_final.json` | reset seeds `9001--9200` |
| blind final manifest | `configs/experiments/reaching_incremental/manifests/s1_blind_final_v1.json` | reset seeds `9401--9600` |
| failure manifests | `configs/experiments/reaching_incremental/manifests/` | candidate-path、R3/R4 和 hard-case 样本桶 |

## 原始结果目录

| 阶段 | 原路径 | 保留内容 |
| --- | --- | --- |
| S0 reaching | `outputs/reaching_recovery_v2/` | 3-seed checkpoints、progress、train metrics、final CSV |
| frozen S0 actor 静态迁移 | `outputs/reaching_incremental/s1_static_obstacle/` | 3-seed final CSV 与 summary |
| S1 fine-tune | `outputs/reaching_incremental/s1_static_obstacle_finetune/` | 训练 checkpoint、日志、3-seed final CSV 与 summary |
| extend +100k | `outputs/reaching_incremental/s1_static_obstacle_finetune_r1_extend100k/` | 3-seed final CSV 与 summary |
| terminal refine | `outputs/reaching_incremental/s1_static_candidate_terminal_refine/` | 大约 412 MB 的 checkpoint、训练日志、final CSV 与当前复跑 summary |
| R3 candidate repair | `outputs/reaching_incremental/s1_r3_candidate_repair/` | checkpoint、日志和 blind final CSV/summary |
| R4 hard-case repair | `outputs/reaching_incremental/s1_r4_hard_case_repair/` | checkpoint、日志和 blind final CSV/summary |
| fine-tune 失败诊断 | `outputs/reaching_incremental/s1_static_failure_diagnostics/` | feasibility、离线分类和评估副本 |
| terminal refine 失败诊断 | `outputs/reaching_incremental/s1_static_terminal_refine_diagnostics/` | feasibility、trace、逐 episode 报告和分类 summary |

## 为什么不搬动这些目录

- summary 和配置写有原始相对路径，移动后会使证据链失效。
- checkpoint 与训练状态用于复核恢复过程，不适合重复进入文档归档。
- 输出目录可能包含用户仍在使用的未提交产物；归档操作不应改写其时间戳或目录结构。
- 本归档已保存论文论证所需的最小结果副本，详细复现实验再按这里的原路径读取。
