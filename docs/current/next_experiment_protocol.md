# P3 冻结后状态

> 更新时间：2026-08-07。原 B1--B5 实验协议已执行完成；本页保留固定口径，并记录其冻结后约束。独立基础 reaching v2 已完成，但不改变 P3 安全方法决议。

## 冻结结论

`outputs/p3_postfix_dev_100k/p3_freeze_decision.json` 的结论为 `do_not_freeze_p3_or_expand_recovery`。严格 B4 留作诊断基线；worst-link recovery 在共享最终集上使 physical contact 从 4/432 上升到 22/432，已拒绝。

因此不得继续运行或扩展 recovery、完整方法 M、OOD 或真机实验。任何后继研究必须另行定义安全目标和独立评估协议，不能将本页固定设置解释为已批准的后续执行命令。

基础 reaching v2 是独立的无障碍执行基线，不属于 P3 V2/G3/G4。其固定结果为全量 `582/600=97.0%`、固定 IK 可达且无障碍候选路径子集 `582/582=100%`；该结果只能用于确认基础 reaching 链路恢复，不能解除本页的 P3 `do_not_freeze_p3_or_expand_recovery` 决议。

## 已完成的目标

在修正连杆速度单位和碰撞事件定义后，重新建立可比较的 P3 基线。B1--B5 只用于验证因子化链路，不构成 P3 方法冻结。

## 固定设置

- 机器人：UR5，frame-corrected 胶囊。
- 障碍物速度：0.05 m/s。
- 关节速度上限：1.0 rad/s。
- 几何裕度：0.03 m。
- 过滤器：strict QP，关闭 recovery/relaxation/maximin。
- P3 回合终止：仅 `pybullet_contact`；capsule overlap 作为独立保守事件。
- 当前阶段过滤器的仿真侧单步耗时门槛为 `300 ms`；必须报告 mean/P95/P99/max 和预算停止率。该门槛用于早期实验筛选，不是最终部署目标。
- 所有速度字段必须携带 `_mps`、`_radps` 或 `_radps2` 后缀。

## 已执行顺序（归档）

1. 运行全量测试和 B4 smoke，检查预测速度/上界字段为有限值。
2. 为 B1--B5 各运行 10k 单 seed 开发轮次，确认训练和评估口径一致。
3. 检查三类碰撞事件、成功率、不可行率和任务误差，不用单一指标冻结设计。
4. 开发轮次通过后，再使用至少 3 个新 train seeds 和独立 validation/eval seeds。
5. 冻结包审计后决定不冻结 P3，且不实现或评估完整方法 M。

10k/100k 开发结果分别写入 `outputs/p3_postfix_dev/` 和 `outputs/p3_postfix_dev_100k/`。禁止写回修正前的 `outputs/p3_dev/`、`outputs/p3_dev_100k/`。

## 共享可行种子集

- 主比较使用 B5 鲁棒预测过滤器的初始可行集：`initial_h_min_m > 0`。
- 固定清单为 `configs/experiments/p3_manifests/b5_robust_feasible_validation.json`（20 个 validation seeds）和 `configs/experiments/p3_manifests/b5_robust_feasible_final.json`（144 个最终评估 seeds）。两者互斥，来源为 `outputs/p3_postfix_dev/initial_feasibility_b5_seed7101_n200.json`。
- B1--B5 均使用这两份清单；checkpoint 选择只使用 validation 清单，最终比较只使用 final 清单。清单不得因方法或 checkpoint 重新采样。
- B4/B5 的无条件 reset 覆盖率及初始不安全比例单独报告，不混入主比较排名，也不改变 B5 的 delay、tracking 或几何裕度。

## 必须记录

- Git commit、完整 config、train/validation/eval seed 和 checkpoint 选择过程。
- 每个评估 CSV 的实际 `seed` 和 `seed_manifest`。
- `predictive_link_velocity_norms_mps`、`predictive_max_link_speed_bound_mps`。
- `collision_capsule_overlap`、`collision_pybullet_contact`、`collision_any`、`termination_collision` 和 `termination_reason`。
- success、最终误差、最小距离、干预率/范数、safe-stop、不可行率和 mean/P95/P99/max 求解时间，以及 `safe_stop_compute_budget` 比率。

## 通过条件

开发轮次只判断链路正确，不形成论文结论。正式冻结至少要求跨 train seed 的任务性能稳定，physical contact 不劣于 B1--B3，capsule overlap 与 physical contact 的差异可解释，且不可行停止不会主导回合；当前阶段还要求观测到的单步过滤耗时不超过 `300 ms`，且无预算停止。

`max_filter_compute_time_s` 是过滤器返回后的 simulator-side 检查，不能中断正在运行的 OSQP、Python 或回退路径。因此“300 ms 通过”不等价于硬实时保证；后续进入部署相关阶段前，仍需实现可中断求解与外部 watchdog。
