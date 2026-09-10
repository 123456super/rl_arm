# 从零实验输出根目录

> 启动日期：2026-09-09  
> 本目录是重启后 R0--R9 唯一允许使用的输出根目录。

建议目录结构：

```text
outputs/restart_2026-09-09/
  r0_protocol/
  r1_geometry/
  r2a_penalty_selection/
  r2_instant_link/
  r3_action_conditioned_prediction/
  r4_one_step_qp/
  r5_fixed_topk_ptqp/
  r6_vg_ptqp/
  r7_heldout_main/
  r8_robustness/
  r9_real_ur5/
```

准入要求：

- 每个阶段保存完整配置、代码 commit、环境信息、seed manifest 和运行命令。
- 训练、validation 与 held-out 分目录保存，不覆盖已有运行。
- 失败、超时、中断和 fallback 输出不得删除。
- 汇总数据必须能追溯到逐 episode 原始 CSV 和 checkpoint 哈希。
- 只有登记到 `docs/experiments/restart_2026-09-09/experiment_results.md` 的结果才能进入论文。
