# 到达优先安全同伦：正式训练前有效验证

> **已终止（2026-09-16）**：旧协议历史记录，仅用于失败追溯；不续训、不参与 v8 模型选择。

日期：2026-09-15  
协议：`task_first_safety_homotopy_v2`（已被 v3 替代）

> 本文记录的是 v2 历史预检。v2 随后的正式 S0 在三个 validation seed 上成功率均为 0；其 55 维输入、持续状态误差成本和终止吸收态补偿已由 `task_first_goal_curriculum_v3` 替代。本文中的奖励数值不得用于判断 v3。

## 当前结论

当前代码已通过目标运动学可达性、失败终止回报排序、replay 严格化奖励重算和短程 SAC 链路验证。这里只保留 v2 修正后的有效结果；旧协议、失败预跑和基于旧 checkpoint 的诊断产物均已删除，不得作为实验依据。

当前可以从随机初始化开始新的 S0 训练。尚未获得通过 S0 Gate 的正式 checkpoint，因此 S1、S2、固定安全系数扫描和严格切换实验仍未开始。

统一有效结果目录：`outputs/thesis_homotopy/preflight_20260915_v2/`。

## V0 自动化测试

命令：

```bash
conda run -n rl pytest -q tests/test_thesis_homotopy.py
```

结果：`8 passed`。覆盖 55 维 observation、低安全系数缩放、终止吸收态补偿、课程门控、replay 严格化、严格 contact 奖励重算、分层采样以及目标位姿 IK/FK 回代。

## V1 目标运动学可达性

目标接受顺序为：

```text
候选关节构型
 -> 关节限位与目标构型自碰撞检查
 -> FK 生成目标位置和姿态
 -> 带位置和姿态约束的 IK
 -> IK 解有限且处于关节限位内
 -> IK 解 FK 回代
 -> 位置误差 <= 0.01 m 且姿态误差 <= 0.05 rad
 -> 接受目标
```

命令：

```bash
conda run -n rl python scripts/preflight_thesis_homotopy.py goal \
  --samples 100 \
  --output outputs/thesis_homotopy/preflight_20260915_v2
```

| 指标 | 结果 |
| --- | ---: |
| 抽样目标 | 100 |
| reset/采样失败 | 0 |
| 完整运动学校验通过率 | 100% |
| 采样尝试次数中位数 / p95 / 最大值 | 23 / 93.05 / 111 |
| IK→FK 最大位置误差 | `7.30e-8 m` |
| IK→FK 最大姿态误差 | `2.27e-7 rad` |

判定：通过。该结果证明目标位姿运动学可达，不等价于已证明任意初始状态到目标之间存在无碰撞路径。

原始数据：

- `outputs/thesis_homotopy/preflight_20260915_v2/goal_summary.json`
- `outputs/thesis_homotopy/preflight_20260915_v2/goal_samples.csv`

## V2 失败终止回报排序

保持宽容期“低碰撞惩罚且碰撞不中止”不变，仅对硬失败和严格任务球碰撞增加有限折扣时域吸收态补偿：

```text
Z_H = (1-gamma^H)/(1-gamma), gamma=0.99, H=240
c_state = 2 rho_p,next^2 + 0.5 rho_R,next^2
c_terminal = Z_H c_state I[硬失败或严格任务球碰撞]
```

宽容期任务球 contact 不是真终止，因此 `c_terminal=0`；`xi=0.02` 时基础 contact 惩罚仍为 `0.68`，到达奖励不缩放。

命令：

```bash
conda run -n rl python scripts/preflight_thesis_homotopy.py reward \
  --samples 100 \
  --output outputs/thesis_homotopy/preflight_20260915_v2
```

| 指标 | 结果 |
| --- | ---: |
| 严格碰撞优于 timeout | 0/100 |
| 硬失败优于 timeout | 0/100 |
| `理想到达 > timeout > max(失败)` | 100/100 |
| 理想 60 步到达回报中位数 | -20.744 |
| timeout 回报中位数 | -233.229 |
| 硬失败回报中位数 | -269.791 |
| 严格碰撞回报中位数 | -285.791 |
| 宽容期单步碰撞回报中位数 | -3.562 |
| 终止吸收态补偿中位数 | 233.229 |

判定：通过。SAC 不再能通过自碰撞、关节越界或严格任务球碰撞来逃避未来的持续位姿误差成本。

原始数据：

- `outputs/thesis_homotopy/preflight_20260915_v2/reward_summary.json`
- `outputs/thesis_homotopy/preflight_20260915_v2/reward_samples.csv`

## V3 SAC 集成冒烟

结果目录：`outputs/thesis_homotopy/preflight_reward_guard_smoke_20260915/`。

使用 v2 协议执行 100 个 S0 transition：

| 指标 | 结果 |
| --- | ---: |
| transition | 100 |
| SAC update | 93 |
| 非有限 reward | 0 |
| checkpoint 协议 | `task_first_safety_homotopy_v2` |

判定：通过。该运行只验证环境、replay、奖励重算、SAC 更新、日志和 checkpoint 链路，不构成收敛或性能结果。

训练器使用根 seed 派生并记录相互独立的环境、课程、replay、warm-up action、Python、NumPy 和 Torch RNG 流。warm-up action 不再调用未显式 seed 的 Gym action space。checkpoint 中的 `q/qdot` 使用 PyBullet 原始双精度保存。

恢复等价性复测比较了“连续运行 120 step”和“运行 100 step、从活动 episode checkpoint 恢复后再运行 20 step”。两侧的 20 条 transition 日志、actor、两个 online/target critic、optimizer、温度、replay、全部 RNG 和活动 episode 状态逐项完全一致。

有效恢复验证目录：

- `outputs/thesis_homotopy/preflight_uninterrupted120_20260915/`
- `outputs/thesis_homotopy/preflight_resume_from100_20260915/`

## V4 S0 Gate 评价链路

严格评价器现在以任务球、自碰撞和环境碰撞的并集计算 `collision_rate`，并分别保留各碰撞分项；Gate 不再遗漏自碰撞或环境碰撞。同时输出终态位置/姿态误差的 mean、p95 和整条轨迹的最小间隙。

使用未训练的 100-step 冒烟 checkpoint 执行 2 个无障碍 episode，评价链路正常完成并正确判为 `gate_pass=false`。该数值不属于性能结果。

有效评价器冒烟目录：`outputs/thesis_homotopy/preflight_s0_evaluator_smoke_20260915/`。

## 下一步顺序

1. 使用 v2 协议和 seed 11001 从随机初始化运行 S0 的首个 100k block。
2. 按 25k 间隔执行 S0 validation Gate；只有合格 checkpoint 才能用于 observation 迁移检查。
3. S0 合格后执行固定 `xi` 扫描，再决定是否进入 S1 完整课程。
4. 获得保持到达能力的高 `xi` checkpoint 后，执行严格终止与 replay 严格化 A/B。
5. 上述预检通过后才运行三个训练 seed 的正式 S0/S1/S2。
