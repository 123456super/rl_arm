# S0 基础 reaching 历史基线

> 归档日期：2026-08-19。状态：已冻结，仅作为 legacy direct-SAC 对照。旧版原文见[归档快照](../documents/reaching_recovery_protocol_pre_hierarchy.md)，当前实验入口见[分层控制通用实验协议](../../../current/hierarchical_control_protocol.md)。

## 1. 目的与固定结果

S0 用于确认 UR5、随机目标、SAC 训练和确定性执行链路具备基础到达能力。v2 三个独立训练 seed `4301/4302/4303` 各训练 300000 environment steps，checkpoint 只按独立 validation manifest 选择。

固定 final manifest `9001--9200` 的结果为：

| 口径 | seed 4301 | seed 4302 | seed 4303 | pooled |
| --- | ---: | ---: | ---: | ---: |
| 完整 200 reset | `194/200=97.0%` | `194/200=97.0%` | `194/200=97.0%` | `582/600=97.0%` |
| 离线候选子集 | `194/194=100%` | `194/194=100%` | `194/194=100%` | `582/582=100%` |

共同失败 reset 为 `9021、9065、9095、9098、9120、9142`，均为 12 s timeout 且无碰撞。有限多初值 IK 未找到候选不等于数学不可达，因此论文必须同时报告完整口径和候选子集口径。

冻结 actor：

| train seed | selected step | checkpoint |
| ---: | ---: | --- |
| 4301 | 240000 | `outputs/reaching_recovery_v2/train/seed_4301/link_fixed_no_obstacle_speed100_seed4301_steps300000/actor_step_240000.pt` |
| 4302 | 280000 | `outputs/reaching_recovery_v2/train/seed_4302/link_fixed_no_obstacle_speed100_seed4302_steps300000/actor_step_280000.pt` |
| 4303 | 220000 | `outputs/reaching_recovery_v2/train/seed_4303/link_fixed_no_obstacle_speed100_seed4303_steps300000/actor_step_220000.pt` |

## 2. 适用边界

- 该结果关闭障碍物、hierarchical control 和 safety filter，只证明旧直接策略的基础 reaching 能力。
- 它不能证明静态/动态避障、安全、泛化或真机性能。
- 旧 actor 的观测和动作都是 direct-SAC 语义，禁止加载到新 hierarchical actor；代码通过 actor signature 拒绝这种混用。
- 新论文可将 S0 作为历史基线，但新方法的无障碍 nominal-only 消融必须单独运行，不能用 S0 替代。

## 3. 冻结资产

- 配置：`configs/experiments/reaching_recovery/v2_seed430{1,2,3}.yaml`
- manifests：`configs/experiments/reaching_recovery/manifests/`
- 输出：`outputs/reaching_recovery_v2/`
- 冻结映射：`configs/experiments/reaching_incremental/frozen_reaching_v2_actors.json`

