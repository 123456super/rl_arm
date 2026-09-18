# 连续实验记录：v8 → v9 → v10 → v11

更新日期：2026-09-18

状态：**v8/v9 已失败；V10 已完成 600k 但未通过 S0；当前 V11 已实现，等待从随机初始化重新训练**

当前协议：`task_first_persistent_eta_replay_homotopy_v11`  
历史基线：`continuous_pose_rfm_mean_smooth_then_safety_homotopy_v8`

结果目录：v8 使用 `outputs/experiments_9_16/`，v9 使用 `outputs/experiments_9_17/`，V10 使用 `outputs/experiments_9/v10/`，V11 使用 `outputs/experiments_9/v11/`。结果目录按协议隔离，文档不再按日期拆分。

本目录记录同一研究问题的连续迭代：v8 是基线和失败诊断，v9 加入连续自碰撞信号，V10 修正 v9 中“姿态学习与强安全约束并行”的课程冲突。各版本 checkpoint/replay 互不兼容；“合并文档”不等于“合并学习状态”。

五篇参考论文的逐项审查、采用与拒绝理由见 [reward_literature_analysis.md](reward_literature_analysis.md)。

## V8 历史结果

seed 11001 从随机初始化连续训练了三个 `100000`-transition block。位置课程和位置巩固完成，姿态课程由 `eta=0` 推进到 `eta=0.65`；block3 内恢复并通过了 `eta=0.5` 和 `eta=0.6`，说明姿态奖励并非完全失效。但最终仍为 `eta=0.65`、`K_pose=0`、确定性位置探针失败、`s0_goal_gate_eligible=false`，未取得 S0 Gate 资格。按预先固定的三 block 停止规则，本实验线不再启动 block4，不进入 S1/S2，也不扩展其他训练 seed。

最大的直接失败类型是自碰撞：`eta=0.65` 的 434 个当前任务 episode 中有 51 个自碰撞、34 个 timeout 和 1 个 joint-limit；最终固定探针的 50 个 episode 中有 2 个碰撞、1 个 timeout。更深层的瓶颈是姿态学习后对固定困难位置目标的保持与泛化不足。S0 的障碍风险输入是空场景占位值，self-collision 只有实际接触后的稀疏硬失败惩罚，没有显式、连续的自碰撞间隙或接近风险信号。这是下一协议版本应优先验证的设计缺口，而不是在当前 v8 中途修改后续训的理由。

## 为什么从 V8 重开实验线

v2～v7 的策略输入把相对姿态压成三维 `SO(3)` 对数向量，缺少末端速度，并且姿态奖励主要依靠距离门控。旧实验已经暴露出姿态课程停滞、位置能力遗忘和价值高估等问题。v8 同时更改 observation 与 reward 的语义，旧网络输入维度、critic 值函数和 replay 奖励都已不兼容，因此必须从随机初始化重新训练。

旧文档和 `outputs/thesis_homotopy/` 保留为历史证据，但状态已标记为“终止”，不删除、不续训、不参与 v8 checkpoint 选择。

## V8 改动摘要

- 策略输入中的末端姿态和目标姿态均使用旋转矩阵前两列的连续 6D 表示。
- 相对姿态输入由三维 `SO(3)` log 改为相对旋转矩阵的 6D 表示；测地角只保留为标量观测、奖励误差和成功判定依据。
- 加入末端线速度、角速度；保留 `q`、`qdot`，并统一用动作速度上限 `0.7 rad/s` 归一化 `qdot` 与速度奖励。
- 平滑正则由六关节归一化速度增量的平方和改为逐关节均方，保留 `0.01` 上限，但不再让绝大多数普通动作都落在同一饱和值。
- 障碍部分显式包含六段连杆各自的接近速度和风险，不只给全局最大风险。
- 到达奖励改为有界、非线性、位置调制姿态的折扣势函数；安全项仍只由 `xi` 调度，因此不会破坏“到达能力优先的安全同伦”。
- v8 observation 为 98 维，输出目录与 v2～v7 物理隔离。

详细定义和失败复盘见 [experiment_plan.md](experiment_plan.md)，实际执行与停止决定见 [experiment_order.md](experiment_order.md)，完整实测记录见 [progress.md](progress.md)。论文中的正式表述以 [`docs/thesis/thesis_outline.md`](../thesis/thesis_outline.md) 前三章为准。

## V9 历史结果

v9 将 observation 从 98 维扩展到 122 维，加入基于 URDF collision mesh 的 self-clearance、self-TTC、self-approach 和 self-risk，并以独立 `lambda_self` 课程加入连续自碰撞代价。预检通过后，正式 S0 Block1 曾在 66646 步意外中断，随后从 50000 步完整 checkpoint 恢复，累计完成 100000 transitions。

Block2 已完成位置巩固和 self-safety ramp，最终为 `K_position=121709`、
`lambda_self=1`、`eta=0.4`、`K_pose=0`。最近固定位置 probe 虽达到 98% success，
仍有 1/50 collision，因而进入 recovery 且无 S0 Gate 资格。按当前课程状态计算，
到 Gate 的 transition 理论下界约 116k，超过原固定预算剩余的 100k。V9 最终决定
不执行 Block3，也不再建立扩展预算分支。详细统计与新旧预算的区分见
[progress.md](progress.md)，执行顺序见 [experiment_order.md](experiment_order.md)。

## V10 历史结果：600k S0

V10 seed 11001 已连续完成 600000 transitions，SAC 数值稳定，但最终停在 `eta=0.75`，`s0_goal_gate_eligible=false`，按协议记为 `hard_budget_stop`。当前档非锚点成功率从 `eta=0.60` 的 `79.6%` 降至 `eta=0.75` 的 `64.2%`，超时升至 `28.9%`；Block6 位置锚点成功率降至 `93.2%`。自碰撞是重要失败类型，但超时和位置—姿态联合收敛失败更主要。

诊断发现 V10 的 60000 条无障碍 FIFO 在长期停留于 `eta=0.75` 后只剩 `eta=0` 锚点与 `eta=0.75` 当前档，中间 `eta=0.1..0.7` 历史数据全部被覆盖。因此 V10 的“历史分层 replay”并未实现持久历史保留。V10 checkpoint/replay 不得用于 V11。

## V11 当前协议

V9 的 `lambda_self` 在位置阶段结束后即与姿态课程同时升到 1，无法隔离“完整位姿尚未掌握”和“连续安全代价阻碍探索”两种原因；其未归一化 self penalty 最高可达每步 `-10`，累计尺度也可能把失败转成保守 timeout。因此 V9 不再训练 Block3，也不从其 checkpoint 续训。

V11 保留 122 维连续 self observation、真实 self-contact 硬终止、奖励与四阶段 S0 课程；核心修复是持久 eta replay。无障碍数据拆为独立的 anchor FIFO、current eta FIFO 和每个已完成 eta 档的冻结历史 snapshot。默认容量为 `15000/30000/3000`（anchor/current/history-per-level），因此长期停留在一个困难档位不会覆盖历史档位。V11 继续使用 `[256,256]` MLP，不加入 LSTM 或未经动力学验证的专家 replay。

V11 预检已通过：定向 `34 passed`、全量 `98 passed`；300-step 真实轨迹加 50-step 恢复保持 replay 和累计 step 连续，V10 checkpoint 因 protocol 不匹配被拒绝。正式 V11 仍必须从随机初始化开始。

补充审查动态惩罚与自由漂浮机械臂奖励论文后，不修改 V10 契约：前者支持现有渐进安全权重，但其 critic-loss 切换阈值不适合动态重标的 SAC；后者的四元数姿态、速度平方和碰撞项已被 V10 等价或更严格覆盖，自由漂浮基座项不适用，未标定白噪声不进入 nominal 训练。逐项依据见 [reward_literature_analysis.md](reward_literature_analysis.md)。

正式训练从随机初始化开始：

```bash
python scripts/train_thesis_homotopy.py \
  --config configs/experiments/thesis_homotopy.yaml \
  --stage s0 --seed 11001 \
  --run-name s0_v11_seed11001_block1
```

Block 是 100k transitions 的保存与分析单元，每 25k 保存 checkpoint，不再是“最多三个”的停止上限。所有正式 seed 和消融都使用每 stage 累计 600k 的统一硬预算。S0 只有连续两个相邻完整 Block 均未推进 `eta`，且当前档 success、自碰撞、timeout、anchor 和固定 probe 均未达到预注册改善阈值，才判平台停止；跨档总体成功率不参与判断。NaN/Inf 或持续 Q/critic 发散单独停止。`s0_goal_gate_eligible=true` 后还必须让同一 checkpoint 在 `41001/42002/43003` 三个 validation seed 上分别通过，才算 S0 成功。
