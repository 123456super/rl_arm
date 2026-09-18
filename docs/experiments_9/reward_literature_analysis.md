# 五篇奖励与课程论文的方法审查

更新日期：2026-09-17

本文记录五篇参考文献的可迁移结论、适用边界，以及它们对 V10/V11 协议的实际影响。V11 只在 V10 的任务优先奖励课程之外修复持久 eta replay，不把 replay 修复误写成新的奖励机制。结论不是把方法机械叠加，而是只采用与固定基座刚性 UR5、全状态观测、离策略 SAC 相容的部分。

## 1. 非线性奖励模块（RFM）

论文：Jiang 等，*Learning Whole-Body Loco-Manipulation for Omni-Directional Task Space Pose Tracking With a Wheeled-Quadrupedal-Manipulator*，IEEE RA-L，2025。

论文的 Reward Fusion Module 有三部分：

1. Reward Prioritization：用乘法 `r_p + r_p r_R` 建立层级，使位置始终有梯度，而姿态贡献随位置质量增加。
2. Micro-Enhancement：将 `r` 改为 `r+r^M`，只显著增加接近目标时的分辨率。
3. 累计误差与阶段融合：把误差积分纳入状态，并用 sigmoid 相位变量在 locomotion/manipulation 奖励组之间平滑切换。

可直接借鉴的是前两项。当前实现使用

```text
H(u) = 0.5 (u + u^4)
Phi = H(u_p) [1 + eta H(u_R)]
r_task = 20 [gamma Phi(s') - Phi(s)] + 20 I_success - r_motion
```

`0.5` 只用于保持 `H` 的范围为 `[0,1]`。乘法使姿态改善不能用明显的位置退化换取总收益；折扣势函数差使用与 SAC 相同的 `gamma`，避免往返刷分。V8/V9 的真实转移审计已经验证了这一方向，因此 V10 保留。

不采用累计误差项。该项会引入一个历史累积状态；若不把它完整加入 observation，环境不再是 Markov 的。当前已有剩余时域、timeout、势函数进展和成功终止，没有证据表明稳态误差是主要瓶颈。论文的 locomotion/manipulation sigmoid 融合也不适用于固定基座 UR5。

## 2. 复杂奖励的两阶段切换

论文：Freitag 等，*Curriculum Reinforcement Learning for Complex Reward Functions*，arXiv:2410.16790v2，2025。

核心算法先使用 `r_base` 学会任务，再切换到 `r_base+w_c r_constraint`。Replay 同时保存基础奖励和完整奖励所需信息，第二阶段重算目标。消融结果表明主要收益来自保留第一阶段网络参数；清空 replay 的影响较小。方法在“基础任务可学，但约束会阻碍探索”时最有效；基础任务本身学不会时无效。论文也观察到切换后任务能力可能退化。

V9 与这一结论冲突：位置巩固结束后，姿态课程和 `lambda_self:0.02->1` 同时开始。也就是说，完整位姿任务尚未学会时，连续自碰撞约束已经逐步增强。V10 改为三个严格顺序的 S0 相位：

```text
位置课程 -> 完整位姿课程 -> 连续自碰撞裕量课程
```

前两相 `lambda_self=0`。这不关闭安全：reset/goal 仍拒绝自碰撞，真实 self-contact 仍逐物理子步检测、立即终止并扣 `10`。第三相仅增加碰撞发生前的 clearance/TTC 连续代价。

不采用论文的 Actor-Q loss 阈值作为切换条件。Q 值尺度受熵温度、reward 尺度和动态重标影响，阈值跨协议不可解释。V10 使用直接任务指标：完整位姿最近 100 episode 成功率、位置锚点最近 100 episode 成功率、25k 完整位姿 transition，以及不接触私有训练数据的确定性位置保持 probe。指标跌破门槛时冻结 `lambda_self`，不继续加约束，也不回退权重。

## 3. 专家经验、课程与 LSTM

论文：Cheng 等，*A collision-free motion planning method for cable-drive redundant manipulators with deep reinforcement learning-based expert guidance and long short-term memory*，Expert Systems with Applications，2026。

论文组合了三项机制：固定目标到随机目标再到障碍场景的三阶段课程；由 APF/RRT/GA 生成且经 MuJoCo 动力学筛选的专家轨迹；主 replay 与专家 replay 混采，并用到最近专家状态的距离 `exp(-k d_min)` 塑形。它还用 30 帧 LSTM 处理柔性缆索的迟滞、振动和执行延迟。论文报告低比例专家样本通常优于过高比例，过多专家数据会造成 policy locking。

可借鉴的是难度递增和“专家比例必须消融、不能越多越好”的原则。当前 `g` 课程已经从局部目标逐步扩展到完整随机目标，S0/S1/S2 再逐步增加障碍复杂度，因此不需要复制其固定目标阶段。

V10 不加入专家 replay，原因如下：

- S0 奖励并不稀疏，V9 位置阶段已达到约 99.6% 成功，瓶颈不是找不到任何有效轨迹。
- 当前没有经过同一 URDF、关节速度限制、逐子步自碰撞检查验证的专家动作数据。
- 目标由一个合法关节构型 FK 得到，但直接使用隐藏的目标关节角生成动作会向策略泄漏 observation 中没有的 IK 分支信息。
- 最近专家“绝对状态”距离若混合不同目标，会奖励接近别的任务轨迹；它也不是保证策略不变的势函数塑形。

V10 也不加入 LSTM。当前刚性 UR5 的 observation 已含 `q/qdot`、末端 twist、障碍位置/速度、剩余时域和连续几何量，在当前仿真假设下是近似 Markov 的；论文中 LSTM 的依据是缆索迟滞和未观测柔性动力学。无依据增加循环网络会同时改变 replay 采样、burn-in、hidden-state checkpoint 和网络容量，无法把收益归因于奖励课程。

## 4. 动态惩罚函数

论文：Yoo、Zavala 与 Lee，*A Dynamic Penalty Function Approach for Constraint-Handling in Reinforcement Learning*，2021。

论文先用 Kreisselmeier-Steinhauser（KS）函数平滑聚合多个不等式约束，再定义只在违反约束时生效的惩罚

```text
p(x,u|mu) = mu KS[g(x,u)] I(KS[g(x,u)] > 0).
```

`mu` 从较小值开始，在神经网络 loss 降至历史最大 loss 的指定比例后乘以常数 `c>1`，直到 `mu_max`。论文的车辆实验使用 `mu:0.05->20`、每次翻倍，并在 100 个随机 seed 中得到 93 个可行策略、83 个“可行且低代价”策略；固定 uniform/linear penalty 对应为 82/42 和 62/48。论文的主要解释是：一开始就施加陡峭大惩罚会让值函数在约束边界附近难以拟合，逐渐增大惩罚可降低近边界逼近偏差。

这个结论支持 V10，但不要求再改代码：

- `lambda_self` 和 `xi_scene` 已经是动态惩罚系数；连续安全组先归一化至 `[0,1]`，再从低权重逐渐增加，正面解决论文所述的陡峭值函数问题。
- V10 比论文多一道任务保持门槛。只有完整位姿成功率、位置锚点和确定性 probe 保持时才推进 `lambda_self`；这比依赖 loss 更直接地验证“基本任务没有被约束压垮”。
- 不采用论文的历史最大 critic loss 比例。SAC 的 critic loss 会随熵温度、target network、replay 分布和在线 reward 重标共同变化；一次早期异常峰值还能永久改变阈值，因此它不是本任务中可跨 seed 解释的课程指标。
- 不把当前逐连杆最坏值聚合改成 KS。当前 reward 使用 `max risk` 与 `min clearance`，直接对应最危险连杆；KS 是近似最大值，可能稀释单一危险 pair，而且改动需要 replay 保存所有 pair 原子量。现有连续代价已经有界且在安全边界处连续，没有论文中“大斜率线性罚”的主要病因。

因此，这篇论文提供的是对 V10/V11 动态安全课程的独立理论与实验支持；V11 的修改理由来自 600k 训练后实际 replay 覆盖审计，而不是论文提出的新奖励项。

## 5. 自由漂浮六自由度机械臂奖励函数

论文：Al Ali 等，*Path planning of 6-DOF free-floating space robotic manipulators using reinforcement learning*，Acta Astronautica，2024。

论文的 DDPG reward 包含位置误差、四元数姿态误差、进入位置/姿态容差后的常数奖励、关节速度平方和，以及低于安全距离时的常数碰撞惩罚：

```text
R = -Kd d - Ktheta theta + rd + rtheta - pc - pj sum(qdot_j^2).
```

其实验显示加入速度项后关节速度平方和下降超过三倍，但开始收敛稍慢；论文还报告了自由漂浮基座质量比和乘性白噪声 observation 下的单独案例，并指出噪声会加剧动作/力矩振荡。

与当前实现逐项比较：

| 论文机制 | V10 状态 | 决策 |
| --- | --- | --- |
| 四元数姿态误差 | V10 用 6D rotation observation 和 `SO(3)` 主值误差，已有四元数符号不变测试 | 保留 V10；不退回原始四元数四维输入 |
| 位置与姿态奖励 | V10 用 RFM 乘法门控的折扣势函数差，避免绝对误差每步累积 | 保留 V10 |
| 容差内持续正奖励 | V10 到达即终止并给一次 `+20`，不存在“到达后保持”的任务定义 | 不采用；否则改变有限时域 MDP，并可能奖励拖延终止 |
| 关节速度平方 | 已有归一化 `mean((qdot/0.7)^2)`，权重 `0.04` | 已采用 |
| 动作平滑 | V10 另有 `mean(((qdot_next-qdot)/0.7)^2)`，权重 `0.01`；比论文只罚速度更完整 | 已采用 |
| 常数碰撞罚 | V10 有逐子步 contact 硬终止、`-10` 和接触前连续 risk/clearance | 不采用论文的较弱常数罚 |
| 自由漂浮基座扰动 | 当前 UR5 固定基座，不存在基座动量耦合 | 不适用 |
| observation 噪声 | 当前第一阶段明确使用仿真真值，论文只给一个特定乘性白噪声案例 | 不混入 nominal；获得传感器误差标定后另建鲁棒性协议 |

论文对当前最有价值的提醒是必须同时报告速度、加速度、jerk 和动作振荡，不能只看成功率。审查时发现总纲虽已要求这些指标，原 evaluator 却没有实现；现已补齐每个 episode 的 command/measured velocity、acceleration、jerk 的 RMS/peak，以及每个场景的 mean/p95/max 汇总。若以后面向 sim-to-real，应先按实际编码器、目标位姿估计和延迟数据定义噪声模型，再建立独立训练/评价分支；随意加入论文的白噪声既不能代表真实传感器，也会破坏当前可归因的 nominal 基线。

## 6. V10 最终设计

V10 保留 122 维 observation 和 `[256,256]` MLP，改变奖励与课程契约：

```text
c_external = (2 R_external + 8 V_external) / 10
c_self     = (2 R_self + 8 V_self) / 10

r = r_task
    - xi_scene c_external
    - lambda_self c_self
    - 10 I_hard
    - 10 I_strict_obstacle_terminal
```

两个连续安全组均在 `[0,1]`，避免一个持续存在的约束项以每步最多 `-10` 压过任务学习。权重 `2:8` 仍表达预测风险与真实安全边界穿入的相对优先级。严格阶段的 obstacle contact 另有一次性 `-10` 终止惩罚，防止策略用提前碰撞规避 timeout；宽容阶段不使用该项。

S0 状态机为：

| 相位 | 条件与行为 |
| --- | --- |
| position | `g:0.03->1`，`eta=0`，`lambda_self=0`；完成 `K_position>=25k` |
| pose | `eta:0->1`，`lambda_self=0`；保持位置锚点和分层 replay |
| self_safety_ramp | 仅当 `eta=1`、`K_pose>=25k`、位姿窗口 `>=0.80`、锚点窗口 `>=0.95` 且 probe 通过时，`lambda_self:0->1`；任一窗口下降即冻结 |
| self_safety_consolidation | `lambda_self=1` 后再累计至少 25k 合格 transition |

Replay 已保存重算奖励所需的 22 个原子量，因此相位切换时保留 actor、critics、optimizer 和 replay，并按当前 `lambda_self` 重标奖励。这与两阶段奖励课程的主要有效机制一致。

这些论文支持任务优先、约束渐增和分层评价，但都不能推出“累计 300k 未通过即学习失败”。V10 因此把样本预算与停滞判据分开：Block 仅作为 100k 分析单位，每 stage 对所有正式 seed 和消融统一使用 600k 硬预算；S0 平台只根据连续两个相邻完整 Block 的同一 `eta` 指标判定。600k 保证比较公平，平台规则用于判断是否仍在学习，两者不得互相替代。

## 7. 必须做的消融

V10 是有理论依据的候选方案，不是已经证明更好的结果。正式结论至少需要同 seed、同目标与同预算比较：

| 组 | 连续 self observation | 连续 self reward | 顺序课程 |
| --- | --- | --- | --- |
| V8-style | 无 | 无 | 无 |
| observation-only | 有 | 无 | 不适用 |
| V9-style | 有 | 未归一化且与姿态并行 | 否 |
| V10 nominal | 有 | 有界且完整位姿后启用 | 是 |

主要指标必须同时报告完整位姿成功率、自碰撞率、timeout、最小 self-clearance、位置保持和达到各课程相位所需 transitions。只降低碰撞但把失败转成 timeout，不算改进。
