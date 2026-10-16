# 面向严格联合位姿约束的高精度机械臂局部收敛增强 SAC 方法

> 版本：v14（冻结对外基线命名、等容量 M3 与互斥 Replay 分类）  
> 日期：2026-10-14

## 1. 核心问题

研究对象不是机械臂能否到达目标附近，而是：

\[
\text{全局可到达}
\rightarrow \text{联合约束持续收紧}
\rightarrow \text{目标邻域局部收敛退化}。
\]

重点分析三个问题：局部方向辨识、连续关键点优化与联合成功条件不完全对齐、精度边界样本不足。

## 2. 方法主线

公共基础是 SAC、显式六维位姿误差、已有三关键点联合位姿误差和 L0–L14 精度课程。基线称为 **Hybrid KP-SAC-Base**（简称 **Base**）：它在显式位置/姿态误差基础上进一步使用三关键点联合几何误差和稠密关键点 tracking/progress 奖励，而不是用关键点完全替代显式位姿表示，并使用公共 precision action scaling 和课程学习。Base 不额外引入针对局部高精度阶段的 threshold-aware precision quality、hold/leave、Success Replay 及本文的局部增强机制。论文正文、图表、公式、方法名和图例统一使用该正式名称。三关键点设计不作为本文独立原创模块。成功必须连续 5 个控制周期同时满足位置和姿态阈值，以区分进入目标区域和稳定局部收敛。课程阶段固定为 L0–L4 联合收紧、L5–L11 固定 0.10 rad 姿态并收紧位置、L12–L14 固定 1 mm 位置并收紧姿态。

\[
\text{Full}=\text{Hybrid KP-SAC-Base}+\text{precision quality/hold/leave}
+\text{Success Replay}+\text{Keypoint Jacobian}
+\text{Joint Constraint Bottleneck Shaping}+\text{Precision Boundary Replay}.
\]

三个复合模块分别对应：

- **Direction**：Keypoint Jacobian 提供局部关节运动方向；
- **Objective**：precision quality、hold/leave 与 Joint Constraint Bottleneck Shaping 共同提供阈值感知、稳定保持和最差约束优化信号；
- **Data**：Success Replay 与 Precision Boundary Replay 共同增加成功轨迹和严格成功边界附近的有效经验。

Objective 中的 Bottleneck 使用 terminal-gated 形式：
\[
\Phi_b(s)=5\exp[-\rho_b(s)/2],\qquad
r_{b,t}=(1-d_t)[\gamma\Phi_b(s_{t+1})-\Phi_b(s_t)],
\]
其中 \(d_t=1\) 表示 success、timeout、collision 或 hard failure terminal。所有 terminal transition 的 \(r_b=0\)，避免引入与目标接近程度相关的隐藏终止奖惩；各类终止结果由已有专门奖励项处理。

论文的一句话研究问题是：显式六维误差与已有三关键点联合几何误差构成的 Hybrid 表示已经提供较强的位姿状态信息，但其在严格联合位姿约束下是否足以支持稳定的局部精细修正仍需检验；本文从方向、目标和数据三个层面研究 SAC 的局部收敛增强。

模型输入也作为方法合同冻结。Hybrid KP-SAC-Base 的 Actor、Critic 均使用 108 维公共状态，包括 6 维关节位置、6 维关节速度、9 维三关键点误差、3 维显式位置误差向量、3 维显式姿态误差向量、2 维误差范数、6 维末端线/角速度，以及 73 维固定的外部障碍与自碰撞几何状态。因此 “Hybrid” 明确指显式六维误差与三关键点误差并存；基础奖励仍为统一关键点奖励，不额外引入一套显式六维奖励。当前无障碍任务中的外部障碍通道使用相同的零值或安全默认值占位；绝对目标位姿和绝对末端位姿不单独输入。Full、M1、M2 在 Actor 端增加世界坐标系下的 \(9\times6\) 三关键点 Jacobian（54 维，对应 observation 切片 `[21:75]`），Actor 因而接收 162 维状态；M3 使用相同的 162 维 Actor 结构，但把该 Jacobian 切片固定为 \(0_{54}\)。因此 Full–M3 只改变 Jacobian 信息，不改变 Actor 第一层参数量。Critic 始终接收去除 Jacobian 槽后的 108 维公共状态和 6 维动作。主要归一化尺度固定为：关键点误差 `0.65 m`、显式位置误差 `0.05 m`、显式姿态误差 `0.50 rad`、关键点 Jacobian `1.1`。完整逐项输入和归一化定义以 `experimental_protocol.md` 的“模型输入合同”为准。

| 方法 | Actor 状态输入 | Critic 状态输入 | Jacobian 作用位置 |
|---|---:|---:|---|
| Hybrid KP-SAC-Base | 108 | 108 | 无 |
| Full | 162 | 108 | 仅 Actor |
| M1：Full−Bottleneck | 162 | 108 | 仅 Actor |
| M2：Full−Boundary Replay | 162 | 108 | 仅 Actor |
| M3：Full−Jacobian | 162（Jacobian 槽为 \(0_{54}\)） | 108 | 无有效 Jacobian 信息 |

## 3. 实验结构

Hybrid KP-SAC-Base 与 Full 使用训练种子 `11001` 和 `11002`；M1、M2、M3 使用预先指定的单一训练种子 `11001`。所有运行的训练预算统一为 `6,000,000 transition`，从随机初始化开始。Hybrid KP-SAC-Base、Full 和 M1–M3 共用相同的 L0–L14 课程、自动升档、probe、确认评估和最终冻结评估协议；具体奖励与 replay 差异按表和实验协议冻结。单种子消融用于机制归因和失败模式分析，不用于估计跨随机种子的方差。

| 方法 | 验证内容 |
|---|---|
| Hybrid KP-SAC-Base | Hybrid 表征、稠密关键点奖励、公共精度执行接口与精度课程的基础能力边界 |
| Full | 完整 Direction–Objective–Data 高精度框架相对基础 Hybrid 基线的整体收益 |
| M1：Full−Bottleneck | 最差归一化约束的连续势函数是否提供额外收益 |
| M2：Full−Boundary Replay | 精度边界经验是否有效 |
| M3：Full−Jacobian | 局部运动方向信息是否有效 |

所有方法保持相同的目标分布，因此远距离和大姿态差结果用于检查 Full 是否以牺牲全局覆盖换取局部精度，而不再引入 semantic replay 或 far-distance restart 作为额外创新。最终测试种子固定为 `41001、41002、41003`，与用于升档的验证种子 `31001、31002、31003` 分离。

所有方法也使用相同的同档 replay 合同：L0–L14 之间不跨档采样。升档后清空当前训练池并先收集 1024 个新档 transition，再恢复更新；旧档快照只用于审计。原因是精度阈值不仅改变 success、Bottleneck 和 precision reward，还通过公共 precision action scaling 改变实际关节速度及下一状态，所以旧档 transition 不能只靠重算 reward/done 严格转换成新档 MDP。成功 episode 的完整序列进入当前成功池；Boundary candidate 仅来自未达到连续成功终止的 episode，且当前状态满足 \(0.8\leq\rho_{\max,t}\leq1.5\)，因此 \(Success\cap Boundary=\varnothing\)。对启用特殊池的 Full 家族配置，三类身份按 Success、Boundary、Ordinary 的优先级互斥分配。Full/M1/M3 的稳定配额为 `20% Success + 20% Boundary + 60% Ordinary`，M2 为 `20% Success + 80% Ordinary`，Hybrid KP-SAC-Base 为 `100% Ordinary`；Base 不启用特殊池，其 Ordinary 流保留当前档全部普通 transition。特殊类别候选不足时，未满足的 quota 回退到 Ordinary。

## 4. 结果解释顺序

Full 与 Hybrid KP-SAC-Base 分别使用两个训练种子在 6M 结束时的最终冻结 Actor，在 L6、L9、L11、L12、L13、L14 上评估并汇总均值和离散程度；M1–M3 使用预先指定的 `11001` 单种子最终 Actor，采用相同冻结评估，但不估计跨种子方差，也不挑选最佳 seed。Hybrid KP-SAC-Base 与 Full 的主比较解释为完整 Direction–Objective–Data 框架的整体收益；M1–M3 只用于 Full 家族内部归因。具体而言，Jacobian、Bottleneck 与 Boundary Replay 的增量作用分别由 Full–M3、Full–M1 和 Full–M2 验证；M1/M2 不用于声称完整 Objective/Data 模块的单项效果。同步记录每个方法首次通过各档位所需的 `T_Lk`，未通过时保持当前档位，达到 L14 后保持 L14 至 6M 结束。随后用 M1–M3 的失败模式归因：方向失配、terminal position-only / orientation-only / joint failure，以及边界附近误差反弹或保持失败。

所有局部分析统一使用

\[
\rho_{\max}=\max\left(\frac{e_p}{\epsilon_p},\frac{e_R}{\epsilon_R}\right),
\]

并将 \(\rho_{\max,t}\leq1.5\) 冻结定义为当前状态位于目标邻域。边界回放只从未稳定成功的 episode 中选择当前状态 \(s_t\) 满足 \(0.8\leq\rho_{\max,t}\leq1.5\) 的 transition，并占 batch 的 20%；成功 episode 的任一 transition 均不属于 Boundary。\(s_{t+1}\) 只用于分析动作结果，不决定几何边界成员资格。局部反向修正率同样只统计当前状态位于目标邻域的 transition：

\[
N_{local}=\sum_t\mathbb I[\rho_{\max,t}\leq1.5],
\]
\[
R_{reverse}=
\frac{
\sum_t\mathbb I[\rho_{\max,t}\leq1.5]\,
\mathbb I[\rho_{\max,t+1}>\rho_{\max,t}]
}{N_{local}}.
\]

这表示 Actor 在已经接近联合成功边界后，反而使最差归一化约束变坏的动作比例。终止 transition 使用 reset 前的真实 terminal state 作为 (t+1)。未稳定成功 episode 的 position-only、orientation-only 和 joint failure 均由 reset 前的真实 terminal state 判定：

\[
\begin{aligned}
\text{Position-only failure}:&\quad e_p>\epsilon_p,\ e_R\le\epsilon_R,\\
\text{Orientation-only failure}:&\quad e_p\le\epsilon_p,\ e_R>\epsilon_R,\\
\text{Joint failure}:&\quad e_p>\epsilon_p,\ e_R>\epsilon_R.
\end{aligned}
\]

三个 failure rate 均以全部评估 episode 数为分母；稳定成功 episode 不计入三类。未稳定成功但 terminal state 单帧同时满足两项阈值的 episode 单独作为 terminal joint-pose hit 诊断。

Jacobian 只作为 Actor augmentation；Critic 使用去除 Jacobian 后的公共状态。诊断使用 DLS 下降方向，而不是未阻尼伪逆：
\[
\dot q^*=+J_{kp}^{\mathsf T}(J_{kp}J_{kp}^{\mathsf T}+\lambda_{DLS}^2I)^{-1}e_{kp},
\qquad \lambda_{DLS}=0.05.
\]
其中 `e_kp=x_kp^goal-x_kp`，cosine similarity 使用实际 action scale 映射后的关节速度计算。

同时报告远距离桶、大姿态差桶、位置和姿态误差 P95、task-cell P10，避免只用单一1 mm成功率作为结论。

## 5. 执行顺序

1. 使用五个正式配置从随机初始化训练；
2. 完成 Full/Base 的两个训练种子，以及 M1–M3 预先指定的 `11001` 单种子，并使用统一 deterministic probe / confirmation；
3. 在 L6、L11、L14 对所有方法的最终 6M Actor 做 1000 episodes 冻结测试；通过宽松、中等和严格三个代表档位保留精度曲线逻辑，升档 probe/confirmation 仍按原协议执行；
4. 先完成 Hybrid KP-SAC-Base 与 Full 的整体主比较，再用 M1–M3 做 Full 家族内部归因；Full–M1、Full–M2、Full–M3 分别验证 Bottleneck、Boundary Replay、Jacobian，M1/M2 不代表删除完整 Objective/Data 模块；
5. 旧实验数据不进入新主表。

论文措辞在实验完成前使用检验性表述：本文检验使用显式六维误差、已有三关键点联合误差、稠密关键点奖励、公共精度执行接口和精度课程的 Hybrid KP-SAC-Base 是否仍会出现局部收敛退化，并比较完整高精度方案的整体改善。三个核心新增机制中的 Jacobian、Bottleneck 与 Boundary Replay 的增量作用分别由 Full–M3、Full–M1 和 Full–M2 验证；Full 与 Hybrid KP-SAC-Base 的比较用于评价完整 Direction–Objective–Data 高精度框架的整体收益，不将 M1/M2 解释为完整 Objective/Data 模块的消融。
