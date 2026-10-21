# 论文实验协议（Hybrid KP-SAC-Base / Full / M1–M3）

> 版本：v14（冻结对外基线命名、等容量 M3 与互斥 Replay 分类）  
> 状态：旧实验全部废弃；新实验从随机初始化开始。  
> 日期：2026-10-14

## 1. 研究对象

论文研究严格联合位姿约束收紧后，SAC 在目标邻域的局部收敛退化：动作方向是否正确、连续关键点优化目标是否与位置和姿态必须同时达标的联合成功条件对齐，以及精度边界样本是否足够。

公共基础为：SAC、显式六维位姿误差、已有三关键点联合位姿误差和精度课程。三关键点表示来自已有工作，不作为本文独立创新。本文基线称为 **Hybrid KP-SAC-Base**：其观测同时保留显式位置/姿态误差与三关键点联合几何误差，而不是用关键点完全替代显式位姿表示；奖励保留稠密关键点 tracking/progress、必要代价与成功奖励，并使用公共精度执行接口和课程学习，不额外引入针对局部高精度阶段的阈值辅助奖励与经验重采样机制。论文正文、图表、公式、方法名和图例统一使用 **Hybrid KP-SAC-Base**（简称 **Base**）。

## 2. 正式方法

| 方法 | 配置 | 定义 |
|---|---|---|
| Hybrid KP-SAC-Base | `paper_kp_sac.yaml` | 显式位置/姿态误差 + 三关键点联合误差 + tracking/progress 关键点奖励 + 公共 precision action scaling + 精度课程；关闭 precision quality、hold/leave、Success Replay、Jacobian、bottleneck shaping 和 Boundary Replay |
| Full | `paper_full.yaml` | 完整基础奖励与 Success Replay + Keypoint Jacobian + Joint Constraint Bottleneck Shaping + Precision Boundary Replay |
| M1 | `paper_m1_no_bottleneck.yaml` | Full 去除 bottleneck shaping |
| M2 | `paper_m2_no_boundary_replay.yaml` | Full 去除 precision boundary replay |
| M3 | `paper_m3_no_jacobian.yaml` | Full 去除 keypoint Jacobian 信息，但保留 54 维全零 Jacobian 槽和相同 Actor 结构 |

Full、M1、M2、M3 均关闭 semantic long-term replay 和 far-distance timeout restart。远距离和大姿态差只作为覆盖评价指标，不作为本文新增机制。

### 2.1 三个复合增强模块

Full 的方法叙事统一为三个层面：

| 复合模块 | 组成 | 作用 |
|---|---|---|
| Direction | Keypoint Jacobian | 告诉 Actor 当前关节动作如何改变关键点误差 |
| Objective | precision quality + hold/leave + terminal-gated bottleneck shaping | 提供阈值感知、稳定保持和最差归一化约束的连续非终止优化信号 |
| Data | Success Replay + Precision Boundary Replay | 强化完整成功轨迹和严格精度边界附近的 transition |

因此方法关系写为
\[
\mathrm{Full}=\mathrm{Hybrid\ KP\!\text{-}\!SAC\!\text{-}\!Base}
+\mathrm{Direction}+\mathrm{Objective}+\mathrm{Data}.
\]
M1 只删除 Objective 中的 bottleneck 子项，M2 只删除 Data 中的 Boundary Replay 子项，M3 删除完整 Direction 信息。Full 与 Base 的比较评价完整 Direction–Objective–Data 高精度框架的整体收益；Full–M1、Full–M2 和 Full–M3 分别验证 Bottleneck、Boundary Replay 和 Jacobian 的增量作用，不能把 M1/M2 表述为删除或验证完整 Objective/Data 模块。

## 3. 公共训练合同

所有方法使用相同的 UR5、PyBullet、动作空间、目标分布、SAC 超参数、课程、transition 预算和评估协议。Full 与 Hybrid KP-SAC-Base 使用训练种子 `11001`、`11002`；M1、M2、M3 使用预先指定的单一训练种子 `11001`。所有运行的训练预算均为 `6,000,000 transition`，从随机初始化开始，不继承旧 Actor、Replay、optimizer 或 curriculum state。

本轮执行冻结为 6M：已经超过该预算的基础基线运行在最近完整 checkpoint 后暂停，超过 6M 的部分仅保留作训练诊断，不纳入固定预算主比较；Full 从最新完整 checkpoint 继续训练至 6M。

Hybrid KP-SAC-Base、Full、M1、M2、M3 的显式位姿误差输入、基础关键点输入、目标采样和 ordinary current-level replay 完全相同。这里的 “Hybrid” 专指观测同时包含显式六维误差与三关键点误差；它不表示额外加入一套显式六维奖励。Hybrid KP-SAC-Base 不包含上述三个复合增强模块；Full 与 M1–M3 按各自定义启用相应组成。因此 Full 对 Hybrid KP-SAC-Base 的主比较衡量完整 Direction–Objective–Data 框架的整体收益，三个关键新增机制的单项归因使用 Full–M3（Jacobian）、Full–M1（Bottleneck）和 Full–M2（Boundary Replay）的成对比较。

### 3.1 模型输入合同

所有方法共享同一组 108 维 Hybrid 公共状态。它并非 Portela 式“关键点替代显式位置/姿态表示”：9 维三关键点联合误差与显式位置误差向量、显式姿态误差向量及两项误差范数同时输入网络。三关键点由末端坐标系中边长为 `0.5 m` 的立方体上三个非共线顶点构造；当前关键点和目标关键点均转换到机器人基座/世界坐标系，并使用 `goal-current` 作为误差方向。公共状态按下表拼接：

| 公共状态量 | 维数 | 归一化或取值约定 |
|---|---:|---|
| 关节位置 \(q\) | 6 | 按各关节上下限线性映射到 `[-1,1]` |
| 关节速度 \(\dot q\) | 6 | 除以 `0.8 rad/s` 后裁剪到 `[-1,1]` |
| 三关键点误差 \(e_{kp}\) | 9 | 世界坐标系；除以 `0.65 m` 后裁剪到 `[-1,1]` |
| 显式位置误差向量 \(e_p^{vec}\) | 3 | `goal-current`；除以 `0.05 m` 后裁剪到 `[-1,1]` |
| 显式姿态误差向量 \(e_R^{vec}\) | 3 | \(SO(3)\) 对数映射；除以 `0.50 rad` 后裁剪到 `[-1,1]` |
| 位置、姿态误差范数 | 2 | 分别除以 `0.05 m`、`0.50 rad` 后裁剪到 `[0,1]` |
| 末端线速度 | 3 | 除以 `0.55 m/s` 后裁剪到 `[-1,1]` |
| 末端角速度 | 3 | 除以 `1.5 rad/s` 后裁剪到 `[-1,1]` |
| 外部障碍相对向量（6 连杆 × 3） | 18 | 每维裁剪到 `[-1,1]` |
| 外部障碍位置、速度 | 6 | 位置按工作空间映射到 `[-1,1]`；速度除以 `0.1 m/s` |
| 外部障碍距离、TTC、接近速度、风险 | 24 | 每类 6 维；按固定物理范围归一化 |
| 外部障碍存在标志 | 1 | `0/1` |
| 自碰撞距离、TTC、接近速度、风险 | 24 | 每类 6 维；按固定物理范围归一化 |
| **公共状态合计** | **108** | |

当前无障碍任务仍保留固定网络结构中的外部障碍通道：相对向量、障碍位置/速度、接近速度和风险填 0，距离和 TTC 填默认安全值，存在标志填 0。所有方法使用完全相同的这些占位量，因此它们不构成方法间变量。

绝对目标位置、目标四元数和绝对末端位姿不作为独立输入；目标信息通过三关键点误差、显式位置误差和显式姿态误差进入网络。

Full、M1、M2 的 Actor 在 108 维公共状态上额外接收 54 维关键点 Jacobian：

\[
J_{kp}=\frac{\partial(x_{kp,1},x_{kp,2},x_{kp,3})}{\partial q}
\in\mathbb R^{9\times6}.
\]

该矩阵与关键点误差使用同一世界坐标系，按关键点顺序 flatten，除以 `1.1` 后裁剪到 `[-1,1]`。在 162 维 Actor observation 中，排列固定为 \(q,\dot q,e_{kp},\operatorname{vec}(J_{kp})\) 和其余公共状态，Jacobian 对应 Python 切片 `[21:75]`。M3 在同一切片填入固定的 \(0_{54}\)，即其 Actor 输入为 \([o_{\rm Hybrid},0_{54}]\) 的等价插槽排列；因此 Full 与 M3 的 Actor 输入维数、网络结构和参数量完全一致，差异仅为是否提供 Jacobian 信息。Jacobian 不输入 Critic；所有 Critic 均接收删除该切片后的 108 维公共状态和 6 维动作。最终输入合同为：

| 方法 | Actor 状态输入 | Critic 状态输入 | Critic 动作输入 |
|---|---:|---:|---:|
| Hybrid KP-SAC-Base | 108 | 108 | 6 |
| Full | 162 = 108 + 54 | 108 | 6 |
| M1：Full−Bottleneck | 162 = 108 + 54 | 108 | 6 |
| M2：Full−Boundary Replay | 162 = 108 + 54 | 108 | 6 |
| M3：Full−Jacobian | 162 = 108 + \(0_{54}\) | 108 | 6 |

执行前检查：`paper_m3_no_jacobian.yaml` 及其环境/训练入口必须实际输出 162 维 Actor observation，并在 `[21:75]` 填入全零；若实现仍返回 108 维，则不得将该运行作为本协议的 M3 结果。

Actor 输出为 6 维归一化关节速度动作。所有五个正式方法共享同一 precision action scaling，它是公共控制基础，不属于 Full 的三个新增模块。令
\[
z_t=\rho_{\max,t}=\max(e_{p,t}/\epsilon_p,e_{R,t}/\epsilon_R),
\qquad \dot q_{\mathrm{actor}}=0.7a_t,
\]
则实际执行命令为
\[
\dot q_{\mathrm{cmd}}=g(z_t)\dot q_{\mathrm{actor}},
\quad
g(z)=
\begin{cases}
0.05+0.05z, & z\leq1,\\
0.10+0.90\min(z-1,1)^2, & z>1.
\end{cases}
\]
因此 \(g(0)=0.05\)、\(g(1)=0.10\)、\(g(2)=1.0\)，并在 \(z\geq2\) 后保持 1.0；随后再执行统一的安全投影和动作裁剪。训练、升档 probe、确认评估和最终测试均使用同一函数。跨 L0–L13 评估得到的是“固定 Actor + 档位相关 precision scaling + 档位相关成功阈值”的 strictness–performance curve，而不是脱离该控制包装器的固定闭环控制律精度极限。

上述输入、排列、归一化和 precision action scaling 在所有方法与所有评估阶段保持一致。

### 3.2 Hybrid KP-SAC-Base 与 Full 奖励合同

五个正式配置均使用统一关键点奖励分支（`keypoint_pose_reward: true`）。令 \(d_t,d_{t+1}\) 为三个关键点平均距离，
\[
Q_{kp,t+1}=\frac{1}{3}\sum_{i=1}^{3}
\exp\left(-\frac{\lVert x_{i,t+1}-x_i^g\rVert_2}{0.05\,\mathrm m}\right),
\qquad \Delta d_t=d_t-d_{t+1}.
\]
当前档位阈值定义
\[
\rho_{p,t}=e_{p,t}/\epsilon_p,\qquad
\rho_{R,t}=e_{R,t}/\epsilon_R,
\]
以及
\[
Q_{\mathrm{prec},t+1}=
\exp\left[-\frac{\rho_{p,t+1}+\rho_{R,t+1}}{2.0}\right].
\]
Hybrid KP-SAC-Base 实际进入 Critic target 的奖励为
\[
\begin{aligned}
r_{\mathrm{Base}}={}&
0.05Q_{kp,t+1}+20\Delta d_t-c_{\mathrm{stop},t}\\
-0.04v_t-0.01u_t+20\mathbb I[\mathrm{task\_reached}]\\
&-c_{\mathrm{safe},t}-c_{\mathrm{terminal},t}
-c_{\mathrm{failure},t}-c_{\mathrm{timeout},t}.
\end{aligned}
\]
该基线固定 keypoint precision reward、hold reward 和 leave penalty 的权重为 0。Full 与 M1–M3 使用的公共高精度辅助项为
\[
r_{\mathrm{aux}}=0.05Q_{\mathrm{prec},t+1}+0.15h_t-0.2\ell_t.
\]
各项激活条件冻结如下：

- \(h_t=1\) 当且仅当 \(s_t,s_{t+1}\) 均同时满足位置与姿态阈值，且没有 hard failure 或 terminal obstacle collision；否则为 0；
- \(\ell_t=1\) 当且仅当 \(s_t\) 同时满足两项阈值而 \(s_{t+1}\) 至少一项越界；否则为 0；
- \(c_{\mathrm{stop},t}=0.10\exp(-e_{p,t+1}/0.03)\exp(-e_{R,t+1}/0.09)v_t\)；
- \(v_t\) 为六关节归一化速度平方的均值，\(u_t\) 为相邻关节速度归一化差平方的均值，两者均裁剪到 \([0,1]\)；
- \(c_{\mathrm{safe},t}=0.05\xi_tC_{\mathrm{ext},t}+0.05\lambda_{self,t}C_{\mathrm{self},t}\)，外部和自碰撞风险/安全距离分别按 \(2:8\) 权重归一化；
- \(C_{\mathrm{ext},t}=(2r_{\mathrm{ext},t}+8c_{\mathrm{ext},t})/10\)，其中 \(r_{\mathrm{ext}}\) 是裁剪到 \([0,1]\) 的外部风险，\(c_{\mathrm{ext}}=[(0.05-d_{\min})/0.05]_0^1\)；\(C_{\mathrm{self},t}=(2r_{\mathrm{self},t}+8c_{\mathrm{self},t})/10\)，\(c_{\mathrm{self}}=[(d_{\mathrm{self,safe}}-d_{\mathrm{self,min}})/d_{\mathrm{self,safe}}]_0^1\)。
- terminal obstacle collision 和 hard failure 的惩罚均为 20；未成功且非 hard failure 的 timeout 惩罚为 2。碰撞项只在对应风险事件发生时激活，timeout 项只在 \(timeout\land\neg task\_reached\land\neg hard\_failure\) 时激活；成功必须连续 \(H=5\) 帧达标，成功 terminal 的基础奖励使用 reset 前真实 terminal state。

配置中的 `joint_precision_progress_scale`、`partial_precision_reward_scale`、split position/orientation progress 与 state-cost 项在正式统一关键点分支中只生成日志诊断量，不进入任何正式方法的训练奖励，不得将其解释为额外训练奖励。

Full 的奖励关系冻结为：
\[
r_{\mathrm{Full}}=r_{\mathrm{Base}}+r_{\mathrm{aux}}+r_b.
\]
M1 去掉 \(r_b\)，M2 与 M3 保持和 Full 完全相同的奖励。Full 系列中的 \(Q_{\mathrm{prec}}\)、hold/leave 以及所有方法共享的离散成功奖励已经提供阈值型联合约束信号。Bottleneck 的增量作用定义为：在这些已有信号之上，为最差归一化约束提供连续的非终止过程优化信号；不声称它首次引入联合约束对齐。

课程使用 L0–L13。完整冻结表如下：

| 档位 | 位置阈值 | 姿态阈值 |
|---|---:|---:|
| L0 | 100 mm | 0.30 rad |
| L1 | 80 mm | 0.24 rad |
| L2 | 64 mm | 0.192 rad |
| L3 | 51.2 mm | 0.1536 rad |
| L4 | 50 mm | 0.10 rad |
| L5 | 25 mm | 0.10 rad |
| L6 | 10 mm | 0.10 rad |
| L7 | 5 mm | 0.10 rad |
| L8 | 2.5 mm | 0.10 rad |
| L9 | 2 mm | 0.10 rad |
| L10 | 1.5 mm | 0.10 rad |
| L11 | 1 mm | 0.10 rad |
| L12 | 1 mm | 0.05 rad |
| L13 | 1 mm | 0.025 rad |

课程分为三个阶段：L0–L4 为位置与姿态联合收紧；L5–L11 姿态固定为 0.10 rad、位置逐步收紧至 1 mm；L12–L13 位置固定为 1 mm、姿态继续收紧至 0.025 rad。远距离和大姿态差目标分布保持不变。

成功定义为连续 `H=5` 个控制周期同时满足位置和姿态阈值，而不是单帧进入阈值区域：

\[
S_t=\mathbb I\left[\bigwedge_{j=t-H+1}^{t}(e_{p,j}\leq\epsilon_p\land e_{R,j}\leq\epsilon_R)\right].
\]

CPU 配置使用 `train.cpu_affinity: false`，由操作系统自由调度。

## 4. 自动升档

统一配置位于 `configs/experiments/paper_auto_promotion_protocol.yaml`：

1. 进入新档位后重新累计该档 transition；
2. 每个档位首次累计 300k 个当前档位 transition 后触发 deterministic probe；
3. probe 未通过时继续留在当前档位，并在之后每增加 300k 个当前档位 transition 时再次 probe；
4. 使用验证种子 `31001、31002、31003`，每个种子 300 episodes；
5. 三种子平均成功率至少 85%，最低单种子至少 80% 后，执行确认评估；
6. 确认阶段使用相同三个验证种子，每个种子 1000 episodes；
7. 确认通过后才升到下一档；达到 L13 后保持 L13 训练直到 6M 预算结束。

在线成功率只用于训练诊断，不能触发升档。

## 5. 最终评估

统一入口：`scripts/core/evaluate_thesis_homotopy.py`。训练期间的 `31001、31002、31003` 只用于验证和升档；最终冻结测试固定使用从未参与训练决策的 `41001、41002、41003`，每个种子运行 1000 episodes。为控制总墙钟时间，正式论文冻结评估只选择 L6、L11、L13 三个代表档位；升档 probe 和 confirmation 合同保持不变。

Full 与 Hybrid KP-SAC-Base 在 6M 结束时分别保存两个训练种子的最终冻结 Actor，不挑选表现最好的种子；M1、M2、M3 保存预先指定的 `11001` 单种子最终 Actor。每个最终 Actor 都在 L6、L9、L11、L12、L13 统一评估。Full/Base 跨 `11001`、`11002` 汇总均值和离散程度；单种子消融只用于机制归因、升档行为和失败模式分析，不用于估计跨随机种子的方差。训练过程中达到各档位时的 checkpoint 仅作为课程进展诊断，不作为跨方法主比较。另记录首次通过每个档位所需的累计 transition `T_Lk`。若某方法未通过升档条件，则一直在当前档位训练至 6M；若提前达到 L13，则一直保持 L13。主指标包括联合成功率、position-only / orientation-only / joint failure 比例、位置和姿态误差 P95、远距离和大姿态差桶成功率，以及 task-cell P10 和最差 10 个 cell 平均成功率。

所有局部机制分析共用同一个归一化坐标：
\[
\rho_p=e_p/\epsilon_p,\qquad \rho_R=e_R/\epsilon_R,\qquad
\rho_{\max}=\max(\rho_p,\rho_R).
\]
目标邻域（precision-local region）冻结定义为 \(\rho_{\max,t}\leq1.5\)。Boundary Replay、Jacobian 方向分析、局部反向修正率及 \(P(\mathrm{success}\mid\mathrm{entered\ boundary})\) 均按 transition 的当前状态 \(s_t\) 使用这一空间。

机制诊断同时记录：

- Jacobian：在 `rho_max <= 1.5` 的目标邻域内，报告 Actor 动作与关键点 DLS 局部下降方向的 cosine similarity，以及局部反向修正率。令
  \[
  N_{local}=\sum_t\mathbb I[\rho_{\max,t}\leq1.5],
  \]
  \[
  R_{reverse}=
  \frac{
  \sum_t\mathbb I[\rho_{\max,t}\leq1.5]\,
  \mathbb I[\rho_{\max,t+1}>\rho_{\max,t}]
  }{N_{local}},
  \]
  其中 (t+1) 使用同一 episode 内动作执行后的真实下一状态；若该动作结束 episode，则使用 reset 前的 terminal state，不能使用 reset 后的新初始状态；
- Bottleneck：终点和最后局部窗口的 `rho_max=max(e_p/epsilon_p,e_R/epsilon_R)` 与 `|rho_p-rho_R|`；
- Boundary Replay：进入归一化边界区域后的最终稳定成功概率 `P(success | entered boundary)`。

主结果比较 Full 与 Hybrid KP-SAC-Base，衡量完整高精度方案相对“Hybrid 表征 + 稠密关键点奖励 + 公共精度执行接口 + 精度课程”基础基线的整体收益。三个核心新增机制中的 Jacobian、Bottleneck 与 Boundary Replay 的增量作用分别由 Full–M3、Full–M1 和 Full–M2 验证；Full 与 Hybrid KP-SAC-Base 的比较用于评价完整 Direction–Objective–Data 高精度框架的整体收益。由于 Base 还关闭 precision quality、hold/leave 和 Success Replay，Full 与 Base 的总差距不能单独归因于 Jacobian、Bottleneck 或 Boundary Replay，也不能用 M1/M2 声称验证了完整 Objective/Data 模块。

对未稳定成功的 episode，失败类型按 reset 前的真实 terminal state 冻结为三个互斥类别：
\[
\begin{aligned}
\text{Position-only failure}:&\quad e_p>\epsilon_p,\ e_R\le\epsilon_R,\\
\text{Orientation-only failure}:&\quad e_p\le\epsilon_p,\ e_R>\epsilon_R,\\
\text{Joint failure}:&\quad e_p>\epsilon_p,\ e_R>\epsilon_R.
\end{aligned}
\]
三个 failure rate 均以全部评估 episode 数为分母；稳定成功 episode 不计入这三类。若未稳定成功 episode 的 terminal state 同时满足两项单帧阈值，则记入 terminal joint-pose hit 诊断，而不误归入上述三类。

## 6. 数据有效性

Precision Boundary Replay 使用当前档位归一化误差定义边界：

\[
\rho_p=e_p/\epsilon_p,\qquad \rho_R=e_R/\epsilon_R,\qquad
\rho_{\max}=\max(\rho_p,\rho_R).
\]

对 transition \((s_t,a_t,r_t,s_{t+1})\)，先按 episode 最终结果划分 Success 与非 Success；Boundary candidate 仅从未达到连续成功终止的 episode 中选择，并由当前状态决定：
\[
\mathcal B=\left\{\tau_t:\neg\mathrm{success}(E(\tau_t))\ \land\ 0.8\leq \rho_{\max,t}\leq1.5\right\},
\]
其中 \(E(\tau_t)\) 表示 transition 所属 episode。该定义表达“尚未稳定成功的 episode 在精度边界时应采取什么动作”。\(s_{t+1}\) 只用于标记该动作随后造成保持、退出、position-only、orientation-only 或 joint failure，不参与几何边界成员资格判断。这与 Jacobian direction analysis 和 reverse rate 均按当前状态选择 transition 的定义一致。

达到连续成功终止时，该成功 episode 的完整 transition 序列会复制到当前成功池，以供审计和 Full 家族采样。Full 与 M1–M3 的成功池按 episode-balanced 方式抽取 transition，并非只采样 terminal success 帧；属于成功 episode 的任一 transition 均不得成为 Boundary candidate，即 \(\mathcal S\cap\mathcal B=\varnothing\)。其 Success quota 使用冷启动日程：当前档成功 episode 少于 5 个时为 0%，达到 5/20/50 个后依次为 5%/15%/20%；不足部分由普通当前档数据补齐。Hybrid KP-SAC-Base 的 success quota、low fraction 和 medium fraction 均固定为 0，成功池即使保留记录也绝不参与梯度 batch。

Full、M1、M3 启用 Boundary Replay，稳定阶段 batch 配额为 `20% Success + 20% Boundary + 60% Ordinary`；M2 关闭 Boundary Replay，稳定阶段为 `20% Success + 80% Ordinary`；Hybrid KP-SAC-Base 固定为 `100% Ordinary`。所有部分都只来自当前 curriculum level。对启用相应特殊池的 Full 家族配置，三类采样身份按 Success 优先、Boundary 次之、Ordinary 最后的顺序互斥分配：成功 episode 的 transition 属于 Success；非成功 episode 中满足上述归一化区间的 transition 属于 Boundary；其余 transition 属于 Ordinary。因此配额表示互斥类别的 batch 比例，而不只是避免在同一 batch 重复抽取同一 transition。Success 或 Boundary 候选不足时，未满足 quota 回退到 Ordinary。Base 不启用任何特殊池，其 `100% Ordinary` 流保留当前档的全部普通 transition，不参与 Success/Boundary 分类。边界样本不建立独立物理 buffer，而是根据当前档 transition 中保存的 episode-success 标志、当前误差和当前档容差在线分层。该相对定义适用于 L0–L13，不使用固定毫米数。

### 6.1 课程档位与 Replay MDP 合同

正式 L0–L13 训练禁止跨档 replay，配置冻结为 `joint_pose_curriculum.cross_level_replay: false`。进入新档位时，上一档当前池被归档用于审计，但不会再参与 Actor 或 Critic 的梯度更新；当前池和当前成功池清空。新档位至少积累 `max(update_after, batch_size)=1024` 个本档 transition 后才恢复 SAC 更新，避免在空当前池阶段用旧档样本补齐 batch。

这一约束同时覆盖 transition dynamics、reward 和 terminal 三层语义。当前实现的 precision action scaling 使用
\[
\rho_{\max}=\max(e_p/\epsilon_p,e_R/\epsilon_R)
\]
决定实际关节速度缩放，因此阈值改变时，同一策略动作可能产生不同的下一状态；仅重算 reward/done 不能把旧 transition 严格转换为新档 MDP。与此同时，precision quality、hold/leave、Joint Constraint Bottleneck Shaping、success bonus、连续 \(H=5\) 成功终止和 bootstrap mask 都依赖当前档阈值或终止合同。因此正式协议选择同档 replay，而不是对不完整历史轨迹做近似重标记。

Replay 仍保存原始当前/下一位置误差、当前/下一姿态误差、关键点距离、物理碰撞/关节越界/timeout 原因、reset 前的真实 next observation、采集档位和采集容差，以支持审计和同档 reward 重构。基础 keypoint tracking 和 keypoint progress 只依赖关键点几何距离，本身不依赖 \((\epsilon_p,\epsilon_R)\)；precision quality、Bottleneck、hold/leave 和 success bonus 依赖课程合同。采样时 reward 由上述原始量重算，但正式训练只对同档 transition 重算，done 保留该档真实物理终止或连续成功终止语义。

Full 的联合约束瓶颈塑形在协议中冻结。Full 家族公共奖励已经通过 precision quality、hold/leave 和连续成功奖励表达部分联合阈值语义；Hybrid KP-SAC-Base 仍保留离散的连续成功奖励。Bottleneck 不被表述为首次引入联合约束。它使用当前 curriculum level 的成功阈值，为两项约束中较差的归一化误差增加连续势函数信号。令
\[
\rho_p=e_p/\epsilon_p,\qquad \rho_R=e_R/\epsilon_R,
\]
\[
\rho_b=\max(\rho_p,\rho_R),\qquad
\Phi_b=\lambda_b\exp(-\rho_b/T_b),
\]
其中 `lambda_b=5.0`、`T_b=2.0`。定义 terminal indicator \(d_t=1\) 当且仅当该 transition 导致 success、timeout、collision 或 hard failure；Bottleneck 额外奖励采用 terminal-gated 形式：
\[
r_{b,t}=(1-d_t)\left[\gamma\Phi_b(s_{t+1})-\Phi_b(s_t)\right].
\]
因此所有 terminal transition 的 \(r_b\) 均为 0，success bonus、timeout penalty、collision penalty 和 hard-failure penalty 分别承担终止结果的奖励或惩罚。这样不会产生“越接近成功边界但最终失败，额外 Bottleneck 惩罚越大”的隐藏终止机制；目标附近的非终止停滞仍由 \((\gamma-1)\Phi_b(s_t)\) 提供轻微时间压力。该项称为 terminal-gated potential-form shaping，不声称标准 PBRS 的策略不变性。Hybrid KP-SAC-Base 和 M1 将 `lambda_b` 固定为 0。

Full 的 Jacobian 输入和关键点误差统一使用世界坐标系。Jacobian 是三个关键点相对于 6 个关节速度的几何 Jacobian，形状为 `9 x 6`，按三个关键点顺序 flatten 为 54 维；输入先除以 `keypoint_jacobian_scale=1.1`，再裁剪到 `[-1,1]`。Hybrid KP-SAC-Base、Full、M1、M2、M3 均固定相同的显式位姿误差、关键点定义、排列顺序和坐标系。Full、M1、M2 的 Actor 接收真实 Jacobian；M3 的同一 54 维切片固定为 0，并保持与 Full 完全相同的 Actor 参数量；Base 使用 108 维基础 Actor。Critic 始终使用去除 Jacobian 槽后的公共状态。因而 Full–M3 只改变 Jacobian 信息，不再同时改变 Actor 第一层参数量。

Jacobian 机制诊断统一使用阻尼最小二乘方向，固定 `lambda_DLS=0.05`：
\[
\dot q^*=+J_{kp}^{\mathsf T}(J_{kp}J_{kp}^{\mathsf T}+\lambda_{DLS}^2I)^{-1}e_{kp}.
\]
其中 `e_kp=x_kp^goal-x_kp`，与代码中的 goal-minus-current 关键点误差定义一致；cosine similarity 使用 action scale 映射后的实际关节速度计算。

旧 M0、旧 M1、旧 M2–M6、semantic replay / far restart 主线及其 checkpoint 和评估结果全部属于历史数据，不进入新论文表格。新结果只能来自本协议列出的五个配置。
