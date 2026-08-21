# 面向动态障碍物的连杆级预测风险约束分层 Residual 学习

## 1. 论文主题

本文面向**单个动态球形障碍物下的机械臂全身避障与目标到达问题**，研究一套可从静态障碍物直接扩展到动态障碍物、后续不再更换主架构的分层控制方法。

核心思路是把全局可达性、名义收敛、动态局部修正和安全约束分别交给合适的模块：

```text
多初值 IK + 全局运动规划
          ↓
确定性轨迹跟踪 + terminal servo
          ↓
风险调节的 Residual SAC
          ↓
连杆级预测风险 + 安全过滤
          ↓
      关节速度命令
```

一句话概括：

**规划基座负责全局方向，确定性控制负责名义收敛，Residual 负责动态局部修正，预测风险负责安全边界。**

## 2. 研究问题

本文重点回答以下问题：

1. 多初值 IK、全局规划和确定性跟踪能否解决直接使用 SAC 时的多路径选择、坏局部 basin 和终端不收敛问题？
2. 连杆级预测风险相较于仅使用末端风险，能否更有效地描述机械臂全身与动态障碍物的碰撞风险？
3. 在已有名义轨迹的条件下，Residual SAC 能否在不显著增加碰撞的前提下改善动态障碍物下的成功率、路径恢复和终端收敛？
4. 全局规划、terminal servo、预测风险、安全过滤和 Residual 学习各自带来多少收益？

## 3. 最终冻结架构

### 3.1 任务规划层

任务规划层低频运行，并支持事件触发重规划：

- 使用多初值 IK 生成多个候选目标关节构型，避免把到达问题绑定到单一 IK 分支。
- 使用 RRT-Connect 或等价的配置空间规划方法搜索无碰撞几何路径。
- 碰撞检查覆盖机械臂各连杆 capsule，而不是只检查末端。
- 对候选路径按可行性、路径长度、最小 clearance 和预计执行时间排序。
- 当路径持续失效、机械臂偏离参考轨迹过大或局部避障无法恢复时，从当前关节状态触发重规划。

全局规划器属于成熟基础模块，主要负责全局拓扑可达性，**不将 RRT-Connect 本身作为论文创新点**。

### 3.2 名义控制层

名义控制层以 20 Hz 跟踪规划器给出的关节空间参考轨迹：

- waypoint 或局部轨迹跟踪器生成名义关节速度 `qdot_nominal`。
- 接近目标且 clearance 足够时，切换或平滑混合到阻尼最小二乘 terminal servo。
- 在没有 Residual 的情况下，该层也应具备稳定的基础到达能力。

确定性控制器负责“沿已知可行路径走到目标”，不再要求 RL 同时学习全局选路和末端伺服。

### 3.3 风险调节的 Residual SAC

Residual SAC 不直接输出完整关节速度，只学习对名义命令的有界修正：

```text
qdot_candidate = qdot_nominal + residual_budget(risk) * Δqdot_RL
```

策略观测至少包含：

- 当前关节位置、关节速度和上一周期执行命令；
- 目标误差、下一 waypoint 误差和轨迹进度；
- `qdot_nominal`；
- 各连杆到障碍物的距离、方向、接近速度、TTC 和预测风险；
- 当前控制模式，即 `TRACK`、`AVOID_HOLD`、`REPLAN`、`SERVO` 或 `PLAN_FAILED`。

Residual 的权限由预测风险和路径状态调节。它负责动态障碍物造成的局部轨迹修正、跟踪误差补偿和安全恢复，不负责从零完成全局运动规划。

### 3.4 连杆级预测风险与安全执行层

最终候选命令必须经过连杆级预测风险检查和安全过滤：

- 使用连杆 capsule 的预测距离、接近速度和 TTC 描述短时碰撞风险。
- 将控制时延、观测年龄、跟踪误差和必要的几何裕量纳入风险计算。
- 使用 QP、CBF 或等价投影方法把候选命令限制在安全集合内。
- 当严格安全约束不可满足时进入安全减速或停止状态，并交由上层决定等待或重规划。
- RL 不得绕过最终安全执行层。

### 3.5 运行状态

静态和动态阶段共用同一套状态机：

```text
TRACK
  路径短时安全：跟踪名义轨迹，并允许 Residual 做小范围修正

AVOID_HOLD
  预测到短时冲突：暂停任务轨迹，由预测 QP 主动增大危险连杆间隙

REPLAN
  冲突持续、偏离过大或局部恢复失败：从当前状态重新规划

SERVO
  接近目标且 clearance 足够：使用 terminal servo 稳定收敛

PLAN_FAILED
  当前 IK/规划预算内未找到路径：输出零 nominal 并周期性重试
```

## 4. 论文创新点定位

本文不把经典 IK、RRT-Connect、DLS 或普通 SAC 单独作为创新。创新重点放在它们围绕动态障碍任务形成的风险约束分层学习方法：

1. **连杆级预测风险约束的分层 Residual 控制框架**

   将全局运动规划、确定性名义控制、Residual 学习和最终安全过滤分层组合，使 RL 专注于动态局部修正，同时保留全身碰撞约束。

2. **面向规划轨迹的风险条件 Residual 策略**

   将名义速度、waypoint 误差、路径进度和各连杆预测风险共同作为策略条件，并依据风险动态限制 residual 权限，降低直接策略在多路径问题中的不稳定性。

3. **面向动态路径失效的局部修正与事件触发重规划机制**

   通过 `TRACK → AVOID_HOLD → REPLAN → SERVO` 的统一运行逻辑，把短时动态避障与持续路径失效分开处理，研究 Residual 修正、安全过滤和重规划之间的分工。

论文最终需要通过消融实验验证这些设计，而不能仅以模块串联或成功率提高来声明创新。

## 5. 实验路线

主架构从静态阶段开始冻结，静态到动态只改变障碍物运动、风险表示和启用模块，不再更换控制层次。

```text
S0：无障碍 reachability 基线
  → 冻结现有纯 actor 结果，确认任务和评估链路有效

S1：静态障碍物下的分层基座
  → 多初值 IK + 全局规划 + 轨迹跟踪 + terminal servo
  → 验证全局可达性、全身无碰撞路径和终端收敛
  → 比较是否加入 Residual，但不把静态阶段的局部参数调优当成主贡献

S2：单个动态障碍物下的完整方法
  → 保持 S1 架构，加入障碍物运动
  → 先做 nominal-only，再在相同规划、跟踪和安全基座上训练 Residual SAC
  → 验证安全减速、等待、重规划与 Residual 局部修正的贡献

S3：鲁棒性与消融
  → 障碍物速度、观测噪声、控制时延和模型误差
  → 拆分验证规划、terminal servo、预测风险、安全过滤和 Residual 的贡献
```

各阶段使用的模块如下：

| 阶段 | 全局规划 | 跟踪/servo | Residual | 风险表示 | 安全过滤/重规划 |
| --- | --- | --- | --- | --- | --- |
| S0 无障碍基线 | 关闭 | 关闭 | 纯 actor 基线 | current | 关闭 |
| S1 静态障碍 | 开启 | 开启 | 消融项 | robust predictive | 开启 |
| S2-N 动态名义基座 | 开启 | 开启 | 关闭 | robust predictive | 开启 |
| S2-R 动态完整方法 | 开启 | 开启 | 开启 | robust predictive | 开启 |
| S3 鲁棒性实验 | 开启 | 开启 | 开启 | robust predictive + 误差/时延 | 开启 |

## 6. 当前证据与架构调整依据

架构调整前的旧文档、核心结果副本、资产路径和 SHA-256 已保存于 [legacy direct-SAC 归档](../archive/legacy_direct_sac/README.md)；以下每条观测到新模块与消融指标的对应关系见 [架构转向证据](../archive/legacy_direct_sac/evidence/architecture_transition_evidence.md)。

- S0 已完成并冻结：固定离线 IK 可达子集为 100%，full set 为 97.0%。
- S1 direct SAC 历史最好 full final 为 `521/600=86.83%`，blind 结果为 `505/600=84.17%`，仍存在较多 timeout。
- 最新失败中 `67/89` 个最终误差大于等于 `0.12 m`，说明主要问题不只是 success threshold 附近的末端推进。
- 当前 feasibility precheck 只对有限个 IK 解检查起点到终点的三次曲线直连；`not_found` 不是完整运动规划意义上的不可达证明。
- 当前 S1 配置关闭了 residual controller，因此不能用 `hold_for_clearance` 或 `link_avoidance` 分支解释这批 timeout。
- 现有 R3/R4 checkpoint 选择落在恢复训练的 warm-up 区间，尚不能作为 repair curriculum 有效或无效的正式结论。

上述证据说明，继续让单个前馈 SAC 同时承担 IK 分支选择、全局绕行、终端伺服和安全控制，不是稳定提高成功率的优先路线。S1 后续应转为验证分层架构，而不是继续叠加 direct SAC 参数。它们只支撑研究假设与职责划分，不替代新架构的正式 S1/S2 结果。

## 7. 评估与消融设计

### 7.1 静态障碍物

至少比较：

1. direct SAC；
2. direct SAC + terminal servo；
3. planner + deterministic tracker + terminal servo；
4. planner + tracker + terminal servo + Residual。

静态阶段分别报告：

- IK found rate 和 plan found rate；
- full-set 与可规划子集 success；
- timeout；
- `collision_any`、capsule overlap 和 physical contact；
- 最终误差、完成时间、路径长度、最小 clearance 和规划时间。

### 7.2 动态障碍物

至少比较：

1. static planner + tracker；
2. planner + tracker + current-risk filter；
3. planner + tracker + predictive-risk filter；
4. planner + tracker + predictive-risk filter + Residual；
5. 完整方法去除重规划或去除 risk-conditioned residual budget 的消融。

除静态指标外，还应报告：

- 动态冲突预测次数和安全过滤介入率；
- `TRACK/AVOID_HOLD/REPLAN/SERVO/PLAN_FAILED` 状态占比与重规划次数；
- Residual 相对 nominal command 的修正幅度；
- 不同障碍物速度、观测延迟和噪声下的 success/collision 变化。

## 8. 架构冻结与止损标准

完成以下验证后冻结主架构，后续只允许优化模块参数、训练方案和实现效率：

1. 全局规划器能在固定静态 manifest 上稳定输出 `IK found / plan found` 统计。
2. planner + tracker + terminal servo 在可规划子集达到至少 95% success。
3. 静态 full-set success 达到 90% 以上，并保持明确的低碰撞口径。
4. 动态 nominal-only 基座能安全执行，允许 timeout，但不得依赖 RL 才能避免频繁碰撞。
5. Residual 输入包含 nominal command、waypoint error、路径进度和连杆级预测风险。
6. 安全过滤器位于最终命令出口，RL 不能绕过。
7. 路径失效后具备 `HOLD/REPLAN` 接口。

若某一模块未达标，应先在同一分层结构内定位规划、跟踪、预测或训练问题，不再更换网络范式和总体控制层次。

## 9. 当前进展与下一步

- S0 无障碍 reaching 基线已经冻结，作为历史基线保留。
- 当前 direct SAC 静态结果作为架构调整前的对照保留，不再作为最终 S1 方法。
- 新 planner/tracker/state-machine、风险调节 residual 接口、最终 predictive safety filter 和 nominal-only 评估入口已经实现并通过代码测试。
- 主动 clearance recovery 接入后，静态和动态各一个 nominal-only 端到端 smoke 均成功且无碰撞，只证明链路可执行，不能替代论文统计。
- 下一步是在固定 manifest 上完成 S1-N nominal-only，再按门槛决定是否训练 S1-R；之后保持相同架构进入 S2 dynamic。

## 10. 预期论文结构

1. 绪论
2. 相关工作：运动规划、轨迹优化、安全控制与 Residual RL
3. 问题定义、连杆级几何建模与评估协议
4. 多初值 IK、全局规划与确定性名义控制基座
5. 连杆级预测风险约束的 Residual 学习方法
6. 静态障碍物下的分层架构验证
7. 动态障碍物下的预测安全、Residual 学习与鲁棒性实验
8. 讨论、局限与未来工作
9. 结论
