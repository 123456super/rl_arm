# 项目文档入口

当前唯一主线是同一套可从静态扩展到动态障碍物的分层控制架构：

```text
多初值 IK + RRT-Connect
        ↓
关节轨迹跟踪 + terminal DLS servo
        ↓
风险调节 Residual SAC
        ↓
连杆级预测风险 + QP/CBF safety filter
        ↓
qdot_cmd
```

`TRACK / AVOID_HOLD / REPLAN / SERVO / PLAN_FAILED` 是静态与动态阶段共用的状态机。旧 direct-SAC、笛卡尔 waypoint residual 和历史 checkpoint 只作为对照，不能加载到新观测与动作语义中。

## 当前文档

1. [当前研究状态](current/research_status.md)：当前证据、代码状态、冻结边界和唯一下一步。
2. [分层控制通用实验协议](current/hierarchical_control_protocol.md)：S1/S2/S3 共用的动作语义、状态机、安全出口与评估规则。
3. [S1 静态障碍物实验协议](current/s1_static_obstacle_protocol.md)：静态阶段的场景条件、实验矩阵、命令与进入 S2 的门槛。
4. [论文主题与结构](design/dynamic_obstacle_thesis_theme.md)：研究问题、创新点、章节与消融设计。
5. [指标和单位定义](reference/metrics_and_units.md)：统一字段、单位与报告口径。
6. [历史资料归档](archive/README.md)：架构调整前的文档快照、结果副本、资产索引和证据来源。

其中 [legacy direct-SAC 架构转向证据](archive/legacy_direct_sac/evidence/architecture_transition_evidence.md) 解释旧失败如何对应新模块，[S0 基础 reaching 历史基线](archive/legacy_direct_sac/evidence/s0_reaching_baseline.md)保存冻结结果和 checkpoint；二者都不是当前执行协议。

## 维护规则

- 新实验必须使用 `configs/experiments/hierarchical/`；`configs/experiments/reaching_incremental/` 是 legacy direct-SAC 历史证据。
- 静态到动态只切换障碍物运动、风险不确定性和 Residual 消融，不更换控制层次、状态机、观测语义或最终安全出口。
- 可规划子集只能辅助解释，必须同时报告完整固定 manifest；`IK not found` 和 `plan not found` 都不是数学不可达证明。
- smoke 或单 episode 只证明链路可运行，不能替代多 seed final 结果。
