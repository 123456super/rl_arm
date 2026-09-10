# 几何修订后 R0–R6 Gate 执行指南

> 适用范围：`conservative_capsule_gap_v2` 及其后续明确登记版本。

## 执行顺序

```text
R0 代码/协议冻结
→ R1 胶囊几何重新标定
→ R2 名义 SAC 从零训练与冻结
→ R3 动作条件预测验证
→ R4 One-Step-QP
→ R5 Predictive-Trajectory-QP
→ R6 全身验收、fallback 与反例测试
```

前一 Gate 未通过时，不进入依赖它的后续阶段。

## R0：实现与协议

- 运行时统一采用 `d = d_raw - 0.04 m`。
- observation、risk、TTC、cost、预测和 QP 必须使用同一保守距离。
- 日志同时保存 `d_min_raw`、`d_min`、`control_min_distance_raw` 和 `control_min_distance`。
- PyBullet contact 独立记录，并作为碰撞终止的唯一事实源。
- 完整测试通过后冻结源码与配置哈希。

## R1：几何校准

- 在六个目标胶囊、随机关节构型和近表面方向上分层抽样。
- 比较原始胶囊距离与 PyBullet collision-shape 距离。
- 分别报告总体和逐连杆误差、危险漏检、碰撞漏检与保守误报。
- `wrist_3` 必须作为显式球形近似单独报告。
- 0.04 m 裕量若仍出现危险漏检，停止并调整几何，不得开始训练。

## R2：名义策略

- 所有 actor 和 replay buffer 从零初始化。
- 训练前冻结开发、validation 和 held-out seed。
- checkpoint 只能按预注册 validation 规则选择。
- 不得恢复或比较修订前 checkpoint 的绝对风险/cost 曲线。

## R3–R6

- R3 验证预测距离、进入时间、危险连杆识别与耗时。
- R4 验证当前几何 One-Step-QP、求解状态、运动边界和逐子步安全。
- R5 只增加固定多时刻预测约束，保持 actor 不变。
- R6 验证全身非线性约束、fallback、反例覆盖和实时性。

## 结果记录

- 任务和 Gate 状态写入 [experiment_tracker.md](experiment_tracker.md)。
- 只有通过数据完整性检查的数值才能写入 [experiment_results.md](experiment_results.md)。
- 任何核心语义变化都必须再次提高实验版本并从受影响阶段重新开始。
