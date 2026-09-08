# 最新论文大纲下的实验进展说明

> 本文件是项目当前实验进展的唯一状态源。实验数据、统计口径和已验证结论统一见 [实验结果事实源](thesis_experiment_results.md)；本文件只记录状态、阻塞项和下一步工作。

## 0. 计划绑定

| 计划编号 | 当前状态 | 说明 |
| --- | --- | --- |
| P0 | 部分完成 | 已有基线结果已整理；后续保持方法命名一致。 |
| P1 | 已完成 | 预测风险计算、单元测试和固定策略离线评估已完成。 |
| P2 | 部分完成，暂不进入 P3 | 预测风险 observation schema 已完成；小规模训练尚未优于当前连杆基线。 |
| P3 | 未完成 | `Instant-Link-SAC` 与 `Predictive-Link-SAC` 的正式多 seed 对比尚未完成。 |
| P4 | 部分完成 | 最小安全 QP 已实现；当前仅有单场景固定 actor 反事实结果，仍需多场景扩展。 |
| P5 | 未完成 | 加速度与 jerk 约束 QP 尚未完成。 |
| P6 | 未完成 | `Proposed` 完整方法主实验尚未完成。 |
| P7 | 未完成 | 鲁棒性与泛化实验尚未完成。 |
| P8 | 未完成 | 真实 UR5 低速验证尚未完成。 |

## 1. 当前论文主线

```text
当前连杆风险基线
-> 预测性连杆风险建模
-> SAC 名义关节速度控制
-> 风险自适应 jerk 约束安全 QP 指令整形
-> 子步监测与低速实机验证
```

当前已完成的基线结果只能支撑“当前连杆级风险是合理起点”。不能把它们写成预测风险、完整 QP 或真实部署已经验证。完整数值和结论边界见 [实验结果事实源](thesis_experiment_results.md)。

## 2. 模块状态

| 模块 | 状态 | 备注 |
| --- | --- | --- |
| UR5 PyBullet 动态障碍物环境 | 已完成 | 固定基座、静态目标、单动态球形障碍物、多场景评估。 |
| 当前连杆级动态风险 | 已完成 | 胶囊体距离、接近速度、TTC 和当前风险聚合。 |
| `ee_fixed` / `link_fixed_penalty1` / `ldrc_fixed` 基线 | 已完成 | 主结果统一见事实源。 |
| 预测性连杆风险 | 已完成离线验证 | 固定策略离线评估有正 Warning Lead Time，但误报偏高。 |
| `Predictive-Link-SAC` | 部分完成 | schema 与 smoke 已通过；小规模训练暂未显示策略收益。 |
| 安全 QP 指令整形器 | 部分完成 | 单场景 smooth-QP 反事实已完成，尚未完成 held-out 多场景验证。 |
| 风险自适应 jerk 约束 QP | 未完成 | 不能写成已验证贡献。 |
| `Proposed` 完整方法 | 未完成 | 尚无多 seed held-out 主比较。 |
| 真实 UR5 低速验证 | 未完成 | 离线预检不能替代现场实验。 |

## 3. 当前阻塞与判定

- `Predictive-Link-SAC` 尚未证明优于 `Instant-Link-SAC`，因此暂不作为主贡献。
- Smooth-QP 仅有单 train seed / eval seed 反事实结果，不能替代 held-out 多 seed 主表。
- 当前 QP 尚未加入显式加速度和 jerk 硬约束，不能宣称“风险自适应 jerk 约束 QP 已验证”。
- 真实机器人控制接口、限速、急停、RGB-D 失效停止、相机外参和碰撞包络仍需现场签核。

## 4. 下一步优先级

1. 将 smooth-QP 扩展到 held-out eval seeds 和 upper arm、elbow、forearm、wrist 压力场景。
2. 加入 `Instant-Link-SAC+QP` 交叉消融，区分预测风险收益与 QP 安全层收益。
3. 在多场景安全收益稳定后，加入加速度/jerk 硬约束和风险自适应连续性权重。
4. 完成 `Predictive-Link-SAC` 与 `Instant-Link-SAC` 的正式多 seed 对比。
5. 运行 `Proposed` 完整方法主实验，再开展鲁棒性、实时性和真实 UR5 低速验证。

## 5. 论文写作边界

当前可以写：

- 当前连杆级风险相较末端风险是更合理的整臂动态避障状态表示；
- `link_fixed_penalty1` 是预测风险和安全 QP 的扩展基线；
- 预测风险在固定策略离线评估中具有正预警提前量；
- 单场景 smooth-QP 反事实结果支持继续扩大安全层验证。

当前不能写：

- `Predictive-Link-SAC` 已经优于 `Instant-Link-SAC`；
- 风险自适应 jerk 约束安全 QP 已经在多场景中降低碰撞；
- 完整 `Proposed` 方法已经得到验证；
- 真实 UR5 实验已经证明方法有效或提供形式化安全保证。

## 6. 相关文档

- [实验结果事实源](thesis_experiment_results.md)：实验数据、数值表格、统计口径和结论边界。
- [实验计划清单](thesis_experiment_plan.md)：P0--P8 的执行任务和通过标准。
- [论文大纲](thesis_outline.md)：研究设计与章节结构，不重复维护结果表。
