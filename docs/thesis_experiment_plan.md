# 最新论文实验计划清单

> 计划日期：2026-09-08  
> 绑定进展文档：`docs/thesis_experiment_progress.md`  
> 对应论文大纲：`docs/thesis_outline.md`  
> 原则：每一步只验证一个关键环节。若结果异常，先停在该步骤定位问题，不把训练、预测、QP 和实机验证混在同一轮实验里。

## 使用方式

本文档是后续实验的执行清单；`docs/thesis_experiment_progress.md` 是状态记录。每完成一个步骤，应同步更新进展文档中的“当前总体进展”和“下一步优先级”。

建议记录规范：

```text
计划编号:
代码版本:
配置文件:
输入 checkpoint:
输出目录:
train seeds:
eval seeds:
是否通过:
主要异常:
下一步处理:
```

## P0：冻结当前基线状态

- [ ] **P0.1 固化已有可用结果**

目标：确认旧结果只作为当前连杆风险基线，不再承担新方法结论。

输入：

```text
outputs/rechecks/heldout_1004_1006/final_3methods/
outputs/paper/final_materials/
docs/thesis_experiment_progress.md
```

验证：

- `ee_fixed`、`link_fixed_penalty1`、`ldrc_fixed` 的主表能复现。
- 新论文中只把 `link_fixed_penalty1` 表述为 `Instant-Link-SAC` 或当前连杆风险基线。
- 不把 `ldrc_adaptive` 和旧 adaptive beta 写成有效贡献。

通过标准：

- 进展文档中明确写清已完成结果的使用边界。
- 后续所有新实验都以 `link_fixed_penalty1` 作为优先扩展起点。

失败定位：

- 如果表格数值和进展文档不一致，先检查 `outputs/paper/final_materials/` 是否由 `final_3methods/` 派生。
- 如果方法名混乱，先统一 `link_fixed_penalty1`、`Instant-Link-SAC`、`Predictive-Link-SAC` 的命名映射。

## P1：离线验证预测风险本身

- [ ] **P1.1 实现预测风险计算模块**

目标：先只实现 `d_pred`、`T_enter`、`Risk_i^pred`，不训练新策略，不接 QP。

输入：

```text
src/rl_risk_sac/utils/risk.py
src/rl_risk_sac/collision/detectors.py
tests/test_risk.py
```

建议新增：

```text
src/rl_risk_sac/utils/predictive_risk.py
tests/test_predictive_risk.py
```

验证：

- 静止障碍物且机械臂静止时，预测距离应与当前距离一致。
- 障碍物匀速接近时，`d_pred <= d_now`，`T_enter` 有限。
- 障碍物远离时，预测风险不应高于接近场景。
- 对相同当前距离、不同接近速度的场景，预测风险应区分危险程度。

通过标准：

- 单元测试通过。
- 输出字段至少包含 `d_pred`、`T_enter`、`risk_pred_per_link`、`risk_pred_body`、`critical_link`。

失败定位：

- 若 `d_pred` 异常，先查胶囊体外推和障碍物外推。
- 若风险跳变严重，先查最近点投影、连杆切换和 softmax 聚合。

- [ ] **P1.2 在固定轨迹上离线评估预测风险**

目标：不改变策略动作，只在已有或新采样轨迹上对比当前风险与预测风险。

输入：

```text
已有 `link_fixed_penalty1` checkpoint
random_crossing / upper_arm / elbow / forearm / wrist 场景
```

输出建议：

```text
outputs/predictive_risk/offline_eval/
```

指标：

```text
Warning Lead Time
未来最小距离误差
风险连杆识别率
误报率
漏报率
```

通过标准：

- 至少在接近型场景中得到正的平均 Warning Lead Time。
- 对“近但远离”的障碍物，误报率不能明显高于当前风险。
- 风险最高连杆与实际最近/越界连杆大体一致。

失败定位：

- 如果 Lead Time 没有提升，先调预测时域和 `d_safe`，不要直接进入训练。
- 如果误报过多，先检查 `w_min`、`w_enter`、`R_horizon` 的权重。

## P2：预测风险进入策略状态

- [ ] **P2.1 扩展 observation schema**

目标：让环境能输出预测风险状态，但先不训练大规模模型。

建议 schema：

```text
link_risk_pred_v1
```

验证：

- observation 维度固定。
- 所有元素有限值。
- 无障碍物或远离障碍物时，预测风险处于低值。
- 旧 `link_risk_v1` baseline 不受影响。

通过标准：

- smoke test 和 observation 单元测试通过。
- `Instant-Link-SAC` 旧配置仍可运行。

失败定位：

- 如果旧 checkpoint 无法加载，说明 schema 与旧配置没有隔离好。
- 如果训练刚开始就 NaN，先查归一化、clip 范围和风险指数项。

- [ ] **P2.2 小规模训练 `Predictive-Link-SAC`**

目标：先确认预测风险状态不会破坏学习，再进行正式三 seed 训练。

建议设置：

```text
train seeds: 101
eval seeds: 1001
训练步数: 10k 至 30k
场景: random_crossing
```

验证：

- reward、cost、success、collision 曲线没有明显异常。
- action、risk、observation 均为有限值。
- 成功率不应明显低于随机或完全停滞策略。

通过标准：

- 小规模训练能稳定结束。
- evaluation CSV 字段完整。

失败定位：

- 如果策略停滞，先降低预测风险奖励权重。
- 如果碰撞率异常升高，先检查预测风险输入方向和尺度。

## P3：预测风险策略的正式对比

- [ ] **P3.1 运行 `Instant-Link-SAC` vs `Predictive-Link-SAC` 主对比**

目标：验证预测风险是否相对当前风险有独立收益。

最低方法：

```text
Instant-Link-SAC
Predictive-Link-SAC
```

建议设置：

```text
train seeds: 101、202、303
validation seed: 2001
held-out eval seeds: 1004、1005、1006
场景: random_crossing，外加至少一个定向压力场景
```

指标：

```text
Success Rate
Collision Rate
Safety Violation Rate
Minimum Distance
Final Position Error
Warning Lead Time
未来最小距离误差
风险连杆识别率
RMS/Peak Jerk
```

通过标准：

- 预测风险相较当前风险具有可重复的正预警提前量。
- 任务成功率没有不可接受下降。
- 若碰撞率没有下降，也要能说明预测改善的是预警而非最终安全。

失败定位：

- 如果离线预测有效但训练无收益，问题多半在状态表示或奖励权重。
- 如果预测风险导致保守停滞，优先调风险权重，而不是扩大预测时域。

## P4：先做最小安全 QP，不加 jerk 自适应

- [ ] **P4.1 实现最小干预安全 QP**

目标：先只验证安全约束方向和可行性，不加入 jerk 自适应复杂度。

QP 初始版本：

```text
min ||q_dot - q_dot_nom||^2 + rho ||xi||^2

subject to:
距离变化率安全约束
关节速度约束
关节位置一步预测约束
xi >= 0
```

建议新增：

```text
src/rl_risk_sac/control/safety_qp.py
tests/test_safety_qp.py
```

验证：

- 远离障碍物时 QP 透传名义动作。
- 接近障碍物且名义动作危险时 QP 修改动作。
- 约束不可行时松弛变量为正，并能触发降级标记。

通过标准：

- 单元测试通过。
- QP 输出速度不越界。
- 约束方向经过构造场景验证。

失败定位：

- 如果越修正越危险，优先检查 `n_i` 方向和 `J_i q_dot` 符号。
- 如果经常不可行，先放宽安全阈值或检查速度/位置边界。

- [ ] **P4.2 固定 actor 反事实评估最小 QP**

目标：只检验安全层，不重训 actor。

对比：

```text
Predictive-Link-SAC
Predictive-Link-SAC + minimal QP
Instant-Link-SAC + minimal QP
```

指标：

```text
Collision Rate
Safety Violation Rate
Minimum Distance
Success Rate
Intervention Rate
CorrectionNorm
QP Infeasible Rate
QP Solve Time
```

通过标准：

- QP 能降低安全违反或碰撞。
- 成功率下降在可解释范围内。
- 介入不是长期保持或频繁停止。

失败定位：

- 如果安全改善来自大量停止，先调整降级策略和安全阈值。
- 如果成功率崩掉，先检查 QP 是否过度保守。

## P5：加入加速度与 jerk 约束

- [ ] **P5.1 加入加速度和 jerk 硬边界**

目标：验证运动连续性约束本身，不立即加入风险自适应权重。

对比：

```text
minimal QP
QP + acceleration bound
QP + acceleration bound + jerk bound
```

指标：

```text
RMS/Peak Acceleration
RMS/Peak Jerk
Collision Rate
Safety Violation Rate
Success Rate
QP Infeasible Rate
```

通过标准：

- jerk 峰值明显下降或被硬边界限制。
- 安全指标不明显恶化。
- QP 不可行率可解释。

失败定位：

- 如果 jerk 降了但碰撞升高，说明运动边界压缩了避障响应，需要调 `a_max`、`j_max` 或提前预警阈值。
- 如果不可行率高，先检查历史速度初始化和 jerk 公式。

- [ ] **P5.2 加入风险自适应连续性权重**

目标：最后再验证自适应权重是否带来额外收益。

对比：

```text
固定连续性权重 QP
风险自适应连续性权重 QP
```

指标：

```text
Intervention Rate
CorrectionNorm
RMS/Peak Jerk
Collision Rate
Safety Violation Rate
Success Rate
```

通过标准：

- 低风险时更平滑。
- 高风险时不削弱避障响应。
- 相比固定权重至少在部分指标上有稳定、可解释收益。

失败定位：

- 如果无明显收益，可将该机制降级为增强模块，不作为独立强创新。
- 如果高风险响应变慢，检查 `lambda_a(R)`、`lambda_j(R)` 是否方向写反。

## P6：完整方法主实验

- [ ] **P6.1 运行最终主比较**

目标：形成新论文第五章的核心结果。

方法矩阵：

```text
EE-SAC
Instant-Link-SAC
Predictive-Link-SAC
Instant-Link-SAC+QP
Proposed
LDRC-SAC
```

最低必须完成：

```text
Instant-Link-SAC
Predictive-Link-SAC
Proposed
Instant-Link-SAC+QP
```

建议设置：

```text
train seeds: 至少 3 个
held-out eval seeds: 与调参完全分离
场景: random_crossing，upper_arm，elbow，forearm，wrist
```

通过标准：

- 每个创新结论都有对应消融。
- `Proposed` 相比无 QP 方法降低碰撞或安全距离违反。
- `Proposed` 相比无 jerk QP 降低 RMS/Peak Jerk。
- QP 求解时间满足控制周期。

失败定位：

- 如果 `Proposed` 不优于 `Predictive-Link-SAC`，回到 P4 检查 QP 是否有效。
- 如果 `Predictive-Link-SAC` 不优于 `Instant-Link-SAC`，回到 P1/P2 检查预测风险。
- 如果只有 `Instant-Link-SAC+QP` 有效，论文贡献应转向安全整形而弱化预测风险。

## P7：鲁棒性与泛化

- [ ] **P7.1 逐项加入扰动**

目标：一次只改变一个扰动因素，避免无法判断失败来源。

顺序：

```text
位置噪声
速度误差
控制延迟
障碍物速度突变
未见速度范围
双球形障碍物
```

通过标准：

- 每个扰动都有单独表格。
- 报告性能下降曲线，而不是只给最终平均值。
- 明确哪些失效来自预测模型，哪些来自 QP 可行域。

失败定位：

- 噪声失败：优先查速度估计和低通滤波。
- 延迟失败：优先查预测时域和安全裕度。
- 双障碍失败：优先查 Top-k 约束数量和 QP 求解时间。

## P8：真实 UR5 低速验证

- [ ] **P8.1 离线部署预检复核**

目标：在碰真实机器人前确认模型、观测和指令链路一致。

输入：

```text
docs/deployment_preflight.md
outputs/deployment_preflight/offline_report.json
```

通过标准：

- checkpoint 可加载。
- 观测维度和训练配置一致。
- 速度、加速度、jerk 限幅配置明确。
- 感知丢失和 QP 失败都有停止策略。

- [ ] **P8.2 真实低速可执行性验证**

目标：只验证低速部署可行性，不故意制造碰撞。

最低对比：

```text
Predictive-Link-SAC
Proposed
```

建议次数：

```text
每组 10 至 20 次
```

指标：

```text
Minimum Distance
Safety Violation Rate
Intervention Rate
CorrectionNorm
Task Success
QP Solve Time
感知丢失次数
人工急停/保护停次数
```

通过标准：

- 能稳定完成低速执行。
- QP 或降级逻辑在高风险时能触发。
- 不宣称绝对安全或高速安全。

失败定位：

- 如果实机比仿真保守，优先查相机外参、安全距离和速度估计延迟。
- 如果执行抖动，优先查底层速度伺服周期和 jerk 边界。

## 最小可写论文路径

如果时间有限，优先完成：

1. P1：预测风险离线验证。
2. P3：`Instant-Link-SAC` vs `Predictive-Link-SAC`。
3. P4：最小安全 QP 固定 actor 反事实。
4. P5：jerk 约束 QP 消融。
5. P6：四方法主比较。

暂缓：

- 双障碍物泛化。
- 大规模真实实验。
- LDRC 深度改进。
- 自适应权重的大范围搜索。

## 与进展文档的绑定规则

`docs/thesis_experiment_progress.md` 中的状态表应使用本文档编号更新：

| 计划编号 | 对应进展项 |
| --- | --- |
| P0 | 已有结果固化与使用边界 |
| P1 | 预测性连杆风险 |
| P2 | `Predictive-Link-SAC` 状态与小规模训练 |
| P3 | 预测风险正式对比 |
| P4 | 最小安全 QP 指令整形器 |
| P5 | 加速度与 jerk 约束 QP |
| P6 | `Proposed` 完整方法 |
| P7 | 鲁棒性与泛化 |
| P8 | 真实 UR5 低速验证 |

每完成一个编号，应在进展文档中追加：

```text
完成日期
输出目录
关键指标
是否通过
是否进入下一步
```
