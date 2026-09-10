# 几何修订后从零实验跟踪

> 重启日期：2026-09-10
> 状态：旧训练、校准和评估记录已清零；尚无新实验结果。

## 重启原因

- `risk.geometry_margin=0.04 m` 先前只出现在配置校验与离线标定中，没有进入运行时风险、TTC、预测或 Safety-QP。
- `wrist_3` 是有意的零长度胶囊（球），但先前没有显式声明，且 `0.050 m` 半径略小于碰撞网格相对 `tool0` 原点约 `0.0511 m` 的最大顶点距离。
- 修订改变 observation、risk、cost、安全距离违反和控制约束的数值语义，因此旧 checkpoint 和旧曲线不得作为新协议证据。

## 冻结的几何语义

- 原始胶囊间隙：`d_raw = center_distance - capsule_radius - obstacle_radius`。
- 保守间隙：`d = d_raw - geometry_margin`。
- `d` 用于 observation、risk、TTC、predictive risk、cost、安全距离判断和 Safety-QP。
- `d_raw` 仅用于诊断与几何误差报告。
- PyBullet contact 仍是碰撞事实源和碰撞终止条件。
- `wrist_3` 显式允许退化为球，半径固定为 `0.055 m`；其他胶囊不允许无意退化。

## 阶段状态

| 阶段 | 状态 | 目标 |
| --- | --- | --- |
| R0 | 进行中 | 冻结修订后的代码、配置、seed、指标及 manifest |
| R1 | 待运行 | 重新校准六段胶囊及 0.04 m 几何裕量 |
| R2 | 阻塞于 R1 | 从头训练名义 SAC；禁止复用旧 checkpoint |
| R3+ | 阻塞于 R2 | 预测、QP、最终比较与鲁棒性实验 |

## R0 验收清单

- [x] 运行时统一使用保守距离，原始距离独立记录。
- [x] `wrist_3` 球形近似显式声明并扩大到 `0.055 m`。
- [x] 普通胶囊退化检查已加入。
- [x] 完整测试集通过（55 tests passed）。
- [ ] 建立干净代码提交或不可变源码快照。
- [ ] 冻结新实验版本号、seed 与输出目录。

## R1 验收清单

- [ ] 每段至少 400 个随机构型/方向样本。
- [ ] 报告 `d_raw - d_bullet` 的分位数及最大单侧误差。
- [ ] 验证保守距离在 `d_safe=0.12 m` 下危险漏检为零。
- [ ] 单独报告 `wrist_3` 球形近似的覆盖与漏检。
- [ ] 标定通过前不得启动任何训练。

## 数据隔离

- 旧 `outputs/restart_2026-09-09/` 已删除，不得恢复其中 checkpoint 作为新结果。
- 新运行必须创建全新 manifest，并记录胶囊配置、风险语义和源码哈希。
- 所有新数值只登记到 [experiment_results.md](experiment_results.md)。
