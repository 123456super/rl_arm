# 几何修订后实验结果事实源

> 重启日期：2026-09-10
> 当前状态：尚无准入结果。

## 准入规则

- 只接收几何修订后的全新运行。
- 运行必须具有完整配置、源码哈希、胶囊定义哈希、依赖版本和 checkpoint 哈希。
- 所有训练必须从随机初始化开始，不得加载修订前 checkpoint 或 replay buffer。
- validation 与 held-out seed 必须在运行前冻结并隔离。
- 失败、碰撞、超时、NaN、solver failure 和 fallback 必须保留在分母中。
- 同时报告保守距离 `d_min` 与原始胶囊距离 `d_min_raw`；碰撞只以 PyBullet contact 为事实源。

## 实验版本

```text
experiment_version: pending-r0-freeze
geometry_semantics: conservative_capsule_gap_v2
geometry_margin_m: 0.04
wrist_3_shape: intentional_sphere
wrist_3_radius_m: 0.055
output_root: outputs/restart_2026-09-09/
```

## 实验登记表

| 编号 | 日期 | 代码版本 | 配置 | 输出目录 | 完整性 | 状态 | 可用于论文 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| — | — | — | — | — | — | 尚未运行 | 否 |

## 已确认结果

尚无。必须先完成 R0 测试与冻结，再重新执行 R1 几何标定。

## 作废边界

2026-09-10 本次几何语义修订前产生的全部 `restart_2026-09-09` 训练、评估和标定数值均已作废，不得引用为新协议结果。
