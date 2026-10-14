# 论文文档索引

当前主线只有五个正式方法：

```text
Hybrid KP-SAC-weak → Full → M1/M2/M3
```

- `experimental_protocol.md`：唯一执行合同，规定配置、预算、升档和评估；
- `joint_pose_reachability_paper_plan.md`：论文问题、方法逻辑、实验归因和结果解释。

Hybrid KP-SAC-weak 同时使用显式六维误差与三关键点联合误差，保留稠密关键点奖励和课程。Full 的三个复合模块为 Direction（Jacobian）、Objective（precision quality + hold/leave + terminal-gated bottleneck）和 Data（Success Replay + Boundary Replay）。Full 对弱基线用于整体比较；M1/M2/M3 分别验证 Bottleneck 子项、Boundary Replay 子项和完整 Direction 模块。Semantic replay、far-distance restart 和旧 M4/M5 不属于新主线。
