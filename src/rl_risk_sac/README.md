# `rl_risk_sac`

这是 Hybrid Keypoint + Jacobian + Auto-PCR 串行协议的核心包。训练线路固定为：

```text
S0 reaching + P1-P5 precision curriculum
  -> S1 static obstacle avoidance
  -> S2 dynamic obstacle avoidance
```

`ThesisHomotopyEnv` 将 actor 的 6 维关节速度动作执行到 PyBullet，并返回末端位姿误差、障碍物/自碰撞风险、速度和奖励字段。`HomotopyCurriculum` 管理精度档位与 S0/S1/S2 完成条件；`HomotopyReplayBuffer` 保存带有原始档位和逐 transition 容差的 replay 数据，并维护跨 P1-P5 的语义长期池。

算法入口是 `ThesisSACAgent`，实现带 Auto-PCR 的 reward-only SAC。旧的通用风险 SAC、动态障碍环境和迁移机制已从正式代码路径移除。

推荐从 `scripts/core/train_thesis_homotopy.py` 和
`configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml` 开始阅读。
四池配置与升档说明见 `docs/experiments_9/four_pool_hybrid_keypoint_jacobian_scheme.md`。
