# 系统架构

项目采用“仿真后端与算法核心解耦”的分层结构。Gym 环境是组装入口，不再直接实现全部算法细节。

```text
TargetProvider + ObstacleProvider + Robot/PyBullet state
                 |
                 v
            RiskDetector
                 |
                 v
 ReachingObservationBuilder / ReachingObjective
                 |
                 v
            SAC policy
                 |
                 v
 JointVelocityRateLimiter -> Butterworth/EMA -> Quintic RTB
                 |
                 v
          PyBullet motor backend
```

## 模块边界

- `collision/`：距离和风险检测接口。当前提供单球体—胶囊连杆实现以及无障碍实现；后续 GJK/FCL、多障碍物和自碰撞实现应放在这里。
- `control/`：策略动作到执行轨迹的管线，包括策略周期速度变化率限制、EMA 或 Butterworth 滤波以及五次插值。
- `scene/`：目标之外的场景对象状态机。当前封装单个球形动态障碍物的采样、命名穿越场景和边界反弹；后续多障碍物或视觉检测结果同步应从这里扩展。
- `tasks/`：任务状态、observation schema、reward/cost 和目标轨迹。默认任务仍是静态到达，`linear_bounce` 可用于动态目标跟踪训练。
- `envs/`：仿真生命周期和模块编排。PyBullet 专用调用应逐步收敛在这一层或未来的 `backends/` 中。
- `algorithms/`：SAC 与约束 SAC，不依赖 PyBullet。

## 兼容性约束

当前 observation schema 为 `link_risk_v1`。只要特征顺序和含义不变，就不得修改该标识；任何 schema 变化必须使用新标识并重新训练。训练时的目标模式、动作变化率限制和 RTB 配置都会随完整配置写入运行目录。

`env.execution.max_policy_velocity_delta=0.1` 是论文式策略动作变化率限制。历史 EMA checkpoint 应通过 `random_crossing_link_fixed_penalty1_ema.yaml` 评估，该配置会把限制设为 `null`，以保持旧执行语义。

## 后续扩展顺序

1. 将 `ObstacleProvider` 从单球扩展为多障碍物，再将 observation 改成固定上限加 mask 或集合编码器。
2. 增加自碰撞、地面和工作台距离检测后端。
3. 提取 `RobotBackend`，使 PyBullet 和真实 UR 控制器共享同一执行管线。
4. 增加带时间戳、坐标系、置信度和有效期的视觉状态同步层。
5. 在实机链路中加入 stale-state watchdog、安全停止和 shadow mode。
