# 指标和单位定义

## 速度与裕度

| 字段 | 单位 | 定义 |
| --- | --- | --- |
| `action_scale` | rad/s | 单关节命令绝对速度上限 |
| `link_velocities_mps` | m/s | 当前胶囊最近点的线速度，按 `J_point qdot` 计算 |
| `max_link_speed_bound_mps` | m/s | 当前姿态、关节速度盒下胶囊端点线速度的局部上界 |
| `obstacle_velocity` | m/s | 障碍物中心线速度 |
| `joint_acceleration_limit_radps2` | rad/s² | 单关节命令加速度上限 |
| `d_pred` | m | 短时预测窗口内最小表面间距 |
| `d_robust` | m | 扣除几何、感知、时延、跟踪裕度后的距离 |
| `h` | m | `d_robust - d_safe` |

禁止用 `action_scale` 数值直接填写任何 `_mps` 字段。当前 `max_link_speed_bound_mps` 是每周期根据当前姿态计算的局部上界，若论文需要全局保证，必须另做全工作空间离线上界审计。

## 碰撞事件

| 字段 | 含义 |
| --- | --- |
| `collision_capsule_overlap` | 障碍物球与保守胶囊包络发生重叠 |
| `collision_pybullet_contact` | PyBullet 返回机器人与障碍物物理接触 |
| `collision_any` | 上述两个事件的逻辑或，用于保守统计 |
| `termination_collision` | 按配置实际导致 episode 终止的碰撞事件 |
| `termination_reason` | `capsule_overlap`、`pybullet_contact` 或空字符串 |
| `collision` | 兼容字段，等同 `collision_any`；新论文禁止单独引用 |

阶段一配置使用 `env.collision.termination=any` 保持历史行为。2026-08-03 后的统一 P3 配置使用 `physical_contact`，避免保守几何包络直接改变回合长度，同时仍完整报告 capsule overlap。

## 成功率

success 仍要求目标误差小于阈值且当前胶囊最小距离大于 `d_safe`。因此 success 不是纯到达率；报告时应同时给出最终位置误差、physical contact 和 capsule overlap。
