# 指标和单位定义

## 命令与风险

| 字段 | 单位 | 定义 |
| --- | --- | --- |
| `action` | 1 | actor 的 `[-1,1]` residual；hierarchical 模式下不是完整速度 |
| `action_scale` | rad/s | 单关节最终命令绝对速度上限 |
| `hierarchical_nominal_qdot` | rad/s | tracker 或 terminal servo 输出 |
| `residual_qdot` | rad/s | budget 和 residual scale 处理后的 actor 修正 |
| `qdot_policy` | rad/s | nominal 与 residual 合成、过滤前的候选命令 |
| `qdot_requested` | rad/s | 送入最终 safety filter 的命令 |
| `qdot_cmd` | rad/s | filter 后实际送往 PyBullet 的唯一命令 |
| `hierarchical_residual_budget` | 1 | 风险条件 residual 权限，范围 `[0,1]` |
| `link_velocities_mps` | m/s | capsule 最近点的线速度 `J_point qdot` |
| `max_link_speed_bound_mps` | m/s | 当前姿态、关节速度盒下 capsule 端点线速度局部上界 |
| `obstacle_velocity` | m/s | 障碍物中心线速度 |
| `joint_acceleration_limit_radps2` | rad/s² | 单关节命令加速度限制 |
| `d_pred` | m | 预测窗口内最小 capsule 表面间距 |
| `d_robust` | m | 扣除几何、感知、时延和跟踪裕量后的距离 |
| `h` | m | `d_robust-d_safe` |

禁止用 `action_scale` 数值填写任何 `_mps` 字段。`max_link_speed_bound_mps` 是逐周期局部上界；需要全局保证时必须另做全工作空间审计。

## 规划与状态机

| 字段 | 定义 |
| --- | --- |
| `hierarchical_plan_reason` | `direct_path`、`rrt_connect`、`ik_not_found` 或 `not_found` |
| `hierarchical_plan_iterations` | 最近一次 RRT-Connect 迭代数；直连为 0 |
| `hierarchical_plan_count` | 当前 episode 总规划尝试次数 |
| `hierarchical_replan_count` | 事件触发重规划次数 |
| `hierarchical_waypoint_count/index` | 当前路径 waypoint 数及跟踪索引 |
| `hierarchical_path_progress` | `index/(count-1)`，范围 `[0,1]`；不是几何弧长比例 |
| `hierarchical_hold_steps` | 当前 episode 累积 AVOID_HOLD 控制步数 |
| `hierarchical_consecutive_hold_steps` | 当前连续 AVOID_HOLD 步数，用于触发 REPLAN |
| `hierarchical_state` | `TRACK`、`AVOID_HOLD`、`REPLAN`、`SERVO`、`PLAN_FAILED` |

`ik_not_found` 与 `not_found` 都表示有限预算内未找到，禁止写成绝对不可达。plan-conditioned success 的分母是 `plan_found` episode；它必须和 full-manifest success 同时报告。

## 碰撞事件

| 字段 | 含义 |
| --- | --- |
| `collision_capsule_overlap` | 障碍物球与保守 capsule 包络重叠 |
| `collision_pybullet_contact` | PyBullet 返回机器人与障碍物物理接触 |
| `collision_any` | 上述两个事件的逻辑或 |
| `termination_collision` | 按配置实际导致 episode 终止的碰撞 |
| `termination_reason` | `capsule_overlap`、`pybullet_contact` 或空字符串 |
| `collision` | 兼容字段，等同 `collision_any`；新论文不得单独引用 |

planner 的 self-collision/contact 检查属于状态有效性，不等同于 episode 的 obstacle collision 指标；若要报告 self-collision，必须新增独立字段。

## 成功、安全与统计

success 要求 TCP 目标误差小于 `success_tolerance`，且当前 `d_min>d_safe`。因此必须同时报告 final error、capsule overlap 和 physical contact。

Safety filter 至少报告 intervention rate/norm、safe-stop rate、infeasible/projection-failure rate、最小 predictive `h` 和 solve time。单个 smoke episode 只能验证链路，不得作为成功率证据。正式结果给出各训练 seed、pooled 分子/分母，并区分 validation、full final 和 plan-conditioned final。
