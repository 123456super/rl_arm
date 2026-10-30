# 当前训练配置

当前新主线为 [单 SAC 参考论文迁移方案](../../docs/毕业论文/单SAC参考论文迁移分析.md)：
`paper_single_sac.yaml` 使用 v2 独立环境，从零训练同时到达与避障的单策略，默认八环境、
ReLU、batch 256、UTD 1、每次更新 Actor/Critic，仿真推进与学习重叠。
动作采用作者 SAC 的固定初始角偏移，组合误差成功阈值为 0.005。
它只包含机器人资产及自身参数，不继承旧交接、课程、奖励或策略权重。

```bash
python scripts/core/train_paper_single_sac.py \
  --config configs/experiments/paper_single_sac.yaml \
  --run-name paper_single_sac_v2_1500k --steps 1500000 --num-envs 8
```

终端每 1000 个总环境步显示进度、无碰撞到达率、到达率、碰撞率、途中/终点保持、误差和速度。
主配置默认不启用 AE；`paper_single_sac_aew.yaml` 单独保留同环境 SAC-AEW 对照，
不宣称等同于论文的 PPO＋AE＋PF。v2 拒绝 v1 断点续训，须新建目录从零训练。
断点为 `outputs/paper_single_sac/<run-name>/checkpoints/step_XXXXXXX.pt`，包含单 SAC、
Replay、优化器、八个仿真状态及随机状态。续训用 `--resume` 和新的 `--run-name`；
不写 `--steps` 时补齐原定 150 万步，写入时表示额外预算。

以下是保留用于历史结果对照的双 SAC 配置。其完整方案、成功条件及旧命令见
[双 SAC 方案](../../docs/毕业论文/双SAC机械臂到达与局部越障方案.md)。

| 配置 | 当前用途 |
|---|---|
| `paper_local_avoidance_geometry.yaml` | 单障碍静态几何越障，样本生成与局部训练的默认配置 |
| `paper_local_avoidance_geometry_multi.yaml` | 多障碍静态几何越障，数量提案分布为 1/2/3 个、50%/30%/20% |
| `paper_reaching_sac.yaml` | 独立 S0 到达训练配置 |

这些旧实验的共用参数放在 `configs/base/`：

- [thesis_common.yaml](../base/thesis_common.yaml)：机器人、观测、到达奖励、课程、执行与 SAC 公共参数。
- [promotion_protocol.yaml](../base/promotion_protocol.yaml)：到达课程晋级与独立测试 seed。
- [dual_sac.yaml](../base/dual_sac.yaml)：两套独立 SAC 的公共连接配置。

几何越障入口直接继承公共配置并定义完整的局部任务，不再依赖轨迹参照实验。
多障碍入口只覆盖数量分布；到达入口只选择 S0 和到达策略。清理前后这三个入口
加载的有效参数保持一致，已有 bank 及检查点任务签名不变。

越障训练使用 `scripts/core/train_local_avoidance.py`，冻结成熟到达策略，单独学习局部越障。
新正式训练要求 bank 已通过整臂路径及局部出口校验；检查只在离线采样、转换或审计阶段执行。
联合运行使用 `scripts/core/evaluate_joint_policy.py`，由瞬时运动意图和安全距离切换策略。

到达交接微调使用 `scripts/core/train_reaching_handoff.py`，固定越障策略，在原始无障碍起点
与真实交接副本上混合训练；联合评估通过 `--reaching-checkpoint` 替换到达 Actor。

正式局部训练默认 8 个环境、batch 1024、UTD 0.25，Actor 每 2 次 Critic 更新一次。
终端每 1000 个累计环境步显示进度和近期在线成功率，默认保留回合与进度摘要，详细更新日志关闭。
完整续训使用相同任务协议、bank 与已有并行环境数，`--steps` 表示额外预算。
