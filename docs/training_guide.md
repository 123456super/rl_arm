# 训练说明

本文档说明当前项目如何启动训练、训练过程中看什么、训练结束后如何评估，以及根据结果决定下一步。当前阶段不建议直接做正式论文实验，先用小步训练确认链路和参数方向。

## 1. 训练前检查

先确认环境和代码链路可用：

```bash
conda run -n rl python scripts/smoke_test.py
```

看到类似下面的信息即可：

```text
{'observation_dim': 67, 'buffer_size': 12, 'risk_global': ..., 'd_min': ..., 'no_obstacle_risk': 0.0, 'actor_loss': ...}
```

如果这里失败，先不要训练。常见原因是没有激活 `rl` 环境、依赖没装完整、配置文件缺字段或 PyBullet 无法加载 URDF。

## 2. 配置入口

默认入口是：

```text
configs/default.yaml
```

它会组合以下配置：

```text
configs/robot/ur5_like.yaml      # 机器人模型、关节、胶囊体、reset 初始状态
configs/environment/sim.yaml     # 仿真环境、目标、障碍物、场景
configs/algo/sac.yaml            # 风险、奖励、平滑、SAC 超参数
configs/run/dev.yaml             # train/eval/smoke 运行参数
```

一般只需要改子配置文件，不要把参数重新塞回 `default.yaml`。如果只是临时试验，可以新建一个入口 YAML，用 `includes` 引用 `default.yaml` 后覆盖少量字段。

如果要使用 GPU，推荐在入口配置里设置：

```yaml
device: cuda:auto
```

`cuda:auto` 会综合每张候选卡的空闲显存占比和算力评分选择最合适的 GPU。相关参数在 `configs/default.yaml` 的 `device_selection` 中：

| 字段 | 含义 |
| --- | --- |
| `candidate_ids` | 候选 GPU id，`null` 表示扫描全部可见 GPU |
| `min_free_memory_gb` | 最小空闲显存要求，低于该值不会被选中 |
| `memory_weight` | 空闲显存占比权重，越大越偏向空闲卡 |
| `compute_weight` | 算力评分权重，越大越偏向高算力卡 |
| `fallback_to_cpu` | 没有可用 GPU 时是否回退 CPU |
| `print_summary` | 是否打印每张候选卡评分和最终选择 |

示例：

```yaml
includes:
  - default.yaml

seed: 101
env:
  obstacle:
    scenario: elbow_crossing
train:
  total_steps: 20000
```

## 3. 训练指令

当前完整方法的训练指令是：

```bash
conda run -n rl python scripts/train.py --config configs/default.yaml --method ldrc_adaptive --total-steps 100000
```

如果使用 `configs/run/dev.yaml` 里的默认 `train.method` 和 `train.total_steps`，可以简化为：

```bash
conda run -n rl python scripts/train.py --config configs/default.yaml
```

四个方法分别是：

```bash
conda run -n rl python scripts/train.py --config configs/default.yaml --method ee_fixed --total-steps 100000
conda run -n rl python scripts/train.py --config configs/default.yaml --method link_fixed --total-steps 100000
conda run -n rl python scripts/train.py --config configs/default.yaml --method ldrc_fixed --total-steps 100000
conda run -n rl python scripts/train.py --config configs/default.yaml --method ldrc_adaptive --total-steps 100000
```

方法含义：

| 方法 | 含义 | 主要用途 |
| --- | --- | --- |
| `ee_fixed` | 末端风险 + 固定风险惩罚 + 固定平滑 | 末端风险基线 |
| `link_fixed` | 连杆级风险 + 固定风险惩罚 + 固定平滑 | 验证连杆级风险是否有用 |
| `ldrc_fixed` | 连杆级风险 + 约束 SAC + 固定平滑 | 验证约束 SAC |
| `ldrc_adaptive` | 连杆级风险 + 约束 SAC + 自适应平滑 | 完整方法 |

## 4. 训练输出在哪里

以 `ldrc_adaptive` 为例，默认输出目录是：

```text
outputs/ldrc_adaptive/
```

主要文件：

| 文件 | 作用 |
| --- | --- |
| `actor.pt` | 评估和部署时加载的策略网络 |
| `agent_state.pt` | critic、alpha、lambda、cost_ema 等训练状态 |
| `config.json` | 本次训练展开后的完整配置快照 |
| `train_metrics.csv` | 每个 episode 的训练指标 |

训练中终端会显示 `tqdm` 进度条。每隔 `train.log_interval` 个 episode 会更新：

| 字段 | 含义 |
| --- | --- |
| `ep` | 已完成 episode 数 |
| `r20` | 最近 20 个 episode 的平均 reward |
| `cost` | 当前 episode 的平均风险代价 |
| `lambda` | 约束 SAC 的拉格朗日乘子 |
| `alpha` | SAC 熵温度 |

更可靠的判断应看 `train_metrics.csv`，不要只看终端瞬时值。

## 5. 训练过程中怎么看

重点看 `outputs/<method>/train_metrics.csv`：

| 字段 | 怎么理解 |
| --- | --- |
| `episode_reward` | 越高通常表示到达任务学得更好，但不能单独作为安全结论 |
| `episode_cost` | 单个 episode 的总风险代价 |
| `safety_violation_rate` | 安全距离违反比例，越低越好 |
| `min_distance` | episode 内最小连杆-障碍物表面距离，应尽量大于 `risk.d_safe` |
| `success` | 是否成功到达目标 |
| `collision` | 是否发生碰撞 |
| `mean_risk` | 平均全局风险 |
| `lambda` | 约束方法中安全约束强度的自适应变化 |
| `alpha` | 熵温度，主要用于判断 SAC 是否还在正常更新 |

建议先每隔一段时间查看最近 20 到 50 个 episode：

```bash
tail -n 50 outputs/ldrc_adaptive/train_metrics.csv
```

也可以用 pandas 快速看均值：

```bash
conda run -n rl python - <<'PY'
import pandas as pd
p = 'outputs/ldrc_adaptive/train_metrics.csv'
df = pd.read_csv(p)
tail = df.tail(50)
print(tail[['episode_reward', 'success', 'collision', 'safety_violation_rate', 'min_distance', 'mean_risk', 'lambda']].mean(numeric_only=True))
PY
```

## 6. 什么时候停止或调整

训练不是看某一个指标，而是看任务和安全是否同时改善。

### 情况 A：reward 上升，但 collision 和 violation 也高

说明策略可能学会了冲向目标，但安全约束不够强。

下一步：

- 对 `ldrc_*`：检查 `lambda` 是否上升。如果 `lambda` 长期接近 0，可能是 `risk.cost.c_safe` 太宽松或 cost 信号偏低。
- 可适度降低 `risk.cost.c_safe`，或提高 `risk.cost.k_violation`、`risk.cost.k_collision`。
- 不要先加大 `reward.collision_penalty`，否则会模糊“任务奖励”和“安全代价”的分离。

### 情况 B：collision 低，但 success 也低

说明策略可能过于保守或目标奖励不够强。

下一步：

- 看 `episode_length` 是否经常到 `env.max_episode_steps`，如果是，策略可能停滞。
- 可适度提高 `reward.w_progress` 或降低 `risk.cost.c_safe` 的严格程度。
- 检查 `env.action_scale` 是否过小，导致机械臂速度不够。

### 情况 C：`lambda` 一直快速增大

说明平均风险代价长期高于 `risk.cost.c_safe`。

下一步：

- 先确认风险代价是否合理：看 `mean_risk`、`safety_violation_rate`、`collision` 是否确实偏高。
- 如果确实偏高，继续训练或加强安全代价。
- 如果 collision 很低但 cost 仍很高，可能是 `d_safe`、`sigma_d`、`c_safe` 标定偏严，需要做小规模标定。

### 情况 D：`lambda` 长期为 0 或接近 0

说明约束没有真正参与策略优化。

下一步：

- 如果安全指标已经很好，这是正常现象。
- 如果 collision/violation 仍高，说明 `c_safe` 太宽松，或 cost 权重太低。
- 优先检查 `risk.cost.c_safe` 和 `risk.cost.k_violation`。

### 情况 E：reward、success、risk 都剧烈震荡

说明训练不稳定。

下一步：

- 先降低学习率：`sac.actor_lr`、`sac.critic_lr`。
- 检查 `sac.batch_size` 是否过小、`warmup_steps` 是否过短。
- 缩小 `env.action_scale` 可以降低动作过激导致的碰撞和震荡。

## 7. 训练后评估

训练结束后先做小规模评估：

```bash
conda run -n rl python scripts/evaluate.py \
  --config configs/default.yaml \
  --method ldrc_adaptive \
  --checkpoint outputs/ldrc_adaptive/actor.pt \
  --episodes 20
```

评估结果会写入：

```text
outputs/ldrc_adaptive/eval_metrics.csv
```

终端会打印摘要：

| 字段 | 含义 |
| --- | --- |
| `success_rate` | 成功率 |
| `collision_rate` | 碰撞率 |
| `non_end_link_collision_rate` | 非末端连杆碰撞率 |
| `mean_final_position_error` | 平均最终位置误差 |
| `mean_min_distance` | 平均 episode 最小距离 |
| `mean_cost` | 平均风险代价 |
| `mean_safety_violation_count` | 平均安全距离违反次数 |
| `mean_action_variation` | 平均动作变化幅度 |
| `mean_rms_acceleration` | 平均关节加速度 RMS |
| `mean_rms_jerk` | 平均 jerk RMS |

如果需要分析典型 episode 曲线：

```bash
conda run -n rl python scripts/evaluate.py \
  --config configs/default.yaml \
  --method ldrc_adaptive \
  --checkpoint outputs/ldrc_adaptive/actor.pt \
  --episodes 3 \
  --trace-output outputs/ldrc_adaptive/traces
```

trace 文件在：

```text
outputs/ldrc_adaptive/traces/episode_0000.csv
```

重点看 `d_min`、`risk_global`、`beta`、`qdot_norm`、`acc_norm`、`jerk_norm`。

## 8. 根据评估结果选择下一步

### 先判断是否具备基础到达能力

建议先做无障碍或低风险场景。可以新建一个配置，例如 `configs/experiments/no_obstacle.yaml`：

```yaml
includes:
  - ../default.yaml

env:
  obstacle:
    enabled: false
```

然后训练或评估完整方法。如果无障碍成功率都很低，先不要做避障对比，应先调任务奖励、动作尺度或目标采样范围。

### 再判断是否具备基本避障能力

使用默认随机障碍物场景训练和评估。若 `collision_rate` 高、`mean_min_distance` 低于 `risk.d_safe`，优先调风险代价和约束参数。

### 再看自适应平滑是否值得保留

对比：

```bash
ldrc_fixed vs ldrc_adaptive
```

如果 `ldrc_adaptive` 的 `mean_rms_jerk` 更低，同时 `collision_rate` 和 `success_rate` 没变差，说明自适应平滑有效。如果 jerk 更低但 collision 上升，说明 `smoothing.beta_min`、`smoothing.beta_max` 或 `smoothing.risk_high` 需要调。

### 最后再做四方法对比

只有当单方法已经能稳定训练时，再依次训练：

```text
ee_fixed
link_fixed
ldrc_fixed
ldrc_adaptive
```

正式对比前要保证：

- 使用相同训练步数。
- 使用相同测试 seed。
- 使用相同测试 episode 数。
- 先做 20 episode 小评估，再扩大到 100 episode。
- 不要只凭一次 seed 下的结果下结论。

## 9. 推荐的当前阶段流程

当前还不是正式实验阶段，建议按这个顺序推进：

1. `smoke_test.py` 通过。
2. `ldrc_adaptive` 跑 5k 到 20k steps，确认训练文件和 checkpoint 正常生成。
3. 用 20 episode 评估，看 success/collision/min_distance 是否有明显问题。
4. 若基础到达能力差，先做 `obstacle.enabled: false` 的无障碍调试。
5. 若到达可以但碰撞高，调整 `risk.cost.c_safe`、`k_violation`、`k_collision`。
6. 若碰撞低但动作抖，调整 `smoothing`。
7. 单方法稳定后，再开始四方法对比和多 seed 训练。
