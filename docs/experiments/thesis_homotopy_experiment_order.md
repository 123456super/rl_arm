# 到达优先安全同伦：实验顺序（粗略版）

> **已终止（2026-09-16）**：本顺序对应 v2～v7 的旧状态/奖励契约，仅供追溯。
> v8 不允许从旧 checkpoint 或 replay 恢复；现行顺序见
> [`docs/experiments_9/experiment_order.md`](../experiments_9/experiment_order.md)。

详细参数、Gate、数据划分和指标定义以 [`docs/thesis/thesis_outline.md`](../thesis/thesis_outline.md) 第 1～3 章为唯一事实源。

## 1. 实现预检

1. 运行 `pytest -q tests/test_thesis_homotopy.py`。
2. 对无障碍、静态球、动态球各做 reset/step 冒烟检查。
3. 用短 S0 run 验证梯度更新、精确 checkpoint 和活跃 episode 恢复。
4. 在多组有效构型、六段胶囊及两个脱离方向上构造 PyBullet 浅接触，验证每个实际 contact 均满足 `d_min<d_safe` 且 `risk_max>=0.79`。
5. 不将预检策略或预检指标写入正式结果表。

## 2. S0：无障碍到达

1. 分别用训练 seed `11001`、`22002`、`33003` 从零训练。
2. S0-P 固定 `orientation_scale=0`，按配置中的 16 个离散 `goal_scale` 台阶训练；每个台阶至少 5000 个非 warm-up transition，并重新累计 100-episode 位置到达窗口，达到 0.80 后才前进一级。
3. 在完整目标范围上继续位置训练至少 25000 transition；随后进入 S0-R，固定 `goal_scale=1`，按配置中的 17 个离散姿态档位把 `eta` 从 0 提高到 1，并将容差从 `pi` 收紧到 `0.10 rad`。
4. 每个姿态档至少训练 5000 个当前任务 transition；换档后重新累计 100 个当前姿态 episode 和 100 个位置锚点 episode，成功率分别达到 0.80 和 0.95。姿态奖励在远离目标时按当前 `eta` 的 20% 生效，并在位置误差 `0.20→0.08 m` 间平滑提升至 100%。
5. 正常状态下 25% episode 为位置锚点，replay 按锚点/当前/历史 `25%/50%/25%` 采样；保持率低于 0.98 时自动提高到 `40%` 和 `40/40/20`，低于 0.95 或确定性探针失败时进入恢复模式，使用 `50%` 和 `50/35/15`，并冻结姿态晋级。
6. 每次姿态晋级还必须通过 50 个私有固定目标的确定性位置探针：成功率不低于 0.95，碰撞和关节越界均为 0。探针失败后至少再训练 5000 个当前档 transition 才能重试。
7. `orientation_scale=1` 后再巩固完整位姿至少 25000 个非锚点 transition。每个 block 100000 transition，每 25000 transition 保存完整 checkpoint。
8. 只有 `goal_scale=1`、`orientation_scale=1`、两次巩固计数达标、当前姿态/位置锚点双窗口达标且最终确定性位置探针通过的 checkpoint 才能运行 validation。
9. 每个 seed 只把通过 Gate 的完整 checkpoint 传给 S1，训练器会在 S0→S1 时再次检查资格。

示例：

```bash
python scripts/train_thesis_homotopy.py \
  --config configs/experiments/thesis_homotopy.yaml \
  --stage s0 --seed 11001 \
  --restart-retention-from outputs/thesis_homotopy/s0_v6_seed11001_pose_block1/checkpoints/step_0025000.pt \
  --run-name s0_v7_seed11001_retention_block1
```

## 3. S1：静态障碍安全同伦

1. 从同一 seed 的 S0 合格完整 checkpoint 继续，不只加载 actor。
2. 按无障碍/静态 `25%/75%` 采样；静态场景从 `xi_static=0.02` 开始，碰撞不终止。
3. 最近 100 个静态 episode 到达率达到 0.90 时累计合格 transition，并在 50000 transition 内把 `xi_static` 提到 1；低于门槛就暂停，绝不降低到达奖励。
4. 达到 1 后规范化静态 replay，切换为碰撞真终止，再训练至少 25000 个静态 transition。
5. 只评价满足严格期资格的 checkpoint，同时检查 S0 retention Gate；失败则从 block 末 checkpoint 连续下一 block。

示例：

```bash
python scripts/train_thesis_homotopy.py \
  --config configs/experiments/thesis_homotopy.yaml \
  --stage s1 --seed 11001 \
  --resume outputs/thesis_homotopy/<S0合格run>/checkpoints/<完整checkpoint>.pt \
  --run-name s1_seed11001_block1
```

## 4. S2：动态障碍安全同伦

1. 从同一 seed 的 S1 合格完整 checkpoint 继续。
2. 按无障碍/静态/动态 `20%/30%/50%` 采样；静态保持 `xi_static=1` 和严格终止，动态从 `xi_dynamic=0.02` 开始且碰撞可继续。
3. 最近 100 个动态 episode 到达率达到 0.80 后才推进动态安全权重；到 1 后规范化动态 replay。
4. 再训练至少 25000 个严格动态 transition，然后同时检查 S0、S1、S2 validation Gate。
5. 三个训练 seed 都必须完成 S2；未通过就按论文规则连续下一 block，最多 3 blocks。

## 5. 冻结、held-out 与消融

1. 在查看 held-out 前，按 validation 的保守聚合规则确定最终 actor，并冻结 actor、71 维输入、固定尺度和动作缩放。
2. 对冻结策略只运行一次 held-out，报告各场景成功、碰撞、关节越界、timeout、最小间隙、平滑性和实时性。
3. 主结论完成后再做动态奖励、输入项和第二阶段安全层消融；消融不得反向改变主模型选择。

严格评价示例：

```bash
python scripts/evaluate_thesis_homotopy.py \
  --config configs/experiments/thesis_homotopy.yaml \
  --checkpoint outputs/thesis_homotopy/<run>/checkpoints/<checkpoint>.pt \
  --scenes none --episodes 100 --seed 41001 \
  --output outputs/thesis_homotopy/<run>/validation_seed41001
```

S0 的每个候选 checkpoint 必须分别使用 validation seed `41001`、`42002`、`43003` 运行无障碍评价，三者均通过才算通过 S0 Gate。评价器的 `collision_rate` 是任务球、自碰撞和环境碰撞的并集；`joint_limit_rate` 单独统计。
