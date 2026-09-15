# 到达优先安全同伦：实验顺序（粗略版）

详细参数、Gate、数据划分和指标定义以 [`docs/thesis/thesis_outline.md`](../thesis/thesis_outline.md) 第 1～3 章为唯一事实源。

## 1. 实现预检

1. 运行 `pytest -q tests/test_thesis_homotopy.py`。
2. 对无障碍、静态球、动态球各做 reset/step 冒烟检查。
3. 用短 S0 run 验证梯度更新、精确 checkpoint 和活跃 episode 恢复。
4. 不将预检策略或预检指标写入正式结果表。

## 2. S0：无障碍到达

1. 分别用训练 seed `11001`、`22002`、`33003` 从零训练。
2. 每个 block 100000 transition，每 25000 transition 保存完整 checkpoint，最多 3 个连续 block。
3. 对每个候选 checkpoint 运行无障碍严格 validation；按论文 Gate 选择，不访问 held-out。
4. 每个 seed 只把通过 Gate 的完整 checkpoint 传给 S1。

示例：

```bash
python scripts/train_thesis_homotopy.py \
  --config configs/experiments/thesis_homotopy.yaml \
  --stage s0 --seed 11001 --run-name s0_seed11001_block1
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

1. 在查看 held-out 前，按 validation 的保守聚合规则确定最终 actor，并冻结 actor、55 维输入、固定尺度和动作缩放。
2. 对冻结策略只运行一次 held-out，报告各场景成功、碰撞、关节越界、timeout、最小间隙、平滑性和实时性。
3. 主结论完成后再做动态奖励、输入项和第二阶段安全层消融；消融不得反向改变主模型选择。

严格评价示例：

```bash
python scripts/evaluate_thesis_homotopy.py \
  --config configs/experiments/thesis_homotopy.yaml \
  --checkpoint outputs/thesis_homotopy/<run>/checkpoints/<checkpoint>.pt \
  --scenes none static dynamic --episodes 100 --seed 41001 \
  --output outputs/thesis_homotopy/<run>/validation_seed41001
```
