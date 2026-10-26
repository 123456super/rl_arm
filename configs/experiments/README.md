# Full 训练配置

唯一正式配置为 `paper_full.yaml`，训练入口为 `scripts/core/train_thesis_homotopy.py`。
Full 使用同一 L0–L13 精度课程；当前阶段配置支持 S0 无障碍、S1 静态障碍及后续动态障碍。

`paper_full_s1_coupling.yaml` 是可选 S1 目标—障碍耦合奖励实验，不替换基础配置。
公式、Replay/示范库契约、验证和试验命令见
[S1 耦合奖励文档](../../docs/experiments/s1_goal_obstacle_coupling.md)。

`thesis.obstacles` 配置混合场景、验收通过的冻结 LSTM 和外部风险参数。
首次运行先执行 `scripts/core/train_obstacle_predictor.py`。
内部阶段 `s0` / 回放场景键 `none` 不决定实际障碍数量。

执行命令见根目录 README；实现和参数约束见 `docs/experiments/full_predictive_avoidance.md`。
