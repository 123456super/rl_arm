# 阶段一复现与新研究开发指南

## 1. 当前阶段与使用边界

阶段一仿真主比较已经完成：`ee_fixed`、`link_fixed_penalty1` 和 `ldrc_fixed` 已完成 3 个 train seeds、checkpoint selection、5 个场景和 3 个 held-out eval seeds 的统一评估。阶段一结论口径见 [experiment_conclusions.md](experiment_conclusions.md)；当前不应重跑主比较或将旧四方法结果写成新主题的正文主表。

本分支的新主线是“不确定性连杆预测风险 + 安全过滤器 + 协同安全强化学习”。方法范围、实验矩阵和实施顺序以 [research_direction.md](research_direction.md) 为准；本指南中的既有命令仅服务于阶段一复现、环境检查和新模块的开发基线。

本指南用于三类工作：

1. 检查环境和复现已有结果材料。
2. 进行与论文主结论隔离的小规模开发或诊断训练。
3. 使用新的 train seeds 和独立 eval seeds 开展后续研究复核。

若目标是实机低速部署，请直接使用 [deployment_preflight.md](deployment_preflight.md)，不要把本指南中的仿真训练命令当作实机控制流程。

## 2. 先做哪一类工作

| 目标 | 推荐入口 | 是否改变主结论 |
| --- | --- | --- |
| 验证代码、URDF 和配置链路 | `scripts/smoke_test.py` | 否 |
| 重建论文表格和图件 | `scripts/prepare_paper_materials.py` | 否 |
| 检查部署候选的离线推理与仿真限幅 | `scripts/deployment_preflight.py` | 否 |
| 调试一个新想法 | 新建 YAML，使用独立输出目录和开发 seed | 否 |
| 复核或挑战论文结论 | 新 train seeds + 新 held-out eval seeds | 可能；需重新完整汇总 |

不要用 5k--20k steps 的开发训练、单个 seed 或旧 `outputs/formal/summary/` 数据更新论文排序。

## 3. 环境检查

从项目根目录运行：

```bash
conda run -n rl python scripts/smoke_test.py
conda run -n rl python scripts/smoke_test.py --config configs/ur5.yaml
```

成功时会输出 observation 维度、replay buffer、风险量和一次 SAC 更新的指标。失败时先处理环境、URDF 或 YAML 配置问题；不要在链路未通过时启动长训练。

如需使用 GPU，在入口 YAML 中设置 `device: cuda:auto`。默认的设备选择会根据空闲显存和算力评分选择候选 GPU；无可用 GPU 时可按 `device_selection.fallback_to_cpu` 回退。

## 4. 配置与方法映射

默认入口为 `configs/default.yaml`，它组合机器人、环境、算法和运行参数：

```text
configs/robot/ur5.yaml           # UR5、关节名、连杆胶囊体
configs/environment/sim.yaml     # PyBullet、目标、障碍物和场景
configs/algo/sac.yaml            # 风险、奖励、平滑和 SAC 参数
configs/run/dev.yaml             # train/eval/smoke 参数
```

一般在子配置或新的实验入口 YAML 中覆盖少数参数，不要把所有参数复制回 `default.yaml`。

| 代码中的 `train.method` / `--method` | 含义 | 当前定位 |
| --- | --- | --- |
| `ee_fixed` | 末端风险 + 固定风险惩罚 + 固定平滑 | 主比较基线 |
| `link_fixed` 且 `sac.fixed_risk_penalty: 1.0` | 连杆级风险 + 固定风险惩罚 + 固定平滑 | 论文中称 `link_fixed_penalty1`；主方法与部署候选 |
| `ldrc_fixed` | 连杆级风险 + 约束 SAC + 固定平滑 | 主比较对象；仅在部分场景碰撞率更低 |
| `ldrc_adaptive` | 连杆级风险 + 约束 SAC + 自适应平滑 | 历史失败消融，不作为部署候选 |

注意：`link_fixed_penalty1` 是论文与结果目录中的配置名称；命令行/代码方法名仍为 `link_fixed`，通过 `sac.fixed_risk_penalty: 1.0` 区分。

## 5. P1/P2 安全链路开发验证

阶段一默认配置的 `env.safety_filter.enabled` 为 `false`，以避免改变已冻结的历史结果。P2 开发入口 [configs/experiments/p2_safety_filter_dev.yaml](../configs/experiments/p2_safety_filter_dev.yaml) 显式启用过滤器，策略命令会经过预测风险、连杆/TCP 约束和安全过滤器后才发送到 PyBullet。

从项目根目录运行：

```bash
conda run -n rl pytest -q tests

conda run -n rl pytest -q \
  tests/test_predictive_risk.py \
  tests/test_safety_filter.py \
  tests/test_safety_filter_integration.py

conda run -n rl python scripts/smoke_test.py \
  --config configs/experiments/p2_safety_filter_dev.yaml
```

集成测试覆盖有效估计直通、无效/过期估计零速度停机、约束不可行停机和“过滤器是唯一命令出口”。运行时 `info` 额外包含 `predictive_risk_status`、`predictive_h_min_m`、`qdot_requested`、`qdot_cmd`、`safety_filter_intervention_norm`、`safety_filter_safe_stop`、`safety_filter_max_constraint_violation` 和 `safety_filter_solve_time_s`。

该链路使用 PyBullet 点雅可比构造连杆与 TCP 约束，避免控制周期内的有限差分状态保存/恢复。P2 的 20 episode、2,693 控制步测量中，过滤器均值为 2.68 ms、P95 为 4.94 ms，未超过 50 ms 仿真控制周期。该数字只覆盖当前 PyBullet 进程内的过滤器计算，不包含 RGB-D、通信或控制器延迟，也不构成真实控制器验收或安全保证。

### 5.1 P3 因子化开发现状

`configs/experiments/p3/` 中的 B1--B5 已用 train seed `4101`、10k step 和 eval seed `5101` 的 20 episode 完成链路检查。B4/B5 的过滤器指标已归档到 `train_metrics.csv`、评估 CSV 和 trace；其求解 P95 分别为 3.89 ms、4.88 ms，均未超 50 ms。

这批单 seed、短预算结果没有产生可冻结的综合候选：B4 的碰撞率/安全距离违反率为 25%/9.40%，B5 的成功率为 0%。因此不要启动新的 train/eval seeds、OOD 或真机工作。下一轮应保持 B1--B5 因子和参数不变，将共同开发训练预算扩展至 100k，并完成独立 checkpoint validation；完整开发预算仍无合理任务--安全折中时，先诊断风险代价、奖励和不可行约束，而不是继续扩大实验矩阵。

## 6. 开发训练：安全的最小流程

短训练只用于确认新配置、日志和 checkpoint 链路。UR5 示例：

```bash
conda run -n rl python scripts/train.py --config configs/experiments/ur5_short_train.yaml
```

无障碍基础到达调试使用：

```bash
conda run -n rl python scripts/train.py --config configs/experiments/no_obstacle.yaml
```

开发实验应遵循以下约束：

- 新建入口 YAML 并设置新的 `train.output_dir` 或 `run_name`，避免写入正式结果目录。
- 在日志中记录 method、风险惩罚、`C_safe`、train seed、eval seed 与代码版本。
- 先进行 20 episode 小评估；只有链路正确且指标合理时才扩展预算。
- 若继续固定惩罚或约束机制研究，使用未参与 `w_R=1.0` 筛选的 train seeds；正式复核还需使用新的 eval seeds。

每次训练会创建独立目录，常见文件如下：

| 文件 | 用途 |
| --- | --- |
| `actor.pt` | 最终策略网络 |
| `actor_step_*.pt` | checkpoint selection 或诊断 |
| `agent_state.pt` | critic、alpha、lambda 等训练状态 |
| `config.json` | 展开后的配置快照 |
| `progress.csv` | step 级进度 |
| `train_metrics.csv` | episode 级训练指标 |

## 7. 训练过程的诊断口径

不要只看 reward。至少同时观察任务、安全与数值稳定性：

| 指标 | 正常关注点 | 异常时的优先检查 |
| --- | --- | --- |
| `success`、`final_position_error` | 目标到达能力 | 目标采样范围、动作尺度、进度奖励 |
| `collision`、`safety_violation_rate`、`min_distance` | 碰撞与距离安全 | `d_safe`、风险代价、障碍物设定、速度限制 |
| `episode_cost`、`mean_risk` | 风险信号是否连续且可区分 | 风险特征、距离/TTC 标定、事件代价 |
| `lambda`（仅 `ldrc_*`） | 是否随长期 episode 代价调整 | `C_safe`、cost 量级和 lambda 学习率 |
| `rms_jerk`、`action_variation` | 执行层平滑性 | `fixed_beta` 或 adaptive 参数；不能只以 jerk 判断安全性 |

常用查看方式：

```bash
RUN_DIR=$(cat outputs/latest_run.txt)
tail -n 50 "$RUN_DIR/train_metrics.csv"
tail -f "$RUN_DIR/progress.csv"
```

`lambda` 长期接近零不一定是错误：先确认 episode 平均风险代价是否确实低于 `C_safe`。反之，`lambda` 持续升高时先检查 risk、violation 和 collision 是否同步偏高，再决定是调整参数还是继续训练。

## 8. 评估与结果归档

单次评估示例：

```bash
conda run -n rl python scripts/evaluate.py \
  --config configs/experiments/random_crossing_link_fixed_penalty1.yaml \
  --method link_fixed \
  --checkpoint outputs/rechecks/link_fixed_penalty1/train/link_fixed/seed_101/link_fixed_seed101_steps100000/actor_step_100000.pt \
  --episodes 20 \
  --seed 3001 \
  --output outputs/dev_eval/link_fixed_penalty1_seed101_eval3001.csv
```

需要典型轨迹时，追加 `--trace-output outputs/dev_eval/traces`，再运行：

```bash
conda run -n rl python scripts/plot_traces.py \
  --trace-dir outputs/dev_eval/traces \
  --output-dir outputs/dev_eval/figures
```

任何用于论文的复核都应满足：相同训练预算、相同场景集合、独立 checkpoint validation、未参与调参的 eval seeds，以及至少 3 个新的 train seeds。统计单位仍是 train seed，不把所有 episode 当成独立训练重复。

## 9. 已完成主结果的复现与论文材料

论文材料使用 held-out 三方法汇总：

```bash
conda run -n rl python scripts/prepare_paper_materials.py
```

它会生成正文 Table 1/2、固定惩罚敏感性图、`ldrc_fixed` 训练诊断图和按 train seed 的附录表，输出到 `outputs/paper/final_materials/`。产物用途和结论边界见 [paper_materials.md](paper_materials.md)。

正式主表的数据源固定为：

```text
outputs/rechecks/heldout_1004_1006/final_3methods/
```

`outputs/formal/summary/`、`outputs/formal/paper_notes/` 与原四方法结果是历史/诊断数据，不能混入正文主表或方法排序。

## 10. 真实低速部署前的离线检查

候选 checkpoint、离线预检命令和现场签核项在 [deployment_preflight.md](deployment_preflight.md)。可重跑：

```bash
conda run -n rl python scripts/deployment_preflight.py
```

该脚本只验证 PyBullet 中的输入维度、checkpoint 加载、动作范围和仿真关节速度命令限幅。真实控制器限速、工作空间围栏、急停/保护停、相机失效安全停止和相机—机器人标定必须现场单独验收。

## 11. 阶段一后续研究优先级（历史参考）

1. 如要验证结果稳健性，优先增加新的 train seeds 和独立 held-out eval seeds。
2. 如要继续约束 SAC，聚焦 upper arm、elbow 的局部碰撞率优势，并同时报告成功率、最小距离、违反率和 jerk 的取舍。
3. 如要继续 adaptive 平滑，先单独解决 actor 退化与 jerk 增大，再做新的多-seed验证；不要把当前结果包装为有效增益。
