# 完整实验顺序（V11 主实验）

更新日期：2026-09-18

当前状态：V8/V9 已停止；V10 的 600k S0 已按 `hard_budget_stop` 归档；V11 代码、持久 replay、预检和恢复检查已完成，正式 S0 尚未启动。本文是命令与执行顺序的唯一入口；方法和阈值以 [`docs/thesis/thesis_outline.md`](../thesis/thesis_outline.md) 为准，实测结果只写入 [progress.md](progress.md)。

## 0. 全局规则

| ID | 子步骤 | 固定规则 | 判定或产物 |
| --- | --- | --- | --- |
| R00 | 环境 | 使用已经启动的 conda `rl` 环境，从仓库根目录运行命令 | 命令不包含 `conda activate` |
| R01 | 协议隔离 | V11 仅写入 `outputs/experiments_9/v11/` | V8/V9/V10 checkpoint、replay、optimizer 不得载入 V11 |
| R02 | Block | 一个完整 Block 为 `100000` environment steps | Block 只是保存和分析单位，不是失败判据 |
| R03 | checkpoint | 每 `25000` steps 保存完整学习状态 | 每个完整 Block 应有 25k、50k、75k、100k 四个 checkpoint |
| R04 | stage 预算 | 每个 stage、正式训练 seed 和消融统一上限 `600000` steps | 由 `stage_total_step` 跨 run 审计；不得给个别 run 追加预算 |
| R05 | 中断恢复 | 从最近完整 checkpoint 以相同 seed、stage 和协议恢复，并使用新 run name | 恢复网络、optimizer、alpha、replay、课程、RNG 和活动 episode |
| R06 | 分档分析 | S0 按 `g`、`eta`、anchor、probe 和 `s0_phase` 分层 | 跨档总体成功率不能用于继续/停止判断 |
| R07 | validation | seeds 固定为 `41001/42002/43003`，确定性动作，每 seed、每场景 100 episodes | 三个 seed 必须分别通过，不能用合并平均掩盖失败 |
| R08 | held-out | seeds 固定为 `91001/92002/93003` | 仅在全部 actor 冻结后访问一次，不得回训或重新选模型 |
| R09 | 契约变化 | observation、reward、几何、课程或 replay 变化必须升级协议 | 新协议从随机初始化开始 |

## 1. V11 预检

| ID | 子步骤 | 命令 | 状态或条件 |
| --- | --- | --- | --- |
| P01 | 定向测试 | `python -m pytest tests/test_analyze_thesis_plateau.py tests/test_thesis_homotopy.py tests/test_evaluate_thesis_homotopy.py -q` | 已通过：39 passed |
| P02 | 全量回归 | `python -m pytest -q` | 已通过：98 passed |
| P03 | 奖励审计 | `python scripts/preflight_thesis_homotopy.py reward --config configs/experiments/thesis_homotopy.yaml --samples 100 --output outputs/experiments_9/v11/preflight/reward` | 复用已通过的 reward 审计 |
| P04 | 目标审计 | `python scripts/preflight_thesis_homotopy.py goal --config configs/experiments/thesis_homotopy.yaml --samples 100 --output outputs/experiments_9/v11/preflight/goal` | 复用已通过的目标审计 |
| P05 | 外部接触审计 | `python scripts/preflight_thesis_homotopy.py contact --config configs/experiments/thesis_homotopy.yaml --samples 100 --output outputs/experiments_9/v11/preflight/contact` | 复用已通过的接触审计 |
| P06 | 自碰撞审计 | `python scripts/preflight_thesis_homotopy.py self-contact --config configs/experiments/thesis_homotopy.yaml --samples 500 --output outputs/experiments_9/v11/preflight/self_contact` | 复用已通过的 mesh/contact 审计 |
| P07 | 场景可行性 | `python scripts/preflight_thesis_homotopy.py feasibility --config configs/experiments/thesis_homotopy.yaml --samples 100 --output outputs/experiments_9/v11/preflight/feasibility` | 复用已通过的可行性审计 |
| P08 | 真实短轨迹 | `python scripts/train_thesis_homotopy.py --config configs/experiments/thesis_homotopy.yaml --stage s0 --seed 91101 --steps 300 --run-name preflight_v11_persistent_replay_300 --validation-mode` | 122 维、22 奖励原子量、持久 replay 可写 |
| P09 | 保存恢复 | 使用上述 300-step checkpoint 恢复 50 steps | replay/stage step 300→350；池库存连续 |
| P10 | 旧协议拒绝 | 尝试载入 V10 `step_0100000.pt` | 必须因 protocol 不匹配拒绝 |

## 2. S0：无障碍完整位姿与自碰撞裕量

### 2.1 seed 11001 训练循环（V11）

| ID | 子步骤 | 命令或动作 | 继续、成功或停止条件 |
| --- | --- | --- | --- |
| S0-01 | Block1 从零训练 | `python scripts/train_thesis_homotopy.py --config configs/experiments/thesis_homotopy.yaml --stage s0 --seed 11001 --run-name s0_v11_seed11001_block1` | 禁止 `--resume`；训练 100k |
| S0-02 | 单 Block 审计 | 检查 `summary.json`、五类 CSV 和四个 checkpoint | 按当前 `g/eta/s0_phase` 报告 success、self-collision、timeout、anchor、probe、Q/alpha；单个 Block 不能判平台 |
| S0-03 | 下一 Block | `python scripts/train_thesis_homotopy.py --config configs/experiments/thesis_homotopy.yaml --stage s0 --seed 11001 --resume <LAST_COMPLETE_CHECKPOINT> --run-name s0_v11_seed11001_block<N>` | 未成功、未平台、未数值失败且 `stage_total_step<600000` 时执行 |
| S0-04 | 两 Block 停止分析 | `python scripts/analyze_thesis_plateau.py --config configs/experiments/thesis_homotopy.yaml --blocks <PREVIOUS_BLOCK_DIR> <CURRENT_BLOCK_DIR> --output outputs/experiments_9/v11/analysis/seed11001_blocks<N-1>_<N>.json` | 只接受两个相邻、各 100k 的完整 S0 run |
| S0-05 | 中断块处理 | 从最近 checkpoint 补齐剩余 steps；分段日志按 `stage_total_step/global_step` 去重合并 | 分段目录不能直接传给 S0-04；平台判断使用后续两个完整 run，或先生成经审计的完整合并数据 |

### 2.2 S0 自动课程子步骤

| ID | 相位 | 自动推进条件 | 必查状态 |
| --- | --- | --- | --- |
| S0-C01 | position 范围课程 | `g` 按 16 档从 0.03 到 1；每档至少 5k 合格 steps且最近 100 episode position success≥0.80 | 当前档计数和窗口不跨档继承；`lambda_self=0` |
| S0-C02 | position 巩固 | `g=1` 后 `K_position>=25000` | 完成后才进入 pose |
| S0-C03 | pose 课程 | `eta` 按 17 档从 0 到 1；每档至少 5k 非锚点 steps且当前档最近 100 episode success≥0.80 | `lambda_self=0`；真实 self-contact 仍硬终止并扣 10 |
| S0-C04 | 位置锚点 | 根据 normal/intermediate/recovery 使用预注册 anchor 概率和 replay 比例 | 最近 100 anchor position success≥0.95 |
| S0-C05 | 固定 probe | 每档使用 50 个私有固定位置目标 | success≥0.95、collision=0、joint-limit=0；失败后至少再训练 5k 当前档 steps |
| S0-C06 | pose 巩固 | `eta=1` 后 `K_pose>=25000` | 位姿窗口≥0.80、anchor≥0.95、最终 probe 通过 |
| S0-C07 | self-safety ramp | C06 持续满足时累计 50k 合格 steps | `lambda_self:0→1`；任务保持下降时冻结 |
| S0-C08 | 满权重巩固 | `lambda_self=1` 后 `K_self_full>=25000` | `s0_phase=complete`、`s0_goal_gate_eligible=true` |

### 2.3 S0 停止判据

| ID | 判据 | 精确定义 | 动作 |
| --- | --- | --- | --- |
| S0-D01 | 平台 | 两个相邻完整 Block endpoint 为同一 `eta`，两块内均未换档；每块≥100 当前档 episode、≥50 anchor且有当前档 probe | 再检查 D02 的所有改善量 |
| S0-D02 | 实质改善 | safe success `+0.02`、self-collision `-0.01`、timeout `-0.02`、anchor success `+0.01`、probe success `+0.02`、probe collision `-0.01`、probe timeout `-0.02` | 任一达到即继续；全部未达到才 `plateau_stop` |
| S0-D03 | 数值异常 | 任一更新诊断 NaN/Inf；或 Q abs p95>1000 且较前块>5倍，同时 critic loss p95>1000 且较前块>5倍 | `numerical_failure` |
| S0-D04 | 硬预算 | `stage_total_step=600000` 且 validation 未全通过 | `hard_budget_stop` |
| S0-D05 | Gate 候选 | `s0_goal_gate_eligible=true` | 先运行三组 validation；该布尔值本身不代表成功 |

### 2.4 S0 validation

对同一 `<S0_GATE_ELIGIBLE_CKPT>` 依次执行：

```bash
python scripts/evaluate_thesis_homotopy.py --config configs/experiments/thesis_homotopy.yaml --checkpoint <S0_GATE_ELIGIBLE_CKPT> --scenes none --episodes 100 --seed 41001 --output outputs/experiments_9/v11/validation/<S0_TAG>_seed41001
python scripts/evaluate_thesis_homotopy.py --config configs/experiments/thesis_homotopy.yaml --checkpoint <S0_GATE_ELIGIBLE_CKPT> --scenes none --episodes 100 --seed 42002 --output outputs/experiments_9/v11/validation/<S0_TAG>_seed42002
python scripts/evaluate_thesis_homotopy.py --config configs/experiments/thesis_homotopy.yaml --checkpoint <S0_GATE_ELIGIBLE_CKPT> --scenes none --episodes 100 --seed 43003 --output outputs/experiments_9/v11/validation/<S0_TAG>_seed43003
```

每个 summary 的 `none.gate_pass` 都必须为 true，即各 seed 分别满足 success≥0.95、collision=0、joint-limit=0、timeout≤0.05。任一失败且停止分析 JSON 中 `plateau_detected=false`、`hard_budget_exhausted=false` 时，从当前 Block 最后 checkpoint 继续；否则按对应原因停止。

## 3. S1：静态障碍物

| ID | 子步骤 | 命令或动作 | 条件 |
| --- | --- | --- | --- |
| S1-01 | Block1 | `python scripts/train_thesis_homotopy.py --config configs/experiments/thesis_homotopy.yaml --stage s1 --seed 11001 --resume <SELECTED_S0_CHECKPOINT> --run-name s1_v10_seed11001_block1` | S0 checkpoint 必须通过三 seed Gate |
| S1-02 | 宽容课程 | none/static episode 和 replay 比例为 25/75；`xi_static` 从 0.02 开始 | 宽容期任务球 contact 可继续，但 self/environment/joint-limit 仍硬失败 |
| S1-03 | safety ramp | static 最近 100 episode task reach≥0.90 时累计 50k static steps | `xi_static:0.02→1`，低于门槛时冻结 |
| S1-04 | replay 严格化 | `xi_static` 首次到 1 时执行一次 strictify | 删除首次 obstacle contact 后不可达 transition并记录哈希 |
| S1-05 | 严格巩固 | strict static steps≥25000 | checkpoint 才有 Gate 资格 |
| S1-06 | 后续 Block | 同 S1-01，改用 `--resume <LAST_COMPLETE_CHECKPOINT>` 和 `block<N>` | Gate 未通过、无数值异常且累计不足 600k 时继续 |
| S1-07 | validation | 对同一候选分别用三个 validation seed，`--scenes none static` | none 保持 S0 Gate；static success≥0.90、collision≤0.03、joint-limit=0、timeout≤0.10 |
| S1-08 | 阶段结论 | 三个 seed 的两个场景均 `gate_pass=true` | 选定 checkpoint 后进入 S2；600k 未通过则失败停止 |

S1 validation 命令模板：

```bash
python scripts/evaluate_thesis_homotopy.py --config configs/experiments/thesis_homotopy.yaml --checkpoint <S1_GATE_ELIGIBLE_CKPT> --scenes none static --episodes 100 --seed <41001_OR_42002_OR_43003> --output outputs/experiments_9/v11/validation/<S1_TAG>_seed<VALIDATION_SEED>
```

## 4. S2：动态障碍物

| ID | 子步骤 | 命令或动作 | 条件 |
| --- | --- | --- | --- |
| S2-01 | Block1 | `python scripts/train_thesis_homotopy.py --config configs/experiments/thesis_homotopy.yaml --stage s2 --seed 11001 --resume <SELECTED_S1_CHECKPOINT> --run-name s2_v10_seed11001_block1` | S1 checkpoint 必须通过三 seed Gate |
| S2-02 | 宽容课程 | none/static/dynamic episode 和 replay 比例 20/30/50；static 固定严格 | `xi_dynamic` 从 0.02 开始 |
| S2-03 | safety ramp | dynamic 最近 100 episode task reach≥0.80 时累计 50k dynamic steps | `xi_dynamic:0.02→1`，低于门槛时冻结 |
| S2-04 | replay 严格化 | `xi_dynamic` 首次到 1 时执行一次 strictify | 记录转换数量和 replay 哈希 |
| S2-05 | 严格巩固 | strict dynamic steps≥25000 | checkpoint 才有 Gate 资格 |
| S2-06 | 后续 Block | 同 S2-01，改用 `--resume <LAST_COMPLETE_CHECKPOINT>` 和 `block<N>` | Gate 未通过、无数值异常且累计不足 600k 时继续 |
| S2-07 | validation | 三个 validation seed，`--scenes none static dynamic` | 三场景分别满足各自 Gate |
| S2-08 | 阶段结论 | 三个 seed 的三场景均 `gate_pass=true` | 冻结该训练 seed 的最终 actor；600k 未通过则失败停止 |

S2 validation 命令模板：

```bash
python scripts/evaluate_thesis_homotopy.py --config configs/experiments/thesis_homotopy.yaml --checkpoint <S2_GATE_ELIGIBLE_CKPT> --scenes none static dynamic --episodes 100 --seed <41001_OR_42002_OR_43003> --output outputs/experiments_9/v11/validation/<S2_TAG>_seed<VALIDATION_SEED>
```

## 5. 训练 seeds 22002 与 33003

| ID | 子步骤 | 命令或动作 | 要求 |
| --- | --- | --- | --- |
| M01 | seed 22002 S0 | `python scripts/train_thesis_homotopy.py --config configs/experiments/thesis_homotopy.yaml --stage s0 --seed 22002 --run-name s0_v11_seed22002_block1` | 从零重复第 2 节 |
| M02 | seed 22002 S1/S2 | 只从该 seed 自己选中的上一阶段 checkpoint 继续 | 重复第 3、4 节 |
| M03 | seed 33003 S0 | `python scripts/train_thesis_homotopy.py --config configs/experiments/thesis_homotopy.yaml --stage s0 --seed 33003 --run-name s0_v11_seed33003_block1` | 从零重复第 2 节 |
| M04 | seed 33003 S1/S2 | 只从该 seed 自己选中的上一阶段 checkpoint 继续 | 重复第 3、4 节 |
| M05 | 跨训练 seed | 汇总三个训练 seed 的独立 validation | 三者都必须通过 S2；不得迁移 checkpoint 或追加个别预算 |

## 6. 冻结与 held-out

| ID | 子步骤 | 命令或动作 | 限制 |
| --- | --- | --- | --- |
| F01 | 冻结清单 | 记录三个训练 seed 的完整 checkpoint、actor、配置、代码、URDF 和输入/动作处理 SHA-256 | 完成后禁止换模型 |
| F02 | held-out 91001 | evaluator 使用 `--scenes none static dynamic --episodes 100 --seed 91001`，输出到 `outputs/experiments_9/v11/held_out/` | 只报告 |
| F03 | held-out 92002 | 同 F02，seed 改为 92002 | 只报告 |
| F04 | held-out 93003 | 同 F02，seed 改为 93003 | 只报告 |
| F05 | 汇总 | 按训练 seed、held-out seed、场景分别报告，再做跨 seed 统计 | 不用 curriculum 混合平均替代分层结果 |

## 7. 消融与独立安全层

| ID | 子步骤 | 当前要求 | 状态 |
| --- | --- | --- | --- |
| A01 | 冻结消融矩阵 | 至少包含 V8-style、observation-only、V9-style、V10 FIFO 和 V11 persistent replay；每项只改变一个因素 | 尚未建立正式配置 |
| A02 | 公平预算 | 使用相同训练/validation seed、目标分布、每 stage 600k 上限和 Gate | 必须预注册 |
| A03 | 独立初始化 | 每个消融从零训练，不复用 nominal replay/checkpoint | 必须执行 |
| A04 | 主要指标 | 完整位姿成功、自碰撞、timeout、最小 self-clearance、位置保持、样本效率 | 禁止只报告总体成功率 |
| L01 | 冻结 actor | 第二阶段不训练或微调 SAC | 等 V11 第一阶段完成 |
| L02 | 安全层对照 | baseline、完整 predictor+QP+monitor 和组件消融使用同一 manifest | 正式驱动脚本尚缺 |
| L03 | 评价 | validation 选配置；冻结后一次性 held-out | 报告安全收益、任务保持、干预率、失败率和周期耗时 |

## 8. 每个正式步骤的记录清单

| ID | 必须记录 | 内容 |
| --- | --- | --- |
| D01 | 配置与版本 | protocol、配置、代码 commit/dirty 状态、依赖、硬件、URDF SHA-256 |
| D02 | 随机性 | 根 seed、派生 RNG streams、training/validation/held-out split |
| D03 | 学习状态 | global/stage/stage-total steps、updates、replay、alpha、Q/target、梯度、非有限值 |
| D04 | 课程 | `g/eta/lambda_self/xi`、窗口、probe、eligible/strict/full-scale 计数 |
| D05 | 任务与安全 | success、timeout、joint-limit、三类 collision、clearance、risk/TTC |
| D06 | 运动质量 | 位置/姿态误差，command/measured velocity、acceleration、jerk |
| D07 | 决策 | `continue`、`run_validation`、成功或具体停止原因 |

## 9. V8/V9/V10 历史摘要（只读）

| 协议 | 已执行范围 | 最终结果 | 后续状态 |
| --- | --- | --- | --- |
| V8 | S0 seed 11001，三个 100k Block | 到 `eta=0.65`，`K_pose=0`，probe 0.94 success并有 collision/timeout | 按当时三 Block 规则失败；不再执行 |
| V9 | S0 seed 11001，Block1 经中断恢复后共 100k，Block2 再训练 100k | 到 `eta=0.4` recovery；SAC 稳定；从该状态到 Gate 理论下界约 116k，超过原剩余 100k | 放弃 Block3；不进入 S1/S2；checkpoint 不得用于 V10 |
| V10 | S0 seed 11001，六个 100k Block | 到 `eta=0.75` 后退化；中间 eta replay 被 FIFO 覆盖；最终 `hard_budget_stop` | 不进入 S1/S2；checkpoint 不得用于 V11 |

历史数值和原始目录见 [progress.md](progress.md)。任何带 `s*_v8`、`s*_v9` 或 `outputs/experiments_9_16`、`outputs/experiments_9_17` 的命令都只属于历史记录，不是当前执行入口。
