# S1 合格绕行示范回放

2026-10-25：第一轮改进只增加独立示范池和约 10% 的 minibatch 示范配额。
不加行为克隆、不增加输入或子目标、不加部署安全过滤器，不改奖励、gamma、控制频率、
时间上限、速度上限或精度升档条件。正式基础配置仍是 `paper_full.yaml`，功能通过 CLI 显式启用。

后续可选耦合奖励实验见 [S1 目标—障碍耦合](s1_goal_obstacle_coupling.md)。
该实验使用独立配置和显式重标的 bank；本文原奖励示范试验仍保持不变，不直接混用两种奖励契约。

## 数据如何生成与筛选

入口：`scripts/calibration/build_s1_expert_bank.py`。

1. 从新训练种子独立采样场景。只生成单静态障碍，提议概率为原有三档的条件比例
   far/near/blocking=18.75%/18.75%/62.5%。不读取此前 27 个诊断案例、固定评估场景或快照。
   场景种子设为高位整数，与项目当前低编号固定评估种子分离；bank 保存具体种子。
2. 使用对应 L0-L13 档位的目标范围、位置/姿态容差、orientation_scale 簿记坐标和连续保持条件。
   L0 的 orientation_scale 为 0，不代表没有姿态目标；显式目标范围仍覆盖该档姿态范围。
3. 从同一初始快照执行冻结的当前 S1 Actor、零动作等待、见证路径跟踪候选。
   候选默认跟踪增益为 3、5、8，始终经过原 `env.step()`、原精度缩放和 0.7 rad/s 动作尺度。
   增益是离线数据生成控制器参数，不是 SAC 超参数，也不在部署时使用。
4. 候选从起点和每一个真实物理子步查询原始网格：外部距离 >=2.5 cm，非相邻自碰撞对距离 >=5 mm。
   任一候选不满足裕度即拒绝；无实际碰撞不等于满足裕度。按 240 Hz 离散检查，不声称连续时间安全证明。
5. 必须在原 500 控制步内严格位姿成功并完成连续 5 步保持；碰撞、越界、超时和部分轨迹不入库。
6. 计算原奖励下的折扣回报。对 lambda_self=0.2 和 1 两个端点，都要求候选优于
   `max(参考 Actor 回报, 等待回报)` 至少 2 分。自碰撞惩罚对权重线性，因此也覆盖区间内的固定轨迹重算。
   不把这项工程筛选阈值解释为理论最优值，也不在奖励里添加“示范额外奖励”。
7. 多候选合格时只保留一条：选择 lambda_self=0.2 下回报最高者，避免相反绕行标签混杂。
   记录真实归一化动作，不把关节角或精度缩放后的关节速度误当作 Actor 动作。

合格数据是经过拒绝筛选的分布，不保证保留原障碍形状、尺寸、三档比例。
在线采集的 none/far/near/blocking=20%/15%/15%/50% 不变；增加示范后，整个 minibatch
不再保证这些比例。日志分别记录在线各池和示范配额。

## 数据契约与回放

- Bank 绑定机器人/碰撞网格、预测器文件校验和、物理参数、观测参数、奖励、gamma 和精度档位。
  不匹配时拒绝加载。不要在生成过程中修改这些文件。
- 只加载自己生成、可信的本地 `bank.pt`；它使用 PyTorch 完整对象序列化，不应加载来源不明的文件。
- 每个精度档位独立保存完整成功 episode。加载时检查观测与动作维度、原始障碍集合、连续链、
  终止标志、保持条件、物理子步检查数量、回报及自碰撞惩罚的重算一致性。
- 只在 S1、当前精度档位采样；不会用 L0 示范替代 L1-L13 数据。
- 当前档没有示范时，明确打印 `no Lx bank; using online-only replay at this level`，
  该档回退原有在线回放，不暂停或放宽精度课程，不静默混用旧档示范。
- 默认 minibatch 1024 中约 102 条示范、922 条原有经验；剩余经验沿用原来的成功/近期回放比例。
  数据不足时示范配额缩小，剩余配额给原有经验。不会让示范绕过原在线 warm-up 条件。
- 示范不进入在线成功池，不增加训练环境步数、episode 数或升档成功窗口。
- 当前安全权重由原 Replay 逻辑重算奖励；不缓存未经重算的训练 reward，也没有 BC loss。
- Bank、筛选元数据、独立采样 RNG 随完整检查点保存。旧版 storage_version=10 检查点仍可加载，
  新字段是可选增量，不改变原始 transition 格式。未启用时维持原采样行为。
- 完整续训默认继承已有示范配额与 RNG；显式指定同一 bank 不重置示范 RNG。
  显式指定不同 bank 会替换独立示范池并记录新校验和，不替换在线 Replay、网络或优化器。

新增 `updates.csv` / `progress.csv` 字段：

- `sample_s1_expert`：最近训练 minibatch 的实际示范条数。
- `stored_s1_expert`：所有档位独立示范池保存的 transition 总数。
- `replay_s1_expert_fraction`：配置的示范比例；缺档时实际条数仍可能为 0，以实际条数为准。

`manifest.json.s1_expert_replay` 记录 bank 来源、SHA256、档位、episode 数、比例与缺档策略。

## 第一轮正式试验命令

在项目根目录、已激活的 conda `rl` 环境中执行。先生成 128 个合格 L0 episode，
再从原 `step_1250000.pt` 完整继承，试训额外 200000 步。先比较固定评估，不直接投入剩余 4.75M。
这是一轮建议的试验预算，不预先保证成功率提升。

### 生成独立示范库

```bash
python scripts/calibration/build_s1_expert_bank.py \
  --config configs/experiments/paper_full.yaml \
  --reference-actor outputs/full/full_seed11001_s1_stratified_success50_6m/checkpoints/actor_step_1250000.pt \
  --output-dir outputs/s1_experts/success50_l0_128_seed20261026 \
  --levels 0 --episodes-per-level 128 --max-scenes-per-level 512 \
  --seed 20261026 --workers 6
```

生成预算耗尽时保存拒绝诊断，但不输出可训练 bank，也不会降低安全或回报门槛。
输出目录必须是新目录；重复执行需要换目录，不能覆盖旧结果。
`generation_report.json` 保存候选、拒绝原因、参考回报、安全间距、检查次数、最终分档数量。

### 完整继承，增加示范回放，试训 200k

```bash
python scripts/core/train_thesis_homotopy.py \
  --config configs/experiments/paper_full.yaml --stage s1 \
  --resume outputs/full/full_seed11001_s1_stratified_success50_6m/checkpoints/step_1250000.pt \
  --s1-expert-bank outputs/s1_experts/success50_l0_128_seed20261026/bank.pt \
  --s1-expert-fraction 0.10 \
  --steps 200000 --num-envs 8 \
  --run-name full_seed11001_s1_success50_expert10_pilot200k_from1250k
```

继承 Actor、Critic、优化器、温度、原 Replay、课程、训练计数和原 RNG。
只新增独立示范池及采样配额；这不等于“所有训练设置完全不变”。`--steps` 是本次额外预算。
若完整跑满，本段最终文件为 `step_0200000.pt` / `actor_step_0200000.pt`，累计 S1 步数为 1450000；
文件名使用本段步数，不要误找 `step_1450000.pt`。
不要从下述 80 步 validation 产物开始正式训练；它只用于实现验收。

公平对照应从同一个原检查点启动、使用相同额外步数，去掉两个 `--s1-expert-*` 参数，
并更换 run-name。比较独立严格位姿成功率、碰撞率、无障碍保持率以及停滞/振荡失败，
不能用示范训练集的成功率替代策略评估。

## 实现验收

已完成的小库只是管线冒烟测试，不是正式训练数据规模，也不是策略性能结果。
最终小库见 `outputs/diagnostics/s1_expert_smoke_20261025_v2/`。
早期无 `_v2` 的开发产物只作记录，不使用；最终契约额外检查了精度档的 orientation_scale 坐标。

最终小库从 5 个已处理的新场景保留 4 个 episode、256 条 transition，覆盖 near=2、blocking=2。
这里只验证管线，不据此估计一般接受率。所有保留轨迹都完成了原始 500 步内的严格位姿到达，
并通过初始状态与每个物理子步的真实网格裕度检查。

从原 `step_1250000.pt` 完整继承，另行完成 80 环境步的实现验证（8 环境、正常 batch=1024）：
日志实际显示每批 102 条示范、922 条原有数据，损失有限，累计 global_step=1250080。
验证目录为 `outputs/diagnostics/s1_expert_integration_20261025/resume1250k_validation_v2/`。
重新加载该完整检查点，不再提供 bank 路径，仍能继承 4 个示范 episode、独立 RNG 和 10% 配额，
并采样 102/1024 条示范。旧原始检查点没有被覆盖，没有长时间正式训练在后台运行。

验证范围包括：安全和回报拒绝规则、完整成功链、同档采样、约 10% 配额、缺档回退、
S0/S2/禁用时采样不变、旧检查点加载、新检查点 RNG 继承，以及只读物理回调不改变轨迹。
正式基础 YAML、奖励函数和 SAC 损失没有因本功能更改。
最终完整测试共 234 项通过（约 91 秒），编译检查和变更空白检查通过。

```bash
PYTHONPATH=src python -m pytest -q tests/core tests/infrastructure
```
