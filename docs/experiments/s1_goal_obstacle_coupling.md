# S1 目标—障碍耦合势函数实验

2026-10-25：增加一个可选的联合过程信号，保留原基础奖励。
这是借鉴目标与障碍联合评价的思路，不是 COR/PAC 公式的逐字复现，不保证策略性能提升。
基础配置 `paper_full.yaml` 默认不启用，实验配置为 `paper_full_s1_coupling.yaml`。

## 定义

位置误差 ep（m）、姿态角误差 eR（rad）、当前机械臂到外部障碍的最小编码胶囊距离 d（m），
有障碍指示量 I=1，无障碍 I=0。使用原始几何距离，不使用预测风险替代距离；它不等同于真实网格间距。

```text
P(s) = 1 / (1 + ep / 0.30 + eR / 1.0)
x(s) = clip((d - 0.025) / (0.10 - 0.025), 0, 1)
S(s) = 3*x^2 - 2*x^3
Phi(s) = beta * I * P(s) * S(s)          beta = 1
F(s,s') = gamma * Phi(s') - Phi(s)       gamma = 0.99
新奖励 = 原奖励 + F
```

- 仅静态障碍 S1 启用；S0/S2/S3 不启用，无障碍 episode 全程新增项为 0。
- 2.5 cm、10 cm 直接读取原风险配置，避免两套阈值漂移。
- 位置和姿态均参与 P。L0 的 orientation_scale=0 不会删掉真实姿态误差。
  0.30 m、1 rad 是本实验固定尺度，不随精度档位变化。
- 离开危险区，即使短时远离目标，也可能得到正过程信号。例如位置误差 0.30→0.35 m、
  姿态误差固定 0.4 rad、间距 0.03→0.05 m，新增单步奖励约 +0.09469136。
- 不是所有绕行步骤都为正：超过 10 cm 后安全项饱和，单纯远离目标仍会降低势能。
- Phi 范围 [0,1]，F 范围 [-1,0.99]。静止时 F=-0.01*Phi，不提供持续站立的正奖励。
- 所有 trainer done（成功、各类碰撞、关节越界、超时）均将下一状态 Phi 强制设为 0。
  最后一步 F=-Phi(s)，不能把整个 F 清零。与 Replay 对超时不 bootstrap 的现有语义一致。

## 能做什么，不能做什么

完整 episode 的折扣新增回报严格为 `sum(gamma^t * F_t) = -Phi(s0)`。
同一起点的成功、绕行、等待、碰撞、超时轨迹都平移同一个常数。
因此本项不改变完整轨迹的回报差，不能修复基础奖励已有的排序反例。
本实验检验更密集的联合信号能否帮助有限样本和函数逼近下的学习，不是增加绕行总收益。
恒等式不保证 SAC 的学习速度或成功率。继承旧 Critic 时有 Q 偏移适应期，不是无扰切换。

这不是安全过滤器、可行性证明或碰撞豁免。原碰撞终止、安全成本、网格检查保持不变。
原 `success_bonus=50`、20 Hz、500 控制步、动作尺度、gamma、网络及输入维度、
场景比例 20%/15%/15%/50%、精度课程和仅有障碍升档条件均不改变。
原 joint-bottleneck 奖励及其终止处理也不修改。

## 环境、Replay 与检查点

环境单步和 Replay 批量重算使用同一个耦合模块。
raw 29D 只记录下一状态外部距离，不能冒充前一状态距离。
重算从已有 current/next 原始 37D 障碍集合的前四列分别取最小值；空集合代表无障碍。
旧 storage_version=10 已保存这两组数据，不新增观测、网络参数或存储列。
缺少有效原始集合时拒绝启用，不将损坏数据静默视为无障碍。

旧奖励完整检查点启用/改动耦合项，必须显式传 `--allow-coupling-reward-change`。
只允许同阶段 S1 的此项变更，不能借此同时更改基础奖励或 gamma。
继承 Actor、Critic、目标网络、优化器、温度、在线 Replay、课程及 RNG；历史经验在采样时重算。
Critic 不重置，也没有做精确 Q 变换，可能短期波动；迁移写入 manifest 和 counters。
新配置检查点同配置再次续训不需要该标志。

新增 `transitions.csv` 字段为 `goal_obstacle_coupling_reward`、
`goal_obstacle_potential`、`next_goal_obstacle_potential`；未启用时为 0。
评估逐步 info 同样保留这三个字段。日志不能替代独立固定场景评估。

## 示范库显式迁移

旧 bank 绑定奖励契约，不能直接混用。仅该 PBRS 变更时，可以复制重标：
物理状态、动作、子步安全检查不变，同起点示范/参考 Actor/等待回报平移相同常数，优势不变。
迁移先完整验证旧库，逐条验算折扣恒等式，再完整验证目标契约；不降低筛选门槛。
物理、基础奖励、阈值有其他变化时拒绝。旧候选诊断可能含部分轨迹，保存在
`source_reward_candidate_diagnostics`，明确仍为源奖励记录，不冒充新奖励完整候选。

项目根目录、已激活 conda `rl` 环境：

```bash
python scripts/calibration/relabel_s1_coupling_bank.py \
  --source outputs/s1_experts/success50_l0_128_seed20261026/bank.pt \
  --config configs/experiments/paper_full_s1_coupling.yaml \
  --output outputs/s1_experts/success50_l0_128_seed20261026_coupling_v1/bank.pt
```

上述迁移已执行，生成 128 episode、11311 transition 的副本，原库未覆盖。
最大恒等式残差约 2.4e-15，见同目录 `bank.relabel.json`。
重复执行须换输出目录，脚本拒绝覆盖。仅加载自己生成的可信 PyTorch 文件。

## 当前单变量试验：仅耦合奖励，不使用示范

此前耦合奖励与 10% 示范同时启用，无法区分两个因素的效果；用户要求终止该训练，
先去掉示范，仅测试耦合奖励。保留示范功能和旧库文件，不删除历史数据。
从原始 `step_1250000.pt` 重新分支，而不是从已经受示范影响的混合试验检查点续训。
已核对该原始检查点不含耦合项及示范池。Actor、Critic、优化器、在线 Replay、课程和 RNG 继承。
不提供 bank，并显式指定示范比例为 0；其余配置不变。先额外试训 200k，下面命令未自动启动。

```bash
python scripts/core/train_thesis_homotopy.py \
  --config configs/experiments/paper_full_s1_coupling.yaml --stage s1 \
  --resume outputs/full/full_seed11001_s1_stratified_success50_6m/checkpoints/step_1250000.pt \
  --allow-coupling-reward-change \
  --s1-expert-fraction 0 \
  --steps 200000 --num-envs 8 \
  --run-name full_seed11001_s1_success50_coupling1_noexpert_pilot200k_from1250k
```

`--steps` 为本段额外预算；跑满本段文件为 `step_0200000.pt`，累计 S1 为 1450000。
无耦合控制组使用同一原始检查点和 `paper_full.yaml`，同样示范比例为 0、不提供 bank，
去掉迁移标志并更换 run-name，其他参数一致。比较相同训练预算，不能把不同起点当作严格对照。
不要从已有示范的检查点只设置比例 0，误以为消除了过去训练的影响。
启动后的 manifest 应记录示范 disabled、fraction=0、transitions=0；
训练日志 `sample_s1_expert` 与 `stored_s1_expert` 都应为 0。
如果将来另做示范实验，须使用匹配奖励契约的 bank，不能静默混用旧内嵌库。
评估严格位姿到达率、碰撞率、无障碍保持率、停滞/振荡、到达用时和最终精度；
不同奖励下的总训练回报不能直接作为策略改进证据。

## 验证

- `tests/core` 与 `tests/infrastructure` 共 256 项通过；覆盖公式、终止、各长度折扣恒等式、
  无障碍为零、单步/批量一致、禁用基线不变、库迁移、续训保护和真实环境接线。
- 对此前 27 个案例的 114 条真实物理轨迹重评分，含成功与各类失败。
  只恢复初始场景查询初始势能，不重新执行动作、不更新策略。
  最大恒等式残差 5.4e-15，最大同起点回报差变化 7.2e-15，确认原排序不变。
  报告为 `outputs/diagnostics/s1_coupling_20261025/recorded_trace_audit.json`。
  不是新安全验收，也不是新策略成功率评估。
- 独立 CPU、8 环境两段各 16 步的短续训验证位于 `outputs/diagnostics/s1_coupling_20261025/`。
  第一段验证显式奖励迁移；第二段不再提供 bank 路径或迁移标志，仍正确继承 128 条示范。
  第二段正常 batch=1024，完成 4 次优化器更新，每批含 102 条示范，Actor/Critic 损失均有限。
  这些短训练产物不得用于正式训练；未启动长期正式实验，未停止或修改正在运行的用户训练。

重评分复现命令（输出须不存在）：

```bash
python scripts/calibration/audit_s1_coupling.py \
  --source outputs/diagnostics/s1_reward_returns_20261025/physical_audit.json \
           outputs/diagnostics/s1_reward_returns_20261025/tracker_sensitivity.json \
  --states outputs/diagnostics/s1_failure_cases_20261025/latest1250k_states.pt \
  --output outputs/diagnostics/s1_coupling_20261025/recorded_trace_audit_repeat.json
```
