# S1：静态障碍物下的 6D 位姿到达

> 依据当前代码及唯一主配置 `configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml`；这是设计与实现说明，不代表已有 S1 实验通过验收。
>
> 训练入口：`scripts/core/train_thesis_homotopy.py`；环境：`src/rl_risk_sac/envs/thesis_homotopy_env.py`；课程与回放：`src/rl_risk_sac/algorithms/homotopy_curriculum.py`、`homotopy_replay.py`。

## 阶段边界与启动

S1 在通过正式完成 gate 的 S0 完整 checkpoint 上继续训练同一个 Hybrid Keypoint + Jacobian + Auto-PCR SAC agent，加入单个静止的外部球形障碍物。每个 episode 的场景概率为 none 25%、static 75%、dynamic 0%；none 用来保留无障碍 6D 到达。S1 不引入动态障碍物，也不改变网络输入输出、奖励定义或 SAC 超参数。

```bash
python scripts/core/train_thesis_homotopy.py \
  --config configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml \
  --stage s1 \
  --resume /absolute/path/to/completed-s0/checkpoints/step_XXXXXXX.pt \
  --seed 11001 \
  --num-envs 1 \
  --run-name s1_seed11001
```

须使用完整的 `step_*.pt`，不能用 actor-only checkpoint。训练器核验来源阶段、protocol、seed/RNG stream 和 S0 最终的 P8 frozen probe、P7 retention、自安全完成 gate；Actor、双 Critic、target Critic、温度 α、优化器及 replay 一起恢复。任务课程和 self-safety 状态继承；static 同伦状态在 S1 初始化。S1/S2 目前只支持单环境采样，配置中的 `train.num_envs: 8` 仅用于 S0，显式指定 S1 多环境会报错。示例路径需换成真实完成 S0 gate 的 checkpoint。

## 固定合同

| 项目 | 当前值 |
|---|---|
| 物理/控制步长 | 1/240 s、0.05 s（每控制步 12 个物理子步） |
| 时限/成功保持 | 最多 500 控制步；位姿在当前容差内连续保持 5 步 |
| 动作 | 6 维归一化关节速度；命令 `0.7 * action` rad/s |
| 精度缩放 | 根据上一步位置、姿态误差，控制命令再乘 [0.1, 1] 系数 |
| 观测 | 162 维 Hybrid Keypoint + Jacobian + 显式位姿误差 |
| SAC | batch 1024、每 transition 0.25 次更新、每 25k 保存 |
| 网络 | 两层 256 宽 ReLU 的 Gaussian actor 和双 Q；`gamma=0.99`、`tau=0.005` |
| 学习率 | actor/critic 各 1e-4，α 3e-4；actor 每 2 次 update 更新一次 |
| Auto-PCR | CHAIN-PCR 开启，自适应系数；目标 penalty/SAC 比 0.003，上限比例 0.1 |

162 维观测由关节位置/速度 12、三关键点位置误差 9、关键点位置 Jacobian 54、显式位置/旋转误差及其模长 8、末端线/角速度 6、外部障碍物信息 49、自碰撞几何信息 24 组成。目标尺度、位置阈值、姿态阈值和剩余时间不再作为网络输入。外部 49 维具体为六连杆相对向量 18、障碍物位置/速度 6、每连杆距离/TTC/接近速度/风险 24、存在标志 1。输入分量经过各自的界限或比例归一化/裁剪，不能当作原始物理量直接解释；none 场景使用安全占位值。Jacobian 是三个关键点各自 3x6 的位置 Jacobian，输入不是逆 Jacobian 控制器。动作最终仍由 SAC 直接给出六关节速度，`self_safety_projection.enabled: false`。

当前任务空间目标范围在各精度档一致：位置距离 0.03–0.5 m、旋转距离 0.03–π rad。距离上限已经收缩，为静态避障绕行保留外围工作空间余量。S1 继承 S0 已通过 gate 的课程档位和对应容差，不因加入障碍物重新回到宽松档；最高档 L6 容差为位置 0.01 m、姿态 0.1 rad。L4–L6 的姿态 scale 均为 1.0，replay 用 level index 区分位置精度档。成功计数取决于目标位姿与 hold，不等同于某个奖励分数。

## 静态障碍物和安全课程

每个 static reset 最多尝试 100 个候选。障碍物为半径 0.075 m 的无质量球体，球心从 `x=[0.22,0.72]`、`y=[-0.48,0.48]`、`z=[0.18,0.62]` m 均匀采样，整个 episode 速度为零；候选距目标位置小于 0.15 m、初始机器人胶囊距离小于 `d_safe=0.12 m` 或 PyBullet 已接触时会被拒绝。固定球体仍会随机器人运动产生非零相对接近速度和变化的 TTC。

static 初始化 `xi=0.02`、成功率门槛 0.90、`strict=false`。最近 100 个 static episode 的 task-reached 比例达门槛后，按合格 episode 的 transitions 累计：

```text
xi = min(1, 0.02 + 0.98 * eligible_steps / 50_000)
```

达到 1 后置 strict。strict 前接触仅计入事件/风险，不单独引发障碍物失败；strict 后，外部障碍物接触会终止 episode 并施加 terminal guard 惩罚。自碰撞、环境碰撞及关节越界始终是 hard failure。没有成功或失败而到 500 步为 timeout。strict 开启后的首个 episode 才开始累计 strict transitions，不能将升档当次 episode 反计进去。

## 奖励与回放

当前 `pose_objective: unified_keypoint`，实际 `r_goal` 使用三关键点 tracking (`0.2 * quality`)、关键点距离进度 (`10 * progress`)、关键点精度 (`0.2 * quality`)，再扣速度与动作平滑代价；成功加 20。升档通过每个 episode 保存的位置/姿态 tolerance 改变精度项，档位编号和 `orientation_scale` 不直接进入 reward 公式。位置、姿态等历史诊断列仍在日志，但不能误写为当前 `r_goal` 的额外双指数 progress 项。hard failure 扣 20、未成功 timeout 扣 2。自安全 penalty 继续生效；外部项为：

```text
clearance = clip((0.12 - min_distance) / 0.12, 0, 1)
external_penalty = xi * 0.05 * (2 * risk_max + 8 * clearance) / 10
reward = r_goal - external_penalty - self_penalty - hard_failure_penalty
         - timeout_penalty - terminal_obstacle_guard
```

S1 replay 按场景分区：每个 1024 batch 目标 `256 none + 768 static`；分区不足则在其他有数据的场景间重新分配。S0 的 none 数据保留为 `s0_anchor/s0_current/s0_history` 等存储；S1 新产生的 none 数据写入 `s0_current`，static 写入独立分区。**四池的 384/256/192/192 配额只用于 S0 的四池采样分支，并非 S1 的 batch 合同**；S1 对 none 使用 `_sample_none_rows` 从已有 anchor/current/history 中按库存抽样，不按四池固定比例抽取，也不抽 frontier/stability-anchor/semantic 专池。S0 四池池子的 checkpoint 状态可能仍随 replay 保留，但不能据此声称 S1 继续按四池训练；`semantic_long_term_replay` 在 S1 不启用。

static 首次 strict 时执行 `replay.strictify("static")`：逐 episode 保留首次障碍物接触 transition 并标为 terminal、清除其成功标签，剔除该 episode 后续 transition；记录整理前后数量及 SHA-256，防止 replay 继续学习穿障后成功的轨迹。

## 完成与核验

```text
S1 complete = static.strict
           and static.replay_strictified
           and static.strict_steps >= 25_000
```

训练日志需要分别检查 static 的到达率、碰撞率、timeout、`xi` 和 strict 转换，及 none 的成功率是否回退；查看 `sample_none/static/dynamic`、replay redistribution、strictify 统计与 checkpoint 的阶段课程状态。该 gate 是代码中的阶段交接条件，不自动证明在独立 frozen evaluation 中的泛化成功率或 S0 能力完全保持。
