# 静态障碍物下的 6D 位姿到达设计

> 文档状态：按当前工作区代码与配置整理
>
> 主配置：`configs/experiments/thesis_serial_v13_5.yaml`
>
> 训练入口：`scripts/train_thesis_homotopy.py`
>
> 相关实现：`src/rl_risk_sac/envs/thesis_homotopy_env.py`、`src/rl_risk_sac/algorithms/homotopy_curriculum.py`、`src/rl_risk_sac/algorithms/homotopy_replay.py`

本文描述串行协议中的 S1 阶段：在已完成的 S0 末端 6D 位姿到达策略基础上，引入单个静态外部障碍物，学习保持到达能力并满足外部安全间隙。S1 不包含动态障碍物；动态障碍物属于 S2。

## 1. 阶段目标与边界

S1 的目标是让同一个策略同时完成：

- 保持 S0 的位置与姿态到达能力；
- 在 episode 内固定的外部球形障碍物旁规划安全关节速度；
- 在障碍物风险逐渐加权后，严格避免障碍物接触；
- 将 S0 的无障碍物能力保留为 25% 的 none 场景锚点。

S1 不重新设计网络、不改变 observation/action 维度、不重新启动 SAC，也不训练移动障碍物预测。静态障碍物的位姿在 reset 时采样，episode 内速度为零。

## 2. S0 到 S1 的完整 checkpoint handoff

S1 必须从完成的 S0 完整 checkpoint 启动：

```bash
python scripts/train_thesis_homotopy.py \
  --config configs/experiments/thesis_serial_v13_5.yaml \
  --stage s1 \
  --resume /absolute/path/to/completed-s0/checkpoints/step_XXXXXXX.pt \
  --run-name s1_seed11001
```

训练器会校验 protocol、root seed、RNG stream 和前一阶段状态。S1 不能使用 `--initialize-actor-from`；没有 `--resume` 时会直接拒绝启动。

继承内容如下：

| 类别 | S0 -> S1 行为 |
|---|---|
| actor | 完整加载，不重新初始化 |
| online critics / target critics | 全部加载，继续更新 |
| `alpha` 与三个 optimizer | 全部加载，保留优化状态 |
| replay buffer | 全部加载，保留 S0 none 数据 |
| 位姿课程状态 | 继承 S0 当前档位、容差、计数和 RNG |
| self-safety 状态 | 继承 `lambda_self` 及其课程计数 |
| Python / NumPy / Torch / 环境 / replay RNG | 从 checkpoint 恢复 |

因此 S1 是同一个 SAC agent 的连续训练，不是用 S0 actor 重新建立 critic 和 replay 的新实验。完成 S0 的 self-safety 权重状态也会继续使用；S1 的主要新变量是外部 static scene。

## 3. 运行方式与采样并行限制

虽然配置中的 `train.num_envs` 为 `8`，当前训练器只允许 S0 使用并行环境采样：

```python
if num_envs > 1 and stage != "s0":
    parser.error("parallel environment collection currently supports S0 only")
```

因此 S1 必须使用单环境、单进程采样。不能把 S1 文档或运行命令理解为 8 个 PyBullet worker 并行训练。checkpoint 中的 replay 和优化状态仍然完整继承，变化的是环境收集方式。

配置中的 S1 stage 参数为：

| 参数 | 值 |
|---|---:|
| physics dt | `1/240 s` |
| control dt | `0.05 s` |
| horizon | `240` 控制步（12 s） |
| action scale | `0.7 rad/s` |
| success hold | `5` 个连续控制步（约 `0.25 s`） |
| batch size | `1024` |
| SAC updates / transition | `0.25` |
| checkpoint 间隔 | `25,000` transitions |
| 静态课程 ramp | `50,000` eligible transitions |
| strict 最少数据 | `25,000` strict transitions |

## 4. S1 场景分布

`HomotopyCurriculum` 按 episode 选择场景：

| 场景 | 概率 | S1 含义 |
|---|---:|---|
| `none` | `0.25` | 无外部障碍物，保持 S0 到达能力 |
| `static` | `0.75` | 一个固定球形障碍物 |
| `dynamic` | `0.00` | S1 禁止，留给 S2 |

replay 采样目标比例与 episode 比例一致。对 batch size `1024`，理想数量约为 `256 none + 768 static + 0 dynamic`。某个分区不足时，`HomotopyReplayBuffer` 会将缺口按可用分区重新分配，并记录 redistribution；这不会引入 dynamic 数据。

## 5. 静态障碍物生成与几何合同

静态场景由 `ThesisHomotopyEnv._create_obstacle()` 生成：

- 障碍物是质量为零的球体，半径 `0.075 m`；
- reset 时采样位置 `x∈[0.22,0.72]`、`y∈[-0.48,0.48]`、`z∈[0.18,0.62] m`；
- 障碍物速度为零，`_advance_obstacle()` 不更新 static 场景；
- 若球心到目标位置小于 `0.15 m`，候选直接拒绝；
- 使用 `compute_thesis_geometry(...)` 检查机器人胶囊几何，初始最小距离必须 `>= d_safe=0.12 m`；
- 还要通过 PyBullet 初始接触检测；
- 每个 reset 最多尝试 `100` 个障碍物候选，全部失败则抛出错误。

这保证障碍物不是 reset 时就与机器人重叠的无效样本，同时保留从当前合法姿态到目标位姿的避障问题。障碍物位置固定，但机器人运动会改变相对距离、接近速度、风险和 TTC。

## 6. Observation 与 Action

S1 沿用 S0 的固定 observation/action 合同，网络输入输出维度不变。

### 6.1 124 维 observation

观测空间仍为 `Box([-1,1], shape=(124,))`，包含：

| 内容 | 维数 | S1 状态 |
|---|---:|---|
| 关节角、关节速度 | 12 | 当前机器人状态 |
| 当前末端位置与 6D 旋转 | 9 | 当前末端位姿 |
| 目标位置与 6D 旋转 | 9 | S0 同一目标采样合同 |
| 位置误差向量、相对旋转 | 9 | 当前到目标的 6D 误差 |
| 位置/姿态误差模长 | 2 | `rho_position`, `rho_orientation` |
| 末端线/角速度 | 6 | 当前运动状态 |
| goal scale、位置/姿态容差、剩余时间 | 4 | 当前 episode 合同 |
| 6 个连杆相对障碍物向量 | 18 | static 场景为真实值 |
| 障碍物位置与速度 | 6 | 速度在 static 中为零 |
| 连杆障碍物距离、TTC、接近速度、风险 | 24 | static 场景为实时几何量 |
| 障碍物存在标志 | 1 | none 为 `0`，static 为 `1` |
| 自碰撞距离、TTC、接近速度、风险 | 24 | 继承 S0 的自安全监测 |
| **合计** | **124** | 固定 schema |

静态障碍物速度为零，所以外部接近速度通常为零；当距离仍大于 `d_safe` 且没有接近速度时，TTC 取上限 `3.0 s`，距离进入 `d_safe` 内时 TTC 置为 `0`。距离风险和安全间隙违反仍会随机器人运动变化。none 样本继续使用 S0 的安全占位值，确保输入维度和字段顺序不变。

### 6.2 6 维 action

actor 输出 6 维归一化关节速度动作 `a∈[-1,1]^6`，环境执行：

```text
policy_command = 0.7 * a  # rad/s
```

配置启用近目标 precision action scaling，按上一步位置/姿态误差将动作幅度缩放到 `[0.10,1.00]`。该缩放不改变 actor 的输出维度，也不替代静态障碍物安全奖励。

## 7. 网络架构与 SAC 继承

S1 使用与 S0 相同的 Gaussian SAC。配置 `hidden_dims: [256,256]`，输入为 124 维，动作输出为 6 维。

### 7.1 Gaussian actor

```text
observation[124]
  -> Linear(124,256) + ReLU
  -> Linear(256,256) + ReLU
  -> Linear(256,256)
       ├-> Linear(256,6): mean
       └-> Linear(256,6): log_std
```

`log_std` 裁剪到 `[-20,2]`，训练时重参数化采样并经过 `tanh`；冻结/评估动作使用 `tanh(mean)`。S1 不增加障碍物专用 encoder 或 recurrent state，障碍物信息直接作为 124 维观测的一部分进入网络。

### 7.2 Twin-Q

两个独立 online Q 和两个 target Q 使用相同结构：

```text
concat(observation[124], action[6]) = 130
  -> Linear(130,256) + ReLU
  -> Linear(256,256) + ReLU
  -> Linear(256,1)
```

S1 继续使用 `gamma=0.99`、`tau=0.005`、actor/critic learning rate `1e-4`、alpha learning rate `3e-4`、初始 `alpha=0.2`、target entropy `-6`。所有优化器状态从 S0 checkpoint 继续。

## 8. 静态场景外部安全同伦

S1 的静态场景初始 contract 为：

```text
xi = 0.02
floor = 0.90
strict = false
replay_strictified = false
```

none 场景始终使用 `xi=1.0` 和 strict contract，但没有实际障碍物。对 static 场景，`xi` 只缩放外部安全 penalty，不改变目标采样、网络结构或 observation/action 维度。

奖励中的外部安全项由风险和间隙违反组成：

```text
clearance_violation = clip((d_safe - distance_min) / d_safe, 0, 1)
external_safety_cost =
    (2 * external_risk + 8 * clearance_violation) / 10
external_safety_penalty = xi * 0.05 * external_safety_cost
```

其中 `d_safe=0.12 m`、`external_safety_scale=0.05`、风险权重为 `2`、间隙权重为 `8`。S1 初期用较小 `xi` 让已学会的到达行为先适应障碍物观测；strict 化后 `xi=1`，完整外部安全权重生效。

## 9. Static strict 晋级

每个 static episode 结束后，课程记录最近 `100` 个 static episode 的 `task_reached`。当窗口已满且滚动任务成功率达到 `0.90` 时，static 进入 eligible 状态，随后按实际 eligible transitions 线性增加 `xi`：

```text
xi = min(1.0, 0.02 + 0.98 * eligible_steps / 50000)
```

当 `xi` 达到 `1.0` 时，static 设置 `strict=True`。这意味着：

- strict 之前，静态障碍物接触会进入风险、奖励和日志，但不会单独触发 obstacle terminal failure；
- strict 之后，障碍物接触触发独立的 `obstacle_failure`，episode 终止并施加 terminal obstacle penalty；它与 self/environment hard failure 分开记录；
- self-collision、环境碰撞和关节越界始终是 hard failure；
- 达到 `240` 步且没有成功或 hard failure 时是 timeout/truncated；
- 6D 位姿成功仍要求位置、姿态同时在当前容差内，并连续保持 `5` 步。

strict 化只针对 static 场景；S1 不开启 dynamic strict。

## 10. Replay 设计与 strictify

S1 的 replay 有两个作用：

1. 从 S0 继承 none 分区，按 S1 的 25% 目标比例抽样，防止加入障碍物后遗忘无障碍物位姿到达；
2. 将 static transition 写入独立的 static 分区，按 75% 目标比例抽样。

`semantic_long_term_replay` 只在 `stage == "s0"` 启用，S1 不新增 semantic long-term replay。none replay 在 S1 中从 S0 已保存的 `anchor`、`current` 和按时间保留的 `history` 分区抽样，作为无障碍物能力锚点；static transition 使用其自身保存的场景、容差和障碍物几何字段重算 reward。

static 首次进入 strict 时，训练器执行：

```python
report = replay.strictify("static")
curriculum.states["static"].replay_strictified = True
```

`strictify` 按 episode 找到第一次障碍物接触：删除接触后的轨迹，并将接触 transition 作为终止失败；同时记录整理前后数量及分区 SHA-256 到 `manifest.json`。这样 strict 开启后，后续 replay 不会继续训练“穿过障碍物后仍可成功”的旧轨迹。

## 11. 奖励组成

S1 保持 S0 的位姿奖励合同：

- 位置的粗/细双指数 progress；
- 姿态的线性全局 progress 和课程容差相关的局部指数 progress；
- joint precision progress 和 partial precision reward；
- near-goal stop cost；
- 关节速度、动作平滑和状态代价；
- 成功奖励 `+20`；
- hard failure 惩罚 `-20`；
- timeout 惩罚 `-2`；
- self-collision safety penalty。

S1 新增实际生效的是外部安全 penalty。静态障碍物接触在 strict 前可以作为非终止风险事件记录，strict 后同时产生 terminal obstacle guard，并结束 episode。障碍物风险不会替换位姿 progress，策略仍必须先学习“到达正确 6D 位姿”，再在同一任务上满足安全约束。

## 12. S1 完成条件

训练器的正式阶段完成判断等价于：

```text
S1 complete =
    static.strict
    and static.replay_strictified
    and static.strict_steps >= 25_000
```

其中 `static.strict` 的前置过程必须包括：

1. 最近 100 个 static episode 的任务成功率达到 `0.90`；
2. static 从 `xi=0.02` 开始，经过 `50,000` eligible transitions 逐步升至 `1.0`；
3. `xi=1.0` 后开启 strict；
4. 首次 strict 触发 static replay strictify；
5. strict 模式继续积累至少 `25,000` transitions。

仅 `xi=1.0` 或仅 rolling success 达标都不代表 S1 完成。

## 13. 训练日志与验证重点

训练期间应重点检查：

- `scene` 的 none/static 比例是否接近 `25%/75%`，dynamic 是否为零；
- `xi`、`strict`、`eligible_steps`、`strict_steps` 是否单调符合课程逻辑；
- `external_safety_penalty`、`external_safety_cost`、`clearance_violation` 是否在 static 中有实际变化；
- strict 前后的 `obstacle_failure`、碰撞率、timeout 和安全成功率；
- none 场景的位姿成功率是否保持，避免 S1 遗忘 S0；
- `manifest.json` 中是否存在 `scene=static` 的 strictification 前后 hash 和统计；
- checkpoint 是否记录 `static.strict=True`、`static.replay_strictified=True` 和足够的 `strict_steps`。

动态障碍物的速度规划、TTC 学习、dynamic strict 和 S2 完成 gate 不属于本文，也不应在 S1 checkpoint 验收中代替静态障碍物条件。
