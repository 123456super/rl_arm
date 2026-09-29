# 动态障碍物下的 6D 位姿到达设计

> 文档状态：按当前工作区代码与配置整理
>
> 主配置：`configs/experiments/thesis_serial_v13_5.yaml`
>
> 训练入口：`scripts/train_thesis_homotopy.py`
>
> 相关实现：`src/rl_risk_sac/envs/thesis_homotopy_env.py`、`src/rl_risk_sac/algorithms/homotopy_curriculum.py`、`src/rl_risk_sac/algorithms/homotopy_replay.py`

本文描述串行协议的最后阶段 S2：在已完成的 S1 静态障碍物策略上引入动态外部障碍物，继续保持完整的末端 6D 位姿到达能力，并学习根据相对速度和 TTC 规避运动中的障碍物。S2 不重新训练一个新 agent，也不改变 S0/S1 已确定的 observation、action 和网络结构。

## 1. 阶段目标与边界

S2 的目标是让同一个策略同时完成：

- 保持无障碍物和静态障碍物场景的位姿到达能力；
- 感知 6 个机器人连杆与移动球形障碍物的相对位置、速度、接近速度和 TTC；
- 在动态风险逐渐加权后避免障碍物接触；
- 在动态障碍物 strict 合同下继续积累足够的有效 transition，形成可供后续验证的 checkpoint。

S2 只使用代码中的单球动态障碍物模型。没有预测网络、历史帧堆叠或额外 recurrent state；动态信息通过固定的 124 维观测直接输入 Gaussian SAC。

## 2. S1 到 S2 的完整 checkpoint handoff

S2 必须从通过 S1 完成 gate 的完整 checkpoint 启动：

```bash
python scripts/train_thesis_homotopy.py \
  --config configs/experiments/thesis_serial_v13_5.yaml \
  --stage s2 \
  --resume /absolute/path/to/completed-s1/checkpoints/step_XXXXXXX.pt \
  --run-name s2_seed11001
```

训练器会验证：

- checkpoint 的前一阶段必须是 `s1`；
- protocol、root seed 和 RNG stream 必须与当前配置一致；
- S1 的 static scene 必须已经 `strict=True`、`replay_strictified=True`，并累计至少 `25,000` 个 strict transitions。

S2 继承的内容：

| 类别 | S1 -> S2 行为 |
|---|---|
| actor | 完整加载，继续训练 |
| 两个 online critic 与两个 target critic | 完整加载，继续更新 |
| `alpha`、actor/critic/alpha optimizer | 完整加载，保留优化状态 |
| replay buffer | 完整加载 none/static/dynamic 分区 |
| 位姿课程状态 | 继承已完成的 S0 目标和精度合同 |
| self-safety 状态 | 继承 S1 的 `lambda_self` 和课程计数 |
| static scene 状态 | 通过 `inherit_scene_state` 继承 S1 的 strict、xi、replay 标记和 strict_steps |
| dynamic scene 状态 | 使用 S2 初始化的 `xi=0.02`、floor `0.80`、非 strict 状态 |
| RNG、active episode 和 checkpoint 计数 | 按完整 checkpoint 恢复 |

因此 S2 的学习重点是 dynamic scene。none 和 static replay 继续作为能力保持数据，而不是被清空后重新收集。

## 3. 运行方式与并行限制

当前训练器只允许 S0 使用并行环境收集：

```python
if num_envs > 1 and stage != "s0":
    parser.error("parallel environment collection currently supports S0 only")
```

所以 S2 必须单环境、单进程运行。配置文件中的 `train.num_envs: 8` 不代表 S2 可以启动 8 个 worker；S2 的 `resolve_num_envs` 会解析为单环境，显式指定大于 1 也会被拒绝。

主要运行参数保持 V13.5 串行协议：

| 参数 | 值 |
|---|---:|
| physics dt | `1/240 s` |
| control dt | `0.05 s` |
| 每个控制步物理子步 | `12` |
| horizon | `240` 控制步（12 s） |
| action scale | `0.7 rad/s` |
| success hold | `5` 个连续控制步（约 `0.25 s`） |
| SAC batch size | `1024` |
| updates / transition | `0.25` |
| dynamic ramp | `50,000` eligible transitions |
| strict 最少 transition | `25,000` |

## 4. S2 场景和 replay 分布

### 4.1 Episode 场景分布

`HomotopyCurriculum` 在 S2 按 episode 选择：

| 场景 | 概率 | S2 作用 |
|---|---:|---|
| `none` | `0.20` | 保持 S0 的无障碍物 6D 到达能力 |
| `static` | `0.30` | 保持 S1 已学会的静态安全能力 |
| `dynamic` | `0.50` | S2 的主要新任务 |

dynamic episode 在 reset 时生成移动障碍物，并在每个物理子步推进。static episode 仍按 S1 的固定球障碍物合同运行，不会被 dynamic 逻辑替换。

### 4.2 Replay batch 分布

`HomotopyReplayBuffer.sample(stage="s2")` 的目标整数比例为 `(51, 77, 128)`，即：

```text
none    = 51 / 256 ≈ 19.92%
static  = 77 / 256 ≈ 30.08%
dynamic = 128 / 256 = 50.00%
```

对配置中的 batch size `1024`，目标约为 `204 none + 308 static + 512 dynamic`。某个分区暂时不足时，replay 会将缺口重新分配到可用分区，并递增 redistribution 计数；dynamic 不足不会凭空生成 transition。

none replay 从 S0 保留的 anchor/current/history 分区抽样，static replay 从 S1 累积的 static 分区抽样，dynamic replay 使用独立的 dynamic 分区。`semantic_long_term_replay` 只在 `stage == "s0"` 启用，S2 不新增该机制。

## 5. 动态障碍物生成与运动模型

实现位于 `ThesisHomotopyEnv._create_obstacle()` 和 `_advance_obstacle()`。

### 5.1 reset 时的初始状态

- 障碍物为质量为零的球体，半径 `0.075 m`；
- 初始位置为 `x∈[0.22,0.72]`、`z∈[0.18,0.62] m`；
- 初始 `y` 在一侧采样：`±Uniform(0.42,0.62) m`；
- 另一侧 waypoint 的 `x∈[0.22,0.72]`、`z∈[0.18,0.62]`，其 `y` 位于相反侧的 `±[0.24,0.48] m`；
- 初始速度方向为“初始位置 -> waypoint”，速度模长由配置 `obstacle_speed=0.1 m/s` 固定；
- 候选必须满足初始几何最小间隙 `>= d_safe=0.12 m`，并通过 PyBullet 初始接触检测；
- 最多尝试 `100` 个候选，失败则终止 reset 并报告错误。

代码只在 reset 时使用 waypoint 计算初速度；没有每一步重新规划到 waypoint 的控制器。episode 中障碍物沿当前速度运动。

### 5.2 每个物理子步的推进和反射

在机器人控制命令执行前，每个物理子步调用 `_advance_obstacle()`：

```text
position <- position + velocity * physics_dt
```

动态障碍物在以下轴向边界内运动并反射速度：

| 轴 | 下界 | 上界 |
|---|---:|---:|
| x | `0.22 m` | `0.72 m` |
| y | `-0.62 m` | `0.62 m` |
| z | `0.18 m` | `0.62 m` |

越过边界时使用镜像位置，并将对应速度分量改为指向区域内部的符号。因此 S2 的障碍物是确定速度、边界反射的单球运动，不是随机加速度模型，也不是动态目标生成器。

## 6. Observation 与 Action 合同

S2 与 S0/S1 保持完全相同的 `124` 维 observation 和 `6` 维 action。这样可以直接加载完整网络和 replay，而不需要输入层迁移或重新初始化 critic。

### 6.1 124 维 observation

观测仍为 `Box([-1,1], shape=(124,))`，主要字段为：

| 内容 | 维数 | S2 状态 |
|---|---:|---|
| 关节角、关节速度 | 12 | 当前机器人状态 |
| 当前末端位置、当前 6D 旋转 | 9 | 当前末端位姿 |
| 目标位置、目标 6D 旋转 | 9 | 继承 S0 的 6D 目标采样合同 |
| 位置误差向量、相对旋转、误差模长 | 11 | 当前位姿误差 |
| 末端线速度、角速度 | 6 | 当前运动状态 |
| goal scale、位置/姿态容差、剩余时间 | 4 | 当前 episode 合同 |
| 6 个连杆到障碍物的相对向量 | 18 | 随移动球更新 |
| 障碍物位置、障碍物速度 | 6 | dynamic 中均为实际值 |
| 连杆距离、TTC、接近速度、风险 | 24 | 包含相对运动信息 |
| 障碍物存在标志 | 1 | dynamic 为 `1` |
| 自碰撞距离、TTC、接近速度、风险 | 24 | 继承 S1 自安全监测 |
| **合计** | **124** | 固定 schema |

动态场景中，几何计算使用障碍物速度和机器人连杆最近点速度得到相对接近速度；TTC 在距离进入 `d_safe` 时为 `0`，无接近速度时取 `3.0 s` 上限，否则按剩余安全距离除以接近速度并裁剪到上限。

### 6.2 6 维关节速度 action

actor 输出 `a∈[-1,1]^6`，环境执行：

```text
policy_command = 0.7 * a  # rad/s
```

近目标 precision action scaling 仍根据上一步位置/姿态误差将动作幅度缩放到 `[0.10,1.00]`。S2 没有额外的避障控制器、Jacobian 逆或速度修正器；动态风险通过观测、奖励和终止合同影响策略学习。

## 7. 网络架构与 SAC 训练

S2 直接加载 S1 的 Gaussian SAC：输入维度 `124`，动作维度 `6`，隐藏层 `[256,256]`，ReLU 激活。

### 7.1 Actor

```text
observation[124]
  -> Linear(124,256) + ReLU
  -> Linear(256,256) + ReLU
  -> Linear(256,256)
       ├-> Linear(256,6): mean
       └-> Linear(256,6): log_std
```

`log_std` 裁剪到 `[-20,2]`；训练时使用重参数化 Gaussian 加 `tanh`，评估时使用 `tanh(mean)`。障碍物速度不是单独送入另一条网络，而是与其余状态拼接在同一个 observation 中。

### 7.2 Twin-Q

两个 online Q 和两个 target Q 均为：

```text
concat(observation[124], action[6]) = 130
  -> Linear(130,256) + ReLU
  -> Linear(256,256) + ReLU
  -> Linear(256,1)
```

S2 继续使用 `gamma=0.99`、`tau=0.005`、actor/critic learning rate `1e-4`、alpha learning rate `3e-4`、初始 `alpha=0.2`、target entropy `-6`。S2 不执行 actor-only 初始化，也不清空 S1 critic 或 optimizer 状态。

## 8. Dynamic 同伦课程与 strict 晋级

S2 初始化时 dynamic scene 的课程状态为：

```text
xi = 0.02
floor = 0.80
strict = false
replay_strictified = false
```

`finish_episode()` 对 dynamic scene 保存最近 `100` 个 episode 的 `task_reached`。当窗口已满且 rolling task success rate `>=0.80` 时，dynamic 进入 eligible 状态，并按 eligible transitions 更新：

```text
xi = min(1.0, 0.02 + 0.98 * eligible_steps / 50000)
```

当 `xi` 达到 `1.0`，dynamic 设置 `strict=True`。S2 期间：

- strict 之前，动态障碍物接触会记录为风险/碰撞事件，但不会单独触发 obstacle terminal failure；
- strict 之后，障碍物接触触发独立的 `obstacle_failure`，episode 终止并施加 terminal obstacle penalty；
- self-collision、环境碰撞和关节越界始终是 hard failure；
- 成功仍要求位置和姿态同时在当前容差内，并连续保持 `5` 步；
- horizon `240` 且没有成功或 failure 时为 timeout/truncated。

static scene 在 S2 中从 S1 继承 strict 状态；none scene 始终使用 `xi=1.0` 的安全合同但没有障碍物实体。

## 9. 动态风险与奖励

S2 使用与 S1 相同的 `homotopy_reward`。位姿部分保持不变：

- 位置的粗/细双指数 progress，以及姿态的线性全局和局部指数 progress；
- joint precision progress、partial precision reward；
- near-goal stop cost、速度和 smoothness cost；
- 成功奖励 `+20`、hard failure 惩罚 `-20`、timeout 惩罚 `-2`；
- self-collision safety penalty。

动态障碍物使外部安全项包含真正的相对运动风险：

```text
clearance_violation = clip((d_safe - distance_min) / d_safe, 0, 1)
external_safety_cost =
    (2 * external_risk + 8 * clearance_violation) / 10
external_safety_penalty = xi * 0.05 * external_safety_cost
```

`external_risk` 由距离、接近速度和 TTC 风险融合得到；动态场景中接近速度来自障碍物速度减去机器人最近点速度。`xi` 从 `0.02` 到 `1.0` 的变化只改变外部安全 penalty 的尺度，不改变网络输入输出或位姿目标范围。

对每根连杆，环境实际使用的融合形式为：

```text
distance_risk = clip(exp(-(distance - d_safe) / 0.12), 0, 1)
velocity_risk = clip(approach_velocity / 0.7, 0, 1)
ttc_risk      = exp(-ttc / 1.0)
link_risk     = clip(
    0.5 * distance_risk
  + 0.2 * velocity_risk
  + 0.3 * ttc_risk,
  0, 1)
```

训练奖励取所有连杆的最大风险 `risk_max`，并以最小表面距离 `distance_min` 计算 clearance violation。这样动态障碍物即使尚未接触，也会因正在接近或 TTC 变小而增加外部安全代价。

## 10. Dynamic replay strictify

dynamic 首次进入 strict 时，训练器执行：

```python
report = replay.strictify("dynamic")
curriculum.states["dynamic"].replay_strictified = True
```

`strictify("dynamic")` 在 dynamic replay 中按 episode 查找第一次障碍物接触：

- 删除接触前已被标记为不可继续的轨迹部分以及接触后的 transition；
- 将第一次接触 transition 设为终止，并清除其 `task_reached` 标志；
- 压缩 dynamic 分区；
- 记录整理前后数量、删除数量和 SHA-256 到 `manifest.json`。

这样 strict 开启后，dynamic batch 不会继续从 replay 中学习“穿过移动障碍物后仍能成功”的旧轨迹。static replay 不会在 S2 重新 strictify；它使用 S1 已完成的 static strict 数据。

## 11. S2 完成条件

代码中的 `stage_complete()` 对 S2 检查 dynamic scene：

```text
S2 complete =
    dynamic.strict
    and dynamic.replay_strictified
    and dynamic.strict_steps >= 25_000
```

完整前置过程是：

1. S1 checkpoint 已通过 static strict/replay/25k gate；
2. dynamic 最近 100 个 episode 的 rolling task success rate 达到 `0.80`；
3. dynamic `xi` 从 `0.02` 经 `50,000` eligible transitions 升至 `1.0`；
4. `xi=1.0` 后开启 dynamic strict；
5. 首次 strict 触发 dynamic replay strictify；
6. strict 模式下继续积累至少 `25,000` dynamic transitions。

仅 dynamic 成功率达标、仅 `xi=1.0` 或仅 replay strictify 均不能单独宣告 S2 完成。

## 12. 日志与验证重点

训练和验收应重点检查：

- episode scene 比例接近 `20%/30%/50%`，dynamic 是否实际被采样；
- `obstacle_position`、`obstacle_velocity` 是否随物理子步推进，并在边界处正确反射；
- dynamic 的 `external_safety_penalty`、`external_safety_cost`、`clearance_violation`、TTC 和 approach velocity 是否出现非静态变化；
- `xi`、`eligible_steps`、`strict`、`strict_steps` 是否符合课程状态机；
- strict 前后的 dynamic obstacle failure、碰撞率、安全成功率和 timeout；
- none/static 场景成功率是否保持，避免 S2 遗忘 S0/S1；
- `manifest.json` 是否包含 dynamic strictification 的 hash 和统计；
- checkpoint 是否记录 `dynamic.strict=True`、`dynamic.replay_strictified=True` 和不少于 `25,000` 的 `strict_steps`。

独立的 `scripts/evaluate_thesis_homotopy.py` 对 dynamic 场景默认参考阈值为：safe success rate `>=0.80`、collision rate `<=0.05`、joint-limit rate `=0`、timeout rate `<=0.20`。这些是评估报告中的场景质量阈值；训练阶段正式完成 gate 仍以 dynamic strict、replay strictify 和 strict transition 数为准。

S2 是当前串行线路的最后阶段。后续若要做部署或论文报告，应使用通过 S2 gate 的完整 checkpoint，并同时报告 none、static、dynamic 三类场景，而不能只报告 dynamic 子集。
