# SAC 名义控制与独立安全层实验说明

## 1. 实验目的与边界

### 1.1 实验目的

本实验采用“任务策略训练—独立安全增强”的两阶段路线。第一阶段通过无障碍、静态障碍物和动态障碍物三级 curriculum 训练 SAC，使其能够独立输出完成目标位姿到达和基础避障的关节速度命令。无障碍策略首次进入一种新障碍物场景时，采用“到达优先—安全渐强—严格收尾”的安全同伦课程：初期保留完整到达奖励，以很低的障碍物风险和碰撞惩罚继续训练，并允许任务球碰撞后继续执行，从而先保住新场景中的到达能力；随后只提高安全惩罚，不降低到达奖励；最终恢复碰撞真终止，并在与正式评价一致的严格 MDP 中完成收尾训练。第二阶段冻结 SAC actor，仅在其输出后增加危险预测、动作修正和执行异常处理，不再训练或微调 SAC。

因此，SAC 是完整的名义任务控制器，安全层是独立的执行保护模块。安全层不学习任务策略，也不替代 SAC 进行全局轨迹规划。

最终需要回答三个实验问题：

1. SAC 能否在单个动态球形障碍物条件下独立、稳定地完成静态目标位姿到达和基础避障？
2. 冻结 SAC 后加入安全层，能否进一步降低碰撞和安全距离违反？
3. 安全层实现上述安全收益时，是否仍能保留可接受的到达成功率、运动平滑性和实时性？

### 1.2 实验边界

| 项目 | 统一定义 |
| --- | --- |
| 机器人 | 固定基座 UR5，控制 6 个旋转关节；URDF 为 `assets/robots/universal_robots/ur_models/ur5.urdf` |
| 任务 | 到达静态目标位姿，不研究动态目标跟踪 |
| 末端参考系 | 受控运动链最后一个实体连杆 `wrist_3_link` 的 URDF link frame；不使用 `flange`、`tool0`、外接工具或额外 TCP 变换 |
| 障碍物 | 单个球体，半径 `0.075 m`；S1 静止，S2 以 `0.1 m/s` 匀速运动并在工作空间边界反射 |
| 几何模型 | 六个主要连杆胶囊体；定义和顺序取自 [`configs/robot/ur5.yaml` 的 `robot.capsules`](../../configs/robot/ur5.yaml#L35) |
| 状态来源 | 关节状态、障碍物状态和 contact 均直接读取仿真真值；不加入感知噪声、估计误差或通信延迟 |
| 动作接口 | SAC 输出 6 维归一化动作，经固定缩放和速度限幅后直接作为关节速度目标 |
| 安全事实源 | 胶囊表面间隙用于风险计算，PyBullet contact 用于判定实际碰撞 |
| 数据隔离 | training、validation 和 held-out 的目标、初始关节状态及障碍物轨迹互不重叠 |

第 3 章给出的数值是当前正式配置。开发阶段允许依据 training 和 validation 建立新配置版本，但必须记录修改原因、配置及相关文件哈希；held-out 仅用于冻结模型的最终评价，不得用于调参、阶段切换或模型选择。

## 2. 总体控制结构

```text
仿真关节状态、目标位姿误差、障碍物位置与速度、逐胶囊相对向量、间隙、TTC、存在标志
        |
        v
SAC actor
        |
        | a_t：六维归一化动作
        v
逐关节固定比例缩放与速度限幅
        |
        | qdot_actor：SAC 输出对应的关节速度命令
        v
仿真器关节速度控制接口
```

第一阶段的执行链止于 `qdot_actor`。该命令在一个 SAC 控制周期 `Delta_T` 内保持不变，由仿真器完成物理子步。除固定动作缩放和速度限幅外，不使用 rate limiter、滤波器、轨迹插值或 Safety-QP，因而不存在 observation 之外的外部控制器状态。

## 3. 第一阶段：训练并冻结任务 SAC

### 3.1 训练原则

S0 从随机初始化的 actor、critic 和 replay buffer 开始；S1、S2 必须从上一阶段通过 Gate 的完整 checkpoint 继续。机器人、目标分布、observation、动作接口和执行链在各阶段保持不变；新场景的任务球碰撞终止语义和安全惩罚系数仅按第 3.4 节的确定性课程变化。后续阶段继续采样已经掌握的简单场景，以减轻能力遗忘。

该课程借鉴 [`Adaptive Reward Shaping`](../references/1-s2.0-S0952197626005658-main.pdf) 根据安全违反逐渐增强避障目标的思想，但不照搬其削弱到达权重的做法：本文的到达误差、进展、平滑和成功奖励在 S0/S1/S2 始终固定，只对新引入场景的风险、距离违反和任务球碰撞惩罚作有界递增。训练期间始终关闭间隙预测器和 Safety-QP，安全层不参与 observation 或动作生成。自碰撞、环境碰撞和关节越界在所有阶段始终是硬失败；只有任务球碰撞在新场景课程初期允许继续。具体奖励、终止语义、阶段比例和 Gate 统一在第 3.4 节定义。

### 3.2 SAC 输入量

SAC 使用 MLP，不使用 LSTM 或固定长度历史。每个控制周期将机器人状态、目标误差、障碍物状态、逐胶囊相对几何和 TTC 拼接为 55 维 observation。除关节量外，所有向量均在机器人基坐标系 `{B}` 中表示。

六个受控关节按以下固定顺序排列：

```text
[shoulder_pan_joint, shoulder_lift_joint, elbow_joint,
 wrist_1_joint, wrist_2_joint, wrist_3_joint]
```

observation 中的 `q`、`qdot`，actor 输出动作，逐关节动作尺度、速度上限、位置上下限以及所有运动学 Jacobian 的关节维度均严格采用该顺序。关节通过 URDF 名称解析为 PyBullet joint ID，不依赖 URDF 加载后的整数编号。

| 索引 | 变量（维度） | 定义与单位 | 固定归一化 | 无障碍填充值 |
| --- | --- | --- | --- | --- |
| `0:6` | `q`（6） | 六个受控关节的仿真位置真值，`rad` | 按各关节上下限仿射映射至 `[-1,1]` | 正常计算 |
| `6:12` | `qdot`（6） | 六个关节的仿真实际速度，`rad/s` | 除以各关节速度上限并裁剪至 `[-1,1]` | 正常计算 |
| `12:15` | `e_p`（3） | `p_goal^B-p_ee^B`，`m` | 除以 `p_error_scale` 并逐分量裁剪至 `[-1,1]` | 正常计算 |
| `15:18` | `e_R`（3） | `Log(R_goal^B (R_ee^B)^T)^vee`，`rad` | 除以 `pi` | 正常计算 |
| `18:36` | `r_1,...,r_6`（18） | `r_i=p_obs^B-c_i^B`，每个向量占连续 3 维，`m` | 除以 `relative_position_scale` 并逐分量裁剪至 `[-1,1]` | 全零 |
| `36:39` | `p_obs^B`（3） | 障碍物球心位置真值，`m` | 按下文固定边界逐轴映射并裁剪至 `[-1,1]` | 全零 |
| `39:42` | `v_obs^B`（3） | 障碍物球心线速度真值，`m/s` | 除以 `obstacle_speed_scale` | 全零 |
| `42:48` | `d_1,...,d_6`（6） | `d_i=||r_i||_2-r_capsule,i-r_obs`，`m` | 在 `[-0.20,0.80] m` 裁剪后映射至 `[-1,1]` | 全部为 `1` |
| `48:54` | `TTC_1,...,TTC_6`（6） | 到达安全边界 `d_safe` 的估计时间，`s` | `clip(TTC_i,0,TTC_max)/TTC_max` | 全部为 `1` |
| `54` | `H`（1） | 障碍物存在标志，无量纲 | 不变 | `0` |

总维度为：

```text
6 + 6 + 3 + 3 + (6 x 3) + 3 + 3 + 6 + 6 + 1 = 55
```

#### 坐标系与位置归一化

当前实验中世界坐标系 `{W}` 与机器人基坐标系 `{B}` 完全重合，即 `T_WB=T_BW=I_4`；因此世界系与基坐标系之间不存在平移或旋转偏置。

目标位姿和当前末端位姿统一表示在 `{B}` 中。末端参考系 `{E}` 固定为 `wrist_3_link` 的 URDF link frame，`p_ee^B` 和 `R_ee^B` 均由该 frame 的正运动学得到。目标生成、observation、reward、成功判定及全部评价使用同一参考系；`flange`、`tool0` 和额外 TCP/工具变换均不参与任务位姿定义。

障碍物球心位置归一化采用覆盖静态和动态障碍物分布的固定边界：

```text
p_obs_low^B  = [ 0.22, -0.62, 0.18] m
p_obs_high^B = [ 0.72,  0.62, 0.62] m

p_obs_norm,k = clip(2 (p_obs,k^B - p_obs_low,k^B)
                         / (p_obs_high,k^B - p_obs_low,k^B) - 1,
                    -1, 1)
```

公式逐轴应用于 `k in {x,y,z}`。该边界包含 S1 的静态球采样范围，并与 S2 的反射边界一致；所有数据 split 共用同一组常数。超出边界的值仅在 observation 中裁剪，原始位置仍用于仿真推进、几何计算和日志。无障碍场景直接写入表中填充值，由 `H=0` 标识，不对占位值作反归一化。

#### 相对几何量的来源

六个胶囊的编号、所属连杆、局部端点偏移、半径、碰撞 link 对应关系和顺序固定取自 [`configs/robot/ur5.yaml` 的 `robot.capsules`](../../configs/robot/ur5.yaml#L35)。本引用仅绑定该配置段，不自动采用同一 YAML 的其他字段。训练和全部评价必须使用同一解析结果，并在 run manifest 中记录源文件及规范化序列化后胶囊表的 SHA-256。

设第 `i` 个胶囊的中轴线端点为 `a_i^B(q)` 和 `b_i^B(q)`，球心为 `p_obs^B`，球半径为 `r_obs=0.075 m`。令 `ell_i=b_i-a_i`，最近点、相对向量和表面间隙定义为：

```text
lambda_i = 0                                                        if ||ell_i||2^2 <= eps_seg
lambda_i = clip(((p_obs - a_i)^T ell_i) / ||ell_i||2^2, 0, 1)      otherwise
c_i      = a_i + lambda_i ell_i
r_i      = p_obs - c_i
d_i      = ||r_i||2 - r_capsule,i - r_obs
```

其中 `eps_seg=1e-12 m^2`；零长度胶囊按以 `a_i` 为球心的球体处理。`p_obs^B` 保留完整空间位置，`r_i` 提供逐胶囊避障方向，`d_i` 提供扣除双方半径后的安全裕度。三者有意保留物理冗余，以减少 MLP 学习正运动学和最近点几何的负担；无需再输入各最近点的绝对位置。

#### 障碍物位置与速度的来源

仿真训练时直接读取球心在世界坐标系中的位置 `p_obs^W` 和线速度 `v_obs^W`。使用机器人基坐标系相对世界坐标系的齐次变换 `T_BW=[R_BW,t_BW;0,1]`，统一转换为：

```text
p_obs^B = R_BW p_obs^W + t_BW
v_obs^B = R_BW v_obs^W
```

球心位置和线速度直接读取仿真真值，不通过逐帧差分估计。S2 中 `||v_obs^B||_2=0.1 m/s`，但 observation 必须保留随轨迹变化的三个方向分量。障碍物转移仅依赖当前 `p_obs^B`、`v_obs^B` 和公开边界，不使用隐藏 waypoint 编号或轨迹相位。

#### TTC 的来源

TTC 不再由多帧距离历史拟合。令 `n_i=r_i/||r_i||2`，最近点速度近似为 `v_c,i^B=J_c,i(q) qdot`，其中 `J_c,i` 是胶囊最近点的线速度雅可比。当前间隙变化率和接近速度定义为：

```text
d_dot_i        = n_i^T (v_obs^B - v_c,i^B)
v_approach_i   = max(-d_dot_i, 0)

TTC_i = 0                                  if d_i <= d_safe
TTC_i = (d_i - d_safe) / v_approach_i     if v_approach_i > eps_v
TTC_i = TTC_max                            otherwise

TTC_obs_i = clip(TTC_i, 0, TTC_max) / TTC_max
```

`TTC_obs_i=0` 表示胶囊已经进入安全边界，数值越小表示风险越紧迫，`TTC_obs_i=1` 表示按当前相对速度在预测时域内不会进入安全边界。`eps_norm=1e-8 m` 只用于判断 `||r_i||` 是否能够安全归一化；若 `||r_i||<=eps_norm`，则直接按已进入安全边界处理。`eps_v=1e-4 m/s` 只用于判断接近速度是否足以计算 TTC；当 `v_approach_i<=eps_v` 时统一取 `TTC_i=TTC_max`。二者不得混用。

TTC 补充相对运动的时间紧迫性。主方案固定使用完整 55 维输入；删除 `p_obs^B`、`d_i` 或 TTC 的版本仅用于消融，不得与主方案共用 checkpoint。

### 3.3 SAC 输出与动作执行

SAC actor 输出六个受控关节的归一化目标速度，不输出关节角度或角度增量。动作及其执行命令定义为：

```text
a_t in [-1,1]^6
qdot_actor,t[j] = clip(a_t[j] * qdot_scale[j], -qdot_limit[j], qdot_limit[j])
```

其中 `0<qdot_scale[j]<=qdot_limit[j]`。同一 run 的 training、validation 和 held-out 共用动作缩放、速度上限和控制周期，实际值写入 run manifest。

在时刻 `t`，环境先由当前仿真状态构造 `o_t`，actor 产生 `a_t`，随后将 `qdot_actor,t` 直接设置为六个关节的速度控制目标，并让仿真器推进一个 SAC 控制周期：

```text
o_t -> a_t -> qdot_actor,t -> simulator.step(N_substeps) -> o_t+1
Delta_T = N_substeps * delta_t_physics
```

`qdot_actor,t` 在 `N_substeps` 个物理子步内保持不变。仿真推进后读取新的机器人状态、障碍物状态、胶囊间隙和 contact，据此计算 reward、终止标志及下一 observation。

replay buffer 保存标准 transition，并附带第 3.4.2 节规定的奖励重算与 episode 结构字段：

```text
(o_t, a_t, r_t, o_{t+1}, terminated_t, truncated_t)
```

replay 中保存实际送入缩放器的 `a_t`，而不是仿真反馈速度；`a_t=0` 表示请求零关节速度。元组中的 `r_t` 是采集时日志值，gradient update 必须从原始分项按当前 `xi_scene` 重算 reward；episode 终止、截断、宽容期规范化和失败优先级按第 3.4.1、3.4.2 节处理。

### 3.4 训练配置与选择协议

本节是第一阶段的唯一配置事实源。一个 run 内，障碍物场景、curriculum 采样比例以及新场景的安全课程状态按下文规则变化，其余定义保持不变。动态奖励调度器属于训练算法状态而不是安全层；其状态必须进入 checkpoint。actor 仅依据严格碰撞终止语义下的 validation Gate 选择并冻结，held-out 不参与训练或选择。

#### 3.4.1 MDP 与奖励

| 项目 | 当前定义 |
| --- | --- |
| observation | 使用第 3.2 节定义的 55 维 `o_t`；只使用固定尺度，不做在线均值方差归一化 |
| 模型与几何 | URDF、末端参考系和胶囊配置按第 1.2、3.2 节定义；run manifest 记录 URDF、胶囊源文件及解析结果哈希。当前 URDF SHA-256 为 `5263b9a27eacc55fadf33abf5c4d8ac95a83f9b6593990c9f290d03c2050d62a` |
| 控制与物理步长 | `delta_t_physics=1/240 s`，`N_substeps=12`，`Delta_T=0.05 s`；每个控制周期动作保持不变 |
| 仿真动力学 | 固定基座、重力 `[0,0,0] m/s^2`、PyBullet velocity control、逐关节最大驱动力 `90 N*m`；关闭命令 rate limit、EMA、Butterworth、quintic 和 Safety-QP |
| 动作尺度 | `qdot_scale=[0.7,...,0.7] rad/s`；URDF 六关节速度硬上限均为 `pi rad/s`，动作执行仍按 3.3 节逐关节裁剪 |
| observation 尺度 | `p_error_scale=1.0 m`，`relative_position_scale=1.0 m`，`obstacle_speed_scale=0.1 m/s`；`p_obs^B` 使用第 3.2 节边界；间隙在 `[-0.20,0.80] m` 裁剪后映射至 `[-1,1]`；`d_safe=0.12 m`，`TTC_max=3.0 s`，`eps_seg=1e-12 m^2`，`eps_norm=1e-8 m`，`eps_v=1e-4 m/s` |
| episode horizon | 最多 `240` 个 SAC 控制周期，即 `12 s` |
| 到达判定 | 若本控制周期未提前触发硬失败，则执行当前动作；当推进后的状态满足 `||e_p,t+1||_2 <= 0.055 m` 且 `||e_R,t+1||_2 <= 0.10 rad` 时置 `I_task_reached=1`。不要求连续保持，不维护 observation 之外的保持计数 |
| 到达终止 | `I_task_reached=1` 时置 `terminated=true`。宽容期允许此前或当前 transition 出现任务球碰撞，因此另记 `collision_assisted_reach`；严格期只有 episode 从未发生任何失败事件时才记 `safe_success` |
| 始终硬终止 | 任一物理子步触发自碰撞或环境碰撞，或任一关节满足 `q_j<q_min,j-eps_q` 或 `q_j>q_max,j+eps_q`，立即停止剩余物理子步，以该子步后状态作为 `o_{t+1}` 并置 `terminated=true`；其中 `eps_q=1e-6 rad`。硬失败始终优先于到达 |
| 任务球碰撞终止 | 对当前正在学习的新障碍场景，当安全课程 `xi_scene<1` 时，任务球 contact 只锁存事件、不中断子步且不终止 episode；当 `xi_scene=1` 时，首次任务球 contact 立即停止剩余子步并真终止，且优先于到达。validation、held-out 和 Gate 始终采用 `xi_scene=1` 的严格语义 |
| 时间截断 | 达到 `240` 步但未真终止时置 `truncated=true`；Bellman target 只用 `terminated` 屏蔽，必须在 `truncated` transition 上继续 bootstrap |
| transition 时序 | 逐子步执行 `a_t`：正常情况及宽容期任务球 contact 均推进全部 `N_substeps`；发生硬失败或严格期任务球 contact 时，在首次事件子步后提前结束。随后用 `o_t`、本 transition 实际执行的全部子步遥测和 `o_{t+1}` 计算奖励原始分项；风险取已执行子步最大值，间隙取已执行子步最小值 |

每个物理子步读取 PyBullet contact。过滤后仅将 `contactDistance<=0 m` 视为有效 contact，并在当前 transition 内锁存。

| 事件 | 判定范围 |
| --- | --- |
| `obstacle_collision` | 任一机器人碰撞 link 与任务球体发生有效 contact |
| `self_collision` | `base_link_inertia`、`shoulder_link`、`upper_arm_link`、`forearm_link`、`wrist_1_link`、`wrist_2_link`、`wrist_3_link` 之间的有效 contact，但排除该有序列表中相邻的 link 对；reset 使用相同规则 |
| `environment_collision` | 任一机器人碰撞 link 与球体以外的外部碰撞体发生有效 contact；固定安装基座自身除外，所有外部 body 写入 run manifest |
| `joint_limit` | 任一关节越过第 3.4.1 节定义的位置边界 |

定义 `I_hard=max(I_self_collision,I_environment_collision,I_joint_limit)`，并定义 `I_obstacle_only=I_obstacle_collision(1-I_hard)`。一个 transition 可以锁存多个事件。宽容期分别报告 `task_reached`、`collision_assisted_reach` 和 `safe_success`，其中 `collision_assisted_reach=1` 当且仅当到达目标且 episode 曾发生任务球 contact，`safe_success=1` 当且仅当到达目标且 episode 从未发生任何碰撞或关节越界。严格训练、validation、held-out 和 Gate 的 episode 主结果统一按 `obstacle_collision > self_collision > environment_collision > joint_limit > safe_success > timeout` 排序；此时报告中的 `success rate` 专指 `safe_success rate`。`collision rate` 不包含 joint limit；所有 rate 的分母均为对应 seed、对应场景层的 episode 总数。

令 `rho_p,t=||e_p,t||_2`、`rho_R,t=||e_R,t||_2`。使用当前 transition 的实际关节速度变化定义 `s_vel,t=||(qdot_t+1-qdot_t)/qdot_scale||_2^2`，因此平滑项只依赖 `(o_t,a_t,o_t+1)`，不依赖未观测的上一条命令。姿态角通过 SO(3) 对数映射取得，目标生成器拒绝初始姿态误差 `rho_R,0>=pi-1e-3` 的数值奇异样本。

逐胶囊风险与全身风险定义为：

```text
R_d,i   = clip(exp(-(d_i-d_safe)/0.12), 0, 1)
R_v,i   = clip(v_approach_i/0.7, 0, 1)
R_ttc,i = exp(-TTC_i/1.0)
R_i     = clip(0.5 R_d,i + 0.2 R_v,i + 0.3 R_ttc,i, 0, 1)
R_max,t = max over all physics substeps and all six capsules of R_i
d_min,t = min over all physics substeps and all six capsules of d_i
```

无障碍 episode 中固定 `R_max,t=0`、安全距离违反标志为 `0`。定义不随课程变化的到达奖励：

```text
r_goal,t = -2.0 rho_p,t+1^2
           -0.5 rho_R,t+1^2
           +18.0 (rho_p,t-rho_p,t+1)
           +4.0  (rho_R,t-rho_R,t+1)
           -0.04 s_vel,t
           +20.0 I_task_reached

c_proximity,t = 1.0 R_max,t
                +3.0 I[d_min,t < d_safe]
```

为避免有限时域任务中“主动触发真终止以逃避后续持续误差代价”的奖励捷径，定义固定折扣时域和终止失败吸收态补偿：

```text
Z_H = sum_{k=0}^{H-1} gamma^k
    = (1-gamma^H)/(1-gamma),  gamma=0.99, H=240

c_state,t = 2.0 rho_p,t+1^2 + 0.5 rho_R,t+1^2

I_terminal_failure = I_hard
                     OR I_strict_obstacle_collision

c_terminal,t = Z_H c_state,t I_terminal_failure
```

该补偿等价于在失败后附加一个保持当前位姿误差的有限时域吸收态：真终止不能再通过删除未来误差项取得更高回报。采用固定完整 `Z_H` 而不使用剩余步数，使同一失败状态在 episode 不同时间具有相同失败代价，也无需把时间索引加入 observation。它是保守补偿，只作用于自碰撞、环境碰撞、关节越界和严格期任务球碰撞；宽容期任务球接触既不终止，也不产生该项。

每种障碍场景具有独立课程系数 `xi_scene in [0.02,1]`。该系数是训练进度状态，不进入 SAC observation；给定 checkpoint、replay 和调度器状态后，其演化完全确定。S0 无障碍场景不使用该系数；S1 中静态场景从 `0.02` 递增到 `1`；S2 开始时静态场景保持 `xi_static=1`，仅动态场景重新从 `xi_dynamic=0.02` 递增。训练 reward 为：

```text
r_t = r_goal,t
      -34.0 I_hard
      -xi_scene [4.0 c_proximity,t + 34.0 I_obstacle_only]
      -c_terminal,t
```

对无障碍 transition 固定安全同伦项为零。`34.0=2.0+4.0x8.0`；当 `xi_scene=0.02` 时，宽容期任务球接触惩罚仍仅为 `0.68`，风险与安全距离违反惩罚也同步缩小，到达奖励完全不变，且 `c_terminal=0`。当 `xi_scene=1` 且任务球接触成为真终止时，才额外启用吸收态补偿。若硬失败与任务球 contact 同时出现，只施加一次硬失败基础惩罚和一次吸收态补偿，不重复叠加任务球基础失败惩罚；当前 transition 的接近风险仍按对应 `xi_scene` 计入。

`I_task_reached` 仅在首次到达且未触发硬失败的终止 transition 上取 `1`；在 `xi_scene<1` 的宽容期，即使 episode 已经或正在与任务球接触，仍可获得到达奖励并终止为 `task_reached`。当 `xi_scene=1` 时任务球碰撞恢复为真终止，失败优先，因此最终严格 MDP 中 `I_task_reached` 与任务球碰撞不会在同一 transition 同时取 `1`。

日志保存三类 contact、joint-limit、`task_reached`、`collision_assisted_reach`、`safe_success`、`xi_scene`、`r_goal`、`c_proximity`、`c_terminal`、各惩罚分项和总 reward。本阶段只训练普通 SAC 的 reward critic，不训练 cost critic 或拉格朗日乘子；jerk 仅作为评价指标，不进入 reward。

#### 3.4.2 到达优先的安全同伦调度

S1 和 S2 分别为新引入的静态、动态障碍场景维护独立调度器。调度器不冻结 actor、critic 或温度，也不降低任何到达奖励；它只在训练表明策略已经能够在新场景中到达后，逐渐增加该场景的安全惩罚。对最近 `100` 个已完成的新场景训练 episode 维护滑动窗口任务到达率 `S_task,100`，碰撞后到达也计为任务到达，硬失败和 timeout 计为未到达。窗口未满时不得推进课程。

令 `K_scene` 为安全课程的合格推进步数，进入新场景时初始化为 `0`。每个新场景 episode 开始时固定本 episode 的 `xi_scene`，中途不得改变。episode 结束 transition 先按该固定系数写入 replay 并完成本步 UTD 更新，然后更新 100-episode 窗口及以下调度器；新系数从下一次 gradient update 和下一新场景 episode 起生效：

```text
S1 static:  S_floor = 0.90
S2 dynamic: S_floor = 0.80
K_ramp = 50000 new-scene control transitions

if the 100-episode window is full and S_task,100 >= S_floor:
    K_scene <- K_scene + number of control transitions in this completed episode
else:
    K_scene <- K_scene

xi_scene <- min(1.0, 0.02 + 0.98 K_scene / K_ramp)
```

因此，只要新场景到达能力低于门槛，安全权重就保持在当前值，不会继续挤压到达行为；到达能力恢复后，安全惩罚才按累计 `50000` 个合格的新场景 transition 从 `2%` 平滑提高到最终值。`xi_scene` 只增不减，以免同一 run 在不同目标之间来回振荡。S1 中无障碍 transition 不推进 `K_static`；S2 中无障碍和静态 transition 均不推进 `K_dynamic`，并且静态 transition 始终使用已经冻结的 `xi_static=1` 和严格碰撞终止。

当一次宽容期 episode 结束后首次使 `xi_scene=1` 时，在下一次环境交互和 gradient update 前原子进入严格期，并规范化该场景 replay：对每个历史 episode 保留首次任务球 contact 之前的 transition，把首次 contact transition 重标为 `terminated=true, truncated=false, I_task_reached=0`，删除其后的 transition；没有任务球 contact 的 episode 原样保留。每条 transition 额外保存执行前的 episode 级 `obstacle_contact_seen` 锁存值，因此即使 FIFO 已覆盖某个 episode 的前缀，也能识别并删除残留的碰撞后 transition。随后重建分区插入顺序、长度和写指针。严格期的新 episode 在首次任务球 contact 后立即终止。该转换只执行一次，过程及转换前后 replay SHA-256 写入 manifest。

严格期至少再收集并训练 `N_strict=25000` 个对应新场景 transition，之后 checkpoint 才有资格参加本阶段 Gate。课程系数达到 `1` 并不自动表示阶段通过；最终选择仍完全依据严格 validation Gate。

为避免非平稳权重与历史标量 reward 冲突，replay 除标准 transition 外必须保存足以重算奖励的原始分项：`rho_p,t`、`rho_p,t+1`、`rho_R,t`、`rho_R,t+1`、`s_vel,t`、`R_max,t`、`d_min,t`、三类 contact、joint-limit、`I_task_reached`、执行前 `obstacle_contact_seen`、场景标签、episode ID 和 episode 内步号。每次抽样均用该场景当前 `xi_scene` 重算 `r_t`；不得直接使用采集时的旧总 reward。`xi_scene`、`K_scene`、滑动窗口内容、宽容/严格状态、严格期步数和 replay 规范化标志均属于必须恢复的学习状态。

#### 3.4.3 SAC 超参数

| 项目 | 当前值或规则 |
| --- | --- |
| 策略 | tanh-squashed diagonal Gaussian policy；actor 输出 6 维 `mu` 和 6 维 `log_std`，`log_std` 裁剪到 `[-20,2]`；tanh log-probability 修正使用 `eps_log=1e-6` |
| 训练动作 | `u=mu+exp(log_std)*epsilon`，`epsilon~N(0,I)`，`a=tanh(u)`；使用重参数化采样，并在 `log pi(a|o)` 中包含 tanh Jacobian 修正 |
| 评估动作 | validation、held-out 和 Gate 一律使用确定性动作 `a=tanh(mu)`；不采样、不附加探索噪声 |
| actor 网络 | `55 -> 256 -> 256 -> (mu,log_std)`，隐藏层 ReLU，无 LayerNorm；线性层使用当前实验所记录 PyTorch 版本的默认初始化 |
| critic 网络 | 两个独立 Q 网络，均为 `(55+6) -> 256 -> 256 -> 1`，隐藏层 ReLU；各自具有 target network，使用 `min(Q1,Q2)` 构造 target |
| 数值与设备 | PyTorch float32；不使用 AMP；训练 run manifest 记录 Python、PyTorch、CUDA、PyBullet、URDF 和配置文件版本及哈希。需要精确续训的 run 启用 PyTorch deterministic algorithms，并记录所有确定性相关环境变量和 backend 开关；若当前硬件或算子无法保证确定性，必须在 manifest 中标记为仅“状态等价恢复”，不能声称逐步复现 |
| discount / target | `gamma=0.99`，Polyak `tau=0.005`；S0 初始化时把两个 online critic 硬复制到对应 target critic，此后每个 gradient update 最后各软更新一次；不存在 target actor |
| optimizer | actor、两个 critic 的联合 optimizer、temperature 均使用 Adam；`actor_lr=critic_lr=alpha_lr=3e-4`，其余 Adam 参数采用当前实验所记录 PyTorch 版本的默认值 |
| entropy | 自动温度调节，`alpha_initial=0.2`，`target_entropy=-6`；S0/S1/S2 均继续继承 `log_alpha` 和 optimizer 状态 |
| replay | 总容量 `300000` transitions；按场景标签建立 FIFO 分区，容量为无障碍 `60000`、静态 `90000`、动态 `150000` |
| batch | `batch_size=256`；S0 从无障碍区抽取 256，S1 按无障碍/静态 `64/192` 抽取，S2 按无障碍/静态/动态 `51/77/128` 抽取；不足的分区只在该阶段初始填充期按其余可用分区比例重分配，并记录实际比例 |
| 探索与更新 | S0 开始的前 `3000` 个环境步执行均匀随机动作；replay 达到 `1000` 且不少于一个 batch 后开始更新；每个环境步执行一次 critic、actor、alpha 和 target 更新，UTD=`1` |
| 梯度处理 | 不裁剪梯度，不做 reward normalization，不使用学习率调度、PER、n-step return 或 HER |

#### 3.4.4 检查点与恢复

| 内容 | 保存与校验规则 |
| --- | --- |
| 学习状态 | actor、两个 critic、两个 target critic 的完整 `state_dict`；actor/critic/alpha optimizer 的参数组、动量和步数；`log_alpha`；后续若增加 scheduler 或 scaler，也保存完整状态 |
| replay 与计数器 | 三个 replay 分区的数组、奖励重算字段、episode ID/步号、有效长度、容量、写指针、场景标签和插入顺序；全局及阶段环境步数、更新次数、episode 数、block、checkpoint 序号、随机探索剩余步数、当前阶段和 minibatch 重分配计数；另保存 `xi_static/xi_dynamic`、`K_scene`、最近 100 个新场景 episode 结果、宽容/严格状态、严格期步数和 replay 规范化标志 |
| 随机状态 | Python、NumPy、PyTorch CPU、全部 CUDA device RNG，以及环境 reset、目标、障碍物、场景选择、动作探索和 replay 抽样的每条独立 RNG 流；同时保存 RNG 名称与派生 seed 的映射 |
| 活动 episode | episode ID/seed、场景、步数、该 episode 固定的 `xi_scene`、目标、初始关节状态、球体参数和反射边界，以及当前 `q`、`qdot`、动作、observation、任务球 contact 锁存、终止状态和累计指标；同时保存 schema 版本 |
| 仿真状态 | PyBullet 可持久化快照，以及所有 body 的逻辑名称、pose、速度、关节状态、动力学参数、碰撞过滤、velocity-control 设置、physics engine 参数和仿真步数；恢复后重新施加配置并逐项校验 |
| 写入完整性 | 仅在 transition 写入 replay、本步全部 UTD 更新以及可能的 episode 结束课程更新/replay 规范化全部完成后保存；先写临时目录和各文件 SHA-256，再写包含 schema、配置、URDF、Git commit、dirty 状态及 diff 哈希的 manifest，最后原子重命名 |
| 加载校验 | 加载前核对 schema、配置、URDF、代码和依赖版本；加载后重算 observation，float32 逐元素绝对误差不得超过 `1e-7`，并核对 replay 指针、计数器、optimizer step 和 RNG 摘要；失败时中止，不允许部分加载或静默 reset |

#### 3.4.5 SAC 更新顺序

从按场景分层抽样得到的 batch `B={(o_t,a_t,raw_reward_fields_t,o_{t+1},terminated_t,truncated_t)}` 中进行一次 gradient update，`|B|=256`。先按每条 transition 的场景标签和该场景当前 `xi_scene` 由第 3.4.1 节公式重算 `r_t`，再进行以下更新。所有 observation 和 action 均为送入网络的归一化值。定义：

```text
m_t = 1 - terminated_t
```

`terminated_t` 对到达、硬失败以及严格期任务球 contact 均为 `1`；宽容期任务球 contact 本身不改变 `terminated_t`。单纯因 240 步时间上限结束时 `terminated_t=0, truncated_t=1`，因此仍然 bootstrap。若真实终止与时间上限在同一步发生，按真实终止处理，即 `terminated_t=1, truncated_t=0`。进入严格期时，旧任务球碰撞 episode 必须已经按第 3.4.2 节规范化，batch 中不得出现严格 MDP 不可达的碰撞后 transition。

对任意 observation `o`，actor 输出 `mu_theta(o)` 和裁剪后的 `log_std_theta(o)`。训练动作及其 log probability 定义为：

```text
sigma_theta(o) = exp(log_std_theta(o))
epsilon ~ N(0,I)
u = mu_theta(o) + sigma_theta(o) * epsilon
a = tanh(u)

log pi_theta(a|o)
  = sum_j [log N(u_j; mu_theta,j(o), sigma_theta,j(o))
             - log(1 - tanh(u_j)^2 + eps_log)]
```

求和覆盖 6 个动作维度。概率和熵均在归一化动作空间 `[-1,1]^6` 中计算，不把后续 `qdot_scale` 的常数 Jacobian 加入 `log pi`。实现时可使用与上式等价的数值稳定 softplus 形式，但必须通过单元测试证明结果一致。

首先在 `no_grad` 下，从当前 actor 为每个 `o_{t+1}` 重新采样 `a'` 并计算 `log pi_theta(a'|o_{t+1})`。使用本次更新开始时的 `alpha=exp(log_alpha)` 和两个 target critic 构造：

```text
y_t = r_t + gamma * m_t *
      (min(Qbar_phi1(o_{t+1},a'), Qbar_phi2(o_{t+1},a'))
       - alpha * log pi_theta(a'|o_{t+1}))

J_Q = mean_B[(Q_phi1(o_t,a_t)-y_t)^2]
      + mean_B[(Q_phi2(o_t,a_t)-y_t)^2]
```

`y_t` 必须停止梯度。两个 online critic 的参数放在同一个 Adam optimizer 中，对 `J_Q` 执行一次 `zero_grad(set_to_none=true) -> backward -> step`；target critic 不接收梯度。

critic 更新完成后，在 `o_t` 上从当前 actor 重新进行一次独立的重参数化采样，得到 `a_pi` 和 `log pi_theta(a_pi|o_t)`：

```text
J_actor = mean_B[
    stopgrad(alpha) * log pi_theta(a_pi|o_t)
    - min(Q_phi1(o_t,a_pi), Q_phi2(o_t,a_pi))
]
```

计算 actor loss 时暂时关闭两个 critic 参数的梯度，但不能 detach critic 对 `a_pi` 的输入梯度；梯度必须通过 Q 对动作的导数回传到 actor。actor optimizer 对 `J_actor` 执行一次更新，critic optimizer 不在此步骤执行。

自动温度使用无约束标量 `log_alpha`，初始化为 `log(0.2)`，并令 `alpha=exp(log_alpha)`。复用本次 actor loss 前向计算得到、但已经停止梯度的 log probability：

```text
H_target = -6

J_alpha = -mean_B[
    log_alpha * stopgrad(log pi_theta(a_pi|o_t) + H_target)
]
```

alpha optimizer 对 `J_alpha` 执行一次更新，不对 actor 反向传播。最后软更新两个 target critic：

```text
Qbar_phi1 <- (1-tau) Qbar_phi1 + tau Q_phi1
Qbar_phi2 <- (1-tau) Qbar_phi2 + tau Q_phi2
```

每个环境步在满足 replay warm-up 条件后严格执行以下顺序一次，UTD=`1`，不延迟 actor 或 alpha 更新：

```text
1. 按阶段比例抽取并固定一个 replay batch，按当前课程系数重算 reward
2. 采样 next-state action，计算 y_t，更新两个 critic
3. 独立采样 current-state action，更新 actor
4. 使用第 3 步的 detached log probability 更新 log_alpha
5. 软更新两个 target critic
6. 记录 batch 索引、各 loss、alpha、Q 均值和 target 均值
```

上述顺序也规定了随机数的消费顺序。每一步更新前仅清零该步骤对应 optimizer 的梯度；loss 均按 batch mean 归约。S1、S2 继承全部 online/target 网络、optimizer、`log_alpha`、replay 和更新计数，不重新执行 target 硬复制。

训练时从高斯策略采样动作，只有评价使用确定性 `tanh(mu)`。S1、S2 禁止仅加载 actor 后重置 critic、温度或 replay。

同阶段中断恢复时继续保存的活动 episode。若 Gate 选中较早 checkpoint 用于阶段切换，则先恢复其全部学习状态和 RNG，再结束旧阶段活动 episode且不追加 transition，随后使用已恢复的场景 RNG 创建新阶段首个 episode。validation/Gate 在独立进程或 checkpoint 副本上运行，不得消费训练 RNG 或改写训练状态。

E1 需要执行一次恢复等价性测试：从同一 checkpoint 独立恢复两次，在相同硬件和软件环境中各继续至少 `100` 个环境步，逐步比较场景 ID、observation、采样动作、reward、终止标志、replay 抽样索引、loss 和网络参数。确定性模式下要求完全一致；若只能达到状态等价而非逐步一致，必须记录首个分歧位置与原因，并相应降低复现性表述。

#### 3.4.6 场景生成与数据划分

| 项目 | 当前值或生成规则 |
| --- | --- |
| 随机种子 | 训练 `[11001,22002,33003]`，validation `[41001,42002,43003]`，held-out `[91001,92002,93003]`；环境、网络、replay 和评价均从根 seed 派生独立 RNG 流 |
| 初始关节状态 | `q_reset=[0,-1.25,1.35,-0.95,-1.10,0] rad + U(-0.22,0.22)^6`，`qdot_reset=0`；越过关节界限、按 3.4.1 节规则检测到初始自碰撞或其他有效 contact，或初始已满足目标的样本拒绝重采样 |
| 静态目标位姿 | 先在 URDF 关节上下限向内收缩 `0.10 rad` 后逐关节均匀采样候选 `q_goal`，通过 FK 生成 `(p_goal,R_goal)`；仅接受位置处于 `x=[0.25,0.78] m`、`y=[-0.45,0.45] m`、`z=[0.18,0.78] m`，且目标构型无自碰撞、IK/FK 回代位置误差不超过 `0.01 m`、姿态误差不超过 `0.05 rad` 的样本 |
| 目标姿态数值条件 | 拒绝初始姿态误差大于等于 `pi-1e-3 rad` 的样本，SO(3) `Log` 统一返回主值旋转向量 |
| 无障碍 S0 | `H=0`，采用 3.2 节规定的无障碍 observation 填充值，不创建球体碰撞体 |
| 静态球 S1 | 半径 `0.075 m`、速度零；球心从 `x=[0.22,0.72] m`、`y=[-0.48,0.48] m`、`z=[0.18,0.62] m` 逐轴均匀采样；拒绝按 3.4.1 节规则检测到有效初始 contact、初始 `d_min<d_safe`、球心距目标位置小于 `0.15 m` 的样本 |
| 动态球 S2 | 半径 `0.075 m`、速度模长恒为 `0.1 m/s`；起点 `x~U(0.22,0.72)`、`z~U(0.18,0.62)`、`y=s U(0.42,0.62)`，其中 `s` 等概率取 `-1/+1`；对侧 waypoint 的 x/z 从相同区间独立采样，`y=-s U(0.24,0.48)`，初速度指向 waypoint |
| 动态转移规则 | waypoint 只用于确定初始速度方向；episode 开始后球体按当前速度作匀速直线运动，并在固定边界 `x=[0.22,0.72] m`、`y=[-0.62,0.62] m`、`z=[0.18,0.62] m` 上作镜面反射，速度模长保持 `0.1 m/s`。球体作为运动学障碍物，不因 contact 改变轨迹；下一状态仅由当前 `p_obs`、`v_obs`、固定边界和公开转移规则决定，不使用隐藏 waypoint 编号、轨迹相位或中途随机改道 |
| 场景可行性 | reset 时用静态几何检查排除初始碰撞，不用待评 actor 筛选 episode，也不按结果删除“困难但有效”的轨迹；若目标被障碍物长时间完全阻断，仍按 timeout/失败统计 |
| 数据集合 | 每个 validation 和 held-out seed、每个场景分层固定生成 `100` 个 episode，即每层各 `300` 个 episode；训练采用对应 seed 的无限确定性 RNG 流，不复用 validation/held-out manifest |
| 隔离与审计 | 三个 split 分别输出目标位姿、初始关节状态、障碍物参数、初始位置和初始速度 manifest；按量化后的完整场景元组检查无重复，并记录生成器版本及 SHA-256 |

静态与动态障碍物的采样必须覆盖 shoulder、upper-arm、forearm 和三个 wrist 胶囊附近区域。E1 使用分层计数验证每个胶囊至少占 validation 动态场景“预测最近胶囊”的 `10%`；若基础随机生成器不满足，只能在正式训练前采用按胶囊分层的拒绝采样并升级协议版本。

#### 3.4.7 Curriculum 与 Gate

| 阶段 | replay minibatch 场景比例 | 到达优先安全课程 | 阶段 Gate |
| --- | --- | --- | --- |
| S0 无障碍 | 无障碍 `100%` | 无安全课程；每 block `100000` 步，最多 `3` blocks，每 `25000` 步保存 checkpoint | 无障碍 success rate `>=0.95`，collision rate `=0`，joint-limit rate `=0`，timeout rate `<=0.05` |
| S1 静态 | 无障碍/静态 `25%/75%` | 从 S0 完整 checkpoint 继续；`xi_static=0.02` 起步，任务球碰撞不终止；到达率门控后逐渐增至 `1`，转换 replay，再以严格碰撞终止训练至少 `25000` 个静态 transition；block/checkpoint 预算同 S0 | 严格静态 success rate `>=0.90`、collision rate `<=0.03`、joint-limit rate `=0`、timeout rate `<=0.10`；同时 S0 retention success rate `>=0.95`、collision rate `=0` 且 joint-limit rate `=0` |
| S2 动态 | 无障碍/静态/动态 `20%/30%/50%` | 从 S1 完整 checkpoint 继续；静态保持 `xi_static=1` 和严格终止；`xi_dynamic=0.02` 起步，按相同规则增至 `1`，转换动态 replay，再以严格碰撞终止训练至少 `25000` 个动态 transition；block/checkpoint 预算同 S0 | 严格动态 success rate `>=0.80`、collision rate `<=0.05`、joint-limit rate `=0`、timeout rate `<=0.20`；同时满足 S1 静态 Gate 和 S0 retention Gate |

每次 Gate 使用确定性动作和最终安全系数，在首次任务球 contact 时立即终止；宽容期训练语义绝不进入 validation。在三个 validation seed 上分别运行对应场景层的 `100` 个固定 episode。各 seed 单独计算指标，只有三者均满足全部阈值时 checkpoint 才通过。Gate 中的 success 一律指 `safe_success`。安全距离违反率、最小间隙、位置/姿态误差和运动学指标虽不作为硬阈值，仍须完整报告。

训练环境在每次 reset 时也按表中的阶段比例抽取 episode 场景；replay 再按同一比例分层抽取 transition，避免不同场景 episode 长度不同而改变实际 minibatch 组成。场景类型在一个 episode 内保持不变。

阶段切换与停止规则固定如下：

1. 每个阶段至少训练一个 `100000` 步 block，每 `25000` 步保存 checkpoint。S0 在 block 结束时评价该 block 的四个 checkpoint；S1/S2 只有 `xi_scene=1` 且已完成至少 `25000` 个对应新场景严格 transition 的 checkpoint 才有资格运行 Gate。其他 checkpoint 只报告训练期 `task_reached`、`collision_assisted_reach`、`safe_success`、碰撞率和课程状态，不得被选择为阶段输出。
2. 若存在多个通过者，先把三个 validation seed 聚合为保守指标：collision、joint-limit、timeout 和两类误差 p95 取跨 seed 最大值，success 取跨 seed 最小值；再按“较低 collision、较低 joint-limit、较高 success、较低 timeout、较低位置误差 p95、较低姿态误差 p95、较早 checkpoint”作字典序选择。
3. 若没有通过者，在同一阶段从该 block 的最后一个 checkpoint 连续训练下一个 block，不得从某个未通过但 validation 较好的早期 checkpoint 分叉。累计三个 blocks 后若课程尚未进入合格严格期，或仍无 checkpoint 通过 Gate，则停止 curriculum，报告该阶段失败，不得提高难度或访问 held-out 集。
4. S2 checkpoint 还必须在 S0、S1、S2 三个 validation 分层上同时通过。选出的 checkpoint 在任何 held-out 运行前冻结 actor、55 维输入处理、归一化常数和动作缩放器，并记录 SHA-256。
5. 三个训练 seed 的 actor 全部冻结后，才允许统一访问 held-out：每个 actor 在三个 held-out seed、每层各 `100` 个 episode 上各执行一次。held-out 不得用于返回训练、调整 reward、改变 Gate 或重新选择 checkpoint。
6. 三个训练 seed 独立执行完整 curriculum。最终结论要求三者均得到通过 S2 Gate 的 actor，且动态场景 validation success rate 的跨训练 seed 极差不超过 `0.10`。若第二阶段只使用一个主 actor，须在访问 held-out 前按第 2 条的保守聚合与字典序规则从三者中选定，之后不得因 held-out 表现更换。

### 3.5 执行流程与产物

```text
E0  固定任务、observation、reward、数据划分和评价协议
 -> E1  验证 IK/FK、胶囊距离、速度直控接口和 checkpoint 恢复等价性
 -> S0  无障碍到达
 -> S1a 静态球低安全惩罚、碰撞可继续，优先恢复到达
 -> S1b 静态球到达率门控的安全惩罚渐增
 -> S1c 静态球严格碰撞终止与收尾训练
 -> S2a 动态球低安全惩罚、碰撞可继续，静态场景保持严格
 -> S2b 动态球到达率门控的安全惩罚渐增
 -> S2c 动态球严格碰撞终止与 S0/S1 能力保留
 -> E2  按 validation Gate 选择并冻结 actor
 -> E3  对冻结 actor 执行一次 held-out 评价
```

validation 和 held-out 均按无障碍、静态障碍物、动态障碍物三层分别报告，不使用 curriculum 混合平均值替代分层结果。每层至少报告：

- success、timeout、joint-limit 和 collision rate，其中 collision 按 obstacle、self、environment 分类；
- 安全距离违反率、episode 全身最小间隙；
- 终点位置误差和姿态误差的 mean、p95、max；
- command 与 measured velocity、acceleration、jerk；
- 各评价 seed 的单独结果及跨 seed 汇总。

训练期另按新障碍场景报告 `task_reached rate`、`collision_assisted_reach rate`、`safe_success rate`、最近 100 个 episode 的门控到达率、`xi_scene`、`K_scene` 和严格期累计步数，用于验证策略是否经历“带碰撞到达—减少碰撞—严格安全到达”的预期转变。这些训练期指标不能替代严格 validation 或 held-out 指标。

第一阶段的最终产物包括：三个训练 seed 各自冻结的 actor、完整 checkpoint、训练配置和依赖 manifest、全部相关文件哈希、独立 validation 报告，以及一次性 held-out 报告。在实验完成前，不在本文预填模型路径、性能数值或通过结论。

## 4. 第二阶段：单动态球形障碍物下的独立安全层

第二阶段不再进行 SAC 训练。固定第一阶段选出的 actor，在静态目标和单个动态球形障碍物场景中，通过安全模块的逐项验证与组合实验判断系统能否在尽量保留到达能力的同时满足安全要求。

### 4.1 动作条件间隙预测器

预测器接收：

- 当前关节位置和速度；
- 上一周期的关节速度命令和关节加速度状态；
- SAC 输出经连续性处理后得到的名义终点速度 `qdot_nom`；
- 障碍物当前位置和速度；
- 与实际执行一致的控制周期、物理子步和 quintic 参数。

预测器对“如果执行当前候选动作会发生什么”进行短时外推，输出：

- 每根胶囊连杆在每个验证时刻的预测表面间隙；
- 每根连杆的未来最小间隙；
- 首次进入安全边界的时刻；
- 对应的危险连杆和危险时刻。

预测器的作用只是为安全层提供动作后果，不改变 SAC，也不生成任务动作。

### 4.2 轨迹一致 Safety-QP

QP 的决策变量是 6 维安全终点关节速度 `qdot_safe`。其目标是在满足约束的前提下，尽量接近 SAC 的名义速度 `qdot_nom`：

```text
minimize    ||qdot_safe - qdot_nom||_W^2
subject to  预测连杆间隙约束
            关节位置约束
            关节速度约束
            acceleration 约束
            jerk 约束
```

QP 不负责寻找一条新的全局路径。它只在当前控制周期内寻找对名义动作的最小安全修正。若障碍物完全阻断路径，局部 QP 可以减速或停止，但不能保证自行找到绕行方向；这种情况必须在实验中作为安全降级或任务失败报告，不能记为成功避障。

quintic 轨迹不是 QP 后的独立平滑器。完整子步速度轨迹由同一个 `qdot_safe` 仿射参数化，因此 QP 中的速度、加速度和 jerk 边界对应随后真正下发的子步命令。

### 4.3 非线性几何验收与约束增广

QP 使用的是局部线性约束，候选解产生后还必须通过完整非线性模型回放：

1. 用 `qdot_safe` 生成本周期全部 quintic 子步。
2. 在预定验证网格上检查所有胶囊连杆与障碍物的间隙。
3. 若发现线性 QP 遗漏的危险连杆或危险时刻，将其加入约束集。
4. 重新线性化并求解，直到全部通过或达到细化次数与时间预算。

Top-k 只能用于初始化约束集，不能代替最终的“全连杆 x 全验证时刻”检查。

### 4.4 逐子步监测与异常降级

模型内验收通过后，执行阶段仍需在每个物理子步读取仿真状态或机器人反馈，检查：

- 实测关节位置、速度、加速度和 jerk；
- 实测全身最小间隙；
- PyBullet contact 或真实碰撞信号；
- 命令与反馈偏差；
- QP 不可行、求解超时和预测失配。

逐子步监测是对建模误差和执行误差的最后防线，不参与 SAC 学习。触发异常后采用冻结的降级规则，并完整记录触发原因。
