# 双 SAC 机械臂到达与局部越障方案

本文统一描述当前代码实现、训练流程和验收标准。系统面向 UR5 六关节机械臂和静态、多形状、可变数量障碍物，采用两套独立 SAC：到达策略完成最终位姿到达，越障策略完成当前阻挡的局部绕行。离线生成起点并校验路径可行性，在线根据整臂安全距离与当前运动方向切换策略。

当前方案使用实测障碍物位置和形状，不调用 LSTM 预测。越障策略的首要目标是越过当前障碍并安全交回到达策略；原目标位姿仍作为输入，使动作具有任务方向信息。

## 1. 系统组成与任务分工

| 模块 | 职责 | 学习或执行方式 |
|---|---|---|
| 到达 SAC | 从当前关节姿态和速度到达原目标位姿 | 使用成熟 S0；越障训练期间冻结，交接微调期间单独更新 |
| 越障 SAC | 绕过当前阻挡、越过局部出口并满足交接检查 | 独立 Actor、双 Critic、目标 Critic、温度参数、优化器和 Replay |
| 几何引导 | 由到达动作计算受阻连杆的局部前进方向及出口 | 一次到达 Actor 推理、正运动学和障碍支撑函数 |
| 联合控制器 | 执行到达与越障的切换 | 每次使用一个 Actor 的动作 |
| 离线路径校验 | 确认训练起点存在整臂安全路径及可达的局部出口 | 关节直连、RRT-Connect、局部目标区域 RRT 和加密复查 |

两个 SAC 不共享学习参数或 Replay。越障训练使用局部奖励；到达微调使用原 S0 到达奖励。离线路径用于筛选场景，不作为越障动作示范，也不进入在线控制。

联合运行流程为：

```text
当前状态与原目标 → 到达 Actor → 当前/预测整臂距离及运动阻挡检查
                                  ↓ 受阻
                           固定局部方向与出口
                                  ↓
                           越障 Actor 执行
                                  ↓ 局部通过且交接检查通过
                           到达 Actor 继续
                                  ↓
                         原目标位置与姿态成功
```

再次受阻时允许再次接管。交接不重置主环境，不改变原目标、关节位置、关节速度或物理时钟。

## 2. 仿真环境与障碍物表示

### 2.1 机器人、时间与最终到达条件

机器人模型为 [UR5 配置](../../configs/robot/ur5.yaml) 指定的 URDF 与原始碰撞网格。控制六个真实关节，实验公共配置将末端连杆覆盖为 `wrist_3_link`；该 URDF 中它与 `tool0` 原点重合。

| 参数 | 当前值 |
|---|---:|
| 物理频率 | 240 Hz |
| 控制频率 | 20 Hz |
| 每控制步物理子步数 | 12 |
| 动作 | 六维归一化关节速度，范围 `[-1,1]` |
| 基础动作尺度 | 0.7 rad/s |
| 单次局部越障上限 | 160 控制步，即 8 s |
| 联合评估默认上限 | 1000 控制步，即 50 s |
| 最终位置容差 | 0.001 m |
| 最终姿态容差 | 0.025 rad |
| 最终成功保持 | 连续 5 个控制步 |

源到达任务使用 L13 分布：初始末端与目标距离为 0.03～0.50 m，姿态差为 0.03～π rad。到达时保留原精度减速；越障执行时暂时关闭该减速，交回后恢复。当前自碰撞安全投影开关为关闭，原始碰撞、自碰撞与关节限位检查保留。

### 2.2 静态障碍物

从仿真读取每个障碍物的身份、位置、姿态、形状和尺寸，身份仅用于场景管理，不作为网络输入。用于集合编码的位置、方向和支持范围特征统一表达在机器人基座坐标系中。

| 形状 | 尺寸参数 | 提案范围 |
|---|---|---|
| 球体 | 半径 | 0.04～0.09 m |
| 圆柱体 | 半径、高度 | 0.04～0.08 m、0.10～0.22 m |
| 长方体 | 三个半尺寸 | 各 0.04～0.08 m |

形状提案等概率。外接球半径限制为 0.04～0.14 m，并使用小、中、大三级尺寸条件采样；长方体和圆柱体姿态角提案范围为每轴 ±30°。障碍物整体保留在 `[-0.85,-0.85,0.05]` 到 `[0.85,0.85,1.10]` m 的工作空间内，生成时按外接半径收缩中心采样边界。

单障碍配置固定生成一个障碍物；多障碍配置的数量提案概率为：

| 数量 | 1 | 2 | 3 |
|---|---:|---:|---:|
| 概率 | 50% | 30% | 20% |

合法性和可行性筛选会改变最终接受样本的统计比例，因此以生成报告中的实际数量分布为准。集合编码器结构不固定障碍槽位数；当前训练分布覆盖 1～3 个，不代表已验证任意数量场景。

### 2.3 胶囊特征与真实安全距离

四个活动连杆组用于外部障碍特征及局部出口计算。当前参数以机器人 YAML 为唯一来源：

| 组名 | 轴线起点与局部偏移 | 轴线终点与局部偏移 | 半径 |
|---|---|---|---:|
| `upper_arm` | `upper_arm_link`，`(0,0,0.13585)` | `forearm_link`，`(0,0,0.13585)` | 0.088 m |
| `forearm` | `forearm_link`，`(0,0,0.0165)` | `wrist_1_link`，`(0,0,-0.09265)` | 0.081 m |
| `wrist_1` | `wrist_1_link`，`(0,0,0)` | `wrist_2_link`，`(0,0,0)` | 0.066 m |
| `wrist_2` | `wrist_2_link`，`(0,0,0)` | `wrist_3_link`，`(0,0,0)` | 0.066 m |

各偏移在其所属 URDF 连杆坐标系中表达。正运动学先转换到世界系，再按查询用途转换到基座系。胶囊距离为表面到表面的有符号距离，分离为正、接触为零、穿透为负；法向由障碍指向机器人。

整臂安全距离另由 PyBullet 原始碰撞几何计算：对完整机器人和全部障碍取最小表面间隙，固定基座及肩部同样参与。自碰撞使用环境配置的非相邻连杆检查对。因此，“整臂距离障碍至少 6 cm”表示所有机器人碰撞几何对所有障碍的最近表面距离至少 0.06 m，不能用末端距离或某一胶囊距离替代。

## 3. 多障碍编码与两套 SAC 输入

### 3.1 单障碍 37 维特征

静态障碍沿用统一凸几何接口，位移区间两端均为零，几何查询集合退化为当前原始形状。每个障碍与四个连杆胶囊分别查询距离、最近点和法向，构造：

$$
x_i=[d_{i,1:4},\ n_{i,1:4},\ c_i,\ e_{i,min},\ e_{i,max},\ \eta_{i,1:12}]\in\mathbb R^{37}.
$$

其中依次为 4 维距离、12 维法向、3 维中心、6 维位移端点和 12 维固定方向支持范围。当前静态场景中两个位移端点均为零，但维数保持一致。

归一化使用固定训练尺度。当前配置仍通过 `thesis.obstacles.predictor_checkpoint` 指向的 `outputs/obstacle_predictor/predictor.pt` 读取 `feature_scale` 等既有标定数据；静态环境不创建位置预测器，也不执行 LSTM 推理。该文件仍是当前配置的输入依赖。

### 3.2 排列不变集合编码

逐障碍共享网络 `37 → 64 → 64` 得到 64 维特征，分别执行平均池化、最大池化和注意力池化，再拼接 `log(1+N)`，形成 193 维集合特征，经 `193 → 128 → 128` 融合为 128 维外部障碍表示。注意力网络为 `37 → 32 → 1`。

Actor、两个 Critic 及其目标网络分别持有编码器参数；编码器随各自 SAC 损失训练。Replay 保存原始 `N×37` 集合和有效 mask，前向时重新编码。障碍排列不改变聚合结果，空集合输出固定的 128 维零向量。

### 3.3 固定状态与维数

到达 Actor 使用 241 维状态，外部障碍编码区段为零，不读取障碍集合。越障 Actor 在同一基础状态上追加 27 维局部上下文，并使用自己的障碍集合编码器替换外部编码区段。

| 特征 | 维数 | Actor 基础状态切片 |
|---|---:|---|
| 六关节位置、速度 | 12 | `[0:12]` |
| 三个关键点的位置误差 | 9 | `[12:21]` |
| 三关键点位置 Jacobian | 54 | `[21:75]` |
| 末端位置、姿态误差向量与范数 | 8 | `[75:83]` |
| 末端线速度、角速度 | 6 | `[83:89]` |
| 外部障碍集合编码 | 128 | `[89:217]` |
| 自碰撞状态 | 24 | `[217:241]` |
| 越障局部上下文 | 27 | `[241:268]`，仅越障策略 |

| 网络 | 到达 SAC | 越障 SAC |
|---|---:|---:|
| Actor 输入 | 241 | 268 |
| Critic 状态输入 | 187 | 214 |
| Critic 拼接六维动作后输入 | 193 | 220 |
| Actor 输出动作 | 6 | 6 |

Jacobian 只供 Actor 使用，Critic 删除 `[21:75]` 区段。Actor 和 Critic 主干隐藏层均为 `[256,256]`。原目标误差仍进入越障网络，但越障奖励和成功判定不要求到达最终目标。

## 4. 局部方向、出口与成功条件

### 4.1 由到达运动意图确定出口

到达 Actor 输出动作后，按基础动作尺度与当前精度控制得到名义关节速度 `u(q)`。在独立查询客户端计算 `q+0.01u` 的连杆端点，与当前端点之差给出局部运动速度；另用 `q+0.05u` 检查名义下一构型的关节限位、自碰撞及外部间隙。计算不推进物理，不向最终目标滚动仿真。

从触发区域内选择沿当前运动方向接近障碍的连杆和障碍对。使用当前最近点固定连杆轴线上的插值比例 `t`，材料点为：

$$
P(q)=(1-t)A_k(q)+tB_k(q).
$$

其名义运动方向归一化为单位轴 `a`。对主阻挡形状 `K` 计算该轴上的最远支撑投影，出口平面为：

$$
a^T P(q)\ge h_K(a)+r_k+0.06.
$$

材料点、轴和出口平面在一次接管期间固定。出口计算及其 24 维端点向量使用世界坐标系，与外部集合特征的基座坐标系分开维护。候选依据接近时间、距离、障碍身份和连杆组确定性排序。最小参考长度为 0.03 m；名义材料点速度至少 0.0001 m/s，接近方向余弦至少 0.05。

### 4.2 可观测进度

将四个胶囊的八个端点拼接为 `v(q)∈R²⁴`。设 `c=sqrt((1-t)²+t²)`，24 维单位权重 `w` 只在选定连杆的两个端点上非零，分别为 `(1-t)a/c` 和 `ta/c`。定义：

$$
L=\frac{h_K(a)+r_k+0.06-a^TP(q_{entry})}{c},\qquad
p(q)=\frac{[v(q)-v(q_{entry})]^Tw}{L}.
$$

`p≥1` 等价于该固定材料点越过出口平面。越障上下文为 `[w, L(1-p), L, 剩余时间比例]`，共 27 维。保存的 `reference_exit_arm=v_entry+Lw` 是平面的代数代表，实际需要到达的是满足出口条件的任意安全构型。

### 4.3 局部成功与交接

一次局部越障成功需要同时满足：

1. 进度 `p≥1`。
2. 当前整臂原始碰撞几何到所有障碍的最小间隙至少 0.06 m。
3. 本回合没有外部碰撞、自碰撞或关节越界。
4. 当前自碰撞检查对间隙至少 0.005 m。
5. 到达 Actor 的名义 `q+0.05u` 构型不越关节限位，到所有障碍的间隙至少 0.025 m，自碰撞间隙至少 0.005 m，且预测进度不小于当前进度减去 `1e-6`。

例如，腕部材料点越过固定出口，但肘部仍距另一障碍 3 cm，就不能交回；单纯后退到安全区域、尚未越过平面也不成功。

0.05 s 检查是局部运动学启发式，检查的是预测端点，不能证明惯性、电机跟踪或整个时间区间的安全。真实执行仍在每个 240 Hz 物理子步检查原始几何。

### 4.4 越障奖励

令 `F(p)=clip(p,-1,2)`，则当前局部奖励为：

$$
r=5[F(p_{next})-F(p)]-C_{ext}-0.2C_{self}-0.02
-0.01\,clip(C_{smooth},0,1)+20I_{pass}-20I_{failure}-2I_{timeout}.
$$

外部和自碰撞风险裁剪到 `[0,1]`。外部风险基于全部障碍对四组胶囊的最近距离，超过 0.10 m 为零，否则使用 `min(1,softplus((0.025-d)/0.02)/5)`，再取各组最大值；训练采用控制周期聚合风险。

进度增量先裁剪势函数再相减，往返运动不能持续刷取正进度奖励。碰撞或关节越界优先判失败；达到 160 步仍未成功判超时。最终位姿到达不单独提供局部成功奖励。

## 5. 训练起点生成与离线可行性校验

### 5.1 冻结到达策略生成真实接手状态

1. 用成熟 S0 在无外部障碍的 L13 任务上运行，要求实际安全到达。
2. 从轨迹中部运动的连杆端点附近提案主障碍，位置偏移标准差为 0.045 m；按数量配置增加上下文障碍。
3. 检查原始起点整臂间隙至少 2.5 cm、源轨迹成功终点至少 6 cm；新增障碍之间保留间隙。
4. 要求主障碍确实与原到达轨迹相交，而不是只在旁边出现。
5. 带障碍重新从原始起点真实执行 S0，按在线共用触发规则捕获接手状态；接近至少 3 个控制步，接近过程外部间隙至少 2.5 cm，且无碰撞。
6. 保存真实关节位置、非零速度、原目标、全部障碍、物理时钟、局部出口和源轨迹身份。

每条源轨迹最多提案 24 次障碍。离线无障碍轨迹只用于布置场景和取得成功终点；局部出口仍由接手时的瞬时运动意图确定。

### 5.2 整臂路径与局部出口路径

起点安全、终点安全和原轨迹受阻并不意味着绕行可行。样本还必须经过独立客户端的离线验证：

1. 以真实接手关节角为起点，以源 S0 成功轨迹最后关节角为全局终点，先检查关节直连；受阻时运行 RRT-Connect。
2. 每个候选构型及连接段检查关节限位、完整机器人对全部障碍间隙至少 2.5 cm，以及配置的非相邻连杆自碰撞间隙至少 5 mm。
3. 找到路径后，以更细关节间隔复查，并寻找满足局部出口、整臂 6 cm 间隙和当前到达 Actor 交接检查的姿态。
4. 第一条全局路径未经过局部出口时，继续用有预算的单树 RRT 搜索局部目标区域，找到后也加密复查。

通过样本保存两条证据路径：接手状态到最终目标，以及接手状态到局部出口。它们共享起点，证明相关构型在已验证的连通区域内，不要求两条路径相同。

| 配置项 | 默认值 |
|---|---:|
| 粗检查最大单关节间隔 | 0.02 rad |
| 加密检查最大单关节间隔 | 0.005 rad |
| RRT 扩展步长 | 0.15 rad |
| 全局、局部搜索各自迭代上限 | 400 |
| 全局与局部共享姿态检查预算 | 4000 次，端点预检另算 |
| 全局与局部合计加密检查预算 | 20000 次 |
| 合计交接测试预算 | 32 次 |

这些参数位于配置顶层 `local_path_verification`。搜索使用按样本内容确定的独立 RNG，不改变障碍提案随机序列，也不改变策略或 Replay 任务签名。

### 5.3 接受规则与证据边界

| 结果 | 处理 |
|---|---|
| `verified` | 全局与局部证据路径均通过，写入新 bank |
| `invalid_start` / `invalid_goal` | 当前端点不满足安全裕量或关节限位，拒绝该提案 |
| `search_unconfirmed` | 预算内未找到全局路径，标记未确认 |
| `local_exit_unconfirmed` | 全局有路径，局部出口仍未确认 |
| `fine_check_rejected` / `local_fine_check_rejected` | 路径未通过加密复查，拒绝该证据 |
| `verification_budget_unconfirmed` / `local_verification_budget_unconfirmed` | 加密复查预算不足，标记未确认 |
| `nominal_endpoint_unconfirmed` | 审计时未重现源 S0 成功终点，标记未确认 |

预算耗尽不是无解证明。只接受已确认样本会改变数据分布，报告必须保留接受比例、未确认原因和实际障碍数量分布。

路径保证限于静态运动学及设定离散分辨率；不证明连续无碰撞，不验证接手速度的制动过程、动力学约束或 160 步时间预算。离线局部出口交接检查使用该姿态下的零速度假设，真实训练起点仍保留原速度。在线后续接手状态也不自动继承 bank 起点的路径证据。

每个通过样本的 `path_verification` 绑定物理场景、任务和局部出口，保存路径、出口姿态、检查分辨率、工作量及耗时。训练加载时做记录一致性检查，不运行规划器。全新正式越障训练要求已校验 bank；完整断点续训保持原 bank 和 Replay，不会自动筛除历史样本。

## 6. 联合控制与到达策略的交接微调

### 6.1 在线策略切换

联合控制从原始任务起点开始，由到达 Actor 执行。当前整臂原始几何间隙或名义下一构型间隙不大于 0.10 m 时，尝试构造当前运动受阻的几何出口。找到有效阻挡对时切换到越障 Actor；没有有效阻挡对时继续到达。

越障期间固定当前出口、累积局部步数，满足第 4 节全部条件后交回到达 Actor。下一步可根据新的阻挡重新接管。10 cm 接管与 6 cm 释放需要同时结合运动方向和交接条件理解，距离阈值本身不保证不会重复切换。

联合任务以原目标位姿成功、碰撞、关节越界或超时结束，局部通过不会单独结束整个任务。如果越障期间已经满足最终目标的保持条件，联合任务也可直接成功。

### 6.2 交接起点微调

到达策略需要适应越障后的姿态与速度。固定当前越障策略，用真实联合运行采集每次交回到达 Actor 前的完整状态，包括首次及后续交接。训练只更新到达 SAC：

- 回合起点以 50% 概率取原始无障碍 L13 任务，50% 概率取真实交接状态；这是回合比例，不是 Replay transition 比例。
- 交接训练副本删除外部障碍及其历史，保留关节位置、速度、原目标和物理时钟，仅重置回合计数。
- 使用原 S0 到达奖励、自碰撞终止、精度控制与 CHAIN-PCR；Actor、Critic、温度学习率分别为 `1e-5`、`3e-5`、`1e-5`。
- 从成熟到达完整 SAC 加载网络、温度和优化器，显式重设学习率，Replay 从空开始。PCR 是相邻更新约束，能力保留仍需实测。
- 先使用 500000 个累计环境步的预算，保持越障策略及切换规则固定。

训练交接只能从训练场景采集。验证、测试源 seed 既排除于交接训练，也排除于微调中的原始随机重置。交接采集每个源场景只遍历一次，不能通过重复相同 rollout 凑重复状态。

微调后的到达 Actor 仍不读取外部障碍，目标是提高交接后的到达能力。其收益必须通过无障碍恢复对照和有障碍联合对照验证。

## 7. 训练执行、日志与保存

越障训练使用 8 个独立仿真进程，子进程持有冻结到达 Actor 的 CPU 副本，主进程集中执行越障推理和 SAC 更新。下一批物理仿真与上一批学习更新重叠。步数按全部环境的 transition 累计；8 个环境各前进一步，预算增加 8。

| 参数 | 越障训练 | 到达交接微调 |
|---|---:|---:|
| batch | 1024 | 1024 |
| 每 transition 的 Critic 更新额度 | 0.25 | 0.25 |
| Actor 更新间隔 | 每 2 次 Critic 更新 | 每 2 次 Critic 更新 |
| Replay 容量 | 50000 | 50000 |
| 随机动作预热 | 1000 步 | 无，使用成熟到达策略采样 |
| Actor / Critic / 温度学习率 | `1e-4 / 1e-4 / 3e-5` | `1e-5 / 3e-5 / 1e-5` |
| gamma / tau | `0.99 / 0.005` | `0.99 / 0.005` |

Replay 满足学习条件后累计更新额度，整数部分执行更新，分数部分保存在断点中。每 1000 步显示进度、近期在线成功率、速度与剩余耗时；近期窗口最多为 100 个已结束回合。到达微调另显示原始起点与交接起点的成功率。在线训练成功率不能代替独立评估。

默认保留 `manifest.json`、`episodes.csv`、`progress.csv` 和结束摘要；详细 `updates.csv` 默认关闭。完整诊断按每 10 次更新采集，避免逐步写出重复大日志。

| 保存文件 | 用途 |
|---|---|
| 越障 `step_XXXXXXX.pt` | 两套 SAC、当前局部 Replay、优化器、温度、累计计数、RNG、并行环境状态及训练场景身份；用于完整续训和联合评估 |
| 越障 `avoidance_actor_step_XXXXXXX.pt` | 仅越障 Actor，含其集合编码器；用于权重迁移或单独部署，不能恢复完整训练 |
| 到达微调 `step_XXXXXXX.pt` | 微调的到达 SAC、新 Replay、RNG、并行状态及源模型绑定信息；不重复保存冻结越障模型 |

越障不单独重复保存 `reaching_actor_step_*.pt` 或通用 `actor_step_*.pt`。续训使用完整 `step` 文件，必须保持相同任务协议、bank、模型契约和已有并行环境数。`--steps` 始终表示额外步数。

当前局部任务协议为 `s0_instantaneous_geometric_passage_v1`。不同任务协议的 Replay 和完整断点不能混用；显式 Actor 初始化属于权重迁移，Critic、Replay、优化器和任务步数重新开始，当前入口还会清零局部上下文输入列。

## 8. 当前模型、样本与执行命令

以下命令在项目根目录执行。新输出目录、运行名和评估 JSON 必须尚不存在；再次执行时更换名字。单障碍和多障碍 bank 必须使用各自匹配的配置，不能在加载时改分布冒充同一任务。

```bash
PYTHON_BIN=/home/c211/anaconda3/envs/rl/bin/python
S0_ACTOR=outputs/full/full_seed11001_from_old_actor_l0_6m_continued/checkpoints/actor_step_4350000.pt
PAIR=outputs/local_avoidance/local_geometric_s1_2m_from400k/checkpoints/step_2000000.pt
CFG=configs/experiments/paper_local_avoidance_geometry.yaml
TRAIN_BANK=outputs/local_starts/geometry_from_train_2048_verified/bank.pt
TEST_BANK=outputs/local_starts/geometric_joint_test_100/bank.pt
```

`S0_ACTOR` 用于原始源策略；`PAIR` 为已完成 200 万步训练的当前联合基线，包含到达完整 SAC 与越障 SAC。微调及覆盖评估须绑定同一份完整 `PAIR` 文件。

### 8.1 审计现有训练起点

```bash
"$PYTHON_BIN" scripts/calibration/audit_local_avoidance_bank.py \
  --config "$CFG" \
  --bank outputs/local_starts/geometry_from_train_2048/bank.pt \
  --reaching-checkpoint "$S0_ACTOR" \
  --output-dir outputs/local_starts/geometry_from_train_2048_verified
```

输出 `audit_report.json` 及只含已确认样本的 `bank.pt`；原库、split、源 seed 与样本 ID 保留。`--limit 10` 可先做小规模检查。筛选后不保证仍有 2048 个样本，空结果只写报告。筛选改变了样本集合时，不能将其作为相同 bank 继续恢复旧 Replay。

### 8.2 生成新的已校验起点

需要重新采样或补充训练集时使用：

```bash
"$PYTHON_BIN" scripts/calibration/build_local_avoidance_bank.py \
  --config "$CFG" --reaching-checkpoint "$S0_ACTOR" \
  --samples 2048 --max-traces 65536 --seed 20261027 --split train \
  --output-dir outputs/local_starts/final_geometry_train_2048
```

该入口自动执行全部可行性检查。若没有获得请求数量，只写生成报告，不写不足额的训练 bank。多障碍生成及训练都将 `CFG` 改为 `configs/experiments/paper_local_avoidance_geometry_multi.yaml`。生成新测试集使用独立 seed、`--split test` 和独立输出目录，不能用于训练。

### 8.3 重新训练越障 SAC

需基于已校验起点重新训练时执行；当前已完成的基线文件不会被此命令覆盖：

```bash
"$PYTHON_BIN" scripts/core/train_local_avoidance.py \
  --config "$CFG" --bank "$TRAIN_BANK" \
  --initialize-reaching-from "$S0_ACTOR" \
  --steps 2000000 --save-interval 50000 --num-envs 8 \
  --run-name local_geometric_verified_2m --device cuda:auto
```

此命令的越障 Actor 从头学习。可选 `--initialize-actor-from <越障Actor或完整断点>` 做显式迁移。相同任务的完整续训使用 `--resume <step文件>`，省去初始化参数，并指定新的运行名。

### 8.4 原始联合基线评估

局部评估从 bank 捕获的接手状态开始，单独检查越障策略：

```bash
"$PYTHON_BIN" scripts/core/evaluate_local_avoidance.py \
  --checkpoint "$PAIR" --bank "$TEST_BANK" \
  --output outputs/joint_evaluation/final_local_test100.json
```

完整联合评估从原始任务起点开始：

```bash
"$PYTHON_BIN" scripts/core/evaluate_joint_policy.py \
  --checkpoint "$PAIR" --bank "$TEST_BANK" \
  --output outputs/joint_evaluation/final_baseline_test100.json \
  --max-episode-steps 1000 --device cpu
```

评估依据源 seed 恢复完整任务原始起点并放入全部障碍，控制器在线自行生成出口，不接收 bank 中预先保存的局部方向、出口或路径。程序拒绝训练样本 ID 或源轨迹 seed 重叠。

当前 `TEST_BANK` 是固定原始 100 场景基准；评估允许保留该集合以比较相同分母。若改用筛选子集或重新生成的测试集，基线和待比较模型都必须在同一集合重测，不能直接与原 100 场景成功率比较。

### 8.5 采集训练交接并微调到达 SAC

```bash
"$PYTHON_BIN" scripts/calibration/collect_reaching_handoffs.py \
  --checkpoint "$PAIR" --source-bank "$TRAIN_BANK" \
  --exclude-bank "$TEST_BANK" \
  --samples 1024 --max-scenes 2048 --split train \
  --output-dir outputs/reaching_handoff_starts/final_train_1024

"$PYTHON_BIN" scripts/core/train_reaching_handoff.py \
  --source-checkpoint "$PAIR" \
  --bank outputs/reaching_handoff_starts/final_train_1024/bank.pt \
  --steps 500000 --num-envs 8 --save-interval 50000 \
  --handoff-fraction 0.5 \
  --actor-lr 0.00001 --critic-lr 0.00003 --alpha-lr 0.00001 \
  --run-name reaching_handoff_verified_500k --device cuda:auto

FT=outputs/reaching_handoff/reaching_handoff_verified_500k/checkpoints/step_0500000.pt
```

交接 bank 必须达到请求数量，采集不足先查看报告，不能把重复状态当新样本。微调完整续训需同时指定相同 `--source-checkpoint`、`--bank`、学习率、比例和环境数，增加 `--resume <微调step文件>`，并更换运行名。

### 8.6 独立恢复对照与联合覆盖评估

```bash
"$PYTHON_BIN" scripts/calibration/collect_reaching_handoffs.py \
  --checkpoint "$PAIR" --source-bank "$TEST_BANK" \
  --exclude-bank "$TRAIN_BANK" \
  --samples 128 --max-scenes 100 --split validation \
  --output-dir outputs/reaching_handoff_starts/final_validation_128

"$PYTHON_BIN" scripts/core/evaluate_reaching_handoff.py \
  --checkpoint "$FT" \
  --bank outputs/reaching_handoff_starts/final_validation_128/bank.pt \
  --normal-episodes 100 \
  --output outputs/joint_evaluation/final_reaching_recovery.json

"$PYTHON_BIN" scripts/core/evaluate_joint_policy.py \
  --checkpoint "$PAIR" --reaching-checkpoint "$FT" --bank "$TEST_BANK" \
  --output outputs/joint_evaluation/final_finetuned_test100.json \
  --max-episode-steps 1000 --device cpu
```

恢复对照在相同无外部障碍交接状态及相同原始随机任务上分别评估原到达与微调到达模型。联合覆盖评估保持越障模型和场景不变，仅替换到达 Actor；先检查原场景来源，再验证微调源完整断点摘要、冻结越障 Actor 摘要及训练/测试隔离。

用于调参的 validation 结果不能同时充当最终独立测试证据；正式报告应另外保留未参与模型选择的 test 场景。

## 9. 验收标准与当前证据

验收同时报告：局部通过率、完整位姿到达成功率、外部碰撞和自碰撞、局部/全局超时、首次和后续越障表现、重复接管、最终误差和计算耗时。到达微调还需报告无障碍交接恢复率及原始到达任务能力保留情况。

当前 200 万步基线在原固定 100 场景上的联合结果为：6 次完整成功、63 次碰撞、29 次局部超时、1 次关节越界、1 次全局超时；293 次越障尝试中完成 207 次局部通过及交接。该结果说明局部通过与最终安全到达之间仍有明显差距，不能把局部训练成功率当作联合完成率。

对同一 100 场景的离线审计全部找到了整臂安全到达路径，其中 60 个还确认了局部出口及交接路径，另 40 个局部任务未确认。未确认组中也有原策略成功案例，因此不能把这 40 个认定为无解。路径及出口检查平均约 0.68 s/场景；包含源 S0 重放的总审计约 168 s。这些是当前机器的离线测量，不是在线周期耗时。

证据文件：

- [原联合评估](../../outputs/joint_evaluation/geometric_2m_joint_test100.json)。
- [100 场景路径审计](../../outputs/diagnostics/path_verification_20261027/test100/audit_report.json)。
- [路径校验实现验证摘要](../../outputs/diagnostics/path_verification_20261027/verification_report.json)。

到达交接微调入口和覆盖评估已实现；正式改善幅度以实际完成的对照报告为准。标记 `validation_mode=true` 的短训和测试通过记录仅证明功能，不能作为学习效果或安全保证。

## 10. 配置与代码索引

`configs/experiments` 只保留单障碍几何越障、多障碍几何越障和独立到达训练三个入口。
公共参数位于 `configs/base`，入口通过 `includes` 继承；移动及合并配置不改变加载结果或当前任务签名。
到达交接微调从源完整断点读取配置，并由命令行指定微调参数，不需要另一套实验 YAML。

| 内容 | 当前入口 |
|---|---|
| 单障碍局部越障配置 | [paper_local_avoidance_geometry.yaml](../../configs/experiments/paper_local_avoidance_geometry.yaml) |
| 多障碍局部越障配置 | [paper_local_avoidance_geometry_multi.yaml](../../configs/experiments/paper_local_avoidance_geometry_multi.yaml) |
| 独立到达训练配置 | [paper_reaching_sac.yaml](../../configs/experiments/paper_reaching_sac.yaml) |
| 共用任务与 SAC 参数 | [thesis_common.yaml](../../configs/base/thesis_common.yaml) |
| 两套独立 SAC | [dual_sac.py](../../src/rl_risk_sac/algorithms/dual_sac.py) |
| 集合编码器 | [obstacle_encoder.py](../../src/rl_risk_sac/algorithms/obstacle_encoder.py) |
| 几何特征与风险 | [convex_sweep.py](../../src/rl_risk_sac/tasks/convex_sweep.py) |
| 瞬时方向、出口和交接检查 | [geometric_passage.py](../../src/rl_risk_sac/tasks/geometric_passage.py) |
| 起点生成与 bank 验证 | [local_samples.py](../../src/rl_risk_sac/tasks/local_samples.py) |
| 离线路径校验 | [local_path_verification.py](../../src/rl_risk_sac/tasks/local_path_verification.py) |
| 局部奖励与任务环境 | [local_avoidance.py](../../src/rl_risk_sac/tasks/local_avoidance.py)、[local_avoidance_env.py](../../src/rl_risk_sac/envs/local_avoidance_env.py) |
| 联合控制器 | [joint_policy.py](../../src/rl_risk_sac/control/joint_policy.py) |
| 起点生成与审计脚本 | [build_local_avoidance_bank.py](../../scripts/calibration/build_local_avoidance_bank.py)、[audit_local_avoidance_bank.py](../../scripts/calibration/audit_local_avoidance_bank.py) |
| 越障训练与并行执行 | [train_local_avoidance.py](../../scripts/core/train_local_avoidance.py)、[train_parallel_local_avoidance.py](../../scripts/core/train_parallel_local_avoidance.py) |
| 交接采集与微调 | [collect_reaching_handoffs.py](../../scripts/calibration/collect_reaching_handoffs.py)、[train_reaching_handoff.py](../../scripts/core/train_reaching_handoff.py) |
| 局部、恢复对照与联合评估 | [evaluate_local_avoidance.py](../../scripts/core/evaluate_local_avoidance.py)、[evaluate_reaching_handoff.py](../../scripts/core/evaluate_reaching_handoff.py)、[evaluate_joint_policy.py](../../scripts/core/evaluate_joint_policy.py) |
