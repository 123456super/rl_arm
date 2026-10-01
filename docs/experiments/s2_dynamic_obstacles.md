# S2：动态障碍物下的 6D 位姿到达

> 依据当前代码及唯一主配置 `configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml`；这是阶段合同，并非已取得的评估结果。
>
> 训练入口：`scripts/core/train_thesis_homotopy.py`；环境：`src/rl_risk_sac/envs/thesis_homotopy_env.py`；课程与回放：`src/rl_risk_sac/algorithms/homotopy_curriculum.py`、`homotopy_replay.py`。

## 阶段边界与启动

S2 从通过 S1 static strict 完成 gate 的完整 checkpoint 接续同一 Hybrid Keypoint + Jacobian + Auto-PCR SAC 策略。episode 场景概率为 none 20%、static 30%、dynamic 50%；前两者用于检验并继续训练已有无障碍和静态安全能力。动态障碍物为单个运动球体，没有预测网络、历史帧或额外 recurrent state。

```bash
python scripts/core/train_thesis_homotopy.py \
  --config configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml \
  --stage s2 \
  --resume /absolute/path/to/completed-s1/checkpoints/step_XXXXXXX.pt \
  --seed 11001 \
  --num-envs 1 \
  --run-name s2_seed11001
```

须使用 S1 的完整 `step_*.pt`，不能用 actor-only checkpoint。训练器要求来源 stage=s1、protocol 和 seed/RNG stream 相符、static 已 strict 且 replay 已 strictify、static strict transitions 至少 25k。Actor、双 Critic、双 target Critic、温度 α、优化器、replay、任务与 self-safety 课程继承；static scene 状态也继承。dynamic 同伦状态在 S2 初始化。S1/S2 都只支持单环境运行；显式指定多环境被拒绝。示例路径须替换为通过 S1 gate 的真实 checkpoint。

## 网络、任务与采样合同

S2 沿用 S1 的 162 维观测和 6 维归一化关节速度动作，不迁移输入层或重建 Critic。关节状态 12、三关键点位置误差 9、三个关键点的 9x6 位置 Jacobian 54、显式位姿误差及模长 8、末端速度 6、外部障碍物 49、自碰撞信息 24，合计 162。目标尺度、位置阈值、姿态阈值和剩余时间不作为网络输入。动态外部 49 维包含六连杆相对向量、障碍物的位置和**速度**、各连杆距离/TTC/接近速度/风险、存在标志；不同字段按代码规定的不同尺度归一化/裁剪。三个关键点的位置 Jacobian 是观测特征，不是执行的逆运动学控制器。

目标任务空间距离始终 0.03–0.7 m、角度 0.03–π rad；继承 S1 已通过 gate 的精度档及位置/姿态容差，最高档 L7 为 `0.005 m / 0.1 rad`。每个 episode 最多 500 控制步，位置和姿态同时满足当前容差并连续保持 5 步算 task reached。环境输出 `action in [-1,1]^6`，关节速度命令首先乘 `0.7 rad/s`，近目标再由上一步位姿误差缩至 [0.1,1] 倍；自安全投影配置关闭。

网络是两层 256 宽 ReLU 的 Gaussian SAC actor、双 Q 和双 target Q；`gamma=0.99`、`tau=0.005`、actor/critic LR 均 1e-4、α LR 3e-4、batch 1024、每 transition 0.25 次更新、actor 每两次 update 更新一次。CHAIN-PCR/Auto-PCR 沿用：目标 penalty/SAC 比 0.003、最大比例 0.1；不是一个新 SAC 结构。控制步 0.05 s（12 个 1/240 s 物理子步），每 25k transitions 保存 checkpoint。

## 动态球体的生成与运动

障碍物为无质量、半径 0.075 m 的球体，速度标称 0.1 m/s。reset 时从 `x=[0.22,0.72]`、`z=[0.18,0.62]` m 和一侧 `|y|=[0.42,0.62]` m 抽取初始位置；在对侧 `|y|=[0.24,0.48]` m 内抽取 waypoint，初始速度指向 waypoint。初始机器人胶囊最小表面距离须不小于 `d_safe=0.12 m`、无 PyBullet 接触，最多重抽 100 次。

waypoint 只用于确定初速度；此后每个物理子步按 `position += velocity * (1/240 s)` 推进，越过 x=[0.22,0.72]、y=[-0.62,0.62]、z=[0.18,0.62] m 的边界时镜像位置并反转对应速度分量。static episode 仍按 S1 的静止球模型运行，none 则无外部球体。动态 TTC 和风险由球与机器人最近点的相对运动计算，不表示模型显式预测未来轨迹。

## 动态 strict 课程与奖励

S2 dynamic 初始化 `xi=0.02`、成功率门槛 0.80、`strict=false`。最近 100 个 dynamic episode 的 task-reached 比例达到门槛后，对合格 episode 计 eligible transitions：

```text
xi = min(1, 0.02 + 0.98 * eligible_steps / 50_000)
```

`xi=1` 后 dynamic strict，之后的完整 strict episode 才增加 strict_steps。strict 前障碍物接触记风险但不单独终止，strict 后接触触发独立的 obstacle failure 与 terminal guard；自碰撞、环境碰撞和关节越界一直是 hard failure。达到 horizon 且未成功/失败则 timeout。S1 继承来的 static 保持 strict 状态，不在 S2 重新走 static ramp。

奖励继续使用 `pose_objective: unified_keypoint`：三关键点 tracking、关键点误差 progress、关键点精度奖励，减速度和动作平滑代价，成功 +20；hard failure -20，未成功 timeout -2；另有自安全代价与当前场景的外部安全项。升档通过 transition 保存的位置/姿态 tolerance 影响精度项，档位编号和 `orientation_scale` 不直接进入 reward 公式。过去位置/姿态 progress 等诊断字段仍记录，但不作为当前 `r_goal` 的额外加项。

```text
clearance = clip((0.12 - min_distance) / 0.12, 0, 1)
external_penalty = xi * 0.05 * (2 * risk_max + 8 * clearance) / 10
reward = r_goal - external_penalty - self_penalty - hard_failure_penalty
         - timeout_penalty - terminal_obstacle_guard
```

dynamic 的 risk_max 来自每连杆距离、相对接近速度及 TTC 风险融合后取最大值；即使尚未碰撞，也可能得到非零 penalty。`xi` 只缩放外部安全 penalty，不调整 actor 输入、目标范围或精度档。

## Replay 与完成条件

replay 按场景分区；对 1024 batch 的目标分配约为 `204 none + 308 static + 512 dynamic`（底层比例 51:77:128 / 256）。场景样本不足时按其他场景剩余库存重新分配，记 redistribution。none 使用从 S0 延续的 `s0_anchor/s0_current/s0_history`，static 使用 S1 延续的 static 分区，dynamic 写入/抽自独立 dynamic 分区。S2 新 none 数据也写 `s0_current`。**S0 的四池 384/256/192/192 配额在 S2 不用于 batch**：none 经 `_sample_none_rows` 抽样，不按四池独立抽 stability-anchor/frontier/semantic；`semantic_long_term_replay` 在 S2 关闭。

dynamic 首次 strict 时调用 `replay.strictify("dynamic")`，每条存储轨迹保留首次障碍物接触并标记 terminal、清除成功标志、剔除后续 transition；记录前后数量及 SHA-256。static 使用 S1 已整理的分区。

```text
S2 complete = dynamic.strict
           and dynamic.replay_strictified
           and dynamic.strict_steps >= 25_000
```

验收时分别检查 dynamic 的接触、回避、TTC、timeout、strict 和回放整理日志，同时冻结评估 none/static 以识别遗忘。正式 stage gate 只检测 dynamic strict、strictify 和数据量，不等同于三个场景的 frozen 成功率均达标。独立评估的其他质量阈值不应被误写成训练器的完成条件。
