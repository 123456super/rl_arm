# S0 输入、目标范围与奖励修正（v3）

> **已终止（2026-09-16）**：旧协议历史记录，仅用于失败追溯；不续训、不参与 v8 模型选择。

日期：2026-09-15  
协议：`task_first_goal_curriculum_v3`

## 修改原因

v2 的 seed 11001 完成 100000 transition 后，三个完整目标 validation seed 的成功率均为 0。训练 transition 中正奖励占约 `0.406%`，最低失败奖励约为 `-983`。检查表明动作、FK/IK 和 replay 链路可用，主要问题是目标分布从训练开始即覆盖完整关节范围、网络需要隐式学习绝对运动学关系，以及终止奖励尺度远大于普通 transition。

## 69 维 observation

在原 55 维输入上增加 14 维：当前末端世界位置 3 维、当前末端世界姿态四元数 4 维、目标世界位置 3 维、目标世界姿态四元数 4 维。四元数统一为 `[x,y,z,w]`，单位化后选择固定的等价符号，避免同一姿态以 `q` 和 `-q` 两种数值进入网络。

完整顺序为：

```text
q(6), qdot(6), p_ee^W(3), Q_ee^W(4), p_goal^W(3), Q_goal^W(4),
e_p(3), e_R(3), capsule-relative(18), p_obs(3), v_obs(3),
clearance(6), TTC(6), obstacle-present(1)
```

## S0 目标范围课程

每个 episode 先在完整关节范围采样 `q_full`，再构造：

```text
q_goal = q_initial + g (q_full - q_initial)
```

`g` 初始为 `0.03`。最近 100 个 S0 episode 的成功率达到 `0.80` 后才累计合格 transition，并在 50000 个合格 transition 内把 `g` 线性增加到 `1.0`；成功率跌破门槛时暂停，尺度不回退。达到 `g=1.0` 后，从下一条完整尺度 transition 开始累计巩固步数，至少达到 25000 才允许运行 S0 Gate 或切换到 S1。目标仍须通过工作空间、自碰撞、IK 和 FK 回代检查。validation、S1、S2 固定使用 `g=1.0`。

## 有界奖励

```text
Delta_p = clip(rho_p,t - rho_p,t+1, -0.10, 0.10)
Delta_R = clip(rho_R,t - rho_R,t+1, -0.25, 0.25)
s_bar   = clip(s_vel, 0, 1)

r_goal = 18 Delta_p + 4 Delta_R - 0.04 s_bar + 20 I_success

R_bar   = clip(R_max, 0, 1)
v_clear = clip((d_safe - d_min) / d_safe, 0, 1)

r = r_goal - 10 I_hard - xi (2 R_bar + 8 v_clear)
```

删除每步剩余位姿误差平方成本和按折扣时域放大的终止吸收态惩罚。安全 shaping 只由连续、受限的风险和安全边界穿入深度构成，上界为 10；硬失败固定扣 10；成功奖励为 20。任务球 contact 仅按课程决定是否终止，不直接制造离散 reward 跳变。

## 验证结果

- 针对性测试：`13 passed`。
- 全量测试：`71 passed`，无失败或跳过。
- 200-step 训练冒烟：69 维 checkpoint 保存成功，reward 范围 `[-0.521, 0.575]`，正奖励比例 `39.5%`，`goal_scale=0.03`。
- checkpoint 恢复与评价器加载成功。
- 100 样本奖励预检：理想到达、静止超时、严格碰撞/硬失败的中位回报约为 `22.87、0、-10`，期望排序通过率 `100%`。
- S0 巩固计数的短训练与 checkpoint 写入通过；未达到 `g=1` 和 25000 巩固步的 checkpoint 被 Gate 正确拒绝。
- 真实浅接触审计覆盖 10 个机器人构型、6 段胶囊和每段 2 个脱离方向，共 120 次 PyBullet contact；`d_min<0.12 m` 与 `risk_max>=0.79` 的一致率为 `100%`，实际最大 `d_min=0.02944 m`、最小风险为 `0.8`。

旧 v2 checkpoint 的网络输入和 replay 均为 55 维，不能恢复到 v3；v3 的 S0 必须从随机初始化重新训练。
