# `link_fixed_penalty1` 低速部署前检查

## 当前状态

离线预检已通过，报告位于 `outputs/deployment_preflight/offline_report.json`。该检查只在 PyBullet 中运行，不连接真实机械臂；它证明 checkpoint 可加载、推理值有限，且仿真输出不超过配置的关节速度限幅。它不构成实机安全验收或形式化安全保证。

## 固化的部署候选

| train seed | selected step | checkpoint |
| --- | ---: | --- |
| 101 | 100000 | `outputs/rechecks/link_fixed_penalty1/train/link_fixed/seed_101/link_fixed_seed101_steps100000/actor_step_100000.pt` |
| 202 | 50000 | `outputs/rechecks/link_fixed_penalty1/train/link_fixed/seed_202/link_fixed_seed202_steps100000/actor_step_50000.pt` |
| 303 | 100000 | `outputs/rechecks/link_fixed_penalty1/train/link_fixed/seed_303/link_fixed_seed303_steps100000/actor_step_100000.pt` |

三个 checkpoint 均由 `validation_seed=2001`、每 checkpoint 20 episodes 的独立验证选择。实机只部署这一固定平滑候选；不部署 `ldrc_fixed` 或 `ldrc_adaptive` 做碰撞性对比。

## 已验证的离线条件

| 检查项 | 当前配置或结果 |
| --- | --- |
| 策略配置 | `link_fixed` 执行实现 + `sac.fixed_risk_penalty=1.0`，论文名称为 `link_fixed_penalty1` |
| 推理输入 | UR5 PyBullet observation 维度匹配，所有候选在 20 个确定性控制步内均为有限值 |
| 策略动作 | 三个候选的归一化动作均未越过 `[-1, 1]` |
| 仿真关节命令 | `env.action_scale=0.7 rad/s`；三个候选的最大绝对命令分别为 0.643、0.678、0.580 rad/s |
| 固定平滑 | `env.fixed_beta=0.35` |
| 风险阈值 | `risk.d_safe=0.12 m` |
| 仿真目标工作空间 | `x=[0.25, 0.78]`、`y=[-0.45, 0.45]`、`z=[0.18, 0.78] m` |

可复跑检查：

```bash
conda run -n rl python scripts/deployment_preflight.py
```

## 实机前必须现场签核

以下项目不在本仓库的 PyBullet 控制链路中，必须在真实控制器与安全单元上验证后才可通电运动：

- 将仿真 `0.7 rad/s` 映射为更保守的真实机器人关节速度、加速度与工作空间限制，并由控制器强制执行。
- 在安全姿态下实测物理急停、保护停、控制器断连和命令超时的停止路径。
- 确认 RGB-D 检测丢失、无效深度、障碍物状态过期或 `Risk_global` 超阈时，控制器在向机械臂发送速度前进入安全停止。
- 完成相机—机械臂外参、TCP、胶囊体保守包络和距离监控的实物复核。
- 先空载、无障碍物、低速度完成单步和短轨迹检查；随后才使用轻质球体进行 10--20 次低速可执行性验证。

## 现场实验边界

只记录可执行性、轨迹、`d_min`、`Risk_global`、关节速度及风险响应。不得安排高风险碰撞性基线对比，也不得将该实验表述为严格安全保证。
