# S1 静态障碍物现状分析

> 更新时间：2026-08-18。本文只讨论当前静态障碍物链路，最新统计以
> `outputs/reaching_incremental/s1_static_terminal_refine_diagnostics/summary.json`
> 为准，目标是解释“为什么没有碰撞，但末端不收敛、导致 timeout”。

## 结论

当前 S1 的主要问题不是碰撞，而是末端收敛。

- 冻结 actor 在静态障碍物下会明显退化，最初是 timeout + 少量碰撞。
- S1 fine-tune 已把 `collision_any` 压到 0，但剩余失败全部变成 timeout。
- `candidate terminal refine` 确实有效，但只把 `candidate_path_found` 桶压到
  `502/525=95.62%`，残余 79 个失败全部是 timeout。
- 4302 是当前最弱 seed，且失败并不集中在单一模式上，而是
  `nonconvergent_timeout`、`near_goal_regression`、`near_goal_timeout` 和
  `low_motion_stall` 都有。

## 当前结果

| stage | pooled success | timeout | collision_any | mean final error |
| --- | ---: | ---: | ---: | ---: |
| frozen S0 actor | `428/600=71.33%` | `156/600=26.0%` | `16/600=2.67%` | `0.0898 m` |
| S1 fine-tune | `501/600=83.50%` | `99/600=16.5%` | `0/600=0%` | `0.0757 m` |
| R1 extend +100k | `501/600=83.50%` | `99/600=16.5%` | `0/600=0%` | `0.0757 m` |
| candidate terminal refine | `521/600=86.83%` | `79/600=13.17%` | `0/600=0%` | `0.0699 m` |
| blind final pooled | `505/600=84.17%` | `95/600=15.83%` | `0/600=0%` | `0.0750 m` |

| seed | S1 fine-tune | candidate terminal refine | blind final |
| --- | ---: | ---: | ---: |
| 4301 | `87.5%` | `90.5%` | `89.0%` |
| 4302 | `82.0%` | `82.0%` | `80.5%` |
| 4303 | `81.0%` | `88.0%` | `83.0%` |

4302 是当前最明显的短板：`164/200=82.0%` 成功，`36/200` 失败，
且四类 timeout 都有出现，terminal shaping 没有把它同步拉齐。

## 为什么会这样

### 1. 障碍物改变的是“终端可达性”，不是单纯的安全性

静态障碍物加入后，策略不再只是朝目标直线收敛，而是要在“绕开障碍”和“最后靠近目标”之间切换。当前链路里，绕障是安全的，但末端推动力不够，容易停在阈值外。最新 summary 里 `candidate_path_found` 成功率已经到
`502/525=95.62%`，而 `not_found` / `not_checked_due_to_ik` 只合计
`19/75` 成功、`56/75` timeout，说明剩余问题已经明显向 hard case
reaching 偏移。

### 2. 当前 reward 对末端没有足够压力

在 `link_fixed` 的基线配置里，`w_terminal_progress=0`，末端进度没有额外 shaping。障碍物一出现，策略更容易学到“保守停住也不容易出错”，但不会自动学到“最后几厘米继续压进去”。

### 3. residual controller 在障碍附近会主动降速或 hold

代码里 `safe_to_use_terminal` 不成立时，会进入 `hold_for_clearance`；`link_avoidance` 也会在障碍附近持续介入。这样可以保安全，但也会把末端推进压弱。

### 4. beta 不是主因

对 4302 的对照显示，调大 `fixed_beta` 只会让动作更猛、jerk 更高，但 success 并没有同步提升。

| setting | success | mean action variation | mean risk |
| --- | ---: | ---: | ---: |
| baseline | `166/200` | `0.224` | `0.112` |
| beta050 | `161/200` | `0.327` | `0.116` |
| beta065 | `166/200` | `0.531` | `0.118` |

说明问题不是“动作太慢”，而是“末端目标压力不足”。

## 4302 的典型失败

### 1. 末端卡住

`episode_0054`：

- 最优误差到 `0.057m`
- 最终误差停在 `0.062m`
- 最后 `qdot` 基本掉到 0
- `d_min` 稳定在安全区，`risk` 也很低

这类样本最像“安全，但不再往前走”。

### 2. 绕障后回不来

`episode_0107`：

- 中途一直高风险、高速度
- 最终误差反而到 `0.603m`
- 说明它不是停住，而是一直没有回到终端 basin

### 3. 快速成功

`episode_0156`：

- `0.6s` 内直接成功
- 说明系统本身能收敛
- 问题是部分 reset 被静态障碍推到了坏 basin

## 失败分布

最新 summary 里，pooled 结果是 `521/600=86.83%` 成功、`79/600`
timeout、`0/600` collision。failure_mode 分布显示：

- `nonconvergent_timeout`：`32`
- `low_motion_stall`：`17`
- `near_goal_regression`：`17`
- `near_goal_timeout`：`13`

这说明失败并不是单一“碰撞”问题，而是两类末端收敛问题叠加：
一类是后段动作变软或停住，另一类是接近目标后又回退。

## 额外证据

- `reset_consistency_counts` 里有 `14` 个 `all_timeout` 和 `30`
  个 `mixed_success_timeout`，说明剩余失败里既有 seed-sensitive 样本，也有更硬的 reset。
- `candidate_path_found` 的 `reset_consistency_candidate_path_found`
  里 `all_timeout=0`、`all_success=153`、`mixed_success_timeout=22`，
  说明这类样本不是绝对不可达，更像 terminal/basin 敏感。
- `not_found` / `not_checked_due_to_ik` 合计贡献了 `56/79` 个 timeout，
  说明当前主要瓶颈已经转向 hard case reachability。

## 代码层解释

关键逻辑在 `src/rl_risk_sac/envs/ur5_dynamic_obstacle_env.py`：

- `safe_to_use_terminal` 不满足时，会进入 `hold_for_clearance`
- `link_avoidance` 会在接近障碍时介入并压低推进
- reward 里的 terminal shaping 只有在 `w_terminal_progress > 0` 时才会起作用

因此，当前静态障碍问题更像是“安全优先把末端推进吃掉了”，不是碰撞控制失效。

## 建议下一步

优先验证 terminal shaping，而不是继续盲调 beta。

```bash
bash scripts/run_s1_static_failure_diagnostics.sh print-summary
bash scripts/run_s1_static_terminal_refine_diagnostics.sh copy-feasibility
bash scripts/run_s1_static_terminal_refine_diagnostics.sh eval-trace-4301
bash scripts/run_s1_static_terminal_refine_diagnostics.sh eval-trace-4302
bash scripts/run_s1_static_terminal_refine_diagnostics.sh eval-trace-4303
bash scripts/run_s1_static_terminal_refine_diagnostics.sh analyze
```

如果要看具体轨迹：

```bash
python scripts/plot_traces.py --trace-dir /tmp/s1_horizon_4302_traces --output-dir /tmp/s1_horizon_4302_plots --episode 54
python scripts/plot_traces.py --trace-dir /tmp/s1_horizon_4302_traces --output-dir /tmp/s1_horizon_4302_plots --episode 107
python scripts/plot_traces.py --trace-dir /tmp/s1_horizon_4302_traces --output-dir /tmp/s1_horizon_4302_plots --episode 156
```

## 参考

- [研究状态](research_status.md)
- [S1 静态障碍物协议](s1_static_obstacle_protocol.md)
