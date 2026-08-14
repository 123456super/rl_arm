# S1 静态障碍物协议与结果

> 更新时间：2026-08-14。状态：已重建基础链路；当前瓶颈为 timeout 和
> terminal convergence。R1 继续训练未提升 full final success。

## 目标

在已冻结的 S0/v2 无障碍 reaching actor 上加入单个静态随机障碍物，
回答两个问题：

1. 冻结 reaching actor 在静态障碍物下退化到什么程度？
2. 仅做静态障碍物 fine-tune 后，是否能恢复绕障与终端收敛能力？

本协议仍不是动态避障、安全过滤器或真机协议。

## 固定条件

- 机器人、随机目标、初始关节状态、20 Hz 控制、12 s episode 上限和
  `0.055 m` success tolerance 沿用 S0/v2。
- 障碍物启用，`scenario=random`，`speed_range=[0.0, 0.0]`。
- 关闭 safety filter、viability monitor、diagnostic logging、
  recovery、constraint relaxation 和 maximin clearance 分支。
- 方法固定为 `link_fixed`。
- validation manifest 固定为
  `configs/experiments/reaching_recovery/manifests/v1_validation.json`
  (`9301--9340`)。
- final manifest 固定为
  `configs/experiments/reaching_recovery/manifests/v1_final.json`
  (`9001--9200`)。

## 产物目录

```text
outputs/reaching_incremental/
  s1_static_obstacle/
    eval/
      seed_4301_final.csv
      seed_4302_final.csv
      seed_4303_final.csv
    summary.json
    episodes_joined.csv
  s1_static_obstacle_finetune/
    train/seed_430{1,2,3}/...
    eval/seed_430{1,2,3}_final.csv
    summary.json
    episodes_joined.csv
  s1_static_obstacle_finetune_r1_extend100k/
    train/seed_430{1,2,3}/...
    eval/seed_430{1,2,3}_final.csv
    summary.json
    episodes_joined.csv
```

## 执行顺序

### 1. 冻结 actor 静态评估

```bash
bash scripts/run_s1_static_chain.sh eval-frozen
```

该命令只评估 S0/v2 冻结 actor，不训练、不选 checkpoint。输出写入
`outputs/reaching_incremental/s1_static_obstacle/`。

### 2. 静态障碍物 fine-tune

如果冻结 actor 静态评估未通过，再运行：

```bash
bash scripts/run_s1_static_chain.sh finetune
```

训练从对应 S0/v2 actor 和 agent state 恢复：

| seed | start step | config |
| ---: | ---: | --- |
| 4301 | 240000 | `configs/experiments/reaching_incremental/s1_static_finetune_seed4301.yaml` |
| 4302 | 280000 | `configs/experiments/reaching_incremental/s1_static_finetune_seed4302.yaml` |
| 4303 | 220000 | `configs/experiments/reaching_incremental/s1_static_finetune_seed4303.yaml` |

### 3. checkpoint 选择

```bash
bash scripts/run_s1_static_chain.sh select-finetune
```

选择规则固定为 validation success rate；同分时按脚本的 physical
contact、capsule overlap、safety violation、final error 和 step
规则打破平局。

### 4. final 复核

```bash
bash scripts/run_s1_static_chain.sh eval-finetune
```

该命令读取每个 seed 的 `selected_checkpoint.csv`，在完整 final
manifest 上复核，并汇总为 `summary.json` 和 `episodes_joined.csv`。

### 5. R1 继续训练诊断

S1 fine-tune 后曾从各自 selected checkpoint 继续训练 `+100000`
environment steps：

| seed | start step | total step | config |
| ---: | ---: | ---: | --- |
| 4301 | 440000 | 540000 | `configs/experiments/reaching_incremental/s1_static_extend100k_seed4301.yaml` |
| 4302 | 480000 | 580000 | `configs/experiments/reaching_incremental/s1_static_extend100k_seed4302.yaml` |
| 4303 | 420000 | 520000 | `configs/experiments/reaching_incremental/s1_static_extend100k_seed4303.yaml` |

R1 结果与 S1 fine-tune final 完全持平，因此 R1 只作为诊断产物，不作为
新的有效提升。

## 实际结果

### 冻结 actor 静态评估

| train seed | success | timeout | collision_any |
| ---: | ---: | ---: | ---: |
| 4301 | `174/200=87.0%` | `22/200=11.0%` | `4/200=2.0%` |
| 4302 | `106/200=53.0%` | `89/200=44.5%` | `5/200=2.5%` |
| 4303 | `148/200=74.0%` | `45/200=22.5%` | `7/200=3.5%` |
| pooled | `428/600=71.33%` | `156/600=26.0%` | `16/600=2.67%` |

结论：冻结 S0 actor 在静态障碍物下明显退化，失败以 timeout 为主，
同时存在少量碰撞。

### S1 fine-tune final

| train seed | selected step | validation success | final success | timeout | collision_any |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4301 | 440000 | `36/40=90.0%` | `175/200=87.5%` | `25/200=12.5%` | `0/200=0%` |
| 4302 | 480000 | `35/40=87.5%` | `164/200=82.0%` | `36/200=18.0%` | `0/200=0%` |
| 4303 | 420000 | `35/40=87.5%` | `162/200=81.0%` | `38/200=19.0%` | `0/200=0%` |
| pooled | - | - | `501/600=83.5%` | `99/600=16.5%` | `0/600=0%` |

S1 fine-tune 将 pooled success 从 `71.33%` 提升到 `83.5%`，并将
`collision_any` 从 `16/600` 降到 `0/600`。但 remaining failures 全部
为 timeout，说明主要瓶颈转为终端收敛。

### R1 extend +100k final

| train seed | selected step | validation success | final success | timeout | collision_any |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4301 | 460000 | `36/40=90.0%` | `175/200=87.5%` | `25/200=12.5%` | `0/200=0%` |
| 4302 | 500000 | `35/40=87.5%` | `164/200=82.0%` | `36/200=18.0%` | `0/200=0%` |
| 4303 | 440000 | `35/40=87.5%` | `162/200=81.0%` | `38/200=19.0%` | `0/200=0%` |
| pooled | - | - | `501/600=83.5%` | `99/600=16.5%` | `0/600=0%` |

R1 selection 均选择续训后的第一个 checkpoint，full final 与 S1
fine-tune 持平；继续简单加步数不再是优先方向。

## 报告口径

必须同时报告：

- full final success：三 seed 各 200 episode，pooled 600 episode。
- timeout 数量：`success=0` 且无碰撞终止的 episode。
- `collision_any`、`collision_capsule_overlap`、`collision_pybullet_contact`
  和 `termination_collision`。
- 平均、P95 和最大 final position error。
- 平均成功 completion time。

不得删除 reset、放宽 success threshold、延长 final episode 时限，或把
validation 数字写成 final 数字。若后续引入静态可行候选子集，该子集只
能作为解释性标签，不能替代 full final。

## 当前问题与后续方向

- 当前 S1 尚不能作为进入 S2 的稳定基础，原因是 full final success 仍
  只有 `83.5%`。
- 剩余失败全部为 timeout；碰撞已清零，因此后续应优化 terminal
  convergence，而不是牺牲安全边界。
- 不建议继续盲目增加训练步数；R1 extend +100k 已显示收益为零。
- 下一步应使用独立 hard-case curriculum 或奖励/终端收敛改造。若使用
  `9001--9200` final 失败样本参与诊断或调参，最终结论需要新的 blind
  final manifest。
