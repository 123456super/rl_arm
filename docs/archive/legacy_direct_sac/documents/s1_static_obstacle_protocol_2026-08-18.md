# S1 静态障碍物协议与结果

> 更新时间：2026-08-18。状态：已完成基础链路重建、candidate terminal
> refine、R3/R4 分层 repair 与 blind final 闭环；当前瓶颈仍是 timeout
> 和 terminal convergence，但 candidate terminal refine 仍是已完成方案
> 里最好的 full final 结果（`86.83%`）。

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
  s1_static_candidate_terminal_refine/
    eval/seed_430{1,2,3}_final.csv
    summary.json
    episodes_joined.csv
  s1_r3_candidate_repair/
    train/seed_430{1,2,3}/...
    eval_blind/seed_430{1,2,3}_blind_final.csv
    eval_blind/summary.json
    eval_blind/episodes_joined.csv
  s1_r4_hard_case_repair/
    train/seed_430{1,2,3}/...
    eval_blind/seed_430{1,2,3}_blind_final.csv
    eval_blind/summary.json
    eval_blind/episodes_joined.csv
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

### 6. candidate terminal refine

针对 `static_candidate_path_status=candidate_path_found` 的样本做 focused
terminal shaping 与 curriculum 修复，不改变 success threshold、不延长
episode 上限，结果如下：

| train seed | final success | timeout | collision_any | mean final error |
| ---: | ---: | ---: | ---: | ---: |
| 4301 | `181/200=90.5%` | `19/200=9.5%` | `0/200=0%` | `0.0612 m` |
| 4302 | `164/200=82.0%` | `36/200=18.0%` | `0/200=0%` | `0.0775 m` |
| 4303 | `176/200=88.0%` | `24/200=12.0%` | `0/200=0%` | `0.0711 m` |
| pooled | - | `521/600=86.83%` | `79/600=13.17%` | `0.0699 m` |

这一步确实压掉了一部分终端不收敛问题，但 4302 仍是短板，
说明当前问题不是单纯“再训练久一点”就能自动解决。

### 7. R3 candidate repair

R3 从 candidate terminal refine 的 selected checkpoint 继续训练，只针对
`static_candidate_path_status=candidate_path_found` 的残余失败做 repair。
训练仍使用固定 validation manifest 选点，但最终结论不再写回原始 final
manifest，而是进入 blind final manifest `9401--9600` 做 held-out 复核。

selected checkpoint 如下：

| train seed | selected step | validation success | mean final error |
| ---: | ---: | ---: | ---: |
| 4301 | 530000 | `37/40=92.5%` | `0.0529 m` |
| 4302 | 500000 | `35/40=87.5%` | `0.0651 m` |
| 4303 | 550000 | `37/40=92.5%` | `0.0591 m` |

### 8. R4 hard-case repair

R4 将 `not_found` / `not_checked_due_to_ik` 的 hard cases 单独建桶，与
candidate-path terminal repair 分开处理。训练仍沿用 R3 的大部分设置，但
focused reset manifest 切换为 hard-case 桶。

selected checkpoint 如下：

| train seed | selected step | validation success | mean final error |
| ---: | ---: | ---: | ---: |
| 4301 | 540000 | `37/40=92.5%` | `0.0529 m` |
| 4302 | 510000 | `35/40=87.5%` | `0.0651 m` |
| 4303 | 560000 | `37/40=92.5%` | `0.0591 m` |

注意：R4-4301 在首次续训时用 `batch_size=256` 触发 CUDA OOM，因此后续
base config 改为 `batch_size=128` 并重跑；4302/4303 已先前完成，仍保留
`256`。因此当前 R4 三个 seed 不是严格同配置实验，这一点必须在定版结论里
显式保留。

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

### candidate terminal refine final

| train seed | success | timeout | collision_any |
| ---: | ---: | ---: | ---: |
| 4301 | `181/200=90.5%` | `19/200=9.5%` | `0/200=0%` |
| 4302 | `164/200=82.0%` | `36/200=18.0%` | `0/200=0%` |
| 4303 | `176/200=88.0%` | `24/200=12.0%` | `0/200=0%` |
| pooled | `521/600=86.83%` | `79/600=13.17%` | `0/600=0%` |

诊断结果显示，残余失败主要分成两类：一类是
`candidate_path_found` 上的 near-goal regression / nonconvergent
timeout，另一类是 `not_found` / `not_checked_due_to_ik` 的 hard cases。
前者通过 focused terminal 修复被明显压缩，但后者仍占了相当一部分，
因此 success 不能继续靠单一 curriculum 盲升。

### blind final：R3 candidate repair

blind final manifest 固定为
`configs/experiments/reaching_incremental/manifests/s1_blind_final_v1.json`
（`9401--9600`），从而避免继续把原始 final 失败样本用于结论本身。

| train seed | selected step | blind success | timeout | collision_any |
| ---: | ---: | ---: | ---: | ---: |
| 4301 | 530000 | `178/200=89.0%` | `22/200=11.0%` | `0/200=0%` |
| 4302 | 500000 | `161/200=80.5%` | `39/200=19.5%` | `0/200=0%` |
| 4303 | 550000 | `166/200=83.0%` | `34/200=17.0%` | `0/200=0%` |
| pooled | - | `505/600=84.17%` | `95/600=15.83%` | `0/600=0%` |

### blind final：R4 hard-case repair

| train seed | selected step | blind success | timeout | collision_any |
| ---: | ---: | ---: | ---: | ---: |
| 4301 | 540000 | `178/200=89.0%` | `22/200=11.0%` | `0/200=0%` |
| 4302 | 510000 | `161/200=80.5%` | `39/200=19.5%` | `0/200=0%` |
| 4303 | 560000 | `166/200=83.0%` | `34/200=17.0%` | `0/200=0%` |
| pooled | - | `505/600=84.17%` | `95/600=15.83%` | `0/600=0%` |

这里最关键的观察不是 “两个 summary 一样”，而是三个 seed 的 blind final
CSV 逐文件完全一致。也就是说，R3 与 R4 在 blind held-out 上不是“接近”，
而是 episode 级结果完全相同。因此当前证据只能支持 “R3≈R4”，不能支持
“R4 hard-case repair 带来了额外泛化收益”。

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

- 当前最好 full final 结果仍是 `candidate terminal refine` 的
  `521/600=86.83%`；R3/R4 blind final 只有 `505/600=84.17%`，因此这轮
  repair 不能写成“已经超越 candidate terminal refine 的新最佳方案”。
- candidate terminal refine 已证明 terminal shaping 有效，但 residual
  failure 仍分成 `candidate_path_found` 的近目标回退/不收敛，以及
  `not_found` / `not_checked_due_to_ik` 的 hard cases，两类问题不能混成
  一个桶。
- blind held-out 上 R3 与 R4 完全等价，说明当前 hard-case curriculum
  还没有转化为额外泛化收益；后续更应先解释为什么 selected checkpoint
  不同但行为结果完全一致。
- 4302 仍是持续短板 seed；如果继续做 S1，优先级应放在 4302 与 blind
  timeout failure cluster 的定向分析，而不是继续统一加训练步数。
- 碰撞仍保持为零，因此后续应继续优化 terminal convergence 和 hard
  case reachability，而不是放宽安全边界。
- 若要把 R4 作为正式对比证据，需要先统一三组 seed 的配置，再重做 blind
  final；当前 mixed-batch 结果更适合作为“方向无额外收益”的诊断结论。
