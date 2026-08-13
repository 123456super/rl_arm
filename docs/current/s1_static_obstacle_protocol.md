# S1 静态障碍物协议

> 更新时间：2026-08-13。状态：待重跑；本协议用于重建 S1 可复现实验链路。

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
