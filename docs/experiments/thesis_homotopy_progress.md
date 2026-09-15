# 到达优先安全同伦：实验进展

更新日期：2026-09-15  
当前协议：`task_first_safety_homotopy_v2`

## 当前状态

论文前三章规定的训练路径已经实现。当前完成的是正式训练前的有效实现验证，尚未产生 S0/S1/S2 正式性能结果。

旧协议结果、失败预跑及其 checkpoint 已从 `outputs/thesis_homotopy/` 清理。训练器会拒绝恢复协议字段不是 `task_first_safety_homotopy_v2` 的 checkpoint。

## 当前有效实现

- 55 维 observation 包含关节状态、目标位姿误差、障碍物与六胶囊相对几何、间隙、TTC 和障碍物存在标志。
- 动作为 20 Hz 原始关节速度命令，每个控制周期执行 12 个 240 Hz 物理子步。
- 目标由候选关节构型 FK 生成，必须通过带姿态 IK、关节限位和 FK 回代误差检查后才能进入 episode。
- 宽容期任务球碰撞不终止，安全惩罚从 `xi=0.02` 起步，到达奖励始终保持不变。
- 自碰撞、环境碰撞、关节越界以及严格期任务球碰撞是真终止，并使用吸收态补偿消除提前终止捷径。
- replay 保存奖励原子量并按当前 `xi` 重算；进入严格期后保留首次 contact、改为 terminal、删除碰撞后 transition，并对首次 contact 重算终止补偿。
- S0/S1/S2 继承完整 actor、critic、target、optimizer、温度、replay、课程状态和 RNG；禁止只加载 actor。

## 已通过验证

统一审计目录：`outputs/thesis_homotopy/preflight_20260915_v2/`。

| 验证 | 有效结果 |
| --- | --- |
| 针对性测试 | `8 passed` |
| 目标运动学 | 100 个目标全部通过；0 次 reset 失败 |
| IK→FK 误差 | 最大位置 `7.30e-8 m`；最大姿态 `2.27e-7 rad` |
| 回报排序 | 100/100 满足 `理想到达 > timeout > 硬失败/严格碰撞` |
| 宽容碰撞语义 | 无终止吸收态补偿，基础 contact 惩罚在 `xi=0.02` 时保持 `0.68` |
| SAC 冒烟 | 100 transition、93 update、全部 reward 有限、v2 checkpoint 可保存 |

详细过程和结果见 `docs/experiments/thesis_homotopy_preflight_20260915.md`。

## 尚未完成

- v2 协议下 seed 11001 的 S0 首个 100k block 及 validation Gate。
- 合格 S0 actor 的障碍 observation 迁移检查和固定 `xi` 扫描。
- S1 静态安全同伦、严格化切换 A/B 与收尾训练。
- S2 动态安全同伦与 S0/S1 retention Gate。
- 三训练 seed 聚合、冻结 actor 后的一次性 held-out 评价及消融实验。

## 入口

- 训练：`scripts/train_thesis_homotopy.py`
- 严格评价：`scripts/evaluate_thesis_homotopy.py`
- 预检：`scripts/preflight_thesis_homotopy.py`
- 固定安全系数探针：`scripts/probe_fixed_xi.py`
- 配置：`configs/experiments/thesis_homotopy.yaml`
- 详细协议：`docs/thesis/thesis_outline.md` 第 1～3 章
