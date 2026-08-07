# 基础 reaching 恢复协议 v1/v2

> 状态：v2 已完成；基础 reaching 基线可冻结。该协议是对冻结 B4 actor 无障碍到达失败的独立诊断，不属于 VAPS G3/G4，不能形成动态避障或安全方法结论。

## 目标

先验证当前 UR5、随机目标、SAC 训练和确定性策略执行链路能否学会基础 reaching。作为 v1 背景诊断，冻结 B4 actor 在同一无障碍 200-reset 清单上仅有 `31/600=5.17%` success；该历史问题促成独立 v2 基线，不能用 v2 无障碍结果替代动态障碍安全证据。

## 固定条件

- UR5、随机目标、初始关节状态、20 Hz 控制、12 s episode 上限和 `0.055 m` success tolerance 均沿用现有环境。
- 明确关闭动态障碍物、安全过滤器、viability monitor、诊断日志和所有 recovery/relaxation/maximin 分支。
- 使用 `link_fixed`，但无障碍时风险为零，固定风险惩罚设为零；该名称只复用既有网络/动作平滑接口，不代表链路风险方法比较。
- v1 独立训练 seeds：`4301, 4302, 4303`；每个训练 `200000` environment steps，并保存每 `20000` step checkpoint。v2 使用相同 seeds、`300000` steps 和相同 validation/final manifests。
- checkpoint 仅按 40 个独立 validation seeds `9301--9340` 的确定性 `success_rate` 选择；同分时遵循现有脚本的碰撞、最终误差和 step 规则。
- 最终结果使用未参与选择的 `9001--9200` 无障碍 reset；该清单与失败 B4 诊断相同，因此可直接比较基础到达能力恢复幅度。

## 决策门槛

v1 的恢复门槛为：任一 train seed 在 200-reset final 清单上的 success 不低于 `80%`，且三 seed 平均 success 不低于 `85%`。该门槛已用于确认基础 reaching 链路恢复。

达到 `99%` 时必须分开报告两个口径：全部 200 个 reset 的 success，以及经固定离线任务预检查标记为 IK 可达且无障碍候选路径找到的条件 success。不得删除 reset 或更改 success threshold 伪造全分布成功率。v1 的 `9001--9200` 清单中 `9021、9065、9095、9098、9120、9142` 均在 32 次固定多初值 IK 搜索中未找到候选（不是数学绝对不可达证明），且被三个 actor 共同超时；因此全分布当前上限为 `194/200=97%`，不可能以这一固定清单声明 `99%`。

v2 仅提高无障碍训练的 `action_scale` 至 `1.0 rad/s` 并增加训练预算至 `300000` steps；该限速与已有统一 P3 设置一致。其目标是在不改变最终清单、目标采样、episode 时限或 success threshold 的前提下，使固定可行候选子集（194 reset）每个 train seed 均达到至少 `193/194=99.48%` 成功。全分布仍完整报告，且不把条件子集写成全分布或安全结论。

## v2 实际结果

- 三个独立训练均完成 `300000` environment steps；checkpoint 仍按 40 个独立 validation seeds 的确定性 `success_rate` 选择。
- 选中的 checkpoint：seed 4301 为 step `240000`，seed 4302 为 step `280000`，seed 4303 为 step `220000`；validation success 均为 `39/40=97.5%`。
- 未参与选择的固定最终清单 `9001--9200` 上：seed 4301、4302、4303 均为全量 `194/200=97.0%`，均为条件可达子集 `194/194=100%`。
- 三 seed pooled 结果为全量 `582/600=97.0%`，条件可达子集 `582/582=100%`。
- 三个 actor 的失败 reset 完全一致：`9021、9065、9095、9098、9120、9142`；均为 12 s 超时，未发生 collision、capsule overlap 或 physical contact。
- 该结果达到 v2 条件门槛（每个 seed 至少 `193/194=99.48%`），但受固定清单中 6 个未找到 IK 候选 reset 限制，全量口径不能达到 99%。

## 冻结结论

- **可以冻结**：将三个选中的 v2 actor 作为无障碍基础 reaching 基线，冻结其配置、checkpoint、validation 清单和 final 清单；后续实验不得静默替换 actor 或改变 success threshold。
- **不能冻结为 99% 全分布结果**：应同时保留并报告全量 `97.0%` 与可达条件 `100%`，不能删除 6 个 reset 或只报告条件口径。
- **不能据此冻结安全结论**：该协议关闭动态障碍、安全过滤器、viability monitor 和 recovery/relaxation 分支；结果只证明基础 reaching 执行链路恢复，不代表安全、泛化、OOD、动态避障或真机性能。

## 配置与产物

- 基础配置：`configs/experiments/reaching_recovery/v1_base.yaml`
- seed 配置：`configs/experiments/reaching_recovery/v1_seed430{1,2,3}.yaml`
- v2 配置：`configs/experiments/reaching_recovery/v2_seed430{1,2,3}.yaml`
- 清单：`configs/experiments/reaching_recovery/manifests/`
- v1 输出根目录：`outputs/reaching_recovery_v1/`
- v2 输出根目录：`outputs/reaching_recovery_v2/`
- v2 final CSV：`outputs/reaching_recovery_v2/eval/seed_430{1,2,3}_final.csv`
