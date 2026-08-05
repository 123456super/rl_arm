# 基础 reaching 恢复协议 v1

> 状态：已配置，等待执行。该协议是对冻结 B4 actor 无障碍到达失败的独立诊断，不属于 VAPS G3/G4，不能形成动态避障或安全方法结论。

## 目标

先验证当前 UR5、随机目标、SAC 训练和确定性策略执行链路能否学会基础 reaching。冻结 B4 actor 在同一无障碍 200-reset 清单上仅有 `31/600=5.17%` success；在此问题排除前，不能解释或推进任何动态障碍安全结果。

## 固定条件

- UR5、随机目标、初始关节状态、20 Hz 控制、12 s episode 上限和 `0.055 m` success tolerance 均沿用现有环境。
- 明确关闭动态障碍物、安全过滤器、viability monitor、诊断日志和所有 recovery/relaxation/maximin 分支。
- 使用 `link_fixed`，但无障碍时风险为零，固定风险惩罚设为零；该名称只复用既有网络/动作平滑接口，不代表链路风险方法比较。
- 独立训练 seeds：`4301, 4302, 4303`；每个训练 `200000` environment steps，并保存每 `20000` step checkpoint。
- checkpoint 仅按 40 个独立 validation seeds `9301--9340` 的确定性 `success_rate` 选择；同分时遵循现有脚本的碰撞、最终误差和 step 规则。
- 最终结果使用未参与选择的 `9001--9200` 无障碍 reset；该清单与失败 B4 诊断相同，因此可直接比较基础到达能力恢复幅度。

## 决策门槛

只要任一 train seed 在 200-reset final 清单上的 success 低于 `80%`，或三 seed 平均 success 低于 `85%`，即判定基础 reaching 尚未恢复；只允许继续定位任务奖励、动作尺度或训练稳定性，仍不得运行动态障碍、V2、OOD、真机或 recovery 分支。

即使全部满足门槛，也只授权重新审查后续安全协议；不得把本协议结果写为安全、泛化或动态避障性能结论。

## 配置与产物

- 基础配置：`configs/experiments/reaching_recovery/v1_base.yaml`
- seed 配置：`configs/experiments/reaching_recovery/v1_seed430{1,2,3}.yaml`
- 清单：`configs/experiments/reaching_recovery/manifests/`
- 输出根目录：`outputs/reaching_recovery_v1/`
