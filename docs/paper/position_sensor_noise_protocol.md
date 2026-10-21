# 冻结策略的末端位置观测噪声实验

本实验只测试末端位置测量不准确时的表现，不训练或微调策略，不施加机械扰动。姿态、关节状态、速度、机器人模型及目标位姿保持原有设置；因此结果不代表编码器噪声、视觉全位姿误差或真机精度。

## 噪声模型与信息边界

每个控制周期（原协议为 0.05 s），在世界坐标系生成一次三维高斯平移测量误差：

\[
\tilde p_t=p_t+\eta_t,\qquad
\eta_t\sim\mathcal N(0,\sigma_p^2 I_3).
\]

`sigma_p` 是每个坐标轴的标准差，不是三维误差范数；三维误差的理论 RMS 为 `sqrt(3) * sigma_p`。默认测试 0、0.1、0.5、1、2 mm。这些等级是仿真压力测试设置，不声明对应某款真实传感器规格。

同一份 `tilde p_t` 同时生成显式位置误差向量、误差范数和三个关键点的位置误差。关键点误差由现有 observation builder 重新计算，再归一化、裁剪。位置噪声不改变旋转或来自关节运动学的 Jacobian。精度动作缩放使用带噪位置误差和原本准确的姿态误差。

仿真物理状态、奖励、碰撞、安全几何、连续成功计数、结束条件以及输出中的位置/姿态误差均使用真实状态。`info` 额外提供 `measured_ee_position`、`measured_position_error_norm` 和 `position_sensor_noise_m` 供审计。策略不能通过 `info` 获取真实误差。原观察中保留的准确关节状态、速度等信息不受本项实验干扰；这是单独扰动末端平移测量通道的实验。

实现位置：

- `src/rl_risk_sac/envs/position_sensor_noise.py`：有单位的噪声生成。
- 两个 `thesis_homotopy_env*.py` 主/精简环境：策略观察与真实评估分开。
- `scripts/core/evaluate_thesis_homotopy.py`：单条件评估参数。
- `scripts/core/evaluate_position_sensor_noise.py`：完整扫描、汇总 CSV、PNG/SVG 曲线。

## 配对与复现

每个测试回合沿用已有评估器的 `reset_seed = test_seed + episode_index` 和固定任务空间分箱。同一个策略在各噪声等级下复用相同初始状态和目标；噪声不消耗任务采样 RNG。不同测试 seed 块的起点必须至少相隔 `episodes`，避免 reset seed 区间重叠。

噪声样本由 `(sensor_noise_seed, reset_seed, control_step)` 确定。同一步重复读取观察不会产生新样本；不同标准差使用相同标准正态样本，只有幅值不同。不同策略/档位/等级成功时间可以不同，所以每个回合实际使用的样本长度可以不同。

扫描器默认从完整 checkpoint 读取训练时保存的 resolved config；也可显式指定 `--config` 覆盖。逐条件 JSON 记录 checkpoint SHA256、源配置 SHA256、训练 seed、测试 seed、噪声 seed 和实际评估容差。原有普通评估入口默认噪声为 0；训练配置没有 `evaluation_sensor_noise` 时不开启噪声。

## 指标与解释

- 真实位置和姿态同时满足原阈值、连续 H=5 步的成功率。
- 全部回合（包括失败回合）的真实终点位置和姿态 P95。
- 碰撞率、超时率与平均回合步数。

所有档位使用同一冻结 checkpoint，显式改变评估容差及其动作缩放接口。L6 为 10 mm / 0.10 rad，L13 为 1 mm / 0.025 rad。与论文中使用训练途中特定档位 checkpoint 的表格比较时，应先确认 checkpoint 相同；本实验零噪声数值不保证等于原表数值。

当前仍在满足原成功条件时结束回合，因此不证明到达后的长期保持精度。需要长期保持结论时，应另加固定保持窗口。

不同测试 seed 只评估测试任务差异，不能作为独立训练 seed。曲线按各训练策略分别绘制，不将测试回合伪装成训练重复，也不以先导结果推断某组件的因果贡献。

## 运行方法

项目环境的 Python 位于 `/home/c211/anaconda3/envs/rl/bin/python`；也可先 `conda activate rl`。

单条件示例（将 checkpoint 路径替换为待测策略）：

```bash
python scripts/core/evaluate_thesis_homotopy.py \
  --checkpoint PATH_TO_COMPLETE_CHECKPOINT \
  --checkpoint-config --level-index 13 \
  --position-noise-std-mm 0.5 --sensor-noise-seed 71001 \
  --episodes 1000 --seed 41001 --num-envs 16 \
  --output outputs/position_sensor_noise/single_condition.json
```

正式扫描示例：

```bash
python scripts/core/evaluate_position_sensor_noise.py \
  --checkpoint \
    outputs/retrain_20261013/full/full_seed11001_6m_l13_resume/checkpoints/step_1475000.pt \
    outputs/retrain_20261013/full/full_seed11002_6m_l13_resume/checkpoints/step_1175000.pt \
  --labels Full_seed11001 Full_seed11002 \
  --levels 6 13 --noise-std-mm 0 0.1 0.5 1 2 \
  --seeds 41001 51001 61001 --episodes 1000 --num-envs 16 \
  --output-dir outputs/position_sensor_noise/full_two_seeds_final
```

输出目录必须不存在，防止覆盖或混合实验。扫描按条件顺序运行，每个条件内部并行仿真。全部策略使用同一套测试任务；添加 Base 等策略时，传入其完整 checkpoint，扫描器会自动选择相应的观察实现。

先导实验使用 `seed=81001`、每条件 100 回合、两个 Full 训练 seed，覆盖 100 个任务空间单元一次，共 20 个条件 / 2000 回合；其输出位于 `outputs/position_sensor_noise/pilot_20261020`。这是调通流程和观察趋势的结果，尚不能替代正式的多任务集、大样本鲁棒性结果。
