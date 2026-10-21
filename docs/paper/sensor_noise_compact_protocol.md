# 精简鲁棒性实验：L13 位置与姿态测量噪声

正文仅增加两类独立噪声实验和一张双子图，不增加固定偏差、联合噪声、临界区加密或新的训练。原位置噪声先导结果保留；本次在独立、更大的测试任务集上重新评估位置和姿态噪声。

## 冻结设置

- 方法：现有两个 Full 策略，训练 seed 为 11001、11002。
- 容差：仅 L13，位置 1 mm、姿态 0.025 rad。
- 每策略、每条件 1000 回合，100 个任务空间分箱各 10 回合。
- 测试 reset seed 为 41001–42000；与先导实验 81001–81100 分离，也不使用课程 probe/confirmation 的 31001 系列。
- 噪声 seed 为 71001，位置和姿态使用不同 RNG 命名空间。
- 原控制周期 0.05 s、horizon 500 步、连续成功 H=5。
- 从各完整 checkpoint 读取保存的配置，不重新训练或微调策略。

## 仅保留的两个实验

| 实验 | 每个分量的高斯标准差 | 另一测量通道 |
|---|---|---|
| 位置测量噪声 | 0、0.1、0.5、1、2 mm | 姿态准确 |
| 姿态测量噪声 | 0、0.0025、0.005、0.01、0.02 rad | 位置准确 |

每个策略的零噪声对照只运行一次，在两图中共用。因此每策略 9 个独立条件，两个策略共 18 个独立条件 / 18000 回合；汇总 CSV 的两类曲线各有 5 个点，总共 20 行，包含共享零噪声行。

位置噪声模型为

\[
\widetilde p_t=p_t+\eta_{p,t},\quad
\eta_{p,t}\sim\mathcal N(0,\sigma_p^2 I).
\]

姿态噪声模型为世界坐标小旋转左乘：

\[
\widetilde R_t=\exp([\eta_{R,t}]_\times)R_t,\quad
\eta_{R,t}\sim\mathcal N(0,\sigma_R^2 I).
\]

实现采用归一化四元数相乘，不直接给四元数四个分量加噪。标准差指三维平移或旋转向量的每个分量，既不是误差范数也不是误差上界；小角度下三维噪声 RMS 为 `sqrt(3) * sigma`。

噪声每控制周期独立采样，同一步重复读取观察返回同一噪声样本。不同等级复用对应通道的标准正态样本、只改变幅值，且不消耗任务采样 RNG。

## 观察与真值的边界

同一带噪位姿重新生成显式位置/姿态误差向量、误差范数和关键点误差，并进入原归一化和裁剪流程。精度动作缩放也使用这份测量，避免保留理想误差接口。

关节测量、速度、目标位姿和基于关节运动学模型的 Jacobian 不加噪。姿态通道加噪时，关键点误差根据带噪姿态计算，但 Jacobian 仍来自准确的关节模型；不把模型特征误称为另一份姿态传感器测量。此协议只评估特定末端测量通道的扰动，不覆盖全部传感器/模型误差。

物理运动状态、碰撞、安全几何、奖励、连续成功计数和终止判定使用真实仿真状态。`info` 分别记录 `measured_ee_position`、`measured_ee_quaternion`、两个 measured error norm 和原真实诊断字段；Actor 不读取 `info`。

## 指标与正文呈现

主图只画两幅联合成功率曲线：左为位置噪声，右为姿态噪声，每图分别画两个训练种子的结果。全部回合的位置/姿态终点 P95、超时率与碰撞率保存在 CSV/JSON，必要时放补充材料，不再增加正文图。

两个训练种子不能代表充分的训练方差估计。测试回合增多是改善测试任务覆盖，不等于增加训练重复。不含 Base 噪声实验，因此不据此声称相较其他算法的鲁棒性优势。达到连续成功就结束回合，也不证明长期保持精度。

## 复现入口

```bash
conda activate rl
python scripts/core/evaluate_sensor_noise.py \
  --checkpoint \
    outputs/retrain_20261013/full/full_seed11001_6m_l13_resume/checkpoints/step_1475000.pt \
    outputs/retrain_20261013/full/full_seed11002_6m_l13_resume/checkpoints/step_1175000.pt \
  --labels Full_seed11001 Full_seed11002 \
  --seeds 41001 --episodes 1000 --num-envs 40 \
  --output-dir outputs/sensor_noise/compact_L13_20261020
```

输出目录必须不存在，以避免混合或覆盖结果。输出包括：

- `sensor_noise_compact.png` / `.svg`：正文双子图。
- `report.md`：结果表、方法段英文草稿及图注。
- `summary.csv`：逐策略指标，并记录共享零噪声结果文件。
- 逐条件 JSON / log：评估结果与原始运行日志。
- `run.json`：运行参数、源代码哈希及完成状态。

底层单条件入口 `evaluate_thesis_homotopy.py` 新增 `--orientation-noise-std-rad`，位置参数仍为 `--position-noise-std-mm`；两者默认 0。

## 验证

`tests/core/test_orientation_sensor_noise.py` 验证 SO(3) 合法性、噪声尺度、世界坐标旋转、幂等采样、关键点与姿态误差一致性、真值奖励/判定及测量驱动的精度缩放。

本次运行前，姿态噪声、原位置噪声及评估器测试合计 25 项通过。
