# UR5 Serial Curriculum

本项目只保留一条正式训练线路：

`S0 到达与姿态精度课程 -> S1 静态障碍物 -> S2 动态障碍物`

唯一正式配置是 `configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml`，
训练入口是 `scripts/core/train_thesis_homotopy.py`。Hybrid Keypoint + Jacobian + Auto-PCR
与四池 replay 的实际配置见
[方案文档](docs/experiments_9/four_pool_hybrid_keypoint_jacobian_scheme.md)。

## 环境

```bash
conda env create -f environment.yml
conda activate rl
python -m pip install -e . --no-deps
```

测试使用 `/home/c211/anaconda3/envs/rl/bin/python` 对应的 `rl` 环境。

## 训练与评估

```bash
python scripts/core/train_thesis_homotopy.py \
  --config configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml \
  --stage s0 --run-name s0_seed11001
```

S0 的四池续训和 L1 升档命令见上述方案文档。冻结评估使用：

```bash
python scripts/core/evaluate_thesis_homotopy.py \
  --config configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml \
  --checkpoint /absolute/path/to/actor_step_XXXXXXX.pt \
  --level-index N --scene none \
  --episodes 1000 \
  --output outputs/eval/latest.json
```

评估固定覆盖 10×10 目标空间网格，报告总体/最弱网格成功率、碰撞、限位、
超时和自碰撞安全距离；默认自动使用可用物理核心，也可用 `--num-envs N`
手动限制并行度。固定 seed 下的任务语义与原串行评估一致。

## 初始化 actor

可选的 V13.4 到达 actor 保存在
`artifacts/initialization/v13_5_reaching_actor.pt`，其来源和 SHA-256 记录在同目录
的 provenance JSON 中。它只用于 actor-only 初始化，不恢复旧训练体系或旧 replay。

## 代码结构

- `src/rl_risk_sac/envs/thesis_homotopy_env.py`：统一的到达、静态障碍和动态障碍环境。
- `src/rl_risk_sac/algorithms/homotopy_curriculum.py`：S0 精度课程及 S1/S2 gate。
- `src/rl_risk_sac/algorithms/homotopy_replay.py`：按场景、精度档位和语义格采样的 replay。
- `src/rl_risk_sac/algorithms/thesis_sac.py`：标准 reward-only SAC。
- `tests/`：串行协议、replay、环境和工具测试。
