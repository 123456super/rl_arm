# 新论文实验配置

正式实验只使用以下五个入口：

| 方法 | 配置 | 训练入口 |
|---|---|---|
| Hybrid KP-SAC-Base | `paper_kp_sac.yaml` | `train_thesis_homotopy_reduced_observation.py` |
| Full | `paper_full.yaml` | `train_thesis_homotopy.py` |
| M1 | `paper_m1_no_bottleneck.yaml` | `train_thesis_homotopy.py` |
| M2 | `paper_m2_no_boundary_replay.yaml` | `train_thesis_homotopy.py` |
| M3 | `paper_m3_no_jacobian.yaml` | `train_thesis_homotopy_reduced_observation.py` |

五个配置使用 L0–L14、6M transition 和统一自动升档协议。所有配置由操作系统自由调度 CPU。统一评估入口是 `scripts/core/evaluate_thesis_homotopy.py`。
