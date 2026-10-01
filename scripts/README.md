# Scripts 分类

当前主线为 Hybrid Keypoint + Jacobian + Auto-PCR。

| 子目录 | 内容 |
| --- | --- |
| `core/` | 训练、冻结评估、预检、checkpoint 对比和可视化入口 |
| `replay/` | 四池 replay、bad-state 起点、Frontier/Anchor 构建与审计 |
| `analysis/` | 大姿态 crossing、alignment、状态来源、replay 覆盖和 plateau 分析 |
| `diagnostics/` | Actor/Critic 漂移、plasticity、timeout、控制层和固定参数探针 |
| `calibration/` | keypoint/Jacobian 标定、observation 饱和审计和几何验证 |

从项目根目录运行脚本，例如：

```bash
python scripts/core/train_thesis_homotopy.py ...
python scripts/core/evaluate_thesis_homotopy.py ...
```

各目录包含 `__init__.py`，跨脚本导入使用完整的 `scripts.<category>.<module>` 路径。
