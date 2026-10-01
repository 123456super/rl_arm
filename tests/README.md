# Tests 分类

| 子目录 | 内容 |
| --- | --- |
| `core/` | 当前主线环境、162D observation、SAC、课程、replay、串行交接和冻结评估测试 |
| `analysis/` | 大姿态分析、bad-state replay 审计、plateau 和 plasticity 分析脚本测试 |
| `infrastructure/` | risk、安全 QP、轨迹平滑、seed 和底层工具测试 |

从项目根目录运行：

```bash
PYTHONPATH=src python -m pytest tests/core tests/analysis tests/infrastructure
```
