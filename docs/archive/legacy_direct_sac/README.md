# Legacy direct-SAC 归档

> 归档日期：2026-08-19  
> 文档快照来源提交：`ab73e567035c194cc2a61563419aadf2e90aad5a`  
> 状态：只读历史基线，不是当前实验协议

本归档对应架构调整前由 `link_fixed` direct actor 直接输出完整关节速度的 S0/S1 路线。它保留失败与改进的真实过程，用于解释为什么论文改为“规划与确定性收敛基座 + risk-conditioned Residual SAC + predictive safety filter”，而不是继续搜索 direct-SAC 参数。

## 内容

1. [旧文档快照](documents/README.md)：从上述 Git 提交逐字提取的旧入口、研究状态、协议、失败分析、论文主题和指标口径。
2. [核心结果副本](results/README.md)：8 份 JSON summary 和 1 份详细失败报告；同时区分历史最好记录与当前磁盘复跑。
3. [架构转向证据](evidence/architecture_transition_evidence.md)：将历史观测映射到新架构模块，并明确证据边界。
4. [S0 基础 reaching 历史基线](evidence/s0_reaching_baseline.md)：冻结结果、checkpoint 和适用边界。
5. [历史资产索引](evidence/asset_index.md)：旧配置、manifest、checkpoint、CSV、trace 和诊断产物的原路径。
6. [结果来源与校验](evidence/result_provenance.md)：结果批次、来源和 SHA-256。

## 使用边界

- `521/600=86.83%` 是旧文档记录的 candidate terminal refine 历史最好结果；当前磁盘同名路径的后续复跑是 `511/600=85.17%`。两者不是同一份 summary。
- R3/R4 的 blind 结果相同不能证明 hard-case curriculum 无效，因为 selected checkpoint 仍处于恢复 warm-up 范围，而且对应 actor 参数相同。
- 历史结果支持改变控制职责和设计消融，不证明新 hierarchical 方法已经超过旧基线，也不证明动态障碍物性能或混合状态机的递归安全性。
