"""碰撞/风险检测子包。

这里导出的 detector 把机械臂胶囊体和障碍物状态转换为 LinkRisk，
环境和 SAC 不需要关心具体几何后端。
"""

from rl_risk_sac.collision.detectors import LinkRiskDetector, NullRiskDetector, RiskDetector

__all__ = ["LinkRiskDetector", "NullRiskDetector", "RiskDetector"]
