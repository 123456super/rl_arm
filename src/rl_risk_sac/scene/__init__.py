"""场景对象状态机子包。

目标之外的动态对象，例如球形障碍物，在这里维护位置、速度和采样规则。
"""

from rl_risk_sac.scene.obstacles import ObstacleProvider, ObstacleState, SphericalObstacleProvider

__all__ = ["ObstacleProvider", "ObstacleState", "SphericalObstacleProvider"]
