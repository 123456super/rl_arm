"""机器人几何抽象子包。

当前使用胶囊体近似 UR5 主要连杆，供连杆级风险计算使用。
"""

from rl_risk_sac.robots.ur5_capsules import UR5CapsuleModel

__all__ = ["UR5CapsuleModel"]
