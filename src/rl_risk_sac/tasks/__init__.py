"""任务定义子包。

reaching 负责 observation/reward/cost，targets 负责目标点的静态或动态轨迹。
"""

from rl_risk_sac.tasks.reaching import ReachingObservationBuilder, ReachingObjective
from rl_risk_sac.tasks.targets import TargetProvider, TargetState, WorkspaceTargetProvider

__all__ = [
    "ReachingObservationBuilder",
    "ReachingObjective",
    "TargetProvider",
    "TargetState",
    "WorkspaceTargetProvider",
]
