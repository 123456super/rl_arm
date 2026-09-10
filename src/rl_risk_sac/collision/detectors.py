from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from rl_risk_sac.robots.ur5_capsules import CapsuleState
from rl_risk_sac.scene import ObstacleState
from rl_risk_sac.utils.risk import LinkRisk, RiskConfig, compute_link_risk


class RiskDetector(Protocol):
    """Geometry backend used by the task environment.

    A detector consumes geometry snapshots rather than simulator handles. This
    keeps observations and rewards independent from PyBullet and leaves a clean
    extension point for GJK/FCL, multiple obstacles, or a real perception stack.

    中文理解：detector 是“几何风险计算接口”。环境先从 PyBullet 读出
    胶囊体和障碍物状态，再把纯 numpy 状态交给 detector；这样以后换
    更真实的碰撞库或视觉检测结果时，不需要重写 SAC。
    """

    def detect(
        self,
        capsules: list[CapsuleState],
        previous_capsules: list[CapsuleState] | None,
        obstacles: Sequence[ObstacleState],
        dt: float | None = None,
    ) -> LinkRisk: ...


@dataclass(frozen=True)
class LinkRiskDetector:
    """Compute link risk against active spherical obstacles.

    end_effector_only=True 时只保留最后一段胶囊体风险，用来实现
    ee_fixed 末端风险基线；否则所有连杆都参与全局风险。
    """

    config: RiskConfig
    obstacle_radius: float
    dt: float
    end_effector_only: bool = False
    no_obstacle_distance: float = 1.5

    def detect(
        self,
        capsules: list[CapsuleState],
        previous_capsules: list[CapsuleState] | None,
        obstacles: Sequence[ObstacleState],
        dt: float | None = None,
    ) -> LinkRisk:
        active_obstacles = [obstacle for obstacle in obstacles if obstacle.enabled]
        if not active_obstacles:
            return _empty_risk(capsules, self.config, self.no_obstacle_distance)

        # 多障碍物时，先分别计算“单个障碍物 vs 所有连杆”的 LinkRisk，
        # 再按连杆聚合出最近距离和最大风险。
        risks = [
            compute_link_risk(
                capsules=capsules,
                prev_capsules=previous_capsules,
                obstacle_center=obstacle.center,
                obstacle_velocity=obstacle.velocity,
                obstacle_radius=self.obstacle_radius,
                dt=self.dt if dt is None else float(dt),
                config=self.config,
                use_end_effector_only=self.end_effector_only,
            )
            for obstacle in active_obstacles
        ]
        return _aggregate_link_risks(risks)


@dataclass(frozen=True)
class NullRiskDetector:
    """Risk detector used when obstacle avoidance is disabled."""

    config: RiskConfig
    no_obstacle_distance: float

    def detect(
        self,
        capsules: list[CapsuleState],
        previous_capsules: list[CapsuleState] | None,
        obstacles: Sequence[ObstacleState],
        dt: float | None = None,
    ) -> LinkRisk:
        del previous_capsules, obstacles, dt
        return _empty_risk(capsules, self.config, self.no_obstacle_distance)


def _aggregate_link_risks(risks: list[LinkRisk]) -> LinkRisk:
    """Merge per-obstacle LinkRisk objects into one scene-level LinkRisk."""
    if len(risks) == 1:
        return risks[0]

    # distance_stack 的形状是 [障碍物数量, 连杆数量]。对每根连杆取最近
    # 的那个障碍物，用它的 closest point/direction/TTC 等几何量。
    distance_stack = np.stack([risk.distances for risk in risks])
    nearest_obstacle_indices = np.argmin(distance_stack, axis=0)
    link_indices = np.arange(distance_stack.shape[1])

    closest_points = np.stack([risk.closest_points for risk in risks])[nearest_obstacle_indices, link_indices]
    distances = distance_stack[nearest_obstacle_indices, link_indices]
    directions = np.stack([risk.directions for risk in risks])[nearest_obstacle_indices, link_indices]
    link_velocities = np.stack([risk.link_velocities for risk in risks])[nearest_obstacle_indices, link_indices]
    approach_velocities = np.stack([risk.approach_velocities for risk in risks])[nearest_obstacle_indices, link_indices]
    ttc = np.stack([risk.ttc for risk in risks])[nearest_obstacle_indices, link_indices]
    per_link_risk = np.max(np.stack([risk.risks for risk in risks]), axis=0)

    # risk_global 取所有连杆风险最大值；d_min 取所有障碍物-连杆组合的
    # 最小表面距离，二者分别服务于控制强度和碰撞/安全判断。
    return LinkRisk(
        closest_points=closest_points.astype(np.float32),
        distances=distances.astype(np.float32),
        directions=directions.astype(np.float32),
        link_velocities=link_velocities.astype(np.float32),
        approach_velocities=approach_velocities.astype(np.float32),
        ttc=ttc.astype(np.float32),
        risks=per_link_risk.astype(np.float32),
        risk_global=float(np.max(per_link_risk)),
        d_min=float(np.min(distance_stack)),
        closest_link=int(np.argmin(distances)),
    )


def _empty_risk(capsules: list[CapsuleState], config: RiskConfig, no_obstacle_distance: float) -> LinkRisk:
    """Return a zero-risk placeholder with the same per-link shape."""
    count = len(capsules)
    return LinkRisk(
        closest_points=np.zeros((count, 3), dtype=np.float32),
        distances=np.full(count, no_obstacle_distance, dtype=np.float32),
        directions=np.zeros((count, 3), dtype=np.float32),
        link_velocities=np.zeros((count, 3), dtype=np.float32),
        approach_velocities=np.zeros(count, dtype=np.float32),
        ttc=np.full(count, config.ttc_max, dtype=np.float32),
        risks=np.zeros(count, dtype=np.float32),
        risk_global=0.0,
        d_min=no_obstacle_distance,
        closest_link=0,
    )
