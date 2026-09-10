from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_risk_sac.robots.ur5_capsules import CapsuleState
from rl_risk_sac.utils.runtime_config import RiskConfig


@dataclass
class LinkRisk:
    """Per-link risk snapshot returned by the detector.

    distances 是胶囊表面到球表面的距离；risks 是每根连杆的风险；
    risk_global 是本时刻用于控制/训练的全局最大风险。
    """
    closest_points: np.ndarray
    # distances are conservative capsule gaps used by control and learning.
    # raw_distances retain the uncalibrated capsule gaps for diagnostics.
    distances: np.ndarray
    raw_distances: np.ndarray
    directions: np.ndarray
    link_velocities: np.ndarray
    approach_velocities: np.ndarray
    ttc: np.ndarray
    risks: np.ndarray
    risk_global: float
    d_min: float
    d_min_raw: float
    closest_link: int


def closest_point_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the closest point on a capsule centerline segment."""
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-12:
        return start.copy(), 0.0
    rho = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
    return start + rho * segment, rho


def compute_link_risk(
    capsules: list[CapsuleState],
    prev_capsules: list[CapsuleState] | None,
    obstacle_center: np.ndarray,
    obstacle_velocity: np.ndarray,
    obstacle_radius: float,
    dt: float,
    config: RiskConfig,
    use_end_effector_only: bool = False,
) -> LinkRisk:
    """Compute dynamic risk between robot link capsules and one spherical obstacle.

    核心逻辑对应论文里的连杆级动态风险：先找球心到每根胶囊中心线的最近点，
    再计算表面距离、相对接近速度和 TTC，最后加权融合成 [0, 1] 风险。
    """
    if dt <= 0:
        raise ValueError("dt must be positive")

    count = len(capsules)
    closest_points = np.zeros((count, 3), dtype=np.float32)
    distances = np.zeros(count, dtype=np.float32)
    raw_distances = np.zeros(count, dtype=np.float32)
    directions = np.zeros((count, 3), dtype=np.float32)
    link_velocities = np.zeros((count, 3), dtype=np.float32)
    approach_velocities = np.zeros(count, dtype=np.float32)
    ttc = np.zeros(count, dtype=np.float32)
    risks = np.zeros(count, dtype=np.float32)

    active_indices = [count - 1] if use_end_effector_only else range(count)

    for i, capsule in enumerate(capsules):
        # rho 是最近点在胶囊中心线上的插值比例；上一帧也用同一个 rho，
        # 近似估计“当前最近点对应的连杆局部位置”的速度。
        closest, rho = closest_point_on_segment(obstacle_center, capsule.start, capsule.end)
        delta = obstacle_center - closest
        center_distance = float(np.linalg.norm(delta))
        direction = delta / (center_distance + config.eps)
        raw_surface_distance = center_distance - capsule.radius - obstacle_radius
        surface_distance = raw_surface_distance - config.geometry_margin

        if prev_capsules is not None and i < len(prev_capsules):
            prev = prev_capsules[i]
            prev_fixed = prev.start + rho * (prev.end - prev.start)
            link_velocity = (closest - prev_fixed) / dt
        else:
            link_velocity = np.zeros(3, dtype=np.float32)

        relative_velocity = obstacle_velocity - link_velocity
        # direction 指向“连杆最近点 -> 障碍物球心”。distance_rate < 0
        # 表示两者沿这条线互相靠近，所以 approach_velocity 取负号后截断。
        distance_rate = float(np.dot(direction, relative_velocity))
        approach_velocity = max(0.0, -distance_rate)

        if surface_distance <= config.d_safe:
            ttc_i = 0.0
        elif approach_velocity <= config.eps_v:
            ttc_i = config.ttc_max
        else:
            ttc_i = min((surface_distance - config.d_safe) / approach_velocity, config.ttc_max)

        distance_risk = float(np.clip(np.exp(-(surface_distance - config.d_safe) / config.sigma_d), 0.0, 1.0))
        velocity_risk = float(np.clip(approach_velocity / config.v_max, 0.0, 1.0))
        ttc_risk = float(np.exp(-ttc_i / config.tau))
        # 最终风险是距离风险、接近速度风险和 TTC 风险的加权和。
        risk = float(
            np.clip(
                config.w_distance * distance_risk
                + config.w_velocity * velocity_risk
                + config.w_ttc * ttc_risk,
                0.0,
                1.0,
            )
        )

        closest_points[i] = closest
        distances[i] = surface_distance
        raw_distances[i] = raw_surface_distance
        directions[i] = direction
        link_velocities[i] = link_velocity
        approach_velocities[i] = approach_velocity
        ttc[i] = ttc_i
        risks[i] = risk if i in active_indices else 0.0

    closest_link = int(np.argmin(distances))
    d_min = float(np.min(distances))
    risk_global = float(np.max(risks))
    return LinkRisk(
        closest_points=closest_points,
        distances=distances,
        raw_distances=raw_distances,
        directions=directions,
        link_velocities=link_velocities,
        approach_velocities=approach_velocities,
        ttc=ttc,
        risks=risks,
        risk_global=risk_global,
        d_min=d_min,
        d_min_raw=float(np.min(raw_distances)),
        closest_link=closest_link,
    )
