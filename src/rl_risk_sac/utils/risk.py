from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_risk_sac.robots.ur5_capsules import CapsuleState


@dataclass
class RiskConfig:
    d_safe: float = 0.12
    sigma_d: float = 0.12
    v_max: float = 0.7
    ttc_max: float = 3.0
    tau: float = 1.0
    eps: float = 1e-8
    eps_v: float = 1e-4
    w_distance: float = 0.5
    w_velocity: float = 0.2
    w_ttc: float = 0.3


@dataclass
class LinkRisk:
    closest_points: np.ndarray
    distances: np.ndarray
    directions: np.ndarray
    link_velocities: np.ndarray
    approach_velocities: np.ndarray
    ttc: np.ndarray
    risks: np.ndarray
    risk_global: float
    d_min: float
    closest_link: int


def closest_point_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, float]:
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
    if dt <= 0:
        raise ValueError("dt must be positive")

    count = len(capsules)
    closest_points = np.zeros((count, 3), dtype=np.float32)
    distances = np.zeros(count, dtype=np.float32)
    directions = np.zeros((count, 3), dtype=np.float32)
    link_velocities = np.zeros((count, 3), dtype=np.float32)
    approach_velocities = np.zeros(count, dtype=np.float32)
    ttc = np.zeros(count, dtype=np.float32)
    risks = np.zeros(count, dtype=np.float32)

    active_indices = [count - 1] if use_end_effector_only else range(count)

    for i, capsule in enumerate(capsules):
        closest, rho = closest_point_on_segment(obstacle_center, capsule.start, capsule.end)
        delta = obstacle_center - closest
        center_distance = float(np.linalg.norm(delta))
        direction = delta / (center_distance + config.eps)
        surface_distance = center_distance - capsule.radius - obstacle_radius

        if prev_capsules is not None and i < len(prev_capsules):
            prev = prev_capsules[i]
            prev_fixed = prev.start + rho * (prev.end - prev.start)
            link_velocity = (closest - prev_fixed) / dt
        else:
            link_velocity = np.zeros(3, dtype=np.float32)

        relative_velocity = obstacle_velocity - link_velocity
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
        directions=directions,
        link_velocities=link_velocities,
        approach_velocities=approach_velocities,
        ttc=ttc,
        risks=risks,
        risk_global=risk_global,
        d_min=d_min,
        closest_link=closest_link,
    )
