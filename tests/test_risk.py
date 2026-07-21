from __future__ import annotations

import numpy as np

from rl_risk_sac.robots.ur5_capsules import CapsuleState
from rl_risk_sac.utils.risk import RiskConfig, closest_point_on_segment, compute_link_risk


def capsule(name: str, start: list[float], end: list[float], radius: float = 0.05) -> CapsuleState:
    return CapsuleState(
        start=np.asarray(start, dtype=np.float32),
        end=np.asarray(end, dtype=np.float32),
        radius=radius,
        name=name,
    )


def test_closest_point_clamps_to_segment() -> None:
    point, rho = closest_point_on_segment(
        np.asarray([2.0, 1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    )

    assert rho == 1.0
    np.testing.assert_allclose(point, [1.0, 0.0, 0.0])


def test_approach_velocity_is_positive_when_obstacle_moves_toward_link() -> None:
    risk = compute_link_risk(
        capsules=[capsule("link", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])],
        prev_capsules=None,
        obstacle_center=np.asarray([0.5, 0.4, 0.0], dtype=np.float32),
        obstacle_velocity=np.asarray([0.0, -0.2, 0.0], dtype=np.float32),
        obstacle_radius=0.05,
        dt=0.05,
        config=RiskConfig(d_safe=0.12, ttc_max=3.0),
    )

    assert risk.approach_velocities[0] > 0.0
    assert risk.ttc[0] < 3.0


def test_approach_velocity_is_zero_when_obstacle_moves_away() -> None:
    risk = compute_link_risk(
        capsules=[capsule("link", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])],
        prev_capsules=None,
        obstacle_center=np.asarray([0.5, 0.4, 0.0], dtype=np.float32),
        obstacle_velocity=np.asarray([0.0, 0.2, 0.0], dtype=np.float32),
        obstacle_radius=0.05,
        dt=0.05,
        config=RiskConfig(d_safe=0.12, ttc_max=3.0),
    )

    assert risk.approach_velocities[0] == 0.0
    assert risk.ttc[0] == 3.0


def test_end_effector_only_masks_non_end_link_risk() -> None:
    risk = compute_link_risk(
        capsules=[
            capsule("base_link", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            capsule("tool_link", [0.0, 2.0, 0.0], [1.0, 2.0, 0.0]),
        ],
        prev_capsules=None,
        obstacle_center=np.asarray([0.5, 0.12, 0.0], dtype=np.float32),
        obstacle_velocity=np.asarray([0.0, -0.2, 0.0], dtype=np.float32),
        obstacle_radius=0.05,
        dt=0.05,
        config=RiskConfig(d_safe=0.12),
        use_end_effector_only=True,
    )

    assert risk.closest_link == 0
    assert risk.risks[0] == 0.0
    assert risk.risk_global == risk.risks[1]
