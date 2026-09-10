from __future__ import annotations

from dataclasses import replace

import numpy as np

from rl_risk_sac.robots.ur5_capsules import CapsuleState
from rl_risk_sac.utils.predictive_risk import PredictiveRiskConfig, compute_predictive_link_risk
from rl_risk_sac.utils.risk import RiskConfig, compute_link_risk
from rl_risk_sac.utils.config import load_config
from rl_risk_sac.utils.runtime_config import RuntimeConfig


RUNTIME_CONFIG = RuntimeConfig.from_mapping(load_config("configs/default.yaml"))


def risk_config(**updates) -> RiskConfig:
    return replace(RUNTIME_CONFIG.risk, **updates)


def predictive_config(**updates) -> PredictiveRiskConfig:
    return replace(RUNTIME_CONFIG.env.observation.predictive_risk, **updates)


def capsule(name: str, start: list[float], end: list[float], radius: float = 0.05) -> CapsuleState:
    return CapsuleState(
        start=np.asarray(start, dtype=np.float32),
        end=np.asarray(end, dtype=np.float32),
        radius=radius,
        name=name,
    )


def test_static_obstacle_and_static_link_matches_current_distance() -> None:
    link = capsule("link", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    obstacle_center = np.asarray([0.5, 0.4, 0.0], dtype=np.float32)
    obstacle_velocity = np.zeros(3, dtype=np.float32)

    current = compute_link_risk(
        capsules=[link],
        prev_capsules=None,
        obstacle_center=obstacle_center,
        obstacle_velocity=obstacle_velocity,
        obstacle_radius=0.05,
        dt=0.05,
        config=risk_config(d_safe=0.12),
    )
    predicted = compute_predictive_link_risk(
        capsules=[link],
        prev_capsules=None,
        obstacle_center=obstacle_center,
        obstacle_velocity=obstacle_velocity,
        obstacle_radius=0.05,
        dt=0.05,
        config=predictive_config(horizon=1.0, step=0.05, d_safe=0.12),
    )

    np.testing.assert_allclose(predicted.d_pred, current.distances, atol=1e-6)
    assert np.isinf(predicted.t_enter[0])
    assert predicted.critical_link == 0


def test_approaching_obstacle_has_smaller_predicted_distance_and_finite_enter_time() -> None:
    link = capsule("link", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    predicted = compute_predictive_link_risk(
        capsules=[link],
        prev_capsules=None,
        obstacle_center=np.asarray([0.5, 0.4, 0.0], dtype=np.float32),
        obstacle_velocity=np.asarray([0.0, -0.4, 0.0], dtype=np.float32),
        obstacle_radius=0.05,
        dt=0.05,
        config=predictive_config(horizon=1.0, step=0.02, d_safe=0.12),
    )

    assert predicted.d_pred[0] <= 0.3
    assert np.isfinite(predicted.t_enter[0])
    assert predicted.risk_pred_per_link[0] > 0.0
    assert predicted.risk_pred_body == predicted.risk_pred_per_link[0]


def test_receding_obstacle_has_lower_risk_than_approaching_obstacle() -> None:
    link = capsule("link", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    config = predictive_config(horizon=1.0, step=0.05, d_safe=0.12)
    kwargs = dict(
        capsules=[link],
        prev_capsules=None,
        obstacle_center=np.asarray([0.5, 0.25, 0.0], dtype=np.float32),
        obstacle_radius=0.05,
        dt=0.05,
        config=config,
    )

    approaching = compute_predictive_link_risk(
        obstacle_velocity=np.asarray([0.0, -0.2, 0.0], dtype=np.float32),
        **kwargs,
    )
    receding = compute_predictive_link_risk(
        obstacle_velocity=np.asarray([0.0, 0.2, 0.0], dtype=np.float32),
        **kwargs,
    )

    assert approaching.risk_pred_per_link[0] > receding.risk_pred_per_link[0]
    assert approaching.d_pred[0] < receding.d_pred[0]


def test_same_current_distance_different_approach_speed_changes_risk() -> None:
    link = capsule("link", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    config = predictive_config(horizon=0.5, step=0.05, d_safe=0.12)
    kwargs = dict(
        capsules=[link],
        prev_capsules=None,
        obstacle_center=np.asarray([0.5, 0.35, 0.0], dtype=np.float32),
        obstacle_radius=0.05,
        dt=0.05,
        config=config,
    )

    slow = compute_predictive_link_risk(
        obstacle_velocity=np.asarray([0.0, -0.1, 0.0], dtype=np.float32),
        **kwargs,
    )
    fast = compute_predictive_link_risk(
        obstacle_velocity=np.asarray([0.0, -0.5, 0.0], dtype=np.float32),
        **kwargs,
    )

    assert fast.risk_pred_per_link[0] > slow.risk_pred_per_link[0]
    assert fast.d_pred[0] < slow.d_pred[0]


def test_end_effector_only_masks_non_end_link_predictive_risk() -> None:
    predicted = compute_predictive_link_risk(
        capsules=[
            capsule("base_link", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            capsule("tool_link", [0.0, 2.0, 0.0], [1.0, 2.0, 0.0]),
        ],
        prev_capsules=None,
        obstacle_center=np.asarray([0.5, 0.25, 0.0], dtype=np.float32),
        obstacle_velocity=np.asarray([0.0, -0.2, 0.0], dtype=np.float32),
        obstacle_radius=0.05,
        dt=0.05,
        config=predictive_config(horizon=1.0, step=0.05, d_safe=0.12),
        use_end_effector_only=True,
    )

    assert predicted.critical_link == 1
    assert predicted.risk_pred_per_link[0] == 0.0
    assert predicted.risk_pred_body == predicted.risk_pred_per_link[1]


def test_predictive_geometry_margin_preserves_raw_distance() -> None:
    kwargs = dict(
        capsules=[capsule("link", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])],
        prev_capsules=None,
        obstacle_center=np.asarray([0.5, 0.4, 0.0], dtype=np.float32),
        obstacle_velocity=np.zeros(3, dtype=np.float32),
        obstacle_radius=0.05,
        dt=0.05,
    )
    nominal = compute_predictive_link_risk(
        config=predictive_config(horizon=0.0, geometry_margin=0.0), **kwargs
    )
    conservative = compute_predictive_link_risk(
        config=predictive_config(horizon=0.0, geometry_margin=0.04), **kwargs
    )

    assert conservative.d_pred_raw is not None
    np.testing.assert_allclose(conservative.d_pred_raw, nominal.d_pred, atol=1e-7)
    np.testing.assert_allclose(conservative.d_pred, nominal.d_pred - 0.04, atol=1e-7)
    assert conservative.risk_pred_body >= nominal.risk_pred_body
