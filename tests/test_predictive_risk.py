from __future__ import annotations

import numpy as np
import pytest

from rl_risk_sac.robots.ur5_capsules import CapsuleState
from rl_risk_sac.utils.predictive_risk import (
    ObstacleStateEstimate,
    PredictionStatus,
    PredictiveRiskConfig,
    compute_predictive_link_risk,
)


def capsule(start: list[float], end: list[float], radius: float = 0.05) -> CapsuleState:
    return CapsuleState(
        start=np.asarray(start, dtype=np.float64),
        end=np.asarray(end, dtype=np.float64),
        radius=radius,
        name="link",
    )


def estimate(**overrides: object) -> ObstacleStateEstimate:
    values: dict[str, object] = {
        "position": np.asarray([0.5, 0.5, 0.0]),
        "velocity": np.asarray([0.0, -0.2, 0.0]),
        "radius_m": 0.05,
        "timestamp_s": 10.0,
        "position_error_bound_m": 0.01,
        "velocity_error_bound_mps": 0.01,
        "valid": True,
    }
    values.update(overrides)
    return ObstacleStateEstimate(**values)  # type: ignore[arg-type]


def test_prediction_accounts_for_relative_link_motion() -> None:
    result = compute_predictive_link_risk(
        [capsule([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])],
        link_velocities_mps=np.asarray([[0.0, 0.1, 0.0]]),
        obstacle=estimate(position_error_bound_m=0.0, velocity_error_bound_mps=0.0),
        now_s=10.0,
        config=PredictiveRiskConfig(prediction_horizon_s=1.0, max_link_speed_mps=0.0),
    )

    assert result.status is PredictionStatus.VALID
    np.testing.assert_allclose(result.predicted_distances_m, [0.10], atol=1e-12)
    np.testing.assert_allclose(result.closest_prediction_times_s, [1.0], atol=1e-12)
    np.testing.assert_allclose(result.link_velocities_mps, [[0.0, 0.1, 0.0]])
    assert result.max_link_speed_bound_mps == 0.0


def test_robust_clearance_subtracts_all_documented_margins() -> None:
    result = compute_predictive_link_risk(
        [capsule([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])],
        link_velocities_mps=np.zeros((1, 3)),
        obstacle=estimate(),
        now_s=10.1,
        config=PredictiveRiskConfig(
            d_safe_m=0.12,
            prediction_horizon_s=1.0,
            control_delay_s=0.05,
            max_link_speed_mps=0.3,
            tracking_error_bound_m=0.02,
            geometry_margin_m=0.01,
        ),
    )

    # Predicted surface distance is 0.20 m.  Margins are geometry=0.01,
    # perception=0.02, delay=(0.20 + 0.01 + 0.30) * 0.15, tracking=0.02.
    assert result.perception_margin_m == pytest.approx(0.02)
    assert result.delay_margin_m == pytest.approx(0.0765)
    np.testing.assert_allclose(result.robust_distances_m, [0.0735], atol=1e-12)
    np.testing.assert_allclose(result.safety_functions_m, [-0.0465], atol=1e-12)


def test_per_link_geometry_margins_are_preserved() -> None:
    result = compute_predictive_link_risk(
        [
            capsule([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            capsule([0.0, 2.0, 0.0], [1.0, 2.0, 0.0]),
        ],
        link_velocities_mps=np.zeros((2, 3)),
        obstacle=estimate(position_error_bound_m=0.0, velocity_error_bound_mps=0.0),
        now_s=10.0,
        config=PredictiveRiskConfig(
            prediction_horizon_s=0.0,
            control_delay_s=0.0,
            max_link_speed_mps=0.0,
            tracking_error_bound_m=0.0,
            geometry_margin_m=(0.01, 0.03),
        ),
    )

    np.testing.assert_allclose(result.geometry_margins_m, [0.01, 0.03])
    np.testing.assert_allclose(result.robust_distances_m, result.predicted_distances_m - [0.01, 0.03])


def test_prediction_uses_the_closest_time_inside_the_horizon() -> None:
    result = compute_predictive_link_risk(
        [capsule([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])],
        link_velocities_mps=np.zeros((1, 3)),
        obstacle=estimate(
            position=np.asarray([0.5, 0.5, 0.0]),
            velocity=np.asarray([0.0, -1.0, 0.0]),
            radius_m=0.05,
            position_error_bound_m=0.0,
            velocity_error_bound_mps=0.0,
        ),
        now_s=10.0,
        config=PredictiveRiskConfig(
            prediction_horizon_s=1.0,
            control_delay_s=0.0,
            max_link_speed_mps=0.0,
            tracking_error_bound_m=0.0,
        ),
    )

    np.testing.assert_allclose(result.closest_prediction_times_s, [0.5], atol=1e-12)
    np.testing.assert_allclose(result.predicted_distances_m, [-0.10], atol=1e-12)


def test_stale_observation_requires_safe_stop() -> None:
    result = compute_predictive_link_risk(
        [capsule([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])],
        link_velocities_mps=np.zeros((1, 3)),
        obstacle=estimate(),
        now_s=10.11,
        config=PredictiveRiskConfig(max_observation_age_s=0.10),
    )

    assert result.status is PredictionStatus.STALE
    assert result.requires_safe_stop
    assert np.isneginf(result.safety_functions_m).all()


@pytest.mark.parametrize(
    ("obstacle", "reason"),
    [
        (estimate(valid=False), "marked invalid"),
        (estimate(timestamp_s=10.1), "future"),
        (estimate(position=np.asarray([np.nan, 0.0, 0.0])), "must be finite"),
    ],
)
def test_invalid_observations_never_produce_a_low_risk_result(
    obstacle: ObstacleStateEstimate, reason: str
) -> None:
    result = compute_predictive_link_risk(
        [capsule([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])],
        link_velocities_mps=np.zeros((1, 3)),
        obstacle=obstacle,
        now_s=10.0,
        config=PredictiveRiskConfig(),
    )

    assert result.status is PredictionStatus.INVALID
    assert reason in result.status_reason
    assert result.requires_safe_stop
    assert np.isneginf(result.safety_functions_m).all()


def test_rejects_mismatched_geometry_margin_count() -> None:
    with pytest.raises(ValueError, match="contain 1 values"):
        compute_predictive_link_risk(
            [capsule([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])],
            link_velocities_mps=np.zeros((1, 3)),
            obstacle=estimate(),
            now_s=10.0,
            config=PredictiveRiskConfig(geometry_margin_m=(0.01, 0.02)),
        )
