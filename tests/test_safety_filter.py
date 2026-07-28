from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rl_risk_sac.utils.predictive_risk import (
    ObstacleStateEstimate,
    PredictiveRiskConfig,
    compute_predictive_link_risk,
)
from rl_risk_sac.utils.safety_filter import (
    LinearVelocityConstraints,
    SafetyFilterConfig,
    SafetyFilterInput,
    SafetyFilterStatus,
    filter_joint_velocity,
)
from rl_risk_sac.robots.ur5_capsules import CapsuleState


def config() -> SafetyFilterConfig:
    return SafetyFilterConfig(
        joint_velocity_limits_radps=np.asarray([1.0]),
        joint_acceleration_limits_radps2=np.asarray([20.0]),
        joint_position_lower_rad=np.asarray([-1.0]),
        joint_position_upper_rad=np.asarray([1.0]),
        control_dt_s=0.1,
        safety_gain=2.0,
    )


def risk_with_safety_function(value: float):
    capsule = CapsuleState(
        start=np.asarray([0.0, 0.0, 0.0]),
        end=np.asarray([1.0, 0.0, 0.0]),
        radius=0.05,
        name="link",
    )
    risk = compute_predictive_link_risk(
        [capsule],
        link_velocities_mps=np.zeros((1, 3)),
        obstacle=ObstacleStateEstimate(
            position=np.asarray([0.5, 0.5, 0.0]),
            velocity=np.zeros(3),
            radius_m=0.05,
            timestamp_s=1.0,
            position_error_bound_m=0.0,
            velocity_error_bound_mps=0.0,
        ),
        now_s=1.0,
        config=PredictiveRiskConfig(prediction_horizon_s=0.0, control_delay_s=0.0, tracking_error_bound_m=0.0),
    )
    return replace(risk, safety_functions_m=np.asarray([value]))


def filter_input(
    risk,
    requested: float,
    jacobian: float | None = 1.0,
    workspace: LinearVelocityConstraints | None = None,
) -> SafetyFilterInput:
    return SafetyFilterInput(
        requested_joint_velocity_radps=np.asarray([requested]),
        joint_positions_rad=np.asarray([0.0]),
        previous_command_radps=np.asarray([0.0]),
        predictive_risk=risk,
        safety_jacobian_m_per_rad=None if jacobian is None else np.asarray([[jacobian]]),
        workspace_constraints=workspace,
    )


def test_unusable_predictive_risk_forces_zero_velocity_stop() -> None:
    capsule = CapsuleState(np.zeros(3), np.ones(3), 0.05, "link")
    stale_risk = compute_predictive_link_risk(
        [capsule],
        link_velocities_mps=np.zeros((1, 3)),
        obstacle=ObstacleStateEstimate(
            position=np.zeros(3), velocity=np.zeros(3), radius_m=0.05,
            timestamp_s=0.0, position_error_bound_m=0.0, velocity_error_bound_mps=0.0,
        ),
        now_s=1.0,
        config=PredictiveRiskConfig(max_observation_age_s=0.1),
    )

    result = filter_joint_velocity(filter_input(stale_risk, requested=0.8), config())

    assert result.status is SafetyFilterStatus.SAFE_STOP_RISK_UNUSABLE
    assert result.requires_safe_stop
    np.testing.assert_array_equal(result.command_joint_velocity_radps, [0.0])


def test_filter_projects_policy_command_onto_predictive_safety_constraint() -> None:
    result = filter_joint_velocity(filter_input(risk_with_safety_function(-0.1), requested=-0.8), config())

    assert result.status is SafetyFilterStatus.FILTERED
    np.testing.assert_allclose(result.command_joint_velocity_radps, [0.2], atol=1e-12)
    assert result.intervention_norm_radps == pytest.approx(1.0)


def test_filter_enforces_workspace_and_command_continuity_constraints() -> None:
    constrained_config = SafetyFilterConfig(
        joint_velocity_limits_radps=np.asarray([1.0]),
        joint_acceleration_limits_radps2=np.asarray([2.0]),
        joint_position_lower_rad=np.asarray([-1.0]),
        joint_position_upper_rad=np.asarray([1.0]),
        control_dt_s=0.1,
    )
    workspace = LinearVelocityConstraints(matrix=np.asarray([[1.0]]), lower_bound=np.asarray([0.15]))
    result = filter_joint_velocity(
        filter_input(risk_with_safety_function(1.0), requested=0.0, workspace=workspace),
        constrained_config,
    )

    assert result.status is SafetyFilterStatus.FILTERED
    np.testing.assert_allclose(result.command_joint_velocity_radps, [0.15], atol=1e-12)


def test_filter_enforces_next_step_joint_position_limit() -> None:
    result = filter_joint_velocity(
        SafetyFilterInput(
            requested_joint_velocity_radps=np.asarray([1.0]),
            joint_positions_rad=np.asarray([0.99]),
            previous_command_radps=np.asarray([0.0]),
            predictive_risk=risk_with_safety_function(1.0),
            safety_jacobian_m_per_rad=np.asarray([[1.0]]),
        ),
        config(),
    )

    assert result.status is SafetyFilterStatus.FILTERED
    np.testing.assert_allclose(result.command_joint_velocity_radps, [0.1], atol=1e-12)


def test_infeasible_safety_constraint_falls_back_to_safe_stop() -> None:
    result = filter_joint_velocity(filter_input(risk_with_safety_function(-0.1), requested=0.4, jacobian=0.0), config())

    assert result.status is SafetyFilterStatus.SAFE_STOP_INFEASIBLE
    assert result.requires_safe_stop
    np.testing.assert_array_equal(result.command_joint_velocity_radps, [0.0])


def test_missing_safety_jacobian_never_allows_raw_policy_command() -> None:
    result = filter_joint_velocity(filter_input(risk_with_safety_function(1.0), requested=0.4, jacobian=None), config())

    assert result.status is SafetyFilterStatus.SAFE_STOP_INVALID_INPUT
    np.testing.assert_array_equal(result.command_joint_velocity_radps, [0.0])
