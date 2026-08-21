from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import rl_risk_sac.utils.safety_filter as safety_filter_module
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
    ViabilityStatus,
    _dykstra_projection,
    assess_strict_viability,
    filter_joint_velocity,
    maximize_linear_velocity,
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
    drift: float | None = None,
    workspace: LinearVelocityConstraints | None = None,
) -> SafetyFilterInput:
    return SafetyFilterInput(
        requested_joint_velocity_radps=np.asarray([requested]),
        joint_positions_rad=np.asarray([0.0]),
        previous_command_radps=np.asarray([0.0]),
        predictive_risk=risk,
        safety_jacobian_m_per_rad=None if jacobian is None else np.asarray([[jacobian]]),
        safety_drift_mps=None if drift is None else np.asarray([drift]),
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


def test_goal_velocity_priority_keeps_hard_constraints_and_selects_forward_command() -> None:
    result = filter_joint_velocity(
        replace(
            filter_input(risk_with_safety_function(-0.1), requested=-0.8),
            goal_velocity_objective=np.asarray([1.0]),
        ),
        replace(
            config(),
            use_qp_solver=True,
            goal_velocity_priority_enabled=True,
            goal_velocity_priority_weight=1.0,
            goal_velocity_intervention_weight=1.0,
        ),
    )

    assert result.status is SafetyFilterStatus.FILTERED
    assert result.reason == "linprog goal-velocity priority optimal"
    np.testing.assert_allclose(result.command_joint_velocity_radps, [1.0], atol=3e-6)


def test_goal_velocity_temporal_tiebreak_stays_near_previous_command() -> None:
    result = filter_joint_velocity(
        replace(
            filter_input(risk_with_safety_function(-0.1), requested=-0.8),
            previous_command_radps=np.asarray([0.2]),
            goal_velocity_objective=np.asarray([1.0]),
        ),
        replace(
            config(),
            goal_velocity_priority_enabled=True,
            goal_velocity_temporal_consistency_enabled=True,
            goal_velocity_near_optimal_tolerance_mps=0.2,
            goal_velocity_continuity_previous_weight=1.0,
            goal_velocity_continuity_requested_weight=0.25,
        ),
    )

    assert result.goal_velocity_secondary_used is True
    assert result.qp_solver_status == "optimal; continuity_tiebreak"
    np.testing.assert_allclose(result.command_joint_velocity_radps, [0.8], atol=3e-6)


def test_preemptive_margin_enforces_an_earlier_predictive_constraint() -> None:
    preemptive_config = replace(config(), preemptive_margin_m=0.05)

    result = filter_joint_velocity(
        filter_input(risk_with_safety_function(0.02), requested=-0.8),
        preemptive_config,
    )

    assert result.status is SafetyFilterStatus.FILTERED
    np.testing.assert_allclose(result.command_joint_velocity_radps, [0.06], atol=1e-12)


def test_filter_compensates_for_closing_obstacle_drift() -> None:
    result = filter_joint_velocity(
        filter_input(risk_with_safety_function(0.1), requested=0.0, drift=-0.3),
        config(),
    )

    assert result.status is SafetyFilterStatus.FILTERED
    np.testing.assert_allclose(result.command_joint_velocity_radps, [0.1], atol=1e-12)


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


def test_maximize_linear_velocity_uses_the_same_strict_constraints() -> None:
    risk = risk_with_safety_function(-0.1)
    input_data = filter_input(risk, requested=-0.8, jacobian=1.0)
    maximum, command, status = maximize_linear_velocity(input_data, config(), np.asarray([1.0]))

    assert status == "optimal"
    assert maximum == pytest.approx(1.0)
    np.testing.assert_allclose(command, [1.0], atol=1e-9)


def test_maximize_linear_velocity_reports_infeasible_constraints() -> None:
    risk = risk_with_safety_function(-1.0)
    input_data = filter_input(risk, requested=0.0, jacobian=0.0)
    maximum, command, status = maximize_linear_velocity(input_data, config(), np.asarray([1.0]))

    assert np.isnan(maximum)
    assert command is None
    assert status.startswith("2:") or status in {"solver_error", "inaccurate_solution"}


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


def test_viability_classifies_strict_infeasibility_by_safe_stop_drift() -> None:
    risk = risk_with_safety_function(-0.1)
    result = filter_joint_velocity(filter_input(risk, requested=0.4, jacobian=0.0), config())

    static_assessment = assess_strict_viability(result, risk, "static_or_slow")
    dynamic_assessment = assess_strict_viability(result, risk, "dynamic_drift")

    assert static_assessment.status is ViabilityStatus.MODEL_INFEASIBLE_STATIC_OR_SLOW
    assert dynamic_assessment.status is ViabilityStatus.MODEL_INFEASIBLE_DYNAMIC
    assert static_assessment.strict_feasible is False
    assert dynamic_assessment.strict_feasible is False


def test_recovery_relaxed_result_is_never_certified_viable() -> None:
    risk = risk_with_safety_function(-0.1)
    result = filter_joint_velocity(
        replace(filter_input(risk, requested=0.4, jacobian=0.0), allow_infeasible_recovery=True),
        config(),
    )

    assessment = assess_strict_viability(result, risk, "dynamic_drift")

    assert result.status is SafetyFilterStatus.RECOVERY_RELAXED
    assert assessment.status is ViabilityStatus.MODEL_INFEASIBLE_STATIC_OR_SLOW
    assert assessment.strict_feasible is False


def test_strict_projection_does_not_enter_iterative_fallback() -> None:
    strict_config = replace(config(), allow_iterative_fallback=False, max_projection_iterations=1)
    result = filter_joint_velocity(
        filter_input(risk_with_safety_function(-0.1), requested=-0.8),
        strict_config,
    )

    assert result.status is SafetyFilterStatus.FILTERED
    assert result.fallback_stage == ""
    assert result.fallback_used is False
    assert result.projection_iterations == 1


def test_primal_infeasible_osqp_status_forces_safe_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        safety_filter_module,
        "_osqp_projection",
        lambda *_args, **_kwargs: (None, "primal infeasible inaccurate"),
    )

    result = filter_joint_velocity(
        filter_input(risk_with_safety_function(1.0), requested=0.4),
        replace(config(), use_qp_solver=True),
    )

    assert result.status is SafetyFilterStatus.SAFE_STOP_INFEASIBLE
    assert result.qp_solver_status == "primal infeasible inaccurate"
    np.testing.assert_array_equal(result.command_joint_velocity_radps, [0.0])


def test_osqp_infeasibility_diagnostics_identify_predictive_barrier() -> None:
    result = filter_joint_velocity(
        filter_input(risk_with_safety_function(-1.0), requested=0.4),
        replace(config(), use_qp_solver=True, infeasibility_diagnostics_enabled=True),
    )

    assert result.status is SafetyFilterStatus.SAFE_STOP_INFEASIBLE
    assert set(result.infeasible_constraint_categories) == {"predictive_barrier", "joint_velocity"}
    assert result.infeasibility_diagnostic_status == "single_category_relaxation"


def test_primal_infeasible_osqp_uses_bounded_recovery_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def osqp_projection(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return None, "primal infeasible"
        return np.asarray([0.2]), "solved"

    monkeypatch.setattr(safety_filter_module, "_osqp_projection", osqp_projection)
    workspace = LinearVelocityConstraints(matrix=np.asarray([[1.0]]), lower_bound=np.asarray([0.1]))
    result = filter_joint_velocity(
        replace(
            filter_input(risk_with_safety_function(-0.1), requested=0.4, workspace=workspace),
            allow_infeasible_recovery=True,
            maximize_min_clearance_recovery=False,
        ),
        replace(config(), use_qp_solver=True),
    )

    assert calls == 2
    assert result.status is SafetyFilterStatus.RECOVERY_RELAXED
    assert result.fallback_stage == "recovery_hard_constraints"
    assert result.projection_iterations == 0
    assert int(result.projection_iterations) == 0
    np.testing.assert_allclose(result.command_joint_velocity_radps, [0.2])


def test_infeasible_recovery_relaxes_only_the_predictive_constraint() -> None:
    result = filter_joint_velocity(
        replace(filter_input(risk_with_safety_function(-0.1), requested=0.4, jacobian=0.0), allow_infeasible_recovery=True),
        config(),
    )

    assert result.status is SafetyFilterStatus.RECOVERY_RELAXED
    np.testing.assert_allclose(result.command_joint_velocity_radps, [0.4], atol=1e-12)
    assert result.max_constraint_category == "predictive_link_0"


def test_maximin_recovery_uses_joint_limited_escape_command() -> None:
    limited_config = replace(
        config(),
        joint_velocity_limits_radps=np.asarray([0.2]),
        qp_time_limit_s=0.02,
    )
    result = filter_joint_velocity(
        replace(
            filter_input(risk_with_safety_function(-0.1), requested=0.0, jacobian=1.0, drift=-0.3),
            allow_infeasible_recovery=True,
            recovery_target_mask=np.asarray([True]),
            maximize_min_clearance_recovery=True,
        ),
        limited_config,
    )

    assert result.status is SafetyFilterStatus.RECOVERY_RELAXED
    assert result.fallback_stage == "recovery_maximin"
    np.testing.assert_allclose(result.command_joint_velocity_radps, [0.2], atol=1e-6)


def test_projection_residual_is_reported_separately_from_confirmed_infeasibility() -> None:
    constrained_config = SafetyFilterConfig(
        joint_velocity_limits_radps=np.asarray([1.0]),
        joint_acceleration_limits_radps2=np.asarray([2.0]),
        joint_position_lower_rad=np.asarray([-1.0]),
        joint_position_upper_rad=np.asarray([1.0]),
        control_dt_s=0.1,
    )
    workspace = LinearVelocityConstraints(matrix=np.asarray([[1.0]]), lower_bound=np.asarray([0.5]))
    risk = risk_with_safety_function(1.0)
    result = filter_joint_velocity(
        filter_input(risk, requested=0.0, workspace=workspace),
        constrained_config,
    )

    assert result.status is SafetyFilterStatus.SAFE_STOP_PROJECTION_FAILED
    assert result.requires_safe_stop
    assert result.max_constraint_category == "workspace_0"
    assert result.constraint_count == 2
    assessment = assess_strict_viability(result, risk)
    assert assessment.status is ViabilityStatus.UNKNOWN_COMPUTE_BUDGET
    assert assessment.strict_feasible is False


def test_result_reports_limiting_joint_box_constraints() -> None:
    result = filter_joint_velocity(filter_input(risk_with_safety_function(1.0), requested=1.0), config())

    assert any(category.startswith("joint_0_") for category in result.active_constraint_categories)


def test_dykstra_projection_finds_a_feasible_box_constrained_command() -> None:
    result = _dykstra_projection(
        requested=np.asarray([0.0]),
        lower=np.asarray([-1.0]),
        upper=np.asarray([1.0]),
        rows=np.asarray([[1.0], [-1.0]]),
        bounds=np.asarray([0.5, -0.8]),
        config=config(),
    )

    assert result is not None
    np.testing.assert_allclose(result, [0.5], atol=1e-6)


def test_solver_status_is_preserved_when_fallback_fails() -> None:
    constrained_config = SafetyFilterConfig(
        joint_velocity_limits_radps=np.asarray([1.0]),
        joint_acceleration_limits_radps2=np.asarray([20.0]),
        joint_position_lower_rad=np.asarray([-1.0]),
        joint_position_upper_rad=np.asarray([1.0]),
        control_dt_s=0.1,
        use_qp_solver=True,
    )
    result = filter_joint_velocity(filter_input(risk_with_safety_function(1.0), requested=0.0), constrained_config)

    assert result.qp_solver_status in {"solved", "unavailable"}


def test_missing_safety_jacobian_never_allows_raw_policy_command() -> None:
    result = filter_joint_velocity(filter_input(risk_with_safety_function(1.0), requested=0.4, jacobian=None), config())

    assert result.status is SafetyFilterStatus.SAFE_STOP_INVALID_INPUT
    np.testing.assert_array_equal(result.command_joint_velocity_radps, [0.0])
