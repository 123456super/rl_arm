"""Fail-safe projection filter for joint-velocity commands.

The filter is deliberately independent from PyBullet and the RL environment.
Callers provide the first-order safety and workspace constraint Jacobians for
the current robot state; this module then makes the smallest practical command
correction through cyclic projection onto the resulting half-spaces.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from itertools import combinations

import numpy as np

from rl_risk_sac.utils.predictive_risk import PredictiveLinkRisk


class SafetyFilterStatus(str, Enum):
    PASSTHROUGH = "passthrough"
    FILTERED = "filtered"
    SAFE_STOP_RISK_UNUSABLE = "safe_stop_risk_unusable"
    SAFE_STOP_INVALID_INPUT = "safe_stop_invalid_input"
    SAFE_STOP_INFEASIBLE = "safe_stop_infeasible"
    SAFE_STOP_PROJECTION_FAILED = "safe_stop_projection_failed"
    SAFE_STOP_COMPUTE_BUDGET = "safe_stop_compute_budget"
    RECOVERY_RELAXED = "recovery_relaxed"


class ViabilityStatus(str, Enum):
    """Protocol-level outcome of a strict safety-filter control cycle."""

    CERTIFIED_VIABLE = "certified_viable"
    MODEL_INFEASIBLE_DYNAMIC = "model_infeasible_dynamic"
    MODEL_INFEASIBLE_STATIC_OR_SLOW = "model_infeasible_static_or_slow"
    INVALID_OR_STALE_OBSERVATION = "invalid_or_stale_observation"
    UNKNOWN_COMPUTE_BUDGET = "unknown_compute_budget"


@dataclass(frozen=True)
class SafetyFilterConfig:
    """Joint-command limits and numerical settings for the safety filter."""

    joint_velocity_limits_radps: np.ndarray
    joint_acceleration_limits_radps2: np.ndarray
    joint_position_lower_rad: np.ndarray
    joint_position_upper_rad: np.ndarray
    control_dt_s: float
    safety_gain: float = 4.0
    preemptive_margin_m: float = 0.0
    max_projection_iterations: int = 80
    constraint_tolerance: float = 1e-8
    projection_failure_tolerance: float = 1e-6
    allow_iterative_fallback: bool = True
    active_set_fallback_enabled: bool = True
    active_set_max_candidate_constraints: int = 12
    fallback_projection_iterations: int = 320
    use_qp_solver: bool = False
    qp_time_limit_s: float | None = None
    infeasibility_diagnostics_enabled: bool = False
    goal_velocity_priority_enabled: bool = False
    goal_velocity_priority_solver: str = "linprog"
    goal_velocity_priority_weight: float = 1.0
    goal_velocity_intervention_weight: float = 1.0
    goal_velocity_objective_scale: float = 1.0
    # Optional second-stage lexicographic objective.  The first LP maximizes
    # the TCP velocity in the current goal direction.  When enabled, a second
    # LP keeps that value within a small certified tolerance and selects the
    # command closest to the requested and previous commands.  This preserves
    # all strict safety constraints while avoiding cycle-to-cycle sign flips
    # between equally useful goal-directed solutions.
    goal_velocity_temporal_consistency_enabled: bool = False
    goal_velocity_near_optimal_tolerance_mps: float = 0.0
    goal_velocity_continuity_requested_weight: float = 0.25
    goal_velocity_continuity_previous_weight: float = 1.0


@dataclass(frozen=True)
class LinearVelocityConstraints:
    """Rows of ``matrix @ qdot >= lower_bound`` supplied by robot kinematics."""

    matrix: np.ndarray
    lower_bound: np.ndarray
    labels: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SafetyFilterInput:
    """One controller-cycle input to the post-policy safety layer.

    ``safety_jacobian_m_per_rad`` must contain one row per element of
    ``predictive_risk.safety_functions_m``.  Optional workspace constraints
    have the same lower-bound form and are normally generated from the TCP
    workspace barrier functions by the robot-control integration.
    """

    requested_joint_velocity_radps: np.ndarray
    joint_positions_rad: np.ndarray
    previous_command_radps: np.ndarray
    predictive_risk: PredictiveLinkRisk
    safety_jacobian_m_per_rad: np.ndarray | None
    safety_drift_mps: np.ndarray | None = None
    allow_infeasible_recovery: bool = False
    recovery_target_mask: np.ndarray | None = None
    maximize_min_clearance_recovery: bool = False
    workspace_constraints: LinearVelocityConstraints | None = None
    # Optional TCP goal-direction objective, expressed as a joint-space row
    # (m/s per rad/s). It is only used when the config explicitly enables the
    # goal-velocity-priority candidate.
    goal_velocity_objective: np.ndarray | None = None


@dataclass(frozen=True)
class SafetyFilterResult:
    command_joint_velocity_radps: np.ndarray
    status: SafetyFilterStatus
    reason: str
    intervention_norm_radps: float
    active_constraint_count: int
    max_constraint_violation: float
    constraint_count: int = 0
    active_constraint_categories: tuple[str, ...] = ()
    max_constraint_category: str = ""
    projection_iterations: int = 0
    fallback_used: bool = False
    qp_solver_used: bool = False
    qp_solver_status: str = ""
    fallback_stage: str = ""
    infeasible_constraint_categories: tuple[str, ...] = ()
    infeasibility_diagnostic_status: str = ""
    goal_velocity_primary_mps: float = float("nan")
    goal_velocity_secondary_used: bool = False
    goal_velocity_secondary_status: str = ""

    @property
    def requires_safe_stop(self) -> bool:
        return self.status in {
            SafetyFilterStatus.SAFE_STOP_RISK_UNUSABLE,
            SafetyFilterStatus.SAFE_STOP_INVALID_INPUT,
            SafetyFilterStatus.SAFE_STOP_INFEASIBLE,
            SafetyFilterStatus.SAFE_STOP_PROJECTION_FAILED,
            SafetyFilterStatus.SAFE_STOP_COMPUTE_BUDGET,
        }


@dataclass(frozen=True)
class ViabilityAssessment:
    """Read-only G0 classification derived from an executed filter result.

    ``one_step_strict_sufficient`` denotes the current implemented sufficient
    condition. It does not introduce a multi-step optimizer or alter the
    command selected by :func:`filter_joint_velocity`.
    """

    status: ViabilityStatus
    strict_feasible: bool
    solver_status: str
    model_assumptions_valid: bool


def maximize_linear_velocity(
    filter_input: SafetyFilterInput,
    config: SafetyFilterConfig,
    objective_joint_velocity: np.ndarray,
) -> tuple[float, np.ndarray | None, str]:
    """Maximize a read-only linear velocity objective under filter constraints.

    The objective is expressed in joint velocity coordinates (for example,
    ``goal_direction @ J_tcp``).  This function rebuilds the exact same box,
    workspace and predictive-barrier constraints used by
    :func:`filter_joint_velocity`; it never changes the command selected by
    the filter.  It is intended for offline/diagnostic logging only.

    Returns ``(maximum, optimizer_command, status)``.  ``maximum`` is NaN and
    the command is ``None`` if the diagnostic solver cannot certify a result.
    """

    objective = np.asarray(objective_joint_velocity, dtype=np.float64)
    if objective.ndim != 1:
        return float("nan"), None, "invalid_objective"
    try:
        lower, upper, _, rows, bounds, _, _, _ = _build_constraints(filter_input, config)
    except (ValueError, TypeError):
        return float("nan"), None, "invalid_constraints"
    if objective.size != lower.size or not np.isfinite(objective).all():
        return float("nan"), None, "invalid_objective"

    # scipy is a declared runtime dependency.  linprog handles the exact
    # linear objective and the same finite joint box as the safety filter.
    try:
        from scipy.optimize import linprog
    except ImportError:
        return float("nan"), None, "scipy_unavailable"
    try:
        result = linprog(
            c=-objective,
            A_ub=-rows,
            b_ub=-bounds,
            bounds=list(zip(lower.tolist(), upper.tolist(), strict=True)),
            method="highs",
        )
    except (ValueError, RuntimeError):
        return float("nan"), None, "solver_error"
    status = str(result.message).lower()
    if not bool(result.success) or result.x is None:
        return float("nan"), None, f"{result.status}:{status}"
    command = np.asarray(result.x, dtype=np.float64)
    if command.shape != objective.shape or not np.isfinite(command).all():
        return float("nan"), None, "invalid_solution"
    # Explicitly verify all inequalities using the same tolerances used by
    # the filter.  This prevents an approximate diagnostic result from being
    # reported as a feasible target velocity.
    max_violation = max(
        float(np.max(bounds - rows @ command, initial=0.0)),
        float(np.max(lower - command, initial=0.0)),
        float(np.max(command - upper, initial=0.0)),
    )
    if max_violation > config.projection_failure_tolerance:
        return float("nan"), None, "inaccurate_solution"
    return float(np.dot(objective, command)), command, "optimal"


def assess_strict_viability(
    result: SafetyFilterResult,
    predictive_risk: PredictiveLinkRisk | None,
    safe_stop_drift_class: str = "not_applicable",
) -> ViabilityAssessment:
    """Classify a completed strict-filter cycle without changing its command."""

    observations_usable = predictive_risk is not None and predictive_risk.usable
    if result.status is SafetyFilterStatus.SAFE_STOP_COMPUTE_BUDGET:
        return ViabilityAssessment(
            status=ViabilityStatus.UNKNOWN_COMPUTE_BUDGET,
            strict_feasible=False,
            solver_status="compute_budget_exceeded",
            model_assumptions_valid=False,
        )
    if not observations_usable or result.status is SafetyFilterStatus.SAFE_STOP_RISK_UNUSABLE:
        return ViabilityAssessment(
            status=ViabilityStatus.INVALID_OR_STALE_OBSERVATION,
            strict_feasible=False,
            solver_status="strict_result_unavailable_observation",
            model_assumptions_valid=False,
        )
    if result.status is SafetyFilterStatus.SAFE_STOP_INVALID_INPUT:
        return ViabilityAssessment(
            status=ViabilityStatus.INVALID_OR_STALE_OBSERVATION,
            strict_feasible=False,
            solver_status="strict_result_unavailable_invalid_input",
            model_assumptions_valid=False,
        )
    if result.status is SafetyFilterStatus.SAFE_STOP_PROJECTION_FAILED:
        return ViabilityAssessment(
            status=ViabilityStatus.UNKNOWN_COMPUTE_BUDGET,
            strict_feasible=False,
            solver_status="strict_projection_failed",
            model_assumptions_valid=False,
        )
    if result.status in {SafetyFilterStatus.PASSTHROUGH, SafetyFilterStatus.FILTERED}:
        return ViabilityAssessment(
            status=ViabilityStatus.CERTIFIED_VIABLE,
            strict_feasible=True,
            solver_status="one_step_strict_sufficient",
            model_assumptions_valid=True,
        )
    if result.status is SafetyFilterStatus.SAFE_STOP_INFEASIBLE:
        if safe_stop_drift_class == "dynamic_drift":
            return ViabilityAssessment(
                status=ViabilityStatus.MODEL_INFEASIBLE_DYNAMIC,
                strict_feasible=False,
                solver_status="strict_infeasible_dynamic_drift",
                model_assumptions_valid=True,
            )
        return ViabilityAssessment(
            status=ViabilityStatus.MODEL_INFEASIBLE_STATIC_OR_SLOW,
            strict_feasible=False,
            solver_status="strict_infeasible_static_or_slow",
            model_assumptions_valid=True,
        )
    return ViabilityAssessment(
        status=ViabilityStatus.MODEL_INFEASIBLE_STATIC_OR_SLOW,
        strict_feasible=False,
        solver_status="recovery_relaxed_not_strict",
        model_assumptions_valid=True,
    )


def filter_joint_velocity(
    filter_input: SafetyFilterInput,
    config: SafetyFilterConfig,
) -> SafetyFilterResult:
    """Return a bounded command or a deterministic zero-velocity safe stop.

    Valid predicted risks are enforced through the first-order condition
    ``J_h qdot + h_drift + safety_gain * (h - preemptive_margin) >= 0``.
    ``h_drift`` captures motion independent of the commanded joints, such as
    the obstacle velocity. Position,
    velocity and acceleration limits are always applied before projecting the
    policy command. A failed input validation or failed feasibility check never
    falls back to the raw policy action.
    """

    risk = filter_input.predictive_risk
    requested = np.asarray(filter_input.requested_joint_velocity_radps, dtype=np.float64)
    if risk.requires_safe_stop:
        return _safe_stop(
            requested,
            SafetyFilterStatus.SAFE_STOP_RISK_UNUSABLE,
            f"predictive risk is {risk.status.value}: {risk.status_reason}",
        )

    try:
        lower, upper, requested, rows, bounds, labels, lower_labels, upper_labels = _build_constraints(
            filter_input, config
        )
    except ValueError as error:
        status = (
            SafetyFilterStatus.SAFE_STOP_INFEASIBLE
            if "limits are infeasible" in str(error)
            else SafetyFilterStatus.SAFE_STOP_INVALID_INPUT
        )
        return _safe_stop(requested, status, str(error))

    zero_row_mask = np.linalg.norm(rows, axis=1) <= 1e-14
    positive_zero_rows = zero_row_mask & (bounds > config.projection_failure_tolerance)
    if np.any(positive_zero_rows):
        if filter_input.allow_infeasible_recovery:
            return _relaxed_recovery_command(
                requested,
                lower,
                upper,
                rows,
                bounds,
                labels,
                lower_labels,
                upper_labels,
                config,
                "zero_sensitivity",
                filter_input=filter_input,
            )
        index = int(np.flatnonzero(positive_zero_rows)[0])
        return _safe_stop(
            requested,
            SafetyFilterStatus.SAFE_STOP_INFEASIBLE,
            f"constraint {labels[index]} has zero sensitivity and a positive lower bound",
            constraint_count=len(bounds),
            max_constraint_violation=float(bounds[index]),
            max_constraint_category=labels[index],
        )

    goal_objective = None
    if (
        config.goal_velocity_priority_enabled
        and filter_input.goal_velocity_objective is not None
        and config.goal_velocity_priority_solver == "linprog"
    ):
        goal_objective = _vector(
            filter_input.goal_velocity_objective,
            "goal_velocity_objective",
            requested.size,
        )
        # The priority objective is a linear TCP velocity.  Solve it directly
        # with HiGHS instead of sending a nearly-linear, badly scaled QP to
        # OSQP.  This keeps every hard constraint identical and removes the
        # maximum-iterations fallback that otherwise silently returns the old
        # minimum-intervention projection.
        goal_command, goal_status, goal_primary, goal_secondary_used = _linprog_goal_projection(
            lower,
            upper,
            rows,
            bounds,
            goal_objective,
            config,
            requested=requested,
            previous=filter_input.previous_command_radps,
        )
        if goal_command is not None:
            violations = bounds - rows @ goal_command
            max_violation = float(max(0.0, np.max(violations, initial=0.0)))
            active_count, active_categories, max_category = _constraint_diagnostics(
                goal_command, lower, upper, violations, labels, lower_labels, upper_labels, config
            )
            intervention = float(np.linalg.norm(goal_command - requested))
            status = (
                SafetyFilterStatus.FILTERED
                if intervention > config.constraint_tolerance
                else SafetyFilterStatus.PASSTHROUGH
            )
            return SafetyFilterResult(
                command_joint_velocity_radps=goal_command,
                status=status,
                reason="linprog goal-velocity priority optimal",
                intervention_norm_radps=intervention,
                active_constraint_count=active_count,
                max_constraint_violation=max_violation,
                constraint_count=len(bounds),
                active_constraint_categories=active_categories,
                max_constraint_category=max_category,
                projection_iterations=0,
                qp_solver_used=False,
                qp_solver_status=goal_status,
                fallback_stage="goal_velocity_linprog",
                goal_velocity_primary_mps=goal_primary,
                goal_velocity_secondary_used=goal_secondary_used,
                goal_velocity_secondary_status=(
                    "continuity_tiebreak" if goal_secondary_used else "disabled"
                ),
            )

    qp_status = "disabled"
    if config.use_qp_solver:
        qp_command, qp_status = _osqp_projection(
            requested,
            lower,
            upper,
            rows,
            bounds,
            config,
            goal_objective=goal_objective,
        )
        if qp_command is not None:
            violations = bounds - rows @ qp_command
            max_violation = float(max(0.0, np.max(violations, initial=0.0)))
            active_count, active_categories, max_category = _constraint_diagnostics(
                qp_command, lower, upper, violations, labels, lower_labels, upper_labels, config
            )
            intervention = float(np.linalg.norm(qp_command - requested))
            status = (
                SafetyFilterStatus.FILTERED
                if intervention > config.constraint_tolerance
                else SafetyFilterStatus.PASSTHROUGH
            )
            return SafetyFilterResult(
                command_joint_velocity_radps=qp_command,
                status=status,
                reason=(
                    "osqp goal-velocity priority optimal"
                    if goal_objective is not None
                    else "osqp optimal"
                ),
                intervention_norm_radps=intervention,
                active_constraint_count=active_count,
                max_constraint_violation=max_violation,
                constraint_count=len(bounds),
                active_constraint_categories=active_categories,
                max_constraint_category=max_category,
                projection_iterations=0,
                qp_solver_used=True,
                qp_solver_status=qp_status,
                fallback_stage="osqp",
            )
        if "infeasible" in qp_status:
            infeasible_categories: tuple[str, ...] = ()
            diagnostic_status = "disabled"
            if config.infeasibility_diagnostics_enabled:
                infeasible_categories, diagnostic_status = _diagnose_infeasible_constraint_categories(
                    requested,
                    lower,
                    upper,
                    rows,
                    bounds,
                    labels,
                    lower_labels,
                    upper_labels,
                    config,
                )
            if filter_input.allow_infeasible_recovery:
                return _relaxed_recovery_command(
                    requested,
                    lower,
                    upper,
                    rows,
                    bounds,
                    labels,
                    lower_labels,
                    upper_labels,
                    config,
                    qp_status,
                    filter_input=filter_input,
                )
            return _safe_stop(
                requested,
                SafetyFilterStatus.SAFE_STOP_INFEASIBLE,
                f"OSQP reported {qp_status}",
                constraint_count=len(bounds),
                max_constraint_category="qp_solver",
                qp_solver_status=qp_status,
                fallback_stage="osqp",
                infeasible_constraint_categories=infeasible_categories,
                infeasibility_diagnostic_status=diagnostic_status,
            )

    command = np.clip(requested, lower, upper)
    iterations_used = 0
    for iteration in range(config.max_projection_iterations):
        iterations_used = iteration + 1
        previous = command.copy()
        for row, bound in zip(rows, bounds, strict=True):
            value = float(np.dot(row, command))
            if value < bound:
                row_norm_sq = float(np.dot(row, row))
                if row_norm_sq <= 1e-14:
                    break
                command += ((bound - value) / row_norm_sq) * row
                command = np.clip(command, lower, upper)
        if np.max(np.abs(command - previous)) <= config.constraint_tolerance:
            break

    violations = bounds - rows @ command
    max_violation = float(max(0.0, np.max(violations, initial=0.0)))
    if max_violation > config.projection_failure_tolerance and config.allow_iterative_fallback:
        fallback_stage = ""
        fallback_command = _dykstra_projection(
            requested,
            lower,
            upper,
            rows,
            bounds,
            config,
        )
        if fallback_command is not None:
            fallback_stage = "dykstra"
        if fallback_command is None:
            fallback_command = _active_set_fallback(
                requested,
                lower,
                upper,
                rows,
                bounds,
                labels,
                lower_labels,
                upper_labels,
                command,
                config,
            )
            if fallback_command is not None:
                fallback_stage = "active_set"
        if fallback_command is not None:
            command = fallback_command
            violations = bounds - rows @ command
            max_violation = float(max(0.0, np.max(violations, initial=0.0)))
            active_count, active_categories, max_category = _constraint_diagnostics(
                command, lower, upper, violations, labels, lower_labels, upper_labels, config
            )
            intervention = float(np.linalg.norm(command - requested))
            status = (
                SafetyFilterStatus.FILTERED
                if intervention > config.constraint_tolerance
                else SafetyFilterStatus.PASSTHROUGH
            )
            return SafetyFilterResult(
                command_joint_velocity_radps=command,
                status=status,
                reason="active-set fallback projection",
                intervention_norm_radps=intervention,
                active_constraint_count=active_count,
                max_constraint_violation=max_violation,
                constraint_count=len(bounds),
                active_constraint_categories=active_categories,
                max_constraint_category=max_category,
                projection_iterations=iterations_used,
                fallback_used=True,
                qp_solver_status=qp_status,
                fallback_stage=fallback_stage,
            )

    active_count, active_categories, max_category = _constraint_diagnostics(
        command, lower, upper, violations, labels, lower_labels, upper_labels, config
    )
    if max_violation > config.projection_failure_tolerance:
        if filter_input.allow_infeasible_recovery:
            return _relaxed_recovery_command(
                requested,
                lower,
                upper,
                rows,
                bounds,
                labels,
                lower_labels,
                upper_labels,
                config,
                qp_status,
                iterations_used,
                filter_input,
            )
        return _safe_stop(
            requested,
            SafetyFilterStatus.SAFE_STOP_PROJECTION_FAILED,
            "cyclic projection did not satisfy all constraints within the failure tolerance",
            active_constraint_count=active_count,
            max_constraint_violation=max_violation,
            constraint_count=len(bounds),
            active_constraint_categories=active_categories,
            max_constraint_category=max_category,
            projection_iterations=iterations_used,
            qp_solver_status=qp_status,
            fallback_stage="failed",
        )

    intervention = float(np.linalg.norm(command - requested))
    status = SafetyFilterStatus.FILTERED if intervention > config.constraint_tolerance else SafetyFilterStatus.PASSTHROUGH
    return SafetyFilterResult(
        command_joint_velocity_radps=command,
        status=status,
        reason="",
        intervention_norm_radps=intervention,
        active_constraint_count=active_count,
        max_constraint_violation=max_violation,
        constraint_count=len(bounds),
        active_constraint_categories=active_categories,
        max_constraint_category=max_category,
        projection_iterations=iterations_used,
        qp_solver_status=qp_status,
    )


def _linprog_goal_projection(
    lower: np.ndarray,
    upper: np.ndarray,
    rows: np.ndarray,
    bounds: np.ndarray,
    objective: np.ndarray,
    config: SafetyFilterConfig,
    *,
    requested: np.ndarray,
    previous: np.ndarray,
) -> tuple[np.ndarray | None, str, float, bool]:
    """Maximize a TCP goal-direction velocity over the strict feasible set."""
    try:
        from scipy.optimize import linprog
    except ImportError:
        return None, "scipy_unavailable", float("nan"), False
    try:
        result = linprog(
            c=-np.asarray(objective, dtype=np.float64),
            A_ub=-np.asarray(rows, dtype=np.float64),
            b_ub=-np.asarray(bounds, dtype=np.float64),
            bounds=list(zip(lower.tolist(), upper.tolist(), strict=True)),
            method="highs",
        )
    except (ValueError, RuntimeError) as error:
        return None, f"solver_error:{error}", float("nan"), False
    status = str(result.message).lower()
    if not bool(result.success) or result.x is None:
        return None, f"{result.status}:{status}", float("nan"), False
    primary_command = np.asarray(result.x, dtype=np.float64)
    if primary_command.shape != objective.shape or not np.isfinite(primary_command).all():
        return None, "invalid_solution", float("nan"), False
    primary_value = float(np.dot(objective, primary_command))

    # Optional lexicographic tie-break.  Keep the primary TCP velocity within
    # a certified near-optimal band, then minimize L1 distance to the current
    # request and previous output. This makes the selected command temporally
    # consistent without changing any strict safety row or bound.
    tolerance = float(config.goal_velocity_near_optimal_tolerance_mps)
    use_secondary = bool(config.goal_velocity_temporal_consistency_enabled) and tolerance > 0.0
    requested = np.asarray(requested, dtype=np.float64)
    previous = np.asarray(previous, dtype=np.float64)
    if use_secondary and requested.shape == objective.shape and previous.shape == objective.shape:
        n = objective.size
        eye = np.eye(n, dtype=np.float64)
        zeros = np.zeros((n, n), dtype=np.float64)
        strict_rows = np.hstack((-rows, zeros, zeros))
        q_req_pos = np.hstack((eye, -eye, zeros))
        q_req_neg = np.hstack((-eye, -eye, zeros))
        q_prev_pos = np.hstack((eye, zeros, -eye))
        q_prev_neg = np.hstack((-eye, zeros, -eye))
        near_optimal = np.hstack((-objective.reshape(1, -1), np.zeros((1, 2 * n))))
        A_ub = np.vstack((strict_rows, q_req_pos, q_req_neg, q_prev_pos, q_prev_neg, near_optimal))
        b_ub = np.concatenate(
            (
                -bounds,
                requested,
                -requested,
                previous,
                -previous,
                np.asarray([-(primary_value - tolerance)], dtype=np.float64),
            )
        )
        c = np.concatenate(
            (
                np.zeros(n, dtype=np.float64),
                np.full(n, float(config.goal_velocity_continuity_requested_weight)),
                np.full(n, float(config.goal_velocity_continuity_previous_weight)),
            )
        )
        try:
            secondary = linprog(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=list(zip(lower.tolist(), upper.tolist(), strict=True)) + [(0.0, None)] * (2 * n),
                method="highs",
            )
        except (ValueError, RuntimeError):
            secondary = None
        if secondary is not None and bool(secondary.success) and secondary.x is not None:
            command = np.asarray(secondary.x[:n], dtype=np.float64)
            if command.shape == objective.shape and np.isfinite(command).all():
                max_violation = max(
                    float(np.max(bounds - rows @ command, initial=0.0)),
                    float(np.max(lower - command, initial=0.0)),
                    float(np.max(command - upper, initial=0.0)),
                )
                if max_violation <= config.projection_failure_tolerance:
                    return command, "optimal; continuity_tiebreak", primary_value, True

    command = primary_command
    max_violation = max(
        float(np.max(bounds - rows @ command, initial=0.0)),
        float(np.max(lower - command, initial=0.0)),
        float(np.max(command - upper, initial=0.0)),
    )
    if max_violation > config.projection_failure_tolerance:
        return None, "inaccurate_solution", primary_value, False
    return command, "optimal", primary_value, False


def _constraint_diagnostics(
    command: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    violations: np.ndarray,
    labels: tuple[str, ...],
    lower_labels: tuple[str, ...],
    upper_labels: tuple[str, ...],
    config: SafetyFilterConfig,
) -> tuple[int, tuple[str, ...], str]:
    active_tolerance = max(config.constraint_tolerance * 10.0, 1e-7)
    active_mask = np.abs(violations) <= active_tolerance
    active_categories_list = [labels[index] for index in np.flatnonzero(active_mask)]
    active_categories_list.extend(
        lower_labels[index] for index in range(command.size) if abs(command[index] - lower[index]) <= active_tolerance
    )
    active_categories_list.extend(
        upper_labels[index] for index in range(command.size) if abs(command[index] - upper[index]) <= active_tolerance
    )
    max_category = labels[int(np.argmax(violations))] if len(labels) else ""
    return int(np.count_nonzero(active_mask)), tuple(dict.fromkeys(active_categories_list)), max_category


def _relaxed_recovery_command(
    requested: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rows: np.ndarray,
    bounds: np.ndarray,
    labels: tuple[str, ...],
    lower_labels: tuple[str, ...],
    upper_labels: tuple[str, ...],
    config: SafetyFilterConfig,
    solver_status: str,
    projection_iterations: int = 0,
    filter_input: SafetyFilterInput | None = None,
) -> SafetyFilterResult:
    """Execute an escape command while relaxing only unsatisfiable link barriers.

    This branch is diagnostic recovery, not a safe command: it preserves joint
    and workspace constraints but records the remaining predictive violation.
    """
    hard_mask = np.asarray([not label.startswith("predictive_link_") for label in labels], dtype=bool)
    command = np.clip(requested, lower, upper)
    fallback_stage = "recovery_joint_bounds"
    if filter_input is not None and filter_input.maximize_min_clearance_recovery:
        recovery_command, recovery_status = _maximin_recovery_projection(
            requested,
            lower,
            upper,
            rows,
            bounds,
            labels,
            hard_mask,
            filter_input,
            config,
        )
        if recovery_command is not None:
            command = recovery_command
            fallback_stage = "recovery_maximin"
            solver_status = f"{solver_status}; recovery={recovery_status}"
    if np.any(hard_mask):
        if fallback_stage != "recovery_maximin":
            projected, hard_status = _osqp_projection(
                command,
                lower,
                upper,
                rows[hard_mask],
                bounds[hard_mask],
                config,
            )
            if projected is not None:
                command = projected
                fallback_stage = "recovery_hard_constraints"
            else:
                solver_status = f"{solver_status}; hard_constraints={hard_status}"
        if fallback_stage == "recovery_joint_bounds":
            hard_projected = _dykstra_projection(
                command,
                lower,
                upper,
                rows[hard_mask],
                bounds[hard_mask],
                config,
            )
            if hard_projected is not None:
                command = hard_projected
                fallback_stage = "recovery_hard_constraints_dykstra"
    violations = bounds - rows @ command
    max_violation = float(max(0.0, np.max(violations, initial=0.0)))
    active_count, active_categories, max_category = _constraint_diagnostics(
        command, lower, upper, violations, labels, lower_labels, upper_labels, config
    )
    return SafetyFilterResult(
        command_joint_velocity_radps=command,
        status=SafetyFilterStatus.RECOVERY_RELAXED,
        reason="recovery command executed with predictive link constraints relaxed",
        intervention_norm_radps=float(np.linalg.norm(command - requested)),
        active_constraint_count=active_count,
        max_constraint_violation=max_violation,
        constraint_count=len(bounds),
        active_constraint_categories=active_categories,
        max_constraint_category=max_category,
        projection_iterations=projection_iterations,
        fallback_used=True,
        qp_solver_used=True,
        qp_solver_status=solver_status,
        fallback_stage=fallback_stage,
    )


def _maximin_recovery_projection(
    requested: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rows: np.ndarray,
    bounds: np.ndarray,
    labels: tuple[str, ...],
    hard_mask: np.ndarray,
    filter_input: SafetyFilterInput,
    config: SafetyFilterConfig,
) -> tuple[np.ndarray | None, str]:
    """Maximize the worst endangered-link clearance derivative under hard bounds."""
    try:
        import osqp
        from scipy import sparse
    except ImportError:
        return None, "unavailable"

    safety_rows = np.asarray([label.startswith("predictive_link_") for label in labels], dtype=bool)
    target_mask = filter_input.recovery_target_mask
    if target_mask is None:
        target_mask = np.ones(int(np.count_nonzero(safety_rows)), dtype=bool)
    else:
        target_mask = _vector(target_mask, "recovery_target_mask", int(np.count_nonzero(safety_rows))).astype(bool)
    selected_rows = np.flatnonzero(safety_rows)[target_mask]
    if selected_rows.size == 0:
        return None, "no_target_links"
    drift = (
        np.zeros(int(np.count_nonzero(safety_rows)), dtype=np.float64)
        if filter_input.safety_drift_mps is None
        else _vector(filter_input.safety_drift_mps, "safety_drift_mps", int(np.count_nonzero(safety_rows)))
    )
    selected_drift = drift[target_mask]
    joint_count = requested.size
    objective_weight = 1.0e-4
    p_matrix = sparse.diags(np.concatenate((np.full(joint_count, objective_weight), [0.0])), format="csc")
    q_vector = np.concatenate((-objective_weight * requested, [-1.0]))
    clearance_rows = np.hstack((rows[selected_rows], -np.ones((selected_rows.size, 1))))
    hard_rows = np.hstack((rows[hard_mask], np.zeros((int(np.count_nonzero(hard_mask)), 1))))
    joint_rows = np.hstack((np.eye(joint_count), np.zeros((joint_count, 1))))
    matrix = sparse.csc_matrix(np.vstack((clearance_rows, hard_rows, joint_rows)))
    lower_bound = np.concatenate((-selected_drift, bounds[hard_mask], lower))
    upper_bound = np.concatenate((np.full(selected_rows.size, np.inf), np.full(np.count_nonzero(hard_mask), np.inf), upper))
    problem = osqp.OSQP()
    setup_kwargs = {}
    if config.qp_time_limit_s is not None:
        setup_kwargs["time_limit"] = config.qp_time_limit_s
    problem.setup(
        P=p_matrix,
        q=q_vector,
        A=matrix,
        l=lower_bound,
        u=upper_bound,
        eps_abs=config.projection_failure_tolerance,
        eps_rel=config.projection_failure_tolerance,
        max_iter=max(400, config.fallback_projection_iterations),
        polish=False,
        verbose=False,
        **setup_kwargs,
    )
    result = problem.solve()
    status = str(result.info.status).lower()
    if "solved" not in status or result.x is None:
        return None, status
    command = np.asarray(result.x[:joint_count], dtype=np.float64)
    if command.shape != requested.shape or not np.isfinite(command).all():
        return None, "invalid_solution"
    return command, status


def _osqp_projection(
    requested: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rows: np.ndarray,
    bounds: np.ndarray,
    config: SafetyFilterConfig,
    goal_objective: np.ndarray | None = None,
) -> tuple[np.ndarray | None, str]:
    """Solve the strict QP, optionally prioritizing TCP goal velocity."""
    try:
        import osqp
        from scipy import sparse
    except ImportError:
        return None, "unavailable"

    matrix = sparse.csc_matrix(np.vstack((rows, np.eye(requested.size))))
    lower_bound = np.concatenate((bounds, lower))
    upper_bound = np.concatenate((np.full(len(bounds), np.inf), upper))
    problem = osqp.OSQP()
    setup_kwargs = {}
    if config.qp_time_limit_s is not None:
        setup_kwargs["time_limit"] = config.qp_time_limit_s
    if goal_objective is None:
        p_matrix = sparse.eye(requested.size, format="csc")
        q_vector = -requested
    else:
        objective = _vector(goal_objective, "goal_velocity_objective", requested.size)
        intervention_weight = float(config.goal_velocity_intervention_weight)
        p_matrix = sparse.diags(
            np.full(
                requested.size,
                2.0 * intervention_weight * float(config.goal_velocity_objective_scale),
            ),
            format="csc",
        )
        q_vector = (
            -float(config.goal_velocity_objective_scale)
            * (
                float(config.goal_velocity_priority_weight) * objective
                + 2.0 * intervention_weight * requested
            )
        )
    problem.setup(
        P=p_matrix,
        q=q_vector,
        A=matrix,
        l=lower_bound,
        u=upper_bound,
        eps_abs=config.projection_failure_tolerance,
        eps_rel=config.projection_failure_tolerance,
        max_iter=max(400, config.fallback_projection_iterations),
        polish=False,
        verbose=False,
        **setup_kwargs,
    )
    result = problem.solve()
    status = str(result.info.status).lower()
    if "solved" not in status or result.x is None:
        return None, status
    command = np.asarray(result.x, dtype=np.float64)
    if command.shape != requested.shape or not np.isfinite(command).all():
        return None, "invalid_solution"
    max_violation = max(
        float(np.max(bounds - rows @ command, initial=0.0)),
        float(np.max(lower - command, initial=0.0)),
        float(np.max(command - upper, initial=0.0)),
    )
    if max_violation > config.projection_failure_tolerance:
        return None, "inaccurate_solution"
    return command, status


def _diagnose_infeasible_constraint_categories(
    requested: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rows: np.ndarray,
    bounds: np.ndarray,
    labels: tuple[str, ...],
    lower_labels: tuple[str, ...],
    upper_labels: tuple[str, ...],
    config: SafetyFilterConfig,
) -> tuple[tuple[str, ...], str]:
    """Find categories whose individual relaxation restores QP feasibility.

    This is deliberately an offline diagnostic: each candidate performs one
    additional OSQP solve with no solver time limit. It must remain disabled
    for any timing-sensitive control path.
    """
    row_categories = np.asarray([_constraint_category(label) for label in labels], dtype=object)
    lower_categories = np.asarray([_constraint_category(label) for label in lower_labels], dtype=object)
    upper_categories = np.asarray([_constraint_category(label) for label in upper_labels], dtype=object)
    categories = tuple(dict.fromkeys((*row_categories, *lower_categories, *upper_categories)))
    diagnostic_config = replace(config, qp_time_limit_s=None)
    restoring_categories: list[str] = []

    for category in categories:
        row_mask = row_categories != category
        lower_probe = lower.copy()
        upper_probe = upper.copy()
        lower_probe[lower_categories == category] = -np.inf
        upper_probe[upper_categories == category] = np.inf
        command, _ = _osqp_projection(
            requested,
            lower_probe,
            upper_probe,
            rows[row_mask],
            bounds[row_mask],
            diagnostic_config,
        )
        if command is not None:
            restoring_categories.append(str(category))

    return (
        tuple(restoring_categories),
        "single_category_relaxation" if restoring_categories else "combined_or_unresolved",
    )


def _constraint_category(label: str) -> str:
    if label.startswith("predictive_link_"):
        return "predictive_barrier"
    if label.startswith("workspace_"):
        return "workspace"
    if label.startswith("joint_"):
        for source in ("velocity", "acceleration", "position"):
            if f"_{source}_" in label:
                return f"joint_{source}"
        return "joint_bound"
    return "other"


def _dykstra_projection(
    requested: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rows: np.ndarray,
    bounds: np.ndarray,
    config: SafetyFilterConfig,
) -> np.ndarray | None:
    """Project onto all half-spaces and the joint box with Dykstra's method."""
    identity = np.eye(requested.size, dtype=np.float64)
    matrix = np.vstack((rows, identity, -identity))
    lower_bound = np.concatenate((bounds, lower, -upper))
    row_norms = np.sum(np.square(matrix), axis=1)
    if np.any(row_norms <= 1e-14):
        return None

    command = np.clip(requested, lower, upper).astype(np.float64, copy=True)
    corrections = np.zeros_like(matrix)
    box_correction = np.zeros_like(command)
    for _ in range(config.fallback_projection_iterations):
        previous = command.copy()
        for index, (row, bound) in enumerate(zip(matrix, lower_bound, strict=True)):
            shifted = command + corrections[index]
            violation = float(bound - np.dot(row, shifted))
            projected = shifted + (violation / row_norms[index]) * row if violation > 0.0 else shifted
            corrections[index] = shifted - projected
            command = projected

        shifted = command + box_correction
        projected = np.clip(shifted, lower, upper)
        box_correction = shifted - projected
        command = projected
        residual = lower_bound - matrix @ command
        if np.max(residual, initial=0.0) <= config.projection_failure_tolerance:
            return command
        if np.max(np.abs(command - previous)) <= config.constraint_tolerance:
            break
    return None


def _active_set_fallback(
    requested: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rows: np.ndarray,
    bounds: np.ndarray,
    labels: tuple[str, ...],
    lower_labels: tuple[str, ...],
    upper_labels: tuple[str, ...],
    current: np.ndarray,
    config: SafetyFilterConfig,
) -> np.ndarray | None:
    """Find a feasible closest command on a small active constraint set.

    This fallback is deliberately bounded: it searches only the most violated
    and near-active rows after cyclic projection, so a difficult control step
    cannot turn the real-time filter into an unbounded combinatorial solver.
    """
    if not config.active_set_fallback_enabled:
        return None
    identity = np.eye(requested.size, dtype=np.float64)
    matrix = np.vstack((rows, identity, -identity))
    lower_bound = np.concatenate((bounds, lower, -upper))
    residual = lower_bound - matrix @ current
    active_tolerance = max(config.constraint_tolerance * 10.0, 1e-7)
    candidate_indices = set(np.flatnonzero(np.abs(residual) <= active_tolerance).tolist())
    ranked = np.argsort(-residual)
    candidate_indices.update(ranked[: config.active_set_max_candidate_constraints].tolist())
    candidate = sorted(candidate_indices)
    if len(candidate) > config.active_set_max_candidate_constraints:
        candidate = sorted(
            candidate,
            key=lambda index: float(residual[index]),
            reverse=True,
        )[: config.active_set_max_candidate_constraints]
    candidate = sorted(candidate)

    best: np.ndarray | None = None
    best_distance = float("inf")
    max_active = min(requested.size, len(candidate))
    for count in range(1, max_active + 1):
        for subset in combinations(candidate, count):
            active_matrix = matrix[list(subset)]
            gram = active_matrix @ active_matrix.T
            if np.linalg.matrix_rank(gram, tol=1e-10) < count:
                continue
            try:
                correction = np.linalg.solve(gram, lower_bound[list(subset)] - active_matrix @ requested)
            except np.linalg.LinAlgError:
                continue
            trial = requested + active_matrix.T @ correction
            if not np.isfinite(trial).all():
                continue
            trial_residual = lower_bound - matrix @ trial
            if np.max(trial_residual) > config.projection_failure_tolerance:
                continue
            distance = float(np.linalg.norm(trial - requested))
            if distance < best_distance:
                best = trial
                best_distance = distance
    return best


def _build_constraints(
    filter_input: SafetyFilterInput,
    config: SafetyFilterConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    requested = _vector(filter_input.requested_joint_velocity_radps, "requested_joint_velocity_radps")
    joint_positions = _vector(filter_input.joint_positions_rad, "joint_positions_rad", requested.size)
    previous = _vector(filter_input.previous_command_radps, "previous_command_radps", requested.size)
    velocity_limits = _positive_vector(config.joint_velocity_limits_radps, "joint_velocity_limits_radps", requested.size)
    acceleration_limits = _positive_vector(
        config.joint_acceleration_limits_radps2,
        "joint_acceleration_limits_radps2",
        requested.size,
    )
    position_lower = _vector(config.joint_position_lower_rad, "joint_position_lower_rad", requested.size)
    position_upper = _vector(config.joint_position_upper_rad, "joint_position_upper_rad", requested.size)
    if (position_lower > position_upper).any():
        raise ValueError("joint position lower limits must not exceed upper limits")
    if not np.isfinite(config.control_dt_s) or config.control_dt_s <= 0.0:
        raise ValueError("control_dt_s must be finite and positive")
    if not np.isfinite(config.safety_gain) or config.safety_gain < 0.0:
        raise ValueError("safety_gain must be finite and non-negative")
    if not np.isfinite(config.preemptive_margin_m) or config.preemptive_margin_m < 0.0:
        raise ValueError("preemptive_margin_m must be finite and non-negative")
    if (
        config.max_projection_iterations <= 0
        or config.constraint_tolerance <= 0.0
        or config.projection_failure_tolerance <= 0.0
        or config.active_set_max_candidate_constraints <= 0
        or config.fallback_projection_iterations <= 0
    ):
        raise ValueError("projection settings must be positive")
    if config.qp_time_limit_s is not None and (
        not np.isfinite(config.qp_time_limit_s) or config.qp_time_limit_s <= 0.0
    ):
        raise ValueError("qp_time_limit_s must be finite and positive when set")
    if (
        config.goal_velocity_priority_solver not in {"linprog", "osqp"}
        or not np.isfinite(config.goal_velocity_priority_weight)
        or config.goal_velocity_priority_weight < 0.0
        or not np.isfinite(config.goal_velocity_intervention_weight)
        or config.goal_velocity_intervention_weight <= 0.0
        or not np.isfinite(config.goal_velocity_objective_scale)
        or config.goal_velocity_objective_scale <= 0.0
        or not np.isfinite(config.goal_velocity_near_optimal_tolerance_mps)
        or config.goal_velocity_near_optimal_tolerance_mps < 0.0
        or not np.isfinite(config.goal_velocity_continuity_requested_weight)
        or config.goal_velocity_continuity_requested_weight < 0.0
        or not np.isfinite(config.goal_velocity_continuity_previous_weight)
        or config.goal_velocity_continuity_previous_weight < 0.0
    ):
        raise ValueError("goal velocity objective weights must be finite and valid")

    dt = config.control_dt_s
    lower_candidates = np.vstack(
        (-velocity_limits, previous - acceleration_limits * dt, (position_lower - joint_positions) / dt)
    )
    upper_candidates = np.vstack(
        (velocity_limits, previous + acceleration_limits * dt, (position_upper - joint_positions) / dt)
    )
    lower = np.max(lower_candidates, axis=0)
    upper = np.min(upper_candidates, axis=0)
    lower_sources = ("velocity", "acceleration", "position")
    upper_sources = ("velocity", "acceleration", "position")
    lower_labels = tuple(
        f"joint_{index}_{lower_sources[int(np.argmax(lower_candidates[:, index]))]}_lower"
        for index in range(requested.size)
    )
    upper_labels = tuple(
        f"joint_{index}_{upper_sources[int(np.argmin(upper_candidates[:, index]))]}_upper"
        for index in range(requested.size)
    )
    if (lower > upper).any():
        raise ValueError("joint position, velocity, and acceleration limits are infeasible")

    safety_values = np.asarray(filter_input.predictive_risk.safety_functions_m, dtype=np.float64)
    if safety_values.ndim != 1 or not np.isfinite(safety_values).all():
        raise ValueError("valid predictive risk must provide finite one-dimensional safety functions")
    safety_jacobian = _matrix(filter_input.safety_jacobian_m_per_rad, "safety_jacobian_m_per_rad", safety_values.size, requested.size)
    safety_drift = (
        np.zeros(safety_values.size, dtype=np.float64)
        if filter_input.safety_drift_mps is None
        else _vector(filter_input.safety_drift_mps, "safety_drift_mps", safety_values.size)
    )
    rows = [safety_jacobian]
    effective_safety_values = safety_values - config.preemptive_margin_m
    bounds = [-config.safety_gain * effective_safety_values - safety_drift]
    labels = [f"predictive_link_{index}" for index in range(safety_values.size)]

    if filter_input.workspace_constraints is not None:
        workspace = filter_input.workspace_constraints
        rows.append(_matrix(workspace.matrix, "workspace_constraints.matrix", None, requested.size))
        bounds.append(_vector(workspace.lower_bound, "workspace_constraints.lower_bound", rows[-1].shape[0]))
        workspace_labels = workspace.labels
        if workspace_labels is None:
            workspace_labels = tuple(f"workspace_{index}" for index in range(rows[-1].shape[0]))
        if len(workspace_labels) != rows[-1].shape[0]:
            raise ValueError("workspace_constraints.labels must match workspace constraint rows")
        labels.extend(workspace_labels)

    return (
        lower,
        upper,
        requested,
        np.concatenate(rows),
        np.concatenate(bounds),
        tuple(labels),
        lower_labels,
        upper_labels,
    )


def _vector(value: np.ndarray, name: str, size: int | None = None) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or (size is not None and vector.size != size) or not np.isfinite(vector).all():
        expected = "a finite one-dimensional vector" if size is None else f"a finite vector of length {size}"
        raise ValueError(f"{name} must be {expected}")
    return vector


def _positive_vector(value: np.ndarray, name: str, size: int) -> np.ndarray:
    vector = _vector(value, name, size)
    if (vector <= 0.0).any():
        raise ValueError(f"{name} must be positive")
    return vector


def _matrix(value: np.ndarray | None, name: str, rows: int | None, columns: int) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is required when predictive risk is usable")
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != columns or (rows is not None and matrix.shape[0] != rows):
        expected_rows = "any number of" if rows is None else str(rows)
        raise ValueError(f"{name} must have shape ({expected_rows}, {columns})")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")
    return matrix


def _safe_stop(
    requested: np.ndarray,
    status: SafetyFilterStatus,
    reason: str,
    active_constraint_count: int = 0,
    max_constraint_violation: float = 0.0,
    constraint_count: int = 0,
    active_constraint_categories: tuple[str, ...] = (),
    max_constraint_category: str = "",
    projection_iterations: int = 0,
    qp_solver_status: str = "",
    fallback_stage: str = "",
    infeasible_constraint_categories: tuple[str, ...] = (),
    infeasibility_diagnostic_status: str = "",
) -> SafetyFilterResult:
    requested_array = np.asarray(requested, dtype=np.float64)
    command = np.zeros_like(requested_array) if requested_array.ndim == 1 else np.zeros(0, dtype=np.float64)
    intervention = float(np.linalg.norm(requested_array)) if requested_array.ndim == 1 and np.isfinite(requested_array).all() else float("nan")
    return SafetyFilterResult(
        command_joint_velocity_radps=command,
        status=status,
        reason=reason,
        intervention_norm_radps=intervention,
        active_constraint_count=active_constraint_count,
        max_constraint_violation=max_constraint_violation,
        constraint_count=constraint_count,
        active_constraint_categories=active_constraint_categories,
        max_constraint_category=max_constraint_category,
        projection_iterations=projection_iterations,
        qp_solver_status=qp_solver_status,
        fallback_stage=fallback_stage,
        infeasible_constraint_categories=infeasible_constraint_categories,
        infeasibility_diagnostic_status=infeasibility_diagnostic_status,
    )
