"""Fail-safe projection filter for joint-velocity commands.

The filter is deliberately independent from PyBullet and the RL environment.
Callers provide the first-order safety and workspace constraint Jacobians for
the current robot state; this module then makes the smallest practical command
correction through cyclic projection onto the resulting half-spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from rl_risk_sac.utils.predictive_risk import PredictiveLinkRisk


class SafetyFilterStatus(str, Enum):
    PASSTHROUGH = "passthrough"
    FILTERED = "filtered"
    SAFE_STOP_RISK_UNUSABLE = "safe_stop_risk_unusable"
    SAFE_STOP_INVALID_INPUT = "safe_stop_invalid_input"
    SAFE_STOP_INFEASIBLE = "safe_stop_infeasible"


@dataclass(frozen=True)
class SafetyFilterConfig:
    """Joint-command limits and numerical settings for the safety filter."""

    joint_velocity_limits_radps: np.ndarray
    joint_acceleration_limits_radps2: np.ndarray
    joint_position_lower_rad: np.ndarray
    joint_position_upper_rad: np.ndarray
    control_dt_s: float
    safety_gain: float = 4.0
    max_projection_iterations: int = 80
    constraint_tolerance: float = 1e-8


@dataclass(frozen=True)
class LinearVelocityConstraints:
    """Rows of ``matrix @ qdot >= lower_bound`` supplied by robot kinematics."""

    matrix: np.ndarray
    lower_bound: np.ndarray


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
    workspace_constraints: LinearVelocityConstraints | None = None


@dataclass(frozen=True)
class SafetyFilterResult:
    command_joint_velocity_radps: np.ndarray
    status: SafetyFilterStatus
    reason: str
    intervention_norm_radps: float
    active_constraint_count: int
    max_constraint_violation: float

    @property
    def requires_safe_stop(self) -> bool:
        return self.status in {
            SafetyFilterStatus.SAFE_STOP_RISK_UNUSABLE,
            SafetyFilterStatus.SAFE_STOP_INVALID_INPUT,
            SafetyFilterStatus.SAFE_STOP_INFEASIBLE,
        }


def filter_joint_velocity(
    filter_input: SafetyFilterInput,
    config: SafetyFilterConfig,
) -> SafetyFilterResult:
    """Return a bounded command or a deterministic zero-velocity safe stop.

    Valid predicted risks are enforced through the first-order condition
    ``J_h qdot + safety_gain * h >= 0``.  Position, velocity and acceleration
    limits are always applied before projecting the policy command.  A failed
    input validation or failed feasibility check never falls back to the raw
    policy action.
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
        lower, upper, requested, rows, bounds = _build_constraints(filter_input, config)
    except ValueError as error:
        return _safe_stop(requested, SafetyFilterStatus.SAFE_STOP_INVALID_INPUT, str(error))

    command = np.clip(requested, lower, upper)
    for _ in range(config.max_projection_iterations):
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
    if max_violation > config.constraint_tolerance:
        return _safe_stop(
            requested,
            SafetyFilterStatus.SAFE_STOP_INFEASIBLE,
            "joint, workspace, or predictive-safety constraints are infeasible",
            active_constraint_count=len(bounds),
            max_constraint_violation=max_violation,
        )

    intervention = float(np.linalg.norm(command - requested))
    status = SafetyFilterStatus.FILTERED if intervention > config.constraint_tolerance else SafetyFilterStatus.PASSTHROUGH
    return SafetyFilterResult(
        command_joint_velocity_radps=command,
        status=status,
        reason="",
        intervention_norm_radps=intervention,
        active_constraint_count=len(bounds),
        max_constraint_violation=max_violation,
    )


def _build_constraints(
    filter_input: SafetyFilterInput,
    config: SafetyFilterConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    if config.max_projection_iterations <= 0 or config.constraint_tolerance <= 0.0:
        raise ValueError("projection settings must be positive")

    dt = config.control_dt_s
    lower = np.maximum.reduce((-velocity_limits, previous - acceleration_limits * dt, (position_lower - joint_positions) / dt))
    upper = np.minimum.reduce((velocity_limits, previous + acceleration_limits * dt, (position_upper - joint_positions) / dt))
    if (lower > upper).any():
        raise ValueError("joint position, velocity, and acceleration limits are infeasible")

    safety_values = np.asarray(filter_input.predictive_risk.safety_functions_m, dtype=np.float64)
    if safety_values.ndim != 1 or not np.isfinite(safety_values).all():
        raise ValueError("valid predictive risk must provide finite one-dimensional safety functions")
    safety_jacobian = _matrix(filter_input.safety_jacobian_m_per_rad, "safety_jacobian_m_per_rad", safety_values.size, requested.size)
    rows = [safety_jacobian]
    bounds = [-config.safety_gain * safety_values]

    if filter_input.workspace_constraints is not None:
        workspace = filter_input.workspace_constraints
        rows.append(_matrix(workspace.matrix, "workspace_constraints.matrix", None, requested.size))
        bounds.append(_vector(workspace.lower_bound, "workspace_constraints.lower_bound", rows[-1].shape[0]))

    return lower, upper, requested, np.concatenate(rows), np.concatenate(bounds)


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
    )
