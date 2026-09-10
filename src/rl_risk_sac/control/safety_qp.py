from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from rl_risk_sac.utils.trajectory_blending import quintic_smoothstep
from rl_risk_sac.utils.runtime_config import SafetyQPConfig


@dataclass(frozen=True)
class SafetyQPResult:
    command: np.ndarray
    nominal_command: np.ndarray
    correction: np.ndarray
    correction_norm: float
    slack: np.ndarray
    max_slack: float
    intervened: bool
    infeasible: bool
    active_constraints: int
    solve_time_ms: float


class SafetyQP:
    """Small velocity-space safety QP with bound constraints.

    Solves the projection form

        min ||qdot - qdot_nom||^2
        s.t. A qdot <= b
             lower <= qdot <= upper

    The implementation is intentionally tiny for P4.1: it iteratively projects
    onto the most violated half-space and clips velocity bounds. Remaining
    violation is reported as slack so callers can log infeasibility.
    """

    def __init__(self, config: SafetyQPConfig) -> None:
        self.config = config

    def solve(
        self,
        nominal_command: np.ndarray,
        constraint_matrix: np.ndarray,
        constraint_bound: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
    ) -> SafetyQPResult:
        start_time = perf_counter()
        nominal = np.asarray(nominal_command, dtype=np.float32)
        lower = np.asarray(lower_bound, dtype=np.float32)
        upper = np.asarray(upper_bound, dtype=np.float32)
        if nominal.shape != lower.shape or nominal.shape != upper.shape:
            raise ValueError("nominal_command, lower_bound, and upper_bound must have the same shape")

        matrix = np.asarray(constraint_matrix, dtype=np.float32)
        bound = np.asarray(constraint_bound, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("constraint_matrix must be a 2D array")
        if matrix.shape[1] != nominal.shape[0]:
            raise ValueError("constraint_matrix column count must match command dimension")
        if bound.shape != (matrix.shape[0],):
            raise ValueError("constraint_bound shape must match constraint rows")
        if np.any(lower > upper):
            raise ValueError("lower_bound cannot exceed upper_bound")

        command = np.clip(nominal, lower, upper).astype(np.float32)
        for _ in range(self.config.max_iterations):
            command = np.clip(command, lower, upper).astype(np.float32)
            if matrix.shape[0] == 0:
                break
            violation = matrix @ command - bound
            worst_index = int(np.argmax(violation))
            worst = float(violation[worst_index])
            if worst <= self.config.violation_tolerance:
                break
            normal = matrix[worst_index]
            denom = float(np.dot(normal, normal))
            if denom <= self.config.eps:
                break
            command = command - (worst / denom) * normal

        command = np.clip(command, lower, upper).astype(np.float32)
        slack = np.maximum(matrix @ command - bound, 0.0).astype(np.float32) if matrix.shape[0] else np.zeros(0)
        correction = (command - nominal).astype(np.float32)
        correction_norm = float(np.linalg.norm(correction))
        max_slack = float(np.max(slack)) if len(slack) else 0.0
        intervened = correction_norm > self.config.violation_tolerance
        infeasible = max_slack > self.config.violation_tolerance
        solve_time_ms = (perf_counter() - start_time) * 1000.0

        return SafetyQPResult(
            command=command,
            nominal_command=nominal.copy(),
            correction=correction,
            correction_norm=correction_norm,
            slack=slack,
            max_slack=max_slack,
            intervened=intervened,
            infeasible=infeasible,
            active_constraints=int(np.count_nonzero(slack <= self.config.violation_tolerance))
            if matrix.shape[0]
            else 0,
            solve_time_ms=solve_time_ms,
        )


def distance_rate_constraint(
    distance_jacobian: np.ndarray,
    distance: float,
    safe_distance: float,
    gain: float,
) -> tuple[np.ndarray, float]:
    """Build A qdot <= b from d_dot + gain * (d - d_safe) >= 0."""

    jacobian = np.asarray(distance_jacobian, dtype=np.float32)
    margin = float(distance) - float(safe_distance)
    return -jacobian, float(gain) * margin


def quintic_endpoint_bounds(
    previous_velocity: np.ndarray,
    previous_acceleration: np.ndarray,
    physics_dt: float,
    sample_count: int,
    max_acceleration: float | None = None,
    max_jerk: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Endpoint bounds that enforce finite-difference limits on every quintic substep."""
    previous_velocity = np.asarray(previous_velocity, dtype=np.float64)
    previous_acceleration = np.asarray(previous_acceleration, dtype=np.float64)
    if previous_velocity.shape != previous_acceleration.shape:
        raise ValueError("previous velocity and acceleration must have the same shape")
    if physics_dt <= 0.0 or sample_count <= 0:
        raise ValueError("physics_dt and sample_count must be positive")
    if max_acceleration is not None and max_acceleration <= 0.0:
        raise ValueError("max_acceleration must be positive or null")
    if max_jerk is not None and max_jerk <= 0.0:
        raise ValueError("max_jerk must be positive or null")

    phase = np.arange(1, sample_count + 1, dtype=np.float64) / float(sample_count)
    blend = np.concatenate(([0.0], np.asarray(quintic_smoothstep(phase), dtype=np.float64)))
    acceleration_coeff = np.diff(blend) / physics_dt
    lower_delta = np.full(previous_velocity.shape, -np.inf, dtype=np.float64)
    upper_delta = np.full(previous_velocity.shape, np.inf, dtype=np.float64)

    def intersect(coeff: float, offset: np.ndarray, limit: float) -> None:
        nonlocal lower_delta, upper_delta
        low = (-limit - offset) / coeff
        high = (limit - offset) / coeff
        lower_delta = np.maximum(lower_delta, np.minimum(low, high))
        upper_delta = np.minimum(upper_delta, np.maximum(low, high))

    if max_acceleration is not None:
        for coeff in acceleration_coeff:
            if abs(coeff) > 1e-12:
                intersect(float(coeff), np.zeros_like(previous_velocity), float(max_acceleration))
    if max_jerk is not None:
        jerk_coeff = np.diff(np.concatenate(([0.0], acceleration_coeff))) / physics_dt
        for index, coeff in enumerate(jerk_coeff):
            if abs(coeff) <= 1e-12:
                continue
            offset = -previous_acceleration / physics_dt if index == 0 else np.zeros_like(previous_velocity)
            intersect(float(coeff), offset, float(max_jerk))

    if np.any(lower_delta > upper_delta):
        raise ValueError("acceleration and jerk limits have no feasible quintic endpoint")
    return (
        (previous_velocity + lower_delta).astype(np.float32),
        (previous_velocity + upper_delta).astype(np.float32),
    )
