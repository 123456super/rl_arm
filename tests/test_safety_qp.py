from __future__ import annotations

import numpy as np

from rl_risk_sac.control import SafetyQP, SafetyQPConfig, distance_rate_constraint


def test_safety_qp_passes_through_safe_nominal_command() -> None:
    qp = SafetyQP()
    result = qp.solve(
        nominal_command=np.asarray([-0.2, 0.0], dtype=np.float32),
        constraint_matrix=np.asarray([[1.0, 0.0]], dtype=np.float32),
        constraint_bound=np.asarray([0.1], dtype=np.float32),
        lower_bound=np.asarray([-0.7, -0.7], dtype=np.float32),
        upper_bound=np.asarray([0.7, 0.7], dtype=np.float32),
    )

    np.testing.assert_allclose(result.command, [-0.2, 0.0])
    assert not result.intervened
    assert not result.infeasible
    assert result.max_slack == 0.0


def test_safety_qp_modifies_dangerous_approaching_command() -> None:
    qp = SafetyQP()
    result = qp.solve(
        nominal_command=np.asarray([0.5, 0.0], dtype=np.float32),
        constraint_matrix=np.asarray([[1.0, 0.0]], dtype=np.float32),
        constraint_bound=np.asarray([0.1], dtype=np.float32),
        lower_bound=np.asarray([-0.7, -0.7], dtype=np.float32),
        upper_bound=np.asarray([0.7, 0.7], dtype=np.float32),
    )

    np.testing.assert_allclose(result.command, [0.1, 0.0], atol=1e-6)
    assert result.intervened
    assert not result.infeasible
    assert result.correction_norm > 0.0


def test_safety_qp_reports_slack_when_constraint_conflicts_with_velocity_bounds() -> None:
    qp = SafetyQP(SafetyQPConfig(max_iterations=4))
    result = qp.solve(
        nominal_command=np.asarray([0.0], dtype=np.float32),
        constraint_matrix=np.asarray([[1.0]], dtype=np.float32),
        constraint_bound=np.asarray([-1.0], dtype=np.float32),
        lower_bound=np.asarray([-0.2], dtype=np.float32),
        upper_bound=np.asarray([0.2], dtype=np.float32),
    )

    np.testing.assert_allclose(result.command, [-0.2], atol=1e-6)
    assert result.intervened
    assert result.infeasible
    assert result.max_slack > 0.0


def test_distance_rate_constraint_uses_safe_direction_sign() -> None:
    matrix_row, bound = distance_rate_constraint(
        distance_jacobian=np.asarray([1.0, 0.0], dtype=np.float32),
        distance=0.2,
        safe_distance=0.1,
        gain=2.0,
    )

    np.testing.assert_allclose(matrix_row, [-1.0, -0.0])
    assert bound == 0.2
