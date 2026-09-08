from __future__ import annotations

import numpy as np

from rl_risk_sac.control import ExecutionPipeline, JointVelocityRateLimiter


def test_joint_velocity_rate_limiter_limits_each_joint_delta() -> None:
    limiter = JointVelocityRateLimiter(joint_count=2, max_delta=0.1)

    command, was_limited = limiter.apply(np.asarray([0.7, -0.4], dtype=np.float32))

    np.testing.assert_allclose(command, [0.1, -0.1])
    assert was_limited


def test_execution_pipeline_combines_rate_limit_and_rtb() -> None:
    pipeline = ExecutionPipeline(
        joint_count=2,
        action_scale=0.7,
        control_dt=0.05,
        smoothing_mode="butterworth_quintic",
        fixed_beta=0.35,
        cutoff_angular_frequency=30.0,
        max_policy_velocity_delta=0.1,
        beta_min=0.12,
        beta_max=0.85,
        risk_high=0.75,
        lambda_beta=0.4,
    )
    pipeline.reset()

    result = pipeline.process(
        normalized_action=np.ones(2, dtype=np.float32),
        previous_command=np.zeros(2, dtype=np.float32),
        risk_global=0.0,
        sample_count=12,
        adaptive=False,
    )

    np.testing.assert_allclose(result.policy_velocity, [0.7, 0.7])
    np.testing.assert_allclose(result.limited_policy_velocity, [0.1, 0.1])
    assert result.rate_limited
    assert result.trajectory.shape == (12, 2)
    assert np.all(np.diff(result.trajectory[:, 0]) >= 0.0)
