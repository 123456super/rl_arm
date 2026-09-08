from __future__ import annotations

import numpy as np

from rl_risk_sac.utils.trajectory_blending import ButterworthQuinticRTB, quintic_smoothstep


def test_quintic_smoothstep_endpoints() -> None:
    assert quintic_smoothstep(0.0) == 0.0
    assert quintic_smoothstep(1.0) == 1.0
    values = quintic_smoothstep(np.linspace(0.0, 1.0, 101))
    assert np.all(np.diff(values) >= 0.0)


def test_rtb_trajectory_reaches_filtered_target_monotonically() -> None:
    blender = ButterworthQuinticRTB(joint_count=2, control_dt=0.05, cutoff_angular_frequency=30.0)
    trajectory = blender.trajectory(
        start_velocity=np.zeros(2, dtype=np.float32),
        policy_velocity=np.ones(2, dtype=np.float32),
        sample_count=12,
    )

    assert trajectory.shape == (12, 2)
    assert np.all(np.diff(trajectory[:, 0]) >= 0.0)
    np.testing.assert_allclose(trajectory[-1], blender.previous_filtered_velocity)
    assert np.all(trajectory[-1] < 1.0)
