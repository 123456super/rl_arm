from __future__ import annotations

import numpy as np

from rl_risk_sac.tasks import WorkspaceTargetProvider
from rl_risk_sac.utils.runtime_config import GoalRuntimeConfig


def test_static_target_does_not_move() -> None:
    provider = WorkspaceTargetProvider(
        GoalRuntimeConfig(mode="static", fixed=True, position=(0.5, 0.0, 0.3), speed_range=(0.02, 0.08)),
        {"x": [0.2, 0.8], "y": [-0.4, 0.4], "z": [0.1, 0.8]},
    )
    initial = provider.reset(np.random.default_rng(1))
    advanced = provider.advance(1.0)

    np.testing.assert_allclose(advanced.position, initial.position)
    np.testing.assert_allclose(advanced.velocity, 0.0)


def test_dynamic_target_exposes_velocity_and_stays_in_workspace() -> None:
    workspace = {"x": [0.2, 0.8], "y": [-0.4, 0.4], "z": [0.1, 0.8]}
    provider = WorkspaceTargetProvider(
        GoalRuntimeConfig(mode="linear_bounce", fixed=True, position=(0.5, 0.0, 0.3), speed_range=(0.08, 0.08)),
        workspace,
    )
    state = provider.reset(np.random.default_rng(2))
    assert np.linalg.norm(state.velocity) > 0.0

    for _ in range(200):
        state = provider.advance(0.1)
        assert all(workspace[axis][0] <= state.position[i] <= workspace[axis][1] for i, axis in enumerate(("x", "y", "z")))
