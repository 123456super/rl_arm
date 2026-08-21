from __future__ import annotations

import numpy as np

from rl_risk_sac.utils.motion_planning import RRTConnectConfig, RRTConnectPlanner


def test_rrt_connect_returns_direct_path_when_edge_is_clear() -> None:
    planner = RRTConnectPlanner(
        np.asarray([-1.0, -1.0]),
        np.asarray([1.0, 1.0]),
        lambda _q: True,
        config=RRTConnectConfig(step_size_rad=0.2, edge_resolution_rad=0.05, max_iterations=20),
        rng=np.random.default_rng(1),
    )

    result = planner.plan(np.asarray([-0.8, 0.0]), np.asarray([0.8, 0.0]))

    assert result.success is True
    assert result.direct_path is True
    assert result.reason == "direct_path"
    np.testing.assert_allclose(result.path[0], [-0.8, 0.0])
    np.testing.assert_allclose(result.path[-1], [0.8, 0.0])


def test_rrt_connect_routes_around_invalid_region() -> None:
    def valid(q: np.ndarray) -> bool:
        return not (-0.25 <= q[0] <= 0.25 and -0.50 <= q[1] <= 0.50)

    planner = RRTConnectPlanner(
        np.asarray([-1.0, -1.0]),
        np.asarray([1.0, 1.0]),
        valid,
        config=RRTConnectConfig(
            step_size_rad=0.15,
            edge_resolution_rad=0.025,
            max_iterations=2000,
            goal_sample_probability=0.15,
        ),
        rng=np.random.default_rng(7),
    )

    result = planner.plan(np.asarray([-0.8, 0.0]), np.asarray([0.8, 0.0]))

    assert result.success is True
    assert result.direct_path is False
    assert len(result.path) >= 3
    assert any(abs(point[1]) > 0.50 for point in result.path)
    for start, end in zip(result.path, result.path[1:]):
        max_delta = np.max(np.abs(end - start))
        steps = max(1, int(np.ceil(max_delta / 0.025)))
        assert all(valid(start + alpha * (end - start)) for alpha in np.linspace(0.0, 1.0, steps + 1))


def test_rrt_connect_reports_invalid_endpoint_without_sampling() -> None:
    planner = RRTConnectPlanner(
        np.asarray([-1.0]),
        np.asarray([1.0]),
        lambda q: bool(q[0] <= 0.5),
    )

    result = planner.plan(np.asarray([0.0]), np.asarray([0.8]))

    assert result.success is False
    assert result.reason == "invalid_goal"
    assert result.sampled_states == 0
