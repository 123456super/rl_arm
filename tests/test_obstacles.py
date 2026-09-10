from __future__ import annotations

import numpy as np
from dataclasses import replace

from rl_risk_sac.scene import SphericalObstacleProvider
from rl_risk_sac.utils.runtime_config import NamedObstacleConfig, ObstacleRuntimeConfig, RandomObstacleConfig


def _config() -> ObstacleRuntimeConfig:
    return ObstacleRuntimeConfig(
        enabled=True,
        episode_enable_probability=1.0,
        scenario="random",
        count=1,
        radius=0.07,
        speed_range=(0.12, 0.12),
        disabled_position=(10.0, 10.0, 10.0),
        bounds={"x": (0.25, 0.78), "y": (-0.7, 0.7), "z": (0.08, 0.85)},
        random=RandomObstacleConfig(
            x_range=(0.22, 0.72),
            z_range=(0.18, 0.62),
            start_y_abs_range=(0.42, 0.62),
            target_y_abs_range=(0.24, 0.48),
        ),
        scenarios={
            "elbow_crossing": NamedObstacleConfig(
                x_range=(0.35, 0.62),
                z_range=(0.42, 0.46),
                start_y_abs=0.58,
                target_y_abs=0.38,
            )
        },
    )


def test_random_obstacle_exposes_motion_state() -> None:
    provider = SphericalObstacleProvider(_config())
    (state,) = provider.reset(np.random.default_rng(3))

    assert state.enabled
    assert np.linalg.norm(state.velocity) > 0.0

    (moved,) = provider.advance(0.1)
    assert not np.allclose(moved.center, state.center)


def test_named_obstacle_uses_scenario_band() -> None:
    config = _config()
    config = replace(config, scenario="elbow_crossing")
    provider = SphericalObstacleProvider(config)

    (state,) = provider.reset(np.random.default_rng(4))

    assert config.scenarios["elbow_crossing"].x_range[0] <= state.center[0]
    assert state.center[0] <= config.scenarios["elbow_crossing"].x_range[1]
    assert abs(abs(state.center[1]) - config.scenarios["elbow_crossing"].start_y_abs) < 1e-6


def test_disabled_obstacle_stays_at_placeholder() -> None:
    config = _config()
    config = replace(config, enabled=False)
    provider = SphericalObstacleProvider(config)

    (initial,) = provider.reset(np.random.default_rng(3))
    (advanced,) = provider.advance(1.0)

    assert not initial.enabled
    np.testing.assert_allclose(initial.center, config.disabled_position)
    np.testing.assert_allclose(advanced.center, initial.center)
    np.testing.assert_allclose(advanced.velocity, 0.0)


def test_episode_dropout_stays_disabled_during_advance() -> None:
    config = _config()
    config = replace(config, episode_enable_probability=0.0)
    provider = SphericalObstacleProvider(config)

    (initial,) = provider.reset(np.random.default_rng(6))
    (advanced,) = provider.advance(1.0)

    assert not initial.enabled
    assert not advanced.enabled
    np.testing.assert_allclose(advanced.center, config.disabled_position)


def test_default_probability_preserves_random_sampling_sequence() -> None:
    first_config = _config()
    second_config = _config()
    second_config = replace(second_config, episode_enable_probability=1.0)
    first = SphericalObstacleProvider(first_config).reset(np.random.default_rng(7))
    second = SphericalObstacleProvider(second_config).reset(np.random.default_rng(7))

    np.testing.assert_array_equal(first[0].center, second[0].center)
    np.testing.assert_array_equal(first[0].velocity, second[0].velocity)


def test_multiple_obstacles_keep_independent_states() -> None:
    config = _config()
    config = replace(config, count=3)
    provider = SphericalObstacleProvider(config)

    states = provider.reset(np.random.default_rng(5))

    assert len(states) == 3
    assert all(state.enabled for state in states)
