from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rl_risk_sac.envs.ur5_dynamic_obstacle_env import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config


def _env(previous_error: float, weight: float = 30.0) -> UR5DynamicObstacleEnv:
    env = object.__new__(UR5DynamicObstacleEnv)
    env.prev_goal_error_norm = previous_error
    env.reward_cfg = {"terminal_goal_radius_m": 0.12, "w_terminal_progress": weight}
    return env


def test_terminal_progress_reward_is_zero_outside_terminal_region() -> None:
    env = _env(0.2)
    assert env._terminal_progress_reward(0.19, 0.01) == 0.0


def test_terminal_progress_reward_rewards_approach_and_penalizes_regression() -> None:
    env = _env(0.1)
    assert np.isclose(env._terminal_progress_reward(0.09, 0.01), 0.3)
    assert np.isclose(env._terminal_progress_reward(0.11, -0.01), -0.3)


def test_terminal_progress_reward_default_is_disabled() -> None:
    env = _env(0.1, weight=0.0)
    assert env._terminal_progress_reward(0.09, 0.01) == 0.0


def test_terminal_reward_config_rejects_negative_radius(tmp_path) -> None:
    config = tmp_path / "terminal_reward.yaml"
    default_config = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    config.write_text(
        f"includes:\n  - {default_config}\n"
        "reward:\n  terminal_goal_radius_m: -0.01\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="terminal_goal_radius_m"):
        load_config(config)
