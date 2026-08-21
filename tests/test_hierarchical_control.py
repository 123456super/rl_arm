from __future__ import annotations

import copy

import numpy as np
import pytest

from rl_risk_sac.algorithms.sac import actor_signature
from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config, validate_config


def _hierarchical_config() -> dict:
    config = copy.deepcopy(load_config("configs/default.yaml"))
    config["device"] = "cpu"
    config["env"]["residual_control"]["enabled"] = False
    config["env"]["hierarchical_control"]["enabled"] = True
    config["env"]["safety_filter"]["enabled"] = True
    config["env"]["obstacle"]["enabled"] = False
    config["env"]["max_episode_steps"] = 3
    for section_name in ("train", "eval", "smoke"):
        config[section_name]["method"] = "hierarchical_residual"
    return config


def test_hierarchical_config_requires_predictive_safety_filter() -> None:
    config = _hierarchical_config()
    config["env"]["safety_filter"]["enabled"] = False

    with pytest.raises(ValueError, match="requires env.safety_filter"):
        validate_config(config)


def test_hierarchical_and_legacy_residual_cannot_both_be_enabled() -> None:
    config = _hierarchical_config()
    config["env"]["residual_control"]["enabled"] = True

    with pytest.raises(ValueError, match="cannot both be enabled"):
        validate_config(config)


def test_hierarchical_control_requires_dedicated_method_name() -> None:
    config = _hierarchical_config()
    config["eval"]["method"] = "link_fixed"

    with pytest.raises(ValueError, match="eval.method must be hierarchical_residual"):
        validate_config(config)


def test_actor_signature_changes_for_hierarchical_action_semantics() -> None:
    legacy = load_config("configs/default.yaml")
    hierarchical = _hierarchical_config()

    assert actor_signature(legacy, "link_fixed") != actor_signature(hierarchical, "hierarchical_residual")


def test_hierarchical_reset_exposes_planner_features_and_zero_residual() -> None:
    env = UR5DynamicObstacleEnv(_hierarchical_config(), method="hierarchical_residual")
    try:
        observation, reset_info = env.reset(seed=123)

        assert observation.shape == env.observation_space.shape
        assert reset_info["hierarchical_control_enabled"] is True
        assert reset_info["hierarchical_state"] in {"TRACK", "PLAN_FAILED"}
        assert reset_info["hierarchical_plan_count"] >= 0
        assert 0.0 <= reset_info["hierarchical_residual_budget"] <= 1.0

        _, _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))

        np.testing.assert_allclose(info["residual_qdot"], 0.0)
        assert info["hierarchical_state"] in {"TRACK", "SERVO", "PLAN_FAILED"}
        assert np.isfinite(info["qdot_cmd"]).all()
    finally:
        env.close()
