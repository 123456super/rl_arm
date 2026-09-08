from __future__ import annotations

import copy

import numpy as np

from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config


def _predictive_config():
    config = load_config("configs/default.yaml")
    config["env"]["observation"]["schema_version"] = "link_risk_pred_v1"
    config["env"]["observation"]["predictive_risk"] = {"horizon": 1.0, "step": 0.05}
    config["env"]["max_episode_steps"] = 4
    config["device"] = "cpu"
    return config


def test_predictive_observation_env_outputs_finite_schema_fields() -> None:
    env = UR5DynamicObstacleEnv(_predictive_config(), method="predictive_link")
    observation, info = env.reset(seed=101)

    assert info["observation_schema"] == "link_risk_pred_v1"
    assert observation.shape == env.observation_space.shape
    assert np.isfinite(observation).all()
    assert "risk_pred_body" in info
    assert "d_pred" in info
    assert "t_enter_pred" in info
    assert "risk_pred_per_link" in info

    next_observation, _, _, _, _, next_info = env.step(env.action_space.sample())
    assert next_info["observation_schema"] == "link_risk_pred_v1"
    assert next_observation.shape == env.observation_space.shape
    assert np.isfinite(next_observation).all()
    env.close()


def test_predictive_observation_keeps_no_obstacle_risk_low() -> None:
    config = _predictive_config()
    config["env"]["obstacle"]["enabled"] = False
    env = UR5DynamicObstacleEnv(config, method="predictive_link")
    observation, info = env.reset(seed=102)

    assert np.isfinite(observation).all()
    assert info["obstacle_enabled"] is False
    assert info["risk_global"] == 0.0
    assert info["risk_pred_body"] == 0.0
    np.testing.assert_allclose(info["risk_pred_per_link"], 0.0)
    env.close()


def test_link_risk_v1_dimension_is_unchanged_when_predictive_config_exists() -> None:
    predictive_config = _predictive_config()
    legacy_config = copy.deepcopy(predictive_config)
    legacy_config["env"]["observation"]["schema_version"] = "link_risk_v1"

    legacy_env = UR5DynamicObstacleEnv(legacy_config, method="link_fixed")
    predictive_env = UR5DynamicObstacleEnv(predictive_config, method="predictive_link")

    assert legacy_env.observation_space.shape[0] < predictive_env.observation_space.shape[0]
    assert legacy_env.observation_builder.dimension(legacy_env.joint_count, legacy_env.capsule_model.count) == 67

    legacy_env.close()
    predictive_env.close()


def test_compact_predictive_observation_omits_per_link_score_from_observation_only() -> None:
    full_config = _predictive_config()
    compact_config = _predictive_config()
    compact_config["env"]["observation"]["predictive_risk"]["include_per_link_score"] = False

    full_env = UR5DynamicObstacleEnv(full_config, method="predictive_link")
    compact_env = UR5DynamicObstacleEnv(compact_config, method="predictive_link")
    observation, info = compact_env.reset(seed=103)

    assert compact_env.observation_space.shape[0] == full_env.observation_space.shape[0] - compact_env.capsule_model.count
    assert compact_env.observation_space.shape[0] == 88
    assert observation.shape == compact_env.observation_space.shape
    assert np.isfinite(observation).all()
    assert "risk_pred_per_link" in info

    full_env.close()
    compact_env.close()
