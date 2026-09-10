from __future__ import annotations

import copy

import numpy as np

from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config


def _predictive_config():
    config = load_config("configs/default.yaml")
    config["env"]["observation"]["schema_version"] = "link_risk_pred_v1"
    config["env"]["observation"]["predictive_risk"].update({"horizon": 1.0, "step": 0.05})
    config["env"]["max_episode_steps"] = 4
    config["device"] = "cpu"
    return config


def test_predictive_observation_env_outputs_finite_schema_fields() -> None:
    config = _predictive_config()
    env = UR5DynamicObstacleEnv(config, method="predictive_link")
    observation, info = env.reset(seed=101)

    assert info["observation_schema"] == "link_risk_pred_v1"
    assert observation.shape == env.observation_space.shape
    assert np.isfinite(observation).all()
    assert "risk_pred_body" in info
    assert "d_pred" in info
    assert "d_pred_raw" in info
    np.testing.assert_allclose(
        info["d_pred_raw"] - info["d_pred"],
        config["risk"]["geometry_margin"],
        atol=1e-6,
    )
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


def test_predictive_risk_penalty_adds_opt_in_dense_reward_signal() -> None:
    base_config = _predictive_config()
    shaped_config = copy.deepcopy(base_config)
    base_config["sac"]["predictive_risk_penalty"] = 0.0
    shaped_config["sac"]["predictive_risk_penalty"] = 0.25

    base_env = UR5DynamicObstacleEnv(base_config, method="predictive_link")
    shaped_env = UR5DynamicObstacleEnv(shaped_config, method="predictive_link")
    base_env.reset(seed=104)
    shaped_env.reset(seed=104)
    action = np.zeros(base_env.action_space.shape, dtype=np.float32)

    _, base_reward, base_cost, _, _, base_info = base_env.step(action)
    _, shaped_reward, shaped_cost, _, _, shaped_info = shaped_env.step(action)

    assert shaped_info["predictive_reward_penalty"] == 0.25 * shaped_info["risk_pred_body"]
    assert base_info["predictive_reward_penalty"] == 0.0
    assert shaped_cost == base_cost
    np.testing.assert_allclose(
        shaped_reward,
        base_reward - shaped_info["predictive_reward_penalty"],
        rtol=1e-6,
        atol=1e-6,
    )

    base_env.close()
    shaped_env.close()


def test_predictive_excess_penalty_uses_only_early_warning_increment() -> None:
    config = _predictive_config()
    config["sac"]["predictive_risk_penalty"] = 0.5
    config["sac"]["predictive_risk_penalty_mode"] = "excess"
    env = UR5DynamicObstacleEnv(config, method="predictive_link")
    env.reset(seed=105)

    _, _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))

    expected_signal = max(info["risk_pred_body"] - info["risk_global"], 0.0)
    np.testing.assert_allclose(info["predictive_reward_signal"], expected_signal, atol=1e-6)
    np.testing.assert_allclose(info["predictive_reward_penalty"], 0.5 * expected_signal, atol=1e-6)
    env.close()
