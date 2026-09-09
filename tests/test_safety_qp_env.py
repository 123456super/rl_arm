from __future__ import annotations

import numpy as np

from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config


def test_safety_qp_env_adds_finite_diagnostics_without_changing_schema() -> None:
    config = load_config("configs/experiments/random_crossing_link_fixed_penalty1_minimal_qp.yaml")
    config["device"] = "cpu"
    config["env"]["max_episode_steps"] = 4
    env = UR5DynamicObstacleEnv(config, method="link_fixed")
    try:
        observation, info = env.reset(seed=31)
        assert info["observation_schema"] == "link_risk_v1"
        assert observation.shape == env.observation_space.shape
        _, _, _, _, _, info = env.step(np.ones(env.action_space.shape, dtype=np.float32))
        assert info["safety_qp_enabled"] is True
        assert np.isfinite(info["safety_qp_correction_norm"])
        assert np.isfinite(info["safety_qp_solve_time_ms"])
        assert info["qdot_policy_safe_target"].shape == info["qdot_policy_limited"].shape
        assert np.max(np.abs(info["qdot_cmd"])) <= config["env"]["action_scale"] + 1e-6
    finally:
        env.close()


def test_safety_qp_is_disabled_by_default() -> None:
    config = load_config("configs/experiments/random_crossing_link_fixed_penalty1.yaml")
    config["device"] = "cpu"
    env = UR5DynamicObstacleEnv(config, method="link_fixed")
    try:
        env.reset(seed=32)
        _, _, _, _, _, info = env.step(np.ones(env.action_space.shape, dtype=np.float32))
        assert info["safety_qp_enabled"] is False
        assert info["safety_qp_intervened"] is False
        assert info["safety_qp_correction_norm"] == 0.0
    finally:
        env.close()


def test_motion_bounded_qp_enforces_physics_substep_limits() -> None:
    config = load_config(
        "configs/experiments/random_crossing_link_fixed_penalty1_qp_acceleration_jerk.yaml"
    )
    config["device"] = "cpu"
    env = UR5DynamicObstacleEnv(config, method="link_fixed")
    try:
        env.reset(seed=33)
        for sign in (1.0, -1.0, 1.0, -1.0):
            _, _, _, terminated, truncated, info = env.step(
                np.full(env.action_space.shape, sign, dtype=np.float32)
            )
            assert info["physics_peak_acceleration"] <= 8.0 + 1e-3
            assert info["physics_peak_jerk"] <= 400.0 + 1e-1
            if terminated or truncated:
                break
    finally:
        env.close()
