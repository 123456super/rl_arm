from __future__ import annotations

import copy

import numpy as np

from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config
from rl_risk_sac.utils.predictive_risk import ObstacleStateEstimate, PredictionStatus
from rl_risk_sac.utils.safety_filter import SafetyFilterResult, SafetyFilterStatus


def test_enabled_filter_is_the_only_command_path() -> None:
    config = copy.deepcopy(load_config("configs/default.yaml"))
    config["env"]["safety_filter"]["enabled"] = True
    env = UR5DynamicObstacleEnv(config, method="link_fixed")
    try:
        env.reset(seed=7)
        safe_stop = SafetyFilterResult(
            command_joint_velocity_radps=np.zeros(env.action_space.shape, dtype=np.float64),
            status=SafetyFilterStatus.SAFE_STOP_RISK_UNUSABLE,
            reason="test safe stop",
            intervention_norm_radps=0.0,
            active_constraint_count=0,
            max_constraint_violation=0.0,
        )
        env._filter_command = lambda _: (safe_stop, None, 0.0)  # type: ignore[method-assign]

        _, _, _, _, _, info = env.step(np.ones(env.action_space.shape, dtype=np.float32))

        np.testing.assert_array_equal(info["qdot_cmd"], np.zeros(env.action_space.shape))
        assert np.linalg.norm(info["qdot_requested"]) > 0.0
        assert info["safety_filter_status"] == SafetyFilterStatus.SAFE_STOP_RISK_UNUSABLE.value
        assert info["safety_filter_safe_stop"] is True
    finally:
        env.close()


def test_enabled_filter_produces_runtime_metrics() -> None:
    config = copy.deepcopy(load_config("configs/default.yaml"))
    config["env"]["safety_filter"]["enabled"] = True
    env = UR5DynamicObstacleEnv(config, method="link_fixed")
    try:
        env.reset(seed=8)
        _, _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))

        assert info["safety_filter_status"] in {
            SafetyFilterStatus.PASSTHROUGH.value,
            SafetyFilterStatus.FILTERED.value,
        }
        assert info["safety_filter_constraint_count"] == 12
        assert 0 <= info["safety_filter_active_constraints"] <= 12
        assert isinstance(info["safety_filter_active_constraint_categories"], str)
        assert isinstance(info["safety_filter_max_constraint_category"], str)
        assert info["safety_filter_projection_iterations"] >= 0
        assert info["safety_filter_safe_stop"] is False
        assert np.isfinite(info["safety_filter_max_constraint_violation"])
        assert info["predictive_risk_status"] == PredictionStatus.VALID.value
        assert np.isfinite(info["predictive_h_min_m"])
        assert info["safety_filter_solve_time_s"] >= 0.0
        assert np.isfinite(info["qdot_cmd"]).all()
    finally:
        env.close()


def test_invalid_perception_estimate_forces_end_to_end_safe_stop() -> None:
    config = copy.deepcopy(load_config("configs/default.yaml"))
    config["env"]["safety_filter"]["enabled"] = True
    env = UR5DynamicObstacleEnv(config, method="link_fixed")
    try:
        env.reset(seed=9)
        env.set_obstacle_state_estimate_override(
            ObstacleStateEstimate(
                position=np.zeros(3),
                velocity=np.zeros(3),
                radius_m=0.07,
                timestamp_s=0.0,
                position_error_bound_m=0.0,
                velocity_error_bound_mps=0.0,
                valid=False,
            )
        )

        _, _, _, _, _, info = env.step(np.ones(env.action_space.shape, dtype=np.float32))

        np.testing.assert_array_equal(info["qdot_cmd"], np.zeros(env.action_space.shape))
        assert info["predictive_risk_status"] == PredictionStatus.INVALID.value
        assert info["safety_filter_status"] == SafetyFilterStatus.SAFE_STOP_RISK_UNUSABLE.value
        assert info["safety_filter_safe_stop"] is True
        assert np.isneginf(info["predictive_h_min_m"])
    finally:
        env.close()


def test_stale_perception_estimate_forces_end_to_end_safe_stop() -> None:
    config = copy.deepcopy(load_config("configs/default.yaml"))
    config["env"]["safety_filter"]["enabled"] = True
    env = UR5DynamicObstacleEnv(config, method="link_fixed")
    try:
        env.reset(seed=10)
        env.set_obstacle_state_estimate_override(
            ObstacleStateEstimate(
                position=np.zeros(3),
                velocity=np.zeros(3),
                radius_m=0.07,
                timestamp_s=-1.0,
                position_error_bound_m=0.0,
                velocity_error_bound_mps=0.0,
            )
        )

        _, _, _, _, _, info = env.step(np.ones(env.action_space.shape, dtype=np.float32))

        np.testing.assert_array_equal(info["qdot_cmd"], np.zeros(env.action_space.shape))
        assert info["predictive_risk_status"] == PredictionStatus.STALE.value
        assert info["safety_filter_status"] == SafetyFilterStatus.SAFE_STOP_RISK_UNUSABLE.value
        assert info["safety_filter_safe_stop"] is True
    finally:
        env.close()
