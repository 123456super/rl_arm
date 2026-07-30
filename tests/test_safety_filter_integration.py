from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np

from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config
from rl_risk_sac.utils.predictive_risk import ObstacleStateEstimate, PredictionStatus
from rl_risk_sac.utils.safety_filter import SafetyFilterResult, SafetyFilterStatus


def _recovery_test_env(jacobian: np.ndarray):
    env = object.__new__(UR5DynamicObstacleEnv)
    env.joint_count = jacobian.shape[1]
    env.action_scale = 1.0
    env.safety_filter_cfg = {
        "recovery_speed_radps": 0.5,
        "recovery_enter_margin_m": 0.06,
        "recovery_exit_margin_m": 0.08,
    }
    env._analytic_constraint_jacobians = lambda _: (jacobian, None, None)  # type: ignore[method-assign]
    return env


def test_recovery_uses_enter_and_exit_hysteresis() -> None:
    env = _recovery_test_env(np.asarray([[1.0, 0.0]]))
    env.recovery_active = False
    env.recovery_triggered = False
    env.recovery_success = False
    env.recovery_steps = 0

    env._update_recovery_state(replace(_predictive_risk_for_test(1), safety_functions_m=np.asarray([0.05])))
    assert env.recovery_active is True
    assert env.recovery_triggered is True

    env._update_recovery_state(replace(_predictive_risk_for_test(1), safety_functions_m=np.asarray([0.07])))
    assert env.recovery_active is True

    env._update_recovery_state(replace(_predictive_risk_for_test(1), safety_functions_m=np.asarray([0.08])))
    assert env.recovery_active is False
    assert env.recovery_success is True


def test_recovery_command_weights_all_below_margin_links() -> None:
    env = _recovery_test_env(np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))
    risk = replace(
        _predictive_risk_for_test(3),
        safety_functions_m=np.asarray([-0.10, -0.05, 0.01]),
    )

    command = env._recovery_command(risk)

    # Deficit weights are 2/3 and 1/3; the third link is already safe.
    np.testing.assert_allclose(command, [1.0 / np.sqrt(5.0), 0.5 / np.sqrt(5.0)], atol=1e-6)


def _predictive_risk_for_test(count: int):
    from rl_risk_sac.utils.predictive_risk import PredictiveLinkRisk

    return PredictiveLinkRisk(
        status=PredictionStatus.VALID,
        status_reason="",
        observation_age_s=0.0,
        closest_prediction_times_s=np.zeros(count),
        predicted_distances_m=np.ones(count),
        robust_distances_m=np.ones(count),
        safety_functions_m=np.ones(count),
        geometry_margins_m=np.zeros(count),
        perception_margin_m=0.0,
        delay_margin_m=0.0,
        tracking_margin_m=0.0,
    )


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
        env._filter_command = lambda *_args, **_kwargs: (safe_stop, None, 0.0)  # type: ignore[method-assign]

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
        assert info["collision_capsule_overlap"] is False
        assert info["collision_pybullet_contact"] is False
        assert info["collision_contact_link_indices"] == ""
        assert info["collision_contact_link_names"] == ""
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
        assert info["safety_filter_predictive_risk_time_s"] >= 0.0
        assert info["safety_filter_jacobian_workspace_time_s"] >= 0.0
        assert info["safety_filter_projection_time_s"] >= 0.0
        assert np.isfinite(info["qdot_cmd"]).all()
        assert info["risk_speed_scale"] == 1.0
    finally:
        env.close()


def test_filter_compute_budget_forces_zero_velocity_after_an_overrun() -> None:
    config = copy.deepcopy(load_config("configs/default.yaml"))
    config["env"]["safety_filter"]["enabled"] = True
    config["env"]["safety_filter"]["max_filter_compute_time_s"] = 1.0e-12
    env = UR5DynamicObstacleEnv(config, method="link_fixed")
    try:
        env.reset(seed=18)
        _, _, _, _, _, info = env.step(np.ones(env.action_space.shape, dtype=np.float32))

        np.testing.assert_array_equal(info["qdot_cmd"], np.zeros(env.action_space.shape))
        assert info["safety_filter_status"] == SafetyFilterStatus.SAFE_STOP_COMPUTE_BUDGET.value
        assert info["safety_filter_safe_stop"] is True
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
