from __future__ import annotations

import numpy as np
import pytest

from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config


def test_fixed_method_runs_with_butterworth_quintic_rtb() -> None:
    config = load_config("configs/experiments/random_crossing_link_fixed_penalty1_rtb.yaml")
    config["device"] = "cpu"
    env = UR5DynamicObstacleEnv(config, method="link_fixed")
    try:
        env.reset(seed=17)
        action = np.ones(env.action_space.shape, dtype=np.float32)
        _, _, _, _, _, info = env.step(action)
        assert info["fixed_smoothing_mode"] == "butterworth_quintic"
        assert info["observation_schema"] == "link_risk_v1"
        assert info["policy_rate_limited"]
        assert np.max(np.abs(info["qdot_policy_limited"])) <= 0.1 + 1e-6
        assert np.isfinite(info["physics_rms_acceleration"])
        assert np.isfinite(info["physics_rms_jerk"])
        assert info["qdot_substeps"].shape == (env.sim_substeps, env.joint_count)
        assert info["measured_qdot_substeps"].shape == (env.sim_substeps, env.joint_count)
        assert info["measured_acc_substeps"].shape == (env.sim_substeps, env.joint_count)
        assert info["measured_jerk_substeps"].shape == (env.sim_substeps, env.joint_count)
        assert info["substep_distance"].shape == (env.sim_substeps,)
        assert info["substep_contact"].shape == (env.sim_substeps,)
        assert np.isclose(info["control_min_distance"], np.min(info["substep_distance"]))
        assert np.isclose(info["control_max_risk"], np.max(info["substep_risk"]))
        assert info["control_collision"] == np.any(info["substep_contact"])
        assert np.max(np.abs(info["qdot_cmd"])) <= config["env"]["action_scale"]
    finally:
        env.close()


def test_rtb_substep_feedback_is_deterministic_for_seed_and_actions() -> None:
    config = load_config("configs/default.yaml")
    config["device"] = "cpu"
    first = UR5DynamicObstacleEnv(config, method="link_fixed")
    second = UR5DynamicObstacleEnv(config, method="link_fixed")
    try:
        first_obs, _ = first.reset(seed=10001)
        second_obs, _ = second.reset(seed=10001)
        np.testing.assert_array_equal(first_obs, second_obs)
        actions = [
            np.linspace(-0.8, 0.8, first.joint_count, dtype=np.float32),
            np.linspace(0.4, -0.4, first.joint_count, dtype=np.float32),
        ]
        for action in actions:
            first_obs, _, _, _, _, first_info = first.step(action)
            second_obs, _, _, _, _, second_info = second.step(action)
            np.testing.assert_array_equal(first_obs, second_obs)
            for key in (
                "qdot_substeps",
                "measured_q_substeps",
                "measured_qdot_substeps",
                "substep_distance",
                "substep_risk",
                "substep_contact",
            ):
                np.testing.assert_array_equal(first_info[key], second_info[key])
    finally:
        first.close()
        second.close()


def test_corrected_main_link_capsules_have_physical_spans() -> None:
    config = load_config("configs/default.yaml")
    config["device"] = "cpu"
    env = UR5DynamicObstacleEnv(config, method="link_fixed")
    try:
        env.reset(seed=19)
        lengths = {capsule.name: float(np.linalg.norm(capsule.end - capsule.start)) for capsule in env._capsules()}
        assert lengths["upper_arm"] > 0.40
        assert lengths["forearm"] > 0.38
        assert lengths["wrist_1"] > 0.09
        assert lengths["wrist_2"] > 0.08
        assert lengths["wrist_3"] <= 1e-6
        wrist_3 = next(capsule for capsule in env._capsules() if capsule.name == "wrist_3")
        assert wrist_3.radius >= 0.052
    finally:
        env.close()


def test_unmarked_degenerate_capsule_is_rejected() -> None:
    config = load_config("configs/default.yaml")
    config["device"] = "cpu"
    config["robot"]["capsules"][-1]["allow_degenerate"] = False
    env = UR5DynamicObstacleEnv(config, method="link_fixed")
    try:
        with pytest.raises(ValueError, match="allow_degenerate=true"):
            env.reset(seed=19)
    finally:
        env.close()
