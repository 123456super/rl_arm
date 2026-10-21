import copy
from pathlib import Path

import numpy as np
import pytest

from rl_risk_sac.envs.position_sensor_noise import PositionSensorNoise
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.envs.thesis_homotopy_env_reduced_observation import (
    ThesisHomotopyEnv as ReducedEnv,
)
from rl_risk_sac.utils.config import load_config


ROOT = Path(__file__).resolve().parents[2]


def config_for(name, std_m=None):
    config = load_config(ROOT / f"configs/experiments/{name}.yaml")
    if std_m is not None:
        config["thesis"]["evaluation_sensor_noise"] = {
            "position_std_m": std_m, "seed": 71001,
        }
    return config


@pytest.mark.parametrize("env_class,name", [
    (ThesisHomotopyEnv, "paper_full"),
    (ReducedEnv, "paper_kp_sac"),
    (ReducedEnv, "paper_m3_no_jacobian"),
])
def test_zero_noise_preserves_trajectory(env_class, name):
    clean = env_class(config_for(name))
    zero = env_class(config_for(name, 0.0))
    try:
        a, _ = clean.reset(seed=41001)
        b, _ = zero.reset(seed=41001)
        np.testing.assert_array_equal(a, b)
        for action in (np.full(6, .1), np.full(6, -.2), np.zeros(6)):
            left, right = clean.step(action), zero.step(action)
            np.testing.assert_array_equal(left[0], right[0])
            assert left[1:5] == right[1:5]
            for key in ("next_rho_position", "next_rho_orientation", "precision_action_scale"):
                assert left[5][key] == right[5][key]
    finally:
        clean.close()
        zero.close()


@pytest.mark.parametrize("env_class,name,jacobian_width", [
    (ThesisHomotopyEnv, "paper_full", 54),
    (ReducedEnv, "paper_kp_sac", 0),
    (ReducedEnv, "paper_m3_no_jacobian", 54),
])
def test_noise_is_consistent_across_position_features_and_does_not_change_task(
    env_class, name, jacobian_width,
):
    clean = env_class(config_for(name))
    noisy = env_class(config_for(name, .0005))
    try:
        clean_obs, true = clean.reset(seed=41001)
        noisy_obs, measured = noisy.reset(seed=41001)
        for key in ("goal_position", "goal_quaternion", "ee_position", "keypoint_errors"):
            np.testing.assert_array_equal(true[key], measured[key])
        assert true["goal_error_norm"] == measured["goal_error_norm"]
        # No clipping near the sampled state: rebuild expected physical errors
        # using the common measurement, rather than editing normalized slots.
        delta = measured["position_sensor_noise_m"]
        assert np.linalg.norm(delta) > 0
        expected_keypoints = np.clip(
            (true["keypoint_errors"] - delta).reshape(-1)
            / noisy.keypoint_error_scale_m, -1, 1,
        )
        np.testing.assert_allclose(noisy_obs[12:21], expected_keypoints, atol=2e-7)
        error_start = 21 + jacobian_width
        expected_error = noisy.goal_position - measured["measured_ee_position"]
        np.testing.assert_allclose(
            noisy_obs[error_start:error_start + 3],
            np.clip(expected_error / .05, -1, 1), atol=2e-7,
        )
        np.testing.assert_array_equal(noisy_obs[:12], clean_obs[:12])
        np.testing.assert_array_equal(noisy_obs[21:error_start], clean_obs[21:error_start])
        # Re-reading a sample must not draw fresh noise, consume task RNG or
        # change the apparent error used by the precision scaling interface.
        state = copy.deepcopy(noisy.rng.bit_generator.state)
        np.testing.assert_array_equal(noisy._observe()[0], noisy_obs)
        assert noisy.rng.bit_generator.state == state
    finally:
        clean.close()
        noisy.close()


def test_precision_scaling_uses_measurement_but_success_uses_truth():
    env = ThesisHomotopyEnv(config_for("paper_full", .02))
    try:
        env.reset(seed=41001)
        true_position, true_quaternion = env.robot.end_effector_pose()
        env.goal_position = env.position_sensor_noise.measure(true_position, 0).copy()
        env.goal_quaternion = true_quaternion.copy()
        env.contract.position_tolerance = .001
        env.contract.orientation_tolerance = .025
        env.success_hold_steps = 1
        _, info, *_ = env._observe()
        assert info["goal_error_norm"] > .001
        assert info["measured_position_error_norm"] < 1e-6
        env.previous_rho_position = info["goal_error_norm"]
        env.previous_rho_orientation = 0.0
        # At the next observation, make the measured position appear at goal
        # again while the physical robot is still far outside the tolerance.
        env.position_sensor_noise.measure = lambda position, step: env.goal_position.copy()
        _, _, _, _, _, after = env.step(np.zeros(6))
        assert after["precision_action_scale"] == pytest.approx(.05, abs=1e-6)
        assert after["measured_position_error_norm"] < 1e-6
        assert after["next_rho_position"] > .001
        assert not after["success"]
        assert not after["strict_pose_reached"]
    finally:
        env.close()


def test_noise_levels_share_draws_and_have_expected_physical_scale():
    low = PositionSensorNoise({"position_std_m": .0005, "seed": 71001})
    high = PositionSensorNoise({"position_std_m": .001, "seed": 71001})
    low.reset(41001)
    high.reset(41001)
    origin = np.zeros(3, dtype=np.float32)
    samples = np.stack([low.measure(origin, step) for step in range(6000)])
    np.testing.assert_allclose(high.measure(origin, 123), 2 * samples[123])
    assert np.max(np.abs(samples.mean(axis=0))) < .000025
    np.testing.assert_allclose(samples.std(axis=0), .0005, rtol=.04)
    assert np.sqrt(np.mean(np.sum(samples ** 2, axis=1))) == pytest.approx(
        np.sqrt(3) * .0005, rel=.04,
    )


@pytest.mark.parametrize("std", [-1, float("nan"), float("inf")])
def test_invalid_noise_rejected(std):
    with pytest.raises(ValueError, match="standard deviation"):
        PositionSensorNoise({"position_std_m": std})
