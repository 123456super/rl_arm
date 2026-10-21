import copy
from pathlib import Path

import numpy as np
import pytest

from rl_risk_sac.envs.orientation_sensor_noise import OrientationSensorNoise
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.envs.thesis_homotopy_env_reduced_observation import (
    ThesisHomotopyEnv as ReducedEnv,
)
from rl_risk_sac.tasks.thesis_reaching import (
    pose_error, pose_keypoint_errors, quaternion_to_matrix, so3_log,
)
from rl_risk_sac.utils.config import load_config


ROOT = Path(__file__).resolve().parents[2]


def config_for(name, std_rad):
    config = load_config(ROOT / f"configs/experiments/{name}.yaml")
    config["thesis"]["evaluation_sensor_noise"] = {
        "orientation_std_rad": std_rad, "seed": 71001,
    }
    return config


def test_rotation_noise_is_on_so3_with_expected_scale_and_shared_draws():
    low = OrientationSensorNoise({"orientation_std_rad": .005, "seed": 71001})
    high = OrientationSensorNoise({"orientation_std_rad": .01, "seed": 71001})
    low.reset(41001)
    high.reset(41001)
    # A non-identity rotation checks world-frame left multiplication.
    q = np.asarray([.3, -.4, .1, .8], dtype=np.float64)
    q /= np.linalg.norm(q)
    original_rotation = quaternion_to_matrix(q)
    samples = []
    for step in range(3000):
        measured = low.measure(q, step)
        rotation = quaternion_to_matrix(measured)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1)
        assert np.linalg.norm(measured) == pytest.approx(1, abs=1e-7)
        samples.append(so3_log(rotation @ original_rotation.T))
    samples = np.asarray(samples)
    assert np.max(np.abs(samples.mean(axis=0))) < .0003
    np.testing.assert_allclose(samples.std(axis=0), .005, rtol=.05)
    high_error = so3_log(quaternion_to_matrix(high.measure(q, 123)) @ original_rotation.T)
    np.testing.assert_allclose(high_error, 2 * samples[123], atol=1e-7)
    np.testing.assert_array_equal(low.measure(q, 123), low.measure(q, 123))
    zero = OrientationSensorNoise({"orientation_std_rad": 0})
    np.testing.assert_array_equal(zero.measure(q, 0), q)


@pytest.mark.parametrize("env_class,name,jacobian_width", [
    (ThesisHomotopyEnv, "paper_full", 54),
    (ReducedEnv, "paper_kp_sac", 0),
    (ReducedEnv, "paper_m3_no_jacobian", 54),
])
def test_orientation_measurement_updates_pose_and_keypoints_but_not_truth(
    env_class, name, jacobian_width,
):
    clean = env_class(config_for(name, 0))
    noisy = env_class(config_for(name, .01))
    try:
        original, true = clean.reset(seed=41001)
        observation, measured = noisy.reset(seed=41001)
        for key in ("goal_position", "goal_quaternion", "ee_position", "ee_quaternion",
                    "keypoint_errors", "orientation_error_vector"):
            np.testing.assert_array_equal(true[key], measured[key])
        assert true["orientation_error_norm"] == measured["orientation_error_norm"]
        q = measured["measured_ee_quaternion"]
        assert not np.array_equal(q, true["ee_quaternion"])
        expected_keypoints = pose_keypoint_errors(
            noisy.goal_position, noisy.goal_quaternion, true["ee_position"], q,
            cube_side_m=noisy.keypoint_cube_side_m,
        )
        np.testing.assert_allclose(observation[12:21], np.clip(
            expected_keypoints.reshape(-1) / noisy.keypoint_error_scale_m, -1, 1,
        ), atol=1e-7)
        error_start = 21 + jacobian_width
        _, expected_error = pose_error(
            noisy.goal_position, noisy.goal_quaternion, true["ee_position"], q,
        )
        np.testing.assert_allclose(observation[error_start + 3:error_start + 6],
                                   np.clip(expected_error / .5, -1, 1), atol=1e-7)
        np.testing.assert_array_equal(observation[:12], original[:12])
        # Jacobians remain computed from the accurate joint/model state;
        # these experiments perturb only the end-effector pose measurement.
        np.testing.assert_array_equal(observation[21:error_start], original[21:error_start])
        np.testing.assert_array_equal(observation[error_start:error_start + 3],
                                      original[error_start:error_start + 3])
        state = copy.deepcopy(noisy.rng.bit_generator.state)
        np.testing.assert_array_equal(noisy._observe()[0], observation)
        assert noisy.rng.bit_generator.state == state
        # With a zero motor command, physical transitions and true rewards
        # must agree despite different policy measurements/scaling signals.
        left, right = clean.step(np.zeros(6)), noisy.step(np.zeros(6))
        assert left[1:5] == right[1:5]
        np.testing.assert_array_equal(left[5]["ee_quaternion"], right[5]["ee_quaternion"])
    finally:
        clean.close()
        noisy.close()


def test_angular_precision_scaling_uses_measurement_success_uses_truth():
    env = ThesisHomotopyEnv(config_for("paper_full", .2))
    try:
        env.reset(seed=41001)
        position, quaternion = env.robot.end_effector_pose()
        env.goal_position = position.copy()
        env.goal_quaternion = env.orientation_sensor_noise.measure(quaternion, 0).copy()
        env.contract.position_tolerance = .001
        env.contract.orientation_tolerance = .025
        env.success_hold_steps = 1
        _, info, *_ = env._observe()
        assert info["orientation_error_norm"] > .025
        assert info["measured_orientation_error_norm"] < 1e-6
        env.previous_rho_position = 0
        env.previous_rho_orientation = info["orientation_error_norm"]
        env.orientation_sensor_noise.measure = lambda q, step: env.goal_quaternion.copy()
        _, _, _, _, _, after = env.step(np.zeros(6))
        assert after["precision_action_scale"] == pytest.approx(.05, abs=1e-6)
        assert after["measured_orientation_error_norm"] < 1e-6
        assert after["next_rho_orientation"] > .025
        assert not after["success"]
    finally:
        env.close()


@pytest.mark.parametrize("std", [-.1, float("nan"), float("inf")])
def test_invalid_orientation_noise_rejected(std):
    with pytest.raises(ValueError, match="orientation noise standard deviation"):
        OrientationSensorNoise({"orientation_std_rad": std})
