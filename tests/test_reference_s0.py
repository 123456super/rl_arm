"""Regression tests for the sole supported FTP4RM reference-PPO path."""

from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from rl_risk_sac.algorithms.thesis_ppo import ThesisPPOAgent
from rl_risk_sac.envs.thesis_homotopy_env_reduced_observation import ThesisHomotopyEnv
from rl_risk_sac.tasks.reference_reaching import reference_ppo_reward
from rl_risk_sac.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class ReferenceS0Tests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/experiments/paper_full.yaml")
        self.config["device"] = "cpu"
        self.config["device_selection"]["print_summary"] = False
        self.env = ThesisHomotopyEnv(self.config)
        self.env.configure_episode("none", xi=1.0, strict=False)

    def tearDown(self):
        self.env.close()

    def test_reset_uses_fixed_orientation_and_31_inputs(self):
        observation, info = self.env.reset(seed=11001)
        self.assertEqual(observation.shape, (31,))
        self.assertTrue(self.env.observation_space.contains(observation))
        np.testing.assert_allclose(
            info["goal_quaternion"], self.env.fixed_goal_quaternion, atol=1.0e-7
        )
        distances = np.asarray(info["reference_point_distances"])
        np.testing.assert_allclose(
            distances,
            np.full(3, info["goal_error_norm"]),
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            observation[-6:-3], info["position_error_vector"], atol=1.0e-7
        )
        np.testing.assert_allclose(
            observation[-3:], info["orientation_error_vector"], atol=1.0e-7
        )
        np.testing.assert_allclose(
            np.linalg.norm(observation[-6:-3]), info["goal_error_norm"], atol=1.0e-7
        )
        np.testing.assert_allclose(
            np.linalg.norm(observation[-3:]), info["orientation_error_norm"], atol=1.0e-7
        )

    def test_action_is_a_joint_angle_increment(self):
        self.env.reset(seed=11001)
        q_before, _ = self.env.robot.joint_state()
        delta = np.asarray([0.10, -0.10, 0.05, -0.05, 0.02, -0.02], dtype=np.float32)
        _, _, _, _, _, info = self.env.step(delta)
        expected = np.clip(
            q_before + delta,
            np.maximum(self.env.robot.joint_lower_limits, -self.env.action_limit),
            np.minimum(self.env.robot.joint_upper_limits, self.env.action_limit),
        )
        np.testing.assert_allclose(info["policy_joint_delta"], delta, atol=1.0e-7)
        np.testing.assert_allclose(info["target_joint_positions"], expected, atol=1.0e-7)

    def test_reference_reward_matches_closed_form(self):
        reward, fields = reference_ppo_reward(
            reference_error=0.2,
            collision=True,
            obstacle_clearances=(),
            collision_penalty=1.0,
        )
        expected_dense = -0.001 * 0.2**2 - np.log(0.2**2 + 1.0e-4)
        self.assertAlmostEqual(fields["r_goal"], expected_dense)
        self.assertAlmostEqual(reward, expected_dense - 1.0)

    def test_non_reference_mode_is_rejected(self):
        config = deepcopy(self.config)
        config["thesis"]["reward"]["reward_mode"] = "removed_mode"
        with self.assertRaisesRegex(ValueError, "only"):
            ThesisPPOAgent(31, 6, config)


if __name__ == "__main__":
    unittest.main()
