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

    def test_reset_uses_fixed_orientation_and_25_inputs(self):
        observation, info = self.env.reset(seed=11001)
        self.assertEqual(observation.shape, (25,))
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

    def test_action_is_an_absolute_joint_target(self):
        self.env.reset(seed=11001)
        target = self.env.fixed_initial_joint_positions.astype(np.float32)
        _, _, _, _, _, info = self.env.step(target)
        np.testing.assert_allclose(info["target_joint_positions"], target, atol=1.0e-7)

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
            ThesisPPOAgent(25, 6, config)


if __name__ == "__main__":
    unittest.main()
