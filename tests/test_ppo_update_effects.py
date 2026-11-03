"""Numerical interpretation of advantage normalization and clipping with Adam."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from rl_risk_sac.algorithms.thesis_ppo import ThesisPPOAgent, _cpu_copy
from rl_risk_sac.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ppo_effect_test", ROOT / "scripts/core/check_ppo_update_effects.py")
EFFECTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EFFECTS)


class PPOUpdateEffectsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ppo-update-effect-test-")
        self.addCleanup(self.temp.cleanup)
        self.config = load_config(ROOT / "configs/experiments/paper_full.yaml")
        self.config["device"] = "cpu"
        self.config["thesis"]["reward"]["reward_mode"] = "reference_ppo"
        self.config["ppo"].update(hidden_dims=[16, 16], epochs=1, minibatch_size=8)
        EFFECTS.seed_all(42)
        self.agent = ThesisPPOAgent(25, 6, self.config)
        obs = np.random.default_rng(42).normal(0, .1, (8, 25)).astype(np.float32)
        _, logp, _, policy = self.agent.act(obs)
        self.batch = dict(obs=obs, actions=policy["raw_actions"], std_scale=policy["std_scale"],
                          logp=logp, advantages=np.linspace(-10, -1, 8, dtype=np.float32),
                          returns=np.linspace(-10000, -1000, 8, dtype=np.float32))
        self.state = _cpu_copy(self.agent.state_dict())

    def run_variant(self, normalize, clip):
        result = EFFECTS.paired_update(self.config, self.state, self.batch, normalize, clip, 10, self.temp.name)
        self.assertIsNone(result["error"])
        self.assertEqual(result["metrics"]["optimizer_steps"], 1)
        return result

    def test_normalization_changes_signs_but_not_single_step_critic_targets_or_updates(self):
        before = deepcopy(self.batch)
        probe = EFFECTS.initial_gradient_probe(self.agent, self.batch, 10)
        self.assertAlmostEqual(probe["normalized_mean"], 0, places=6)
        self.assertAlmostEqual(probe["normalized_std"], 1, places=6)
        self.assertAlmostEqual(probe["advantage_sign_flip_fraction"], .5)
        raw = self.run_variant(False, .5)
        normalized = self.run_variant(True, .5)
        self.assertEqual(raw["critic_mse_before"], normalized["critic_mse_before"])
        self.assertEqual(raw["critic_mse_after"], normalized["critic_mse_after"])
        self.assertEqual(raw["critic_parameter_delta_l2"], normalized["critic_parameter_delta_l2"])
        for key in self.batch:
            np.testing.assert_array_equal(self.batch[key], before[key])

    def test_adam_parameter_update_is_not_reduced_in_proportion_to_clipped_gradient(self):
        unclipped = self.run_variant(True, None)
        clipped = self.run_variant(True, .5)
        gradient = clipped["gradients"]["critic"][0]
        self.assertAlmostEqual(gradient["post_norm"], .5, places=5)
        self.assertLess(gradient["retained_fraction"], .001)
        # First-step Adam divides by its RMS estimate: a uniform gradient
        # rescale does not imply an equally rescaled parameter step.
        step_ratio = clipped["critic_parameter_delta_l2"] / unclipped["critic_parameter_delta_l2"]
        self.assertGreater(step_ratio, .95)
        self.assertLess(step_ratio, 1.01)


if __name__ == "__main__":
    unittest.main()
