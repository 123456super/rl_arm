"""Regression tests for bounded absolute-angle PPO and keypoint task semantics."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.distributions import Normal

from rl_risk_sac.algorithms.thesis_ppo import ThesisPPOAgent
from rl_risk_sac.envs.thesis_homotopy_env_reduced_observation import ThesisHomotopyEnv
from rl_risk_sac.tasks.thesis_reaching_reduced_observation import homotopy_reward
from rl_risk_sac.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ppo_trainer", ROOT / "scripts/core/train_thesis_ppo.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class KeypointPPOV2Tests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/experiments/paper_full.yaml")
        self.config["device"] = "cpu"
        self.config["thesis"]["simple_reaching_mode"] = True

    def env(self, reference=False):
        config = deepcopy(self.config)
        if reference:
            config["thesis"]["reward"]["reward_mode"] = "reference_ppo"
        env = ThesisHomotopyEnv(config)
        self.addCleanup(env.close)
        env.configure_episode("none", xi=1.0, strict=False)
        return env

    def reward(self, distance, **flags):
        env = self.env()
        return homotopy_reward(
            **{**env.reward_parameters, "rho_position": distance,
               "next_rho_position": distance, "rho_orientation": 0.0,
               "next_rho_orientation": 0.0, "smooth_velocity": 0.0,
               "velocity_magnitude": 0.0, "orientation_scale": 1.0,
               "task_reached": False, "hard_failure": False,
               "obstacle_collision": False, "risk_max": 0.0,
               "distance_min": 0.8, "xi": 1.0, "lambda_self": 0.0,
               "keypoint_distance": distance, "next_keypoint_distance": distance,
               **flags}
        )[0]

    def test_reward_is_negative_bounded_and_monotonic(self):
        values = [self.reward(d) for d in [0.0, 0.005, 0.01, 0.1, 0.5, 3.0]]
        self.assertAlmostEqual(values[0], -0.1)
        self.assertTrue(all(-1.0 <= r <= -0.1 for r in values))
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))

    def test_terminal_rewards_replace_dense_reward(self):
        self.assertEqual(self.reward(0.005, task_reached=True), 100.0)
        self.assertEqual(self.reward(0.5, hard_failure=True), -250.0)
        self.assertEqual(self.reward(0.5, timeout=True), -50.0)

    def test_hold_reset_cycle_is_worse_than_success(self):
        inside, outside = self.reward(.005), self.reward(.0101)
        successful = [inside] * 4 + [100.0]
        cycle = [inside if t % 5 != 4 else outside for t in range(100)]
        cycle[-1] = -50.0
        discounted = lambda xs: sum(.99 ** t * r for t, r in enumerate(xs))
        self.assertGreater(discounted(successful), discounted(cycle))
        # Even 99 worst ordinary steps including self-contact cost, then
        # timeout, are preferable to immediate hard failure.
        self.assertGreater(99 * -2.0 - 50.0, -250.0)

    def test_keypoint_observation_retains_large_errors(self):
        env = self.env(); env.reset(seed=11001)
        pos, quat = env.robot.end_effector_pose()
        env.goal_position = pos + np.array([2.5, 0.0, 0.0])
        env.goal_quaternion = quat.copy()
        obs, *_ = env._observe()
        np.testing.assert_allclose(obs[12:21:3], 2.5, atol=1e-6)
        self.assertTrue(env.observation_space.contains(obs))
        np.testing.assert_array_equal(obs[-2:], [1.0, 0.0])

    def test_hold_observation_contains_post_transition_counter(self):
        env = self.env(); env.reset(seed=11001)
        env._set_joint_state(env.goal_joint_positions)
        for step in range(5):
            obs, reward, _, term, trunc, info = env.step(env.goal_joint_positions)
            self.assertAlmostEqual(float(obs[-1]), (step + 1) / 5.0, places=6)
            self.assertAlmostEqual(float(obs[-2]), (99-step) / 100.0, places=6)
            self.assertEqual(info["success_hold_count"], step + 1)
            self.assertEqual(term, step == 4)
            self.assertFalse(trunc)

    def test_keypoint_timeout_matches_reference_truncation(self):
        env = self.env(); env.horizon = 1; env.reset(seed=11001)
        _, reward, _, term, trunc, info = env.step(env.fixed_initial_joint_positions)
        self.assertFalse(term); self.assertTrue(trunc)
        self.assertTrue(info["timeout"])
        self.assertEqual(info["termination_reason"], "timeout")
        # Reward remains deliberately mode-specific.
        self.assertEqual(reward, -50.0)

    def test_reference_timeout_retains_truncation_and_dense_reward(self):
        env = self.env(reference=True); env.horizon = 1; obs, _ = env.reset(seed=11001)
        _, reward, _, term, trunc, info = env.step(np.zeros(6))
        self.assertEqual(obs.shape, (25,))
        self.assertFalse(term); self.assertTrue(trunc)
        self.assertTrue(info["timeout"])
        self.assertNotEqual(reward, -50.0)

    def test_reference_keeps_shared_reset_and_uses_reference_dynamics(self):
        keypoint = self.env()
        reference = self.env(reference=True)
        key_obs, key_reset = keypoint.reset(seed=11001)
        ref_obs, ref_reset = reference.reset(seed=11001)
        self.assertEqual(key_obs.shape, (38,))
        self.assertEqual(ref_obs.shape, (25,))
        for name in ("goal_position", "goal_quaternion", "goal_joint_positions"):
            np.testing.assert_array_equal(key_reset[name], ref_reset[name])
        np.testing.assert_array_equal(ref_obs[-3:], np.zeros(3, dtype=np.float32))
        action = np.asarray([.9, -1.4, 1.3, .1, .6, -1.4], dtype=np.float32)
        _, key_reward, _, key_term, key_trunc, key_info = keypoint.step(action)
        _, ref_reward, _, ref_term, ref_trunc, ref_info = reference.step(action)
        # Initial pose and target generation remain shared. Reference physics
        # deliberately uses the paper's gravity and per-joint motor forces.
        self.assertNotEqual(key_reward, ref_reward)
        np.testing.assert_array_equal(
            key_info["target_joint_positions"], ref_info["target_joint_positions"]
        )
        self.assertFalse(np.allclose(
            keypoint.robot.joint_state_exact()[0],
            reference.robot.joint_state_exact()[0], atol=1e-7,
        ))
        self.assertEqual(key_info["executed_substeps"], 100)
        self.assertEqual(ref_info["executed_substeps"], 100)

    def test_s0_target_bounds_match_reference_and_invalid_action_is_rejected(self):
        env = self.env(); env.reset(seed=11001)
        with self.assertRaises(ValueError):
            env.step(np.full(6, np.nan))
        _, _, _, _, _, info = env.step(np.full(6, 3.14))
        np.testing.assert_allclose(info["target_joint_positions"], 3.14, atol=1e-6)

    def test_full_obstacle_stages_keep_38_inputs_and_absolute_targets(self):
        for scene in ["static", "dynamic"]:
            config = deepcopy(self.config)
            config["thesis"]["simple_reaching_mode"] = False
            config["thesis"]["blocking_obstacle_mode"] = True
            env = ThesisHomotopyEnv(config)
            self.addCleanup(env.close)
            env.configure_episode(scene, xi=1.0, strict=scene == "dynamic")
            obs, _ = env.reset(seed=11001)
            self.assertEqual(obs.shape, (38,))
            self.assertIsNotNone(env.obstacle_id)
            obs, reward, _, _, _, info = env.step(env.fixed_initial_joint_positions)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertTrue(np.isfinite(reward))
            np.testing.assert_allclose(info["target_joint_positions"], env.fixed_initial_joint_positions, atol=1e-6)

    def test_gae_uses_terminal_mask_and_does_not_cross_reset(self):
        rewards = np.array([[-50.0], [1.0]], dtype=np.float32)
        values = np.zeros_like(rewards)
        next_values = np.array([[999.0], [10.0]], dtype=np.float32)
        term = np.array([[1.0], [0.0]], dtype=np.float32)
        adv, returns = trainer.compute_gae(rewards, values, next_values, term, term, np.full_like(rewards, .99), .95)
        self.assertAlmostEqual(float(returns[0, 0]), -50.0)
        self.assertAlmostEqual(float(returns[1, 0]), 10.9, places=5)
        trunc_adv, _ = trainer.compute_gae(rewards[:1], values[:1], next_values[:1], np.zeros((1, 1)), np.ones((1, 1)), np.full((1, 1), .99), .95)
        self.assertAlmostEqual(float(trunc_adv[0, 0]), -50.0 + .99 * 999.0, places=4)

    def test_actor_mean_bounded_and_initial_home(self):
        agent = ThesisPPOAgent(38, 6, self.config)
        with torch.no_grad():
            agent.actor[-1].weight.zero_()
            mean = agent._dist(torch.zeros(1, 38)).mean[0]
            np.testing.assert_allclose(mean.numpy(), self.config["thesis"]["fixed_orientation_goal"]["initial_joint_positions_rad"], atol=1e-6)
            agent.actor[-1].bias.fill_(100.0)
            self.assertTrue(bool((agent._dist(torch.zeros(1, 38)).mean.abs() <= agent.action_limit).all()))

    def test_reference_actor_initial_mean_follows_shared_home_without_changing_exploration(self):
        config = deepcopy(self.config)
        config["thesis"]["reward"]["reward_mode"] = "reference_ppo"
        agent = ThesisPPOAgent(25, 6, config)
        with torch.no_grad():
            dist = agent._dist(torch.zeros(1, 25))
            np.testing.assert_allclose(dist.mean[0].numpy(), config["thesis"]["fixed_orientation_goal"]["initial_joint_positions_rad"], atol=1e-6)
            np.testing.assert_allclose(dist.stddev.numpy(), 1.0)
        self.assertIsNone(agent.max_grad_norm)
        self.assertEqual(agent.max_abs_log_ratio, 20.)

    def test_raw_samples_are_saved_before_clipping(self):
        agent = ThesisPPOAgent(38, 6, self.config)
        obs = np.zeros((2, 38), dtype=np.float32)
        with patch.object(Normal, "sample", return_value=torch.full((2, 6), 5.0)):
            commands, logp, _, policy = agent.act(obs)
        np.testing.assert_allclose(commands, 3.14)
        np.testing.assert_allclose(policy["raw_actions"], 5.0)
        expected = agent._log_prob(torch.tensor(obs), torch.tensor(policy["raw_actions"]), torch.tensor(policy["std_scale"]))
        np.testing.assert_allclose(logp, expected.detach().numpy())

    def test_both_s0_modes_share_ensemble_probability_contract(self):
        for reference, obs_dim in [(False, 38), (True, 25)]:
            config = deepcopy(self.config)
            if reference:
                config["thesis"]["reward"]["reward_mode"] = "reference_ppo"
            agent = ThesisPPOAgent(obs_dim, 6, config)
            obs = np.zeros((1, obs_dim), dtype=np.float32)
            with patch("numpy.random.binomial", return_value=2):
                _, logp, _, policy = agent.act(obs, episode_fraction=.5)
            np.testing.assert_allclose(policy["std_scale"], 1 / np.sqrt(2), atol=1e-7)
            expected = agent._log_prob(
                torch.tensor(obs), torch.tensor(policy["raw_actions"]),
                torch.tensor(policy["std_scale"]),
            )
            np.testing.assert_allclose(logp, expected.detach().numpy(), atol=1e-6)

    def test_gradient_update_and_checkpoint_contract(self):
        config = deepcopy(self.config); config["ppo"]["keypoint"].update(epochs=1, minibatch_size=8)
        agent = ThesisPPOAgent(38, 6, config)
        obs = np.random.default_rng(10).normal(0.0, .1, (8, 38)).astype(np.float32)
        _, logp, _, policy = agent.act(obs)
        before = agent.actor[-1].weight.detach().clone()
        metrics = agent.update(dict(obs=obs, actions=policy["raw_actions"], std_scale=policy["std_scale"], logp=logp, advantages=np.linspace(-1, 1, 8, dtype=np.float32), returns=np.arange(8, dtype=np.float32)))
        self.assertTrue(all(np.isfinite(v) for v in metrics.values() if v is not None))
        self.assertFalse(torch.equal(before, agent.actor[-1].weight))
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in agent.value.parameters()))
        restored = ThesisPPOAgent(38, 6, config); restored.load_state_dict(agent.state_dict())
        with self.assertRaisesRegex(ValueError, "incompatible checkpoint"):
            restored.load_state_dict({"actor": agent.actor.state_dict()})

    def test_reference_uses_reference_main_experiment_ppo_settings(self):
        keypoint = ThesisPPOAgent(38, 6, self.config)
        reference_config = deepcopy(self.config)
        reference_config["thesis"]["reward"]["reward_mode"] = "reference_ppo"
        reference = ThesisPPOAgent(25, 6, reference_config)
        for field in ("gamma", "gae_lambda", "clip_ratio", "minibatch_size",
                      "actor_lr", "critic_lr", "entropy_coef", "value_coef",
                      "action_std_init"):
            self.assertEqual(getattr(keypoint, field), getattr(reference, field), field)
        self.assertEqual(reference.epochs, 20)
        self.assertFalse(reference.normalize_advantage)
        self.assertIsNone(reference.max_grad_norm)
        self.assertIsNone(reference.target_kl)
        self.assertFalse(reference.adaptive_kl_lr)
        self.assertTrue(reference.lr_decay_enabled)

    def test_s0_limit_detection_matches_reference_nonterminal_semantics(self):
        env = self.env(); env.reset(seed=11001)
        real_joint_state = env.robot.joint_state
        for overshoot, expected_failure in [(0.0001, False), (0.002, True)]:
            env.reset(seed=11001)
            q, qdot = real_joint_state()
            q[2] = env.robot.joint_upper_limits[2] + overshoot
            calls = 0
            def measured_state():
                nonlocal calls
                calls += 1
                return real_joint_state() if calls == 1 else (q.copy(), qdot.copy())
            with patch.object(env.robot, "joint_state", side_effect=measured_state):
                _, _, _, terminated, _, info = env.step(env.fixed_initial_joint_positions)
            self.assertEqual(info["joint_limit"], expected_failure)
            self.assertFalse(terminated)
            self.assertEqual(info["executed_substeps"], 100)


if __name__ == "__main__":
    unittest.main()
