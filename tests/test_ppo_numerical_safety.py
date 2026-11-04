"""Fail-fast PPO updates, KL stopping, and recoverable diagnostic artifacts."""
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rl_risk_sac.algorithms.thesis_ppo import PPONumericalError, ThesisPPOAgent
from rl_risk_sac.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("safety_ppo_trainer", ROOT / "scripts/core/train_thesis_ppo.py")
TRAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINER)


class PPONumericalSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="test-ppo-safety-")
        self.addCleanup(self.directory.cleanup)
        torch.manual_seed(10)

    def agent(self, device="cpu"):
        config = load_config(ROOT / "configs/experiments/paper_full.yaml")
        config["device"] = device
        config["ppo"].update(
            hidden_dims=[16, 16], epochs=3, minibatch_size=8, target_kl=.02
        )
        return ThesisPPOAgent(31, 6, config)

    def batch(self, agent):
        obs = np.random.default_rng(10).normal(0, .1, (8, agent.obs_dim)).astype(np.float32)
        _, logp, _, policy = agent.act(obs)
        return dict(obs=obs, actions=policy["raw_actions"], std_scale=policy["std_scale"],
                    logp=logp, advantages=np.linspace(-2, 2, 8, dtype=np.float32),
                    returns=np.linspace(-3, 3, 8, dtype=np.float32))

    def update(self, agent, batch):
        return agent.update(batch, diagnostic_dir=self.directory.name,
                            diagnostic_context={"step": 123, "update": 4})

    def same_state(self, actual, expected):
        if isinstance(actual, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(actual, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                self.same_state(actual[key], expected[key])
        elif isinstance(actual, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.same_state(left, right)
        else:
            self.assertEqual(actual, expected)

    def artifact(self, error):
        self.assertTrue(error.diagnostic_path.is_file())
        data = torch.load(error.diagnostic_path, weights_only=False, map_location="cpu")
        self.assertEqual(data["context"]["step"], 123)
        self.assertIn("obs", data["batch"])
        self.assertIn("failed_state", data)
        self.assertIn("rng_state", data)
        return data

    def test_reference_optimizer_defaults(self):
        config = load_config(ROOT / "configs/experiments/paper_full.yaml")
        config["device"] = "cpu"
        config["ppo"]["hidden_dims"] = [16, 16]
        agent = ThesisPPOAgent(31, 6, config)
        self.assertTrue(agent.normalize_advantage)
        self.assertIsNone(agent.max_grad_norm)
        self.assertEqual(agent.epochs, 20)
        self.assertIsNone(agent.target_kl)
        self.assertEqual(agent.max_abs_log_ratio, 20)
        self.assertTrue(agent.actor_std_lr_adaptation)
        self.assertEqual(agent.actor_std_lr_ema_decay, .95)
        self.assertEqual(agent.actor_lr_min, 1e-5)

    def test_training_uses_configured_fixed_discount(self):
        rewards = np.zeros((100, 8), dtype=np.float32)
        gammas = TRAINER.fixed_discount_factors(rewards, .99)
        self.assertEqual(gammas.shape, rewards.shape)
        self.assertEqual(gammas.dtype, np.float32)
        np.testing.assert_array_equal(gammas, np.full_like(rewards, .99))
        with self.assertRaisesRegex(ValueError, "gamma"):
            TRAINER.fixed_discount_factors(rewards, 0.0)

    def test_actor_lr_tracks_smoothed_std_while_critic_remains_linear(self):
        config = load_config(ROOT / "configs/experiments/paper_full.yaml")
        config["device"] = "cpu"
        config["ppo"].update(hidden_dims=[16, 16], actor_std_lr_ema_decay=.95)
        agent = ThesisPPOAgent(31, 6, config)
        with torch.no_grad():
            agent.actor_log_std.fill_(np.log(.04))
        agent._update_actor_std_ema()
        actor_lr, critic_lr = agent._apply_learning_rate_schedule(.25)
        expected_ema = .95 * .08 + .05 * .04
        self.assertAlmostEqual(agent.actor_std_ema, expected_ema, places=8)
        self.assertAlmostEqual(actor_lr, 3e-4 * expected_ema / .08, places=10)
        self.assertAlmostEqual(critic_lr, 3e-4 * .75, places=10)

    def test_old_checkpoint_initializes_actor_std_ema_from_learned_std(self):
        agent = self.agent()
        state = deepcopy(agent.state_dict())
        state.pop("actor_std_ema")
        state["actor_log_std"].fill_(np.log(.012))
        agent.load_state_dict(state)
        self.assertAlmostEqual(agent.actor_std_ema, .012, places=7)

    def test_disabled_gradient_clipping_still_reports_finite_raw_norms(self):
        agent = self.agent()
        batch = self.batch(agent)
        with patch(
            "rl_risk_sac.algorithms.thesis_ppo.nn.utils.clip_grad_norm_",
            side_effect=AssertionError("gradient clipping must remain disabled"),
        ):
            metrics = self.update(agent, batch)
        self.assertGreater(metrics["actor_grad_norm"], 0.0)
        self.assertGreater(metrics["critic_grad_norm"], 0.0)
        self.assertEqual(metrics["actor_gradient_clip_fraction"], 0.0)
        self.assertEqual(metrics["critic_gradient_clip_fraction"], 0.0)

    def test_extreme_log_differences_fail_before_exp_in_both_directions(self):
        for change in (-100.0, 100.0):
            agent = self.agent()
            batch = self.batch(agent)
            batch["logp"] += change
            before = deepcopy(agent.state_dict())
            with patch("rl_risk_sac.algorithms.thesis_ppo.torch.exp") as exponential:
                with self.assertRaisesRegex(PPONumericalError, "abnormal abs") as caught:
                    self.update(agent, batch)
                exponential.assert_not_called()
            self.assertTrue(caught.exception.rolled_back)
            self.same_state(agent.state_dict(), before)
            data = self.artifact(caught.exception)
            self.assertEqual(data["context"]["phase"], "before_ratio_exp")
            self.assertGreater(float(data["intermediates"]["log_ratio"].abs().max()), 99)

    def test_nonfinite_batch_is_saved_and_rejected(self):
        for key in ("obs", "actions", "std_scale", "logp", "advantages", "returns"):
            agent = self.agent(); batch = self.batch(agent)
            batch[key].flat[0] = np.nan
            with self.assertRaisesRegex(PPONumericalError, "non-finite batch") as caught:
                self.update(agent, batch)
            self.assertTrue(np.isnan(self.artifact(caught.exception)["batch"][key].flat[0]))

    def test_nonfinite_new_log_probability_is_rejected_before_exp(self):
        agent = self.agent(); batch = self.batch(agent)
        with patch.object(agent, "_log_prob", return_value=torch.full((8,), float("nan"))):
            with patch("rl_risk_sac.algorithms.thesis_ppo.torch.exp") as exponential:
                with self.assertRaisesRegex(PPONumericalError, "non-finite log_probabilities") as caught:
                    self.update(agent, batch)
                exponential.assert_not_called()
        self.artifact(caught.exception)

    def test_kl_stop_before_first_step_has_finite_explicit_metrics(self):
        agent = self.agent(); batch = self.batch(agent)
        batch["logp"] -= .5  # exp(.5) - 1 - .5 = .14872 > .03
        before = deepcopy(agent.state_dict())
        metrics = self.update(agent, batch)
        self.assertTrue(metrics["kl_early_stopped"])
        self.assertEqual(metrics["optimizer_steps"], 0)
        self.assertEqual(metrics["actor_optimizer_attempts"], 0)
        self.assertEqual(metrics["critic_optimizer_steps"], 3)
        self.assertEqual(metrics["critic_epochs_completed"], 3)
        self.assertEqual(metrics["epochs_completed"], 0)
        self.assertEqual(metrics["stop_epoch"], 1)
        self.assertAlmostEqual(metrics["max_approx_kl"], np.expm1(.5) - .5, places=6)
        self.assertTrue(all(np.isfinite(x) for x in metrics.values() if x is not None))
        self.same_state(agent.actor.state_dict(), before["actor"])
        self.assertTrue(any(
            not torch.equal(value, before["value"][key])
            for key, value in agent.value.state_dict().items()
        ))
        self.assertEqual(list(Path(self.directory.name).glob("*.pt")), [])

    def test_post_step_kl_rejects_actor_but_critic_finishes_all_epochs(self):
        agent = self.agent(); batch = self.batch(agent)
        original = agent._log_prob
        calls = 0

        def diverging_log_prob(*args):
            nonlocal calls
            calls += 1
            return original(*args) + (0.5 if calls >= 3 else 0.0)

        actor_before = deepcopy(agent.actor.state_dict())
        actor_optimizer_before = deepcopy(agent.actor_optimizer.state_dict())
        value_before = deepcopy(agent.value.state_dict())
        with patch.object(agent, "_log_prob", side_effect=diverging_log_prob):
            metrics = self.update(agent, batch)
        self.assertEqual(calls, 3)
        self.assertTrue(metrics["kl_early_stopped"])
        self.assertTrue(metrics["actor_update_rejected"])
        self.assertEqual(metrics["optimizer_steps"], 0)
        self.assertEqual(metrics["actor_optimizer_attempts"], 1)
        self.assertEqual(metrics["rejected_actor_steps"], 1)
        self.assertEqual(metrics["critic_optimizer_steps"], 3)
        self.assertEqual(metrics["critic_epochs_completed"], 3)
        self.assertEqual(metrics["stop_epoch"], 1)
        self.same_state(agent.actor.state_dict(), actor_before)
        # Adam moments from the rejected Actor proposal are rolled back.
        self.same_state(agent.actor_optimizer.state_dict(), actor_optimizer_before)
        self.assertTrue(any(
            not torch.equal(value, value_before[key])
            for key, value in agent.value.state_dict().items()
        ))
        self.assertAlmostEqual(agent.actor_optimizer.param_groups[0]["lr"], 3.0e-4)
        agent.validate_training_state()

    def test_nonfinite_initial_parameter_and_optimizer_state_are_rejected(self):
        for corrupt_optimizer in (False, True):
            agent = self.agent()
            if corrupt_optimizer:
                self.update(agent, self.batch(agent))
                next(iter(agent.value_optimizer.state.values()))["exp_avg"].fill_(float("inf"))
            else:
                with torch.no_grad():
                    agent.actor[-1].bias.fill_(float("nan"))
            # Generate valid input without using the already-corrupted actor.
            batch = self.batch(self.agent())
            with self.assertRaisesRegex(PPONumericalError, "non-finite training_state") as caught:
                self.update(agent, batch)
            data = self.artifact(caught.exception)
            self.assertFalse(caught.exception.rolled_back)
            self.assertIsNone(data["clean_state"])
            self.assertEqual(data["context"]["phase"], "before_update")

    def test_after_step_checks_catch_bad_parameters_and_adam_moments_and_roll_back(self):
        for optimizer_name in ("actor_optimizer", "value_optimizer"):
            for corrupt_moment in (False, True):
                with self.subTest(optimizer=optimizer_name, moment=corrupt_moment):
                    agent = self.agent()
                    self.update(agent, self.batch(agent))  # initialize Adam state
                    batch = self.batch(agent)
                    before = deepcopy(agent.state_dict())
                    optimizer = getattr(agent, optimizer_name)
                    original_step = optimizer.step

                    def bad_step(*args, **kwargs):
                        result = original_step(*args, **kwargs)
                        if corrupt_moment:
                            next(iter(optimizer.state.values()))["exp_avg"].fill_(float("nan"))
                        else:
                            with torch.no_grad():
                                optimizer.param_groups[0]["params"][0].fill_(float("nan"))
                        return result

                    with patch.object(optimizer, "step", side_effect=bad_step):
                        with self.assertRaisesRegex(PPONumericalError, "non-finite training_state") as caught:
                            self.update(agent, batch)
                    self.assertTrue(caught.exception.rolled_back)
                    self.same_state(agent.state_dict(), before)
                    agent.validate_training_state()
                    data = self.artifact(caught.exception)
                    self.assertEqual(data["context"]["phase"], "actor_optimizer_step" if optimizer_name.startswith("actor") else "critic_optimizer_step")

    def test_nonfinite_gradients_are_preserved_without_an_optimizer_step(self):
        agent = self.agent(); batch = self.batch(agent)
        before = deepcopy(agent.state_dict())
        hook = agent.actor[-1].weight.register_hook(lambda grad: grad * float("nan"))
        self.addCleanup(hook.remove)
        with self.assertRaises(PPONumericalError) as caught:
            self.update(agent, batch)
        self.same_state(agent.state_dict(), before)
        data = self.artifact(caught.exception)
        self.assertTrue(any(isinstance(g, torch.Tensor) and bool(torch.isnan(g).any()) for g in data["gradients"].values()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_update_and_full_rollback_after_a_later_minibatch_failure(self):
        agent = self.agent(device="cuda:0")
        self.update(agent, self.batch(agent))
        batch = self.batch(agent)
        before = deepcopy(agent.state_dict())
        original_step = agent.value_optimizer.step
        calls = 0

        def bad_second_step(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = original_step(*args, **kwargs)
            if calls == 2:
                next(iter(agent.value_optimizer.state.values()))["exp_avg_sq"].fill_(float("nan"))
            return result

        with patch.object(agent.value_optimizer, "step", side_effect=bad_second_step):
            with self.assertRaises(PPONumericalError) as caught:
                self.update(agent, batch)
        self.assertEqual(calls, 2)
        self.same_state(agent.state_dict(), before)
        agent.validate_training_state()
        self.assertTrue(caught.exception.rolled_back)
        self.assertIn("cuda", self.artifact(caught.exception)["rng_state"])

    def test_corrupt_checkpoint_rejected_without_loading_any_weights(self):
        agent = self.agent(); before = deepcopy(agent.state_dict())
        bad = deepcopy(before)
        bad["actor_log_std"].fill_(float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "non-finite checkpoint"):
            agent.load_state_dict(bad)
        self.same_state(agent.state_dict(), before)

    def test_trainer_failure_preserves_diagnostic_and_never_overwrites_checkpoint(self):
        output = Path(self.directory.name) / "trainer-failure"
        checkpoint = output / "checkpoints/step_0000008.pt"
        info = dict(reference_point_distances=[.1, .1, .1], goal_error_norm=.1,
                    orientation_error_norm=.2, next_rho_position=.1,
                    next_rho_orientation=.2, success_hold_count=0,
                    reference_error=.2)

        class FakeTemplate:
            def __init__(self, config):
                self.observation_space = SimpleNamespace(shape=(31,))
                self.action_space = SimpleNamespace(shape=(6,))

            def close(self):
                pass

        class FakePool:
            closed = False

            def __init__(self, config, seeds):
                torch.save({"known_good": True}, checkpoint)

            def reset_many(self, contracts):
                return {i: (np.zeros(31, dtype=np.float32), info) for i in contracts}

            def step_many(self, actions):
                return {i: (np.zeros(31, dtype=np.float32), -1., 0., False, False, info) for i in actions}

            def close(self):
                FakePool.closed = True

        def bad_logp(agent, obs, actions, std_scale):
            return torch.full((len(obs),), 1000., device=agent.device)

        argv = ["train_thesis_ppo.py", "--steps", "8",
                "--num-envs", "8", "--device", "cpu", "--output-dir", self.directory.name,
                "--run-name", "trainer-failure"]
        with patch("sys.argv", argv), patch.object(TRAINER, "ParallelThesisEnvPool", FakePool), \
                patch("rl_risk_sac.envs.thesis_homotopy_env_reduced_observation.ThesisHomotopyEnv", FakeTemplate), \
                patch.object(ThesisPPOAgent, "_log_prob", bad_logp), \
                redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            with self.assertRaises(PPONumericalError):
                TRAINER.main()
        self.assertTrue(FakePool.closed)
        self.assertEqual(torch.load(checkpoint, weights_only=False), {"known_good": True})
        self.assertFalse((output / "summary.json").exists())
        diagnostics = list((output / "diagnostics").glob("*.pt"))
        self.assertEqual(len(diagnostics), 1)
        artifact = torch.load(diagnostics[0], weights_only=False)
        self.assertEqual(artifact["context"]["step"], 8)
        self.assertEqual(artifact["batch"]["obs"].shape, (8, 31))


if __name__ == "__main__":
    unittest.main()
