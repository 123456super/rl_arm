from __future__ import annotations

import copy
import json
import random

import numpy as np
import pytest
import torch

from rl_risk_sac.algorithms.replay_buffer import Batch, ReplayBuffer
from rl_risk_sac.algorithms.sac import SACAgent
from rl_risk_sac.utils.config import load_config
from rl_risk_sac.utils.seeding import capture_rng_state, restore_rng_state, set_seed
from scripts.train import replay_action_signature, reset_training_environment, validate_resume_checkpoint_steps


def _small_config() -> dict:
    config = load_config("configs/default.yaml")
    config["device"] = "cpu"
    config["sac"]["hidden_dims"] = [8, 8]
    config["sac"]["batch_size"] = 4
    config["sac"]["actor_anchor_weight"] = 1.0
    return config


def _batch(obs_dim: int = 3, action_dim: int = 2) -> Batch:
    return Batch(
        observations=torch.randn(4, obs_dim),
        actions=torch.tanh(torch.randn(4, action_dim)),
        rewards=torch.randn(4, 1),
        costs=torch.rand(4, 1),
        next_observations=torch.randn(4, obs_dim),
        dones=torch.zeros(4, 1),
    )


def test_time_limit_truncation_keeps_bellman_bootstrap() -> None:
    replay = ReplayBuffer(obs_dim=2, action_dim=1, capacity=4, device="cpu")
    replay.add(
        np.zeros(2),
        np.zeros(1),
        0.0,
        0.0,
        np.ones(2),
        done=False,
        truncated=True,
    )
    batch = replay.sample(1)
    assert batch.dones.item() == 0.0
    assert replay.truncateds[0].item() == 1.0


def test_replay_stratifies_success_and_failure_episodes() -> None:
    replay = ReplayBuffer(obs_dim=1, action_dim=1, capacity=8, device="cpu", stratified_fraction=1.0)
    success = [replay.add(np.array([1.0]), np.zeros(1), 0.0, 0.0, np.zeros(1), False)]
    failure = [replay.add(np.array([-1.0]), np.zeros(1), 0.0, 0.0, np.zeros(1), False)]
    replay.label_episode(success, success=True)
    replay.label_episode(failure, success=False)

    indices = replay._sample_indices(20)
    outcomes = replay.episode_outcomes[indices]
    assert np.count_nonzero(outcomes == 1) == 10
    assert np.count_nonzero(outcomes == 0) == 10


def test_replay_round_trip_preserves_ring_order_and_labels(tmp_path) -> None:
    replay = ReplayBuffer(
        obs_dim=1,
        action_dim=1,
        capacity=4,
        device="cpu",
        stratified_fraction=0.5,
        action_signature="full-action",
    )
    for value in range(5):
        index = replay.add(
            np.array([value], dtype=np.float32),
            np.array([value], dtype=np.float32),
            float(value),
            0.0,
            np.array([value + 1], dtype=np.float32),
            done=False,
            truncated=value == 4,
        )
        replay.label_episode([index], success=value % 2 == 0)
    path = tmp_path / "replay.npz"
    replay.save(path)

    restored = ReplayBuffer(obs_dim=1, action_dim=1, capacity=6, device="cpu", action_signature="full-action")
    restored.load(path)

    assert len(restored) == 4
    assert restored.ptr == 4
    assert restored.observations[:4, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert restored.truncateds[:4, 0].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert restored.episode_outcomes[:4].tolist() == [0, 1, 0, 1]


def test_replay_action_signature_mismatch_is_rejected(tmp_path) -> None:
    replay = ReplayBuffer(obs_dim=1, action_dim=1, capacity=4, device="cpu", action_signature="full-action")
    replay.add(np.zeros(1), np.zeros(1), 0.0, 0.0, np.zeros(1), done=False)
    path = tmp_path / "replay.npz"
    replay.save(path)

    restored = ReplayBuffer(obs_dim=1, action_dim=1, capacity=4, device="cpu", action_signature="residual")
    with pytest.raises(ValueError, match="action semantics"):
        restored.load(path)


def test_replay_action_signature_reads_missing_signature_as_legacy(tmp_path) -> None:
    path = tmp_path / "legacy_replay.npz"
    np.savez_compressed(
        path,
        capacity=np.asarray(1),
        obs_dim=np.asarray(1),
        action_dim=np.asarray(1),
        ptr=np.asarray(0),
        size=np.asarray(0),
        stratified_fraction=np.asarray(0.0),
        observations=np.zeros((0, 1), dtype=np.float32),
        actions=np.zeros((0, 1), dtype=np.float32),
        rewards=np.zeros((0, 1), dtype=np.float32),
        costs=np.zeros((0, 1), dtype=np.float32),
        next_observations=np.zeros((0, 1), dtype=np.float32),
        dones=np.zeros((0, 1), dtype=np.float32),
    )

    assert replay_action_signature(path) == ""


def test_agent_checkpoint_restores_targets_optimizers_and_training_state(tmp_path) -> None:
    config = _small_config()
    source = SACAgent(3, 2, config, method="link_fixed")
    source.update(_batch())
    source.save(tmp_path, training_state={"step": 123, "episode": 7})

    restored_config = copy.deepcopy(config)
    restored_config["sac"]["actor_lr"] = 1.0e-5
    restored = SACAgent(3, 2, restored_config, method="link_fixed")
    load_info = restored.load(tmp_path / "actor.pt", tmp_path / "agent_state.pt")

    assert load_info["reward_critics_loaded"] is True
    assert load_info["cost_critics_loaded"] is True
    assert load_info["optimizers_loaded"] is True
    assert load_info["training_state"] == {"step": 123, "episode": 7}
    assert restored.actor_optimizer.param_groups[0]["lr"] == 1.0e-5
    for expected, actual in zip(source.reward_target_q1.parameters(), restored.reward_target_q1.parameters()):
        assert torch.equal(expected, actual)
    assert restored.actor_reference is not None


def test_changed_reward_reinitializes_only_reward_critics(tmp_path) -> None:
    source_config = _small_config()
    source = SACAgent(3, 2, source_config, method="link_fixed")
    source.save(tmp_path)

    changed_config = copy.deepcopy(source_config)
    changed_config["reward"]["w_terminal_progress"] = 30.0
    restored = SACAgent(3, 2, changed_config, method="link_fixed")
    load_info = restored.load(tmp_path / "actor.pt", tmp_path / "agent_state.pt")

    assert load_info["reward_signature_match"] is False
    assert load_info["reward_critics_loaded"] is False
    assert load_info["cost_signature_match"] is True
    assert load_info["cost_critics_loaded"] is True


def test_changed_risk_reinitializes_reward_critics(tmp_path) -> None:
    source_config = _small_config()
    source = SACAgent(3, 2, source_config, method="link_fixed")
    source.save(tmp_path)

    changed_config = copy.deepcopy(source_config)
    changed_config["risk"]["weights"]["distance"] = 0.6
    restored = SACAgent(3, 2, changed_config, method="link_fixed")
    load_info = restored.load(tmp_path / "actor.pt", tmp_path / "agent_state.pt")

    assert load_info["reward_signature_match"] is False
    assert load_info["reward_critics_loaded"] is False
    assert load_info["cost_signature_match"] is False
    assert load_info["cost_critics_loaded"] is False


def test_residual_control_change_requires_actor_semantics_reset(tmp_path) -> None:
    source_config = _small_config()
    source = SACAgent(3, 2, source_config, method="link_fixed")
    source.save(tmp_path)

    changed_config = copy.deepcopy(source_config)
    changed_config["env"]["residual_control"]["enabled"] = True
    restored = SACAgent(3, 2, changed_config, method="link_fixed")

    with pytest.raises(ValueError, match="action semantics"):
        restored.load(tmp_path / "actor.pt", tmp_path / "agent_state.pt")


def test_actor_only_load_rejects_config_action_semantics_mismatch(tmp_path) -> None:
    source_config = _small_config()
    source = SACAgent(3, 2, source_config, method="link_fixed")
    source.save(tmp_path)
    (tmp_path / "agent_state.pt").unlink()
    (tmp_path / "config.json").write_text(json.dumps(source_config), encoding="utf-8")

    changed_config = copy.deepcopy(source_config)
    changed_config["env"]["residual_control"]["enabled"] = True
    restored = SACAgent(3, 2, changed_config, method="link_fixed")

    with pytest.raises(ValueError, match="action semantics"):
        restored.load_actor(tmp_path / "actor.pt")


def test_agent_load_without_state_rejects_actor_action_semantics_mismatch(tmp_path) -> None:
    source_config = _small_config()
    source = SACAgent(3, 2, source_config, method="link_fixed")
    source.save(tmp_path)

    changed_config = copy.deepcopy(source_config)
    changed_config["env"]["residual_control"]["enabled"] = True
    restored = SACAgent(3, 2, changed_config, method="link_fixed")

    with pytest.raises(ValueError, match="action semantics"):
        restored.load(tmp_path / "actor.pt", state_path=None)


def test_actor_only_load_rejects_agent_state_action_semantics_mismatch(tmp_path) -> None:
    source_config = _small_config()
    source = SACAgent(3, 2, source_config, method="link_fixed")
    source.save(tmp_path, suffix="_step_10")

    changed_config = copy.deepcopy(source_config)
    changed_config["env"]["residual_control"]["enabled"] = True
    restored = SACAgent(3, 2, changed_config, method="link_fixed")

    with pytest.raises(ValueError, match="action semantics"):
        restored.load_actor(tmp_path / "actor_step_10.pt")


def test_actor_only_action_semantics_transfer_expands_added_observation_inputs(tmp_path) -> None:
    source_config = _small_config()
    source = SACAgent(3, 2, source_config, method="link_fixed")
    source.save(tmp_path)

    changed_config = copy.deepcopy(source_config)
    changed_config["env"]["residual_control"]["enabled"] = True
    restored = SACAgent(5, 2, changed_config, method="link_fixed")

    restored.load_actor(
        tmp_path / "actor.pt",
        allow_action_semantics_transfer=True,
    )

    source_weight = source.actor.state_dict()["backbone.0.weight"]
    restored_weight = restored.actor.state_dict()["backbone.0.weight"]
    torch.testing.assert_close(restored_weight[:, : source_weight.shape[1]], source_weight)
    assert torch.count_nonzero(restored_weight[:, source_weight.shape[1] :]) == 0


def test_rng_state_round_trip() -> None:
    set_seed(123)
    state = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(1))
    restore_rng_state(state)
    actual = (random.random(), np.random.random(), torch.rand(1))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_rng_state_round_trip_accepts_tensor_subclasses() -> None:
    set_seed(456)
    state = capture_rng_state()
    state["torch"] = torch.as_tensor(state["torch"], device="cpu", dtype=torch.uint8).clone()
    if "torch_cuda" in state:
        state["torch_cuda"] = [
            torch.as_tensor(item, device="cpu", dtype=torch.uint8).clone()
            for item in state["torch_cuda"]
        ]
    expected = (random.random(), np.random.random(), torch.rand(1))
    restore_rng_state(state)
    actual = (random.random(), np.random.random(), torch.rand(1))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_legacy_checkpoint_filenames_must_match_start_step() -> None:
    validate_resume_checkpoint_steps(
        "actor_step_520000.pt",
        "agent_state_step_520000.pt",
        "replay_step_520000.npz",
        520000,
    )
    with pytest.raises(ValueError, match="agent-state checkpoint step 500000"):
        validate_resume_checkpoint_steps(
            "actor_step_520000.pt",
            "agent_state_step_500000.pt",
            None,
            520000,
        )


class _ResetEnv:
    def __init__(self) -> None:
        self.config = {
            "train": {
                "focused_reset_jitter": {
                    "enabled": True,
                    "goal_radius_m": 0.02,
                }
            }
        }
        self.calls = []

    def reset(self, *, seed=None, options=None):
        self.calls.append({"seed": seed, "options": options})
        return np.zeros(1, dtype=np.float32), {}


def test_focused_reset_jitter_is_applied_only_to_focused_resets(monkeypatch) -> None:
    env = _ResetEnv()
    draws = iter([1234])
    monkeypatch.setattr(np.random, "random", lambda: 0.0)
    monkeypatch.setattr(np.random, "choice", lambda values: values[0])
    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: next(draws))

    _, _, seed, source, jitter_seed = reset_training_environment(env, [9001], 1.0)

    assert seed == 9001
    assert source == "focused"
    assert jitter_seed == 1234
    assert env.calls[0]["options"]["reset_jitter"]["seed"] == 1234

    env = _ResetEnv()
    monkeypatch.setattr(np.random, "random", lambda: 1.0)
    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 4567)

    _, _, seed, source, jitter_seed = reset_training_environment(env, [9001], 0.0)

    assert seed == 4567
    assert source == "random"
    assert jitter_seed is None
    assert env.calls[0]["options"] is None
