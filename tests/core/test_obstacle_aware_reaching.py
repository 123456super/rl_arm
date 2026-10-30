from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from rl_risk_sac.algorithms.dual_sac import ReachingSACAgent
from rl_risk_sac.algorithms.replay_buffer import Batch
from rl_risk_sac.envs.parallel_local_avoidance_env import FrozenCPUReachingPolicy
from rl_risk_sac.tasks.obstacle_observation import ObstacleObservation, obstacle_set, stack_observations
from rl_risk_sac.tasks.reaching_handoff import (
    OBSTACLE_AWARE_TRAIN_PROTOCOL, JointHandoffReplay, ObstacleAwareJointHandoffReplay,
    ReachingDecisionReturn, obstacle_aware_reaching_config, recovery_config,
)
from rl_risk_sac.utils.config import load_config


@pytest.fixture
def config():
    cfg = load_config("configs/experiments/paper_local_avoidance_geometry.yaml")
    cfg = recovery_config(cfg)
    cfg["device"] = "cpu"
    cfg["sac"].update(hidden_dims=[16, 16], actor_update_interval=1)
    return cfg


def observations():
    rng = np.random.default_rng(9)
    state = rng.normal(size=241).astype(np.float32)
    return [ObstacleObservation(state, rng.normal(size=(n, 37))) for n in (0, 1, 3)]


def test_enabling_perception_preserves_mature_outputs_and_independent_networks(config):
    agent = ReachingSACAgent(241, 6, config)
    inputs = observations()
    actions = agent.select_actions(inputs, deterministic=True)
    backbones = {name: {k: v.detach().clone() for k, v in getattr(agent, name).state_dict().items()
                       if not k.startswith("obstacle_encoder.")}
                 for name in ("actor", "q1", "q2", "target_q1", "target_q2")}
    assert agent.initialize_obstacle_awareness(np.arange(37) + 1)
    np.testing.assert_array_equal(actions, agent.select_actions(inputs, deterministic=True))
    # All state-dependent reaching computation survives; only encoder outputs
    # are zero at transfer. Independent trainable encoder parameters remain.
    pointers = []
    for name in backbones:
        module = getattr(agent, name)
        for key, expected in backbones[name].items():
            torch.testing.assert_close(module.state_dict()[key], expected, rtol=0, atol=0)
        assert not module.obstacle_encoder.fusion[-1].weight.any()
        pointers.append(module.obstacle_encoder.fusion[-1].weight.data_ptr())
    assert len(set(pointers)) == 5
    assert agent.state_dict()["reaching_obstacle_aware"]
    assert not agent.initialize_obstacle_awareness(np.ones(37))  # Never reset a trained encoder.


def test_actor_critics_and_targets_learn_from_obstacles_and_cpu_matches(config):
    torch.manual_seed(42)
    agent = ReachingSACAgent(241, 6, config)
    agent.initialize_obstacle_awareness(np.ones(37))
    rng = np.random.default_rng(8)
    replay = ObstacleAwareJointHandoffReplay(32, "cpu", 1)
    for i in range(16):
        obs = ObstacleObservation(rng.normal(size=241), rng.normal(size=(i % 4, 37)))
        following = ObstacleObservation(rng.normal(size=241), rng.normal(size=((i + 1) % 4, 37)))
        replay.add(obs, rng.normal(size=6).clip(-1, 1), float(i), following, False, .99)
    before = {name: getattr(agent, name).obstacle_encoder.fusion[-1].weight.detach().clone()
              for name in ("actor", "q1", "q2", "target_q1", "target_q2")}
    for _ in range(3):
        batch = replay.sample(16)
        metrics = agent.update(batch, reference_observations=batch)
        assert all(np.isfinite(value) for value in metrics.values())
    for name, value in before.items():
        assert not torch.equal(value, getattr(agent, name).obstacle_encoder.fusion[-1].weight)
    inputs = observations()
    actions = agent.select_actions(inputs, True)
    assert not np.array_equal(actions[0], actions[1])
    assert not np.array_equal(actions[1], actions[2])
    np.testing.assert_allclose(actions, agent.select_actions(stack_observations(inputs), True), atol=1e-7)
    permutation = ObstacleObservation(np.asarray(inputs[2]), obstacle_set(inputs[2])[::-1])
    np.testing.assert_allclose(agent.select_action(inputs[2], True), agent.select_action(permutation, True), atol=1e-7)
    cpu = FrozenCPUReachingPolicy(copy.deepcopy(agent.actor))
    for obs in inputs:
        np.testing.assert_allclose(cpu.select_action(obs), agent.select_action(obs, True), atol=1e-7)
    saved = copy.deepcopy(agent.state_dict())
    restored = ReachingSACAgent(241, 6, config)
    restored.load_state_dict(saved, actor_only=True)
    assert restored.obstacle_aware
    np.testing.assert_array_equal(actions, restored.select_actions(inputs, True))


def test_old_checkpoint_load_remains_blind_even_into_perception_config(config):
    old = ReachingSACAgent(241, 6, config)
    cfg = obstacle_aware_reaching_config(config)
    restored = ReachingSACAgent(241, 6, cfg)
    restored.load_state_dict(copy.deepcopy(old.state_dict()))
    assert not restored.obstacle_aware
    np.testing.assert_array_equal(old.select_actions(observations(), True), restored.select_actions(observations(), True))
    assert cfg["thesis"]["obstacles"]["enabled"] and cfg["thesis"]["observation"]["reaching_obstacle_aware"]
    assert not config["thesis"]["obstacles"]["enabled"]


def test_variable_obstacle_replay_roundtrip_and_pending_return_preserve_both_sets():
    replay = ObstacleAwareJointHandoffReplay(3, "cpu", 1)
    inputs = observations()
    for i in range(5):  # Exercise ring overwrites, not just append-only storage.
        obs, following = inputs[i % 3], inputs[(i + 1) % 3]
        proposed = np.full(6, i / 10)
        decision = ReachingDecisionReturn(obs, proposed)
        decision.advance(1, .9)  # The controller may execute a different backup action.
        decision.advance(2, .9)
        decision.store(replay, following, False, .9)
    state = copy.deepcopy(replay.state_dict())
    assert state["protocol"] == OBSTACLE_AWARE_TRAIN_PROTOCOL
    assert replay.size == 3 and replay.ptr == 2
    for i in range(3):
        batch = replay._batch(np.array([i]))
        np.testing.assert_array_equal(batch.obstacle_features[0, batch.obstacle_mask[0]].numpy(), replay.obstacle_sets[i])
        np.testing.assert_array_equal(batch.next_obstacle_features[0, batch.next_obstacle_mask[0]].numpy(), replay.next_obstacle_sets[i])
    restored = ObstacleAwareJointHandoffReplay(3, "cpu", 99)
    restored.load_state_dict(state)
    a, b = replay.sample(7), restored.sample(7)
    for name in ("observations", "actions", "rewards", "dones", "bootstrap_discounts", "obstacle_features",
                 "obstacle_mask", "next_obstacle_features", "next_obstacle_mask"):
        torch.testing.assert_close(getattr(a, name), getattr(b, name))
    with pytest.raises(ValueError, match="requires raw obstacle sets"):
        restored.load_state_dict(JointHandoffReplay(3, "cpu", 1).state_dict())
    bad = copy.deepcopy(state)
    bad["obstacle_sets"].pop()
    with pytest.raises(ValueError, match="counts differ"):
        restored.load_state_dict(bad)
    # A pending return snapshots its set independently of future observation mutation.
    obs = inputs[1]
    pending = ReachingDecisionReturn(obs, np.ones(6))
    expected = obstacle_set(obs).copy()
    obs.obstacle_features[:] = 999
    np.testing.assert_array_equal(obstacle_set(pending.observation), expected)
