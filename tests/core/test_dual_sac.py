from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from rl_risk_sac.algorithms.dual_sac import (
    DualSACAgent, build_thesis_sac, checkpoint_policy_state,
    validate_dual_sac_config,
)
from rl_risk_sac.algorithms.replay_buffer import Batch
from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.tasks.obstacle_observation import ObstacleObservation
from rl_risk_sac.utils.config import load_config
from scripts.core.train_thesis_homotopy import (
    apply_stage_profile, checkpoint_actor_state, initialize_frozen_reaching,
)


@pytest.fixture
def config(tmp_path):
    config = load_config("configs/base/dual_sac.yaml")
    config["device"] = "cpu"
    config["sac"]["hidden_dims"] = [16, 16]
    config["sac"]["actor_update_interval"] = 1
    apply_stage_profile(config, "s1")
    # Static S1 loads normalization only, never a temporal predictor.
    scale = tmp_path / "scale.pt"
    torch.save({"format_version": 1, "accepted": True,
                "feature_scale": np.ones(37, np.float32)}, scale)
    config["thesis"]["obstacles"]["predictor_checkpoint"] = str(scale)
    return config


def batch():
    return Batch(
        observations=torch.randn(8, 241), actions=torch.randn(8, 6).tanh(),
        rewards=torch.randn(8, 1), costs=torch.zeros(8, 1),
        next_observations=torch.randn(8, 241), dones=torch.zeros(8, 1),
        obstacle_features=torch.randn(8, 2, 37),
        obstacle_mask=torch.tensor([[True, False], [True, True]] * 4),
        next_obstacle_features=torch.randn(8, 2, 37),
        next_obstacle_mask=torch.ones(8, 2, dtype=torch.bool),
    )


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for x, y in zip(left, right):
            assert_state_equal(x, y)
    else:
        assert left == right


@pytest.mark.parametrize("role", ["reaching", "avoidance"])
def test_only_selected_sac_updates_including_optimizers_and_targets(config, role):
    config["dual_sac"]["training_policy"] = role
    apply_stage_profile(config, "s0" if role == "reaching" else "s1")
    agent = DualSACAgent(241, 6, config)
    inactive = agent.avoidance if role == "reaching" else agent.reaching
    before = copy.deepcopy(inactive.state_dict())
    actor_before = copy.deepcopy(agent.actor.state_dict())
    critic_before = copy.deepcopy(agent.q1.state_dict())
    sample = batch()
    metrics = agent.update(sample, reference_observations=sample)
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(value, actor_before[key])
               for key, value in agent.actor.state_dict().items())
    assert any(not torch.equal(value, critic_before[key])
               for key, value in agent.q1.state_dict().items())
    assert agent.update_steps == 1
    assert inactive.update_steps == 0
    assert not inactive.log_alpha.requires_grad
    assert_state_equal(before, inactive.state_dict())
    active_storage = {p.data_ptr() for module in
                      (agent.actor, agent.q1, agent.q2, agent.target_q1, agent.target_q2)
                      for p in module.parameters()}
    inactive_storage = {p.data_ptr() for module in
                        (inactive.actor, inactive.q1, inactive.q2, inactive.target_q1, inactive.target_q2)
                        for p in module.parameters()}
    assert active_storage.isdisjoint(inactive_storage)
    assert agent.log_alpha.data_ptr() != inactive.log_alpha.data_ptr()


def test_reaching_action_ignores_obstacles_and_selection_is_explicit(config):
    agent = DualSACAgent(241, 6, config)
    state = np.random.default_rng(4).normal(size=241).astype(np.float32)
    empty = ObstacleObservation(state)
    blocked = ObstacleObservation(state, np.full((2, 37), -1, np.float32))
    empty_action = agent.select_action(empty, True, policy="reaching")
    blocked_action = agent.select_action(blocked, True, policy="reaching")
    np.testing.assert_array_equal(empty_action, blocked_action)
    np.testing.assert_array_equal(empty_action, agent.reaching.select_action(blocked, True))
    np.testing.assert_array_equal(
        agent.select_action(blocked, True), agent.avoidance.select_action(blocked, True))
    with pytest.raises(ValueError, match="unknown SAC policy"):
        agent.select_action(empty, policy="auto")


def test_reaching_update_external_features_do_not_change_learning(config):
    config["dual_sac"]["training_policy"] = "reaching"
    apply_stage_profile(config, "s0")
    agent = DualSACAgent(241, 6, config)
    twin = DualSACAgent(241, 6, config)
    twin.load_state_dict(copy.deepcopy(agent.state_dict()))
    sample = batch()
    empty = replace(sample, obstacle_features=None, obstacle_mask=None,
                    next_obstacle_features=None, next_obstacle_mask=None)
    torch.manual_seed(2026)
    agent.update(sample, reference_observations=sample)
    torch.manual_seed(2026)
    twin.update(empty, reference_observations=empty)
    assert_state_equal(agent.state_dict(), twin.state_dict())


def test_complete_roundtrip_restores_both_sacs_and_named_actor_exports(config, tmp_path):
    agent = DualSACAgent(241, 6, config)
    sample = batch()
    agent.update(sample, reference_observations=sample)
    saved = copy.deepcopy(agent.state_dict())
    restored = DualSACAgent(241, 6, config)
    restored.load_state_dict(saved)
    assert_state_equal(saved, restored.state_dict())
    assert not restored.reaching.actor.backbone[0].weight.requires_grad
    path = tmp_path / "actor_step_0000001.pt"
    restored.save_actor(path)
    for role in ("reaching", "avoidance"):
        actor = torch.load(path.with_name(f"{role}_{path.name}"), weights_only=False)
        assert_state_equal(actor, restored.policy(role).actor.state_dict())
    full = tmp_path / "checkpoint.pt"
    torch.save({"agent": saved}, full)
    assert_state_equal(checkpoint_actor_state(full, torch.device("cpu"), "reaching"),
                       agent.reaching.actor.state_dict())


def test_checkpoint_rejects_single_dual_and_role_confusion(config):
    agent = DualSACAgent(241, 6, config)
    single = ThesisSACAgent(241, 6, config)
    with pytest.raises(ValueError, match="both policies"):
        agent.load_state_dict(single.state_dict())
    with pytest.raises(ValueError, match="matching dual_sac"):
        single.load_state_dict(agent.state_dict())
    other = copy.deepcopy(config)
    other["dual_sac"]["training_policy"] = "reaching"
    apply_stage_profile(other, "s0")
    with pytest.raises(ValueError, match="training_policy differs"):
        DualSACAgent(241, 6, other).load_state_dict(agent.state_dict())
    corrupt = agent.state_dict()
    del corrupt["dual_sac"]["policies"]["reaching"]
    with pytest.raises(ValueError, match="both complete policies"):
        checkpoint_policy_state(corrupt)


def test_import_complete_s0_preserves_reaching_without_touching_avoidance(config, tmp_path):
    source_config = copy.deepcopy(config)
    apply_stage_profile(source_config, "s0")
    source = ThesisSACAgent(241, 6, source_config)
    sample = replace(batch(), obstacle_features=None, obstacle_mask=None,
                     next_obstacle_features=None, next_obstacle_mask=None)
    source.update(sample, reference_observations=sample)
    checkpoint = tmp_path / "s0.pt"
    torch.save({"agent": source.state_dict(), "curriculum": {"stage": "s0"}}, checkpoint)
    dual = DualSACAgent(241, 6, config)
    avoidance = copy.deepcopy(dual.avoidance.state_dict())
    metadata = initialize_frozen_reaching(dual, checkpoint)
    assert metadata["complete_sac"] and metadata["frozen"]
    assert_state_equal(source.state_dict(), dual.reaching.state_dict())
    assert_state_equal(avoidance, dual.avoidance.state_dict())
    assert not dual.reaching.log_alpha.requires_grad


def test_dual_config_enforces_original_reward_scene_separation(config):
    validate_dual_sac_config(config)
    config["dual_sac"]["training_policy"] = "reaching"
    with pytest.raises(ValueError, match="obstacle-free S0"):
        validate_dual_sac_config(config)
    apply_stage_profile(config, "s0")
    validate_dual_sac_config(config)
    config["dual_sac"]["training_policy"] = "avoidance"
    with pytest.raises(ValueError, match="obstacle-enabled"):
        validate_dual_sac_config(config)


def test_disabled_dual_config_keeps_existing_single_sac(config):
    config.pop("dual_sac")
    assert isinstance(build_thesis_sac(241, 6, config), ThesisSACAgent)
