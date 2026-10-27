"""Execution optimizations must preserve SAC math and empty-set semantics."""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from rl_risk_sac.algorithms.obstacle_encoder import ObstacleSetEncoder
from rl_risk_sac.algorithms.replay_buffer import Batch
from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.tasks.obstacle_observation import cache_obstacle_rows, pad_obstacle_sets
from rl_risk_sac.utils.config import load_config
from rl_risk_sac.utils.local_training import local_training_schedule, sac_resume_contract


DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]


def test_local_schedule_uses_old_sac_batch_and_fractional_utd_without_changing_bank_contract():
    from rl_risk_sac.tasks.local_avoidance import contract_digest
    config = load_config("configs/experiments/paper_local_avoidance_geometry.yaml")
    # Reconstruct the trajectory task explicitly for compatibility tests.
    config["local_avoidance"].pop("reference_mode")
    config["local_avoidance"].pop("geometry")
    schedule = local_training_schedule(config)
    assert schedule["batch_size"] == 1024
    assert schedule["updates_per_transition"] == .25
    assert schedule["diagnostic_interval"] == 10
    previous = copy.deepcopy(config)
    previous["sac"].update(batch_size=256, updates_per_transition=1.)
    assert contract_digest(config) == contract_digest(previous)
    assert sac_resume_contract(config) == sac_resume_contract(previous)
    previous["sac"]["actor_lr"] *= 2
    assert sac_resume_contract(config) != sac_resume_contract(previous)


@pytest.mark.parametrize("rate", [0., -.25, float("nan"), float("inf")])
def test_local_schedule_rejects_invalid_update_rates(rate):
    config = load_config("configs/experiments/paper_local_avoidance_geometry.yaml")
    # Reconstruct the trajectory task explicitly for compatibility tests.
    config["local_avoidance"].pop("reference_mode")
    config["local_avoidance"].pop("geometry")
    config["sac"]["updates_per_transition"] = rate
    with pytest.raises(ValueError, match="finite and positive"):
        local_training_schedule(config)


@pytest.mark.parametrize("device", DEVICES)
def test_sparse_diagnostics_preserve_all_learning_state_and_metrics_at_collection(device):
    config = load_config("configs/base/thesis_common.yaml")
    config["device"] = device
    config["sac"].update(hidden_dims=[16, 16], actor_update_interval=2)
    reference = ThesisSACAgent(162, 6, config)
    sparse = ThesisSACAgent(162, 6, config)
    sparse.load_state_dict(copy.deepcopy(reference.state_dict()))
    batch = Batch(torch.randn(16, 162, device=device), torch.randn(16, 6, device=device).tanh(),
                  torch.randn(16, 1, device=device), torch.zeros(16, 1, device=device),
                  torch.randn(16, 162, device=device), torch.zeros(16, 1, device=device))
    for update in range(1, 13):
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        expected = reference.update(batch, reference_observations=batch)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        collect = update == 1 or update % 10 == 0
        actual = sparse.update(batch, reference_observations=batch,
                               collect_diagnostics=collect, check_finite=True)
        assert actual == (expected if collect else {})
        for name in ("actor", "q1", "q2", "target_q1", "target_q2", "chain_reference_actor"):
            for key, value in getattr(sparse, name).state_dict().items():
                torch.testing.assert_close(value, getattr(reference, name).state_dict()[key], rtol=0, atol=0)
        assert sparse.chain_pcr_sac_loss_ema == reference.chain_pcr_sac_loss_ema
        assert sparse.chain_pcr_loss_ema == reference.chain_pcr_loss_ema


def test_sparse_diagnostics_still_reject_nonfinite_learning_values_on_cpu():
    config = load_config("configs/base/thesis_common.yaml")
    config["device"] = "cpu"
    config["sac"].update(hidden_dims=[8, 8], chain_pcr_enabled=False, actor_update_interval=2)
    agent = ThesisSACAgent(162, 6, config)
    batch = Batch(torch.zeros(8, 162), torch.zeros(8, 6), torch.full((8, 1), float("nan")),
                  torch.zeros(8, 1), torch.zeros(8, 162), torch.zeros(8, 1))
    with pytest.raises(RuntimeError, match="nonfinite local SAC update"):
        agent.update(batch, collect_diagnostics=False, check_finite=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("counts", [(1, 2, 3), (0, 3, 0, 1), (0, 0)])
def test_cached_rows_match_fallback_outputs_gradients_and_empty_sets(device, counts):
    torch.manual_seed(318)
    optimized = ObstacleSetEncoder(np.ones(37)).to(device)
    reference = copy.deepcopy(optimized)
    # Include fully masked positive padding, which must keep the old no-graph
    # behavior (None gradients, so Adam does not advance unused parameters).
    maximum = max(1, max(counts))
    features = torch.randn(len(counts), maximum, 37, device=device, requires_grad=True)
    other = features.detach().clone().requires_grad_(True)
    raw_mask = torch.arange(maximum, device=device)[None] < torch.tensor(counts, device=device)[:, None]
    cached_mask = cache_obstacle_rows(raw_mask.clone(), counts)
    actual, expected = optimized(features, cached_mask), reference(other, raw_mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.requires_grad == expected.requires_grad
    if expected.requires_grad:
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(features.grad, other.grad, rtol=0, atol=0)
        for left, right in zip(optimized.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    else:
        assert features.grad is other.grad is None
        assert all(parameter.grad is None for parameter in optimized.parameters())
    empty, mask = pad_obstacle_sets([np.empty((0, 37), np.float32)] * len(counts), device)
    assert optimized(empty, mask).shape == (len(counts), 128)
    assert not optimized(empty, mask).any()


@pytest.mark.parametrize("device", DEVICES)
def test_cached_encoder_forward_avoids_dynamic_nonzero_and_scalar_reads(device):
    encoder = ObstacleSetEncoder(np.ones(37)).to(device)
    for counts in ((1, 3), (0, 3), (0, 0)):
        values = [np.ones((count, 37), np.float32) for count in counts]
        features, mask = pad_obstacle_sets(values, device)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            encoder(features, mask)
        operations = {event.key for event in profiler.key_averages()}
        assert "aten::nonzero" not in operations
        assert "aten::_local_scalar_dense" not in operations


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("pattern", [(1, 2, 3), (0, 1, 3)])
def test_cached_rows_preserve_complete_sac_updates_with_noncontiguous_downstream_gradients(device, pattern):
    config = load_config("configs/base/thesis_common.yaml")
    config["device"] = device
    config["sac"].update(hidden_dims=[16, 16], actor_update_interval=1)
    optimized = ThesisSACAgent(241, 6, config)
    reference = ThesisSACAgent(241, 6, config)
    reference.load_state_dict(copy.deepcopy(optimized.state_dict()))
    counts = [pattern[i % len(pattern)] for i in range(256)]
    sets = [np.random.default_rng(i).normal(size=(count, 37)).astype(np.float32)
            for i, count in enumerate(counts)]
    features, mask = pad_obstacle_sets(sets, device)
    batch = Batch(torch.randn(256, 241, device=device), torch.randn(256, 6, device=device).tanh(),
                  torch.randn(256, 1, device=device), torch.zeros(256, 1, device=device),
                  torch.randn(256, 241, device=device), torch.zeros(256, 1, device=device),
                  obstacle_features=features, obstacle_mask=mask,
                  next_obstacle_features=features, next_obstacle_mask=mask)
    fallback = replace(batch, obstacle_mask=mask.clone(), next_obstacle_mask=mask.clone())
    for _ in range(4):
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        expected = reference.update(fallback, reference_observations=fallback)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        actual = optimized.update(batch, reference_observations=batch)
        assert actual == expected
        for name in ("actor", "q1", "q2", "target_q1", "target_q2", "chain_reference_actor"):
            left, right = getattr(optimized, name), getattr(reference, name)
            if left is not None:
                for key, value in left.state_dict().items():
                    torch.testing.assert_close(value, right.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["ordinary", "capturable", "fused"])
def test_sac_resume_hosts_adam_steps_according_to_optimizer_mode(device, mode):
    config = load_config("configs/base/thesis_common.yaml")
    config["device"] = device
    config["sac"].update(hidden_dims=[8, 8], chain_pcr_enabled=False, actor_update_interval=1)
    agent = ThesisSACAgent(162, 6, config)
    batch = Batch(torch.randn(8, 162, device=device), torch.randn(8, 6, device=device).tanh(),
                  torch.randn(8, 1, device=device), torch.zeros(8, 1, device=device),
                  torch.randn(8, 162, device=device), torch.zeros(8, 1, device=device))
    fresh = copy.deepcopy(agent.state_dict())
    restored = ThesisSACAgent(162, 6, config)
    restored.load_state_dict(fresh)
    assert all(not optimizer.state for optimizer in (
        restored.actor_optimizer, restored.q_optimizer, restored.alpha_optimizer))
    agent.update(batch)
    state = copy.deepcopy(agent.state_dict())
    for name in ("actor_optimizer", "q_optimizer", "alpha_optimizer"):
        for group in state[name]["param_groups"]:
            group.update(capturable=mode == "capturable", fused=mode == "fused")
        if torch.cuda.is_available():
            for values in state[name]["state"].values():
                values["step"] = values["step"].cuda()  # Older GPU-hosted checkpoints.
    restored.load_state_dict(state)
    for optimizer in (restored.actor_optimizer, restored.q_optimizer, restored.alpha_optimizer):
        assert optimizer.state
        for parameter, values in optimizer.state.items():
            expected_device = "cpu" if mode == "ordinary" else parameter.device.type
            assert values["step"].device.type == expected_device
            assert values["step"].item() == 1
            assert values["exp_avg"].device == parameter.device
            assert values["exp_avg_sq"].device == parameter.device
    if mode == "ordinary":
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        expected = agent.update(batch)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        actual = restored.update(batch)
        assert actual == expected
        for left, right in zip(agent.actor.parameters(), restored.actor.parameters(), strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        for left, right in zip(agent.q1.parameters(), restored.q1.parameters(), strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
