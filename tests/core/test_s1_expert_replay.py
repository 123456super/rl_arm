"""Synthetic plumbing tests, not physical demonstrations or policy benchmarks."""
import copy

import numpy as np
import pytest
import torch

from rl_risk_sac.algorithms.homotopy_replay import HomotopyReplayBuffer, _Partition, _vectorized_replay_rewards
from rl_risk_sac.algorithms.s1_expert_replay import (
    candidate_rejection, contract_sha256, expert_contract, validate_expert_bank,
)
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.tasks.obstacle_observation import ObstacleObservation
from rl_risk_sac.utils.config import load_config
from scripts.core.train_thesis_homotopy import apply_stage_profile, configure_s1_expert_replay


@pytest.fixture(scope="module")
def setup():
    config = load_config("configs/experiments/paper_full.yaml")
    apply_stage_profile(config, "s1")
    env = ThesisHomotopyEnv(config)
    parameters = dict(env.reward_parameters)
    env.close()
    return config, parameters, expert_contract(config)


def raw_row(level=0, terminal=False):
    raw = np.zeros(29, np.float32)
    raw[[0, 1]] = .01
    raw[[2, 3]] = .02
    raw[5] = terminal
    raw[12] = .2
    raw[16] = 0. if level == 0 else .25
    raw[18] = .25
    raw[23] = level
    raw[24:26] = [.1, .3] if level == 0 else [.08, .24]
    raw[26:29] = [.02, .02, .8]
    return raw


@pytest.fixture
def bank(setup):
    _, parameters, contract = setup
    part = _Partition(241, 6, 200)
    obs = ObstacleObservation(np.full(241, 99, np.float32), np.zeros((1, 37), np.float32))
    episodes = []
    for episode in range(10):
        for step in range(1, 21):
            part.add(obs, np.zeros(6), obs, step == 20, raw_row(terminal=step == 20), episode, step)
        indices = part.episode_indices(episode)
        rewards, _ = _vectorized_replay_rewards(part.raw[indices], part.done[indices],
                                               np.ones(20), 1., .99, parameters)
        episodes.append(dict(
            episode_id=episode, level=0, seed=0x80000000 + episode, steps=20,
            cause="success", physical_substep_margin_passed=True, fresh_training_scene=True,
            min_advantage=2., min_external_mesh_distance_m=.08, min_self_mesh_distance_m=.02,
            physics_substep_checks=241,
            discounted_return=float(np.dot(.99 ** np.arange(20), rewards)),
            discounted_self_penalty=0., reference=dict(discounted_return=-100., discounted_self_penalty=0.),
            waiting=dict(discounted_return=-100., discounted_self_penalty=0.),
        ))
    return dict(format_version=1, stage="s1", complete=True, contract=copy.deepcopy(contract),
                contract_sha256=contract_sha256(contract), reward_parameters=parameters,
                partitions={0: part}, episodes=episodes)


def replay(setup, level=0):
    _, parameters, _ = setup
    result = HomotopyReplayBuffer(
        241, 6, "cpu", seed=11, s0_joint_pose=True, cross_level_replay=False,
        s0_anchor_fraction=0., s0_current_fraction=.8, s0_success_fraction=.2,
        s0_current_capacity=1200, s0_anchor_capacity=1, s0_success_capacity=1,
        reward_parameters=parameters,
    )
    result.begin_orientation_level(level)
    obs = ObstacleObservation(np.full(241, -2, np.float32))
    for i in range(1200):
        result.s0_current.add(obs, np.zeros(6), obs, False, raw_row(level), 1000 + i // 20, i % 20 + 1)
    return result


def configure(replay, bank, fraction=.1):
    replay.configure_s1_experts(bank, fraction=fraction, source="synthetic-test-only", sha256="a" * 64)


def sample(replay, stage="s1", level=0):
    return replay.sample(stage, {"none": 1.}, 1024, curriculum_level=level, lambda_self=.2)


def test_valid_bank(setup, bank):
    validate_expert_bank(bank, setup[2])


def test_coupling_bank_migration_is_valid_non_destructive_and_preserves_advantage(setup, bank):
    from rl_risk_sac.algorithms.s1_expert_replay import relabel_coupling_bank
    from rl_risk_sac.tasks.goal_obstacle_coupling import coupling_parameters
    for part in bank["partitions"].values():
        for values in (*part.obstacle_sets, *part.next_obstacle_sets):
            values[:, :4] = .08
    config = copy.deepcopy(setup[0])
    config["thesis"]["reward"]["goal_obstacle_coupling"] = dict(enabled=True, beta=1.)
    parameters = dict(setup[1], **coupling_parameters(config))
    original = bank["episodes"][0]["discounted_return"]
    migrated = relabel_coupling_bank(bank, config, parameters)
    validate_expert_bank(migrated, expert_contract(config))
    assert bank["episodes"][0]["discounted_return"] == original
    assert migrated["episodes"][0]["discounted_return"] < original
    a, b = bank["episodes"][0], migrated["episodes"][0]
    assert a["discounted_return"] - a["waiting"]["discounted_return"] == pytest.approx(
        b["discounted_return"] - b["waiting"]["discounted_return"])
    assert migrated["coupling_relabel"]["max_telescoping_residual"] < 1e-6
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_expert_bank(bank, expert_contract(config))
    config["thesis"]["reward"]["success_bonus"] += 1
    with pytest.raises(ValueError, match="base reward"):
        relabel_coupling_bank(bank, config, parameters)


@pytest.mark.parametrize("mutation,match", [
    (lambda b: b.update(complete=False), "quota"),
    (lambda b: b["episodes"][0].update(physics_substep_checks=21), "physical-substep"),
    (lambda b: b["episodes"][0].update(min_external_mesh_distance_m=.02), "physical-substep"),
    (lambda b: b["partitions"][0].raw.__setitem__((0, 23), 1), "precision"),
    (lambda b: b["partitions"][0].raw.__setitem__((0, 16), 1), "precision"),
    (lambda b: b["partitions"][0].done.__setitem__((5, 0), 1), "terminal"),
    (lambda b: b["partitions"][0].obs.__setitem__((1, 0), 0), "discontinuous"),
    (lambda b: b["partitions"][0].action.__setitem__((0, 0), 2), "normalized"),
    (lambda b: b["episodes"][0].update(discounted_return=500.), "stored return"),
    (lambda b: b["episodes"][0].update(discounted_self_penalty=500.), "stored self penalty"),
])
def test_bad_bank_is_rejected(bank, mutation, match):
    mutation(bank)
    with pytest.raises(ValueError, match=match):
        validate_expert_bank(bank)


def test_reward_contract_mismatch(setup, bank):
    changed = copy.deepcopy(setup[0])
    changed["thesis"]["reward"]["success_bonus"] = 60.
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_expert_bank(bank, expert_contract(changed))


def test_exact_minibatch_quota_without_online_contamination(setup, bank):
    buffer = replay(setup)
    configure(buffer, bank)
    batch = sample(buffer)
    assert int((batch.observations[:, 0] == 99).sum()) == 102
    assert buffer.last_s1_expert_count == 102
    assert buffer.last_orientation_sample_counts["current_recent"] == 922
    assert len(buffer) == 1200
    assert buffer.current_success_episode_count == 0
    assert not batch.protected_mask.any()
    assert batch.obstacle_features.shape[1:] == (1, 37)
    assert int(batch.obstacle_mask.sum()) == 102


@pytest.mark.parametrize("stage,fraction", [("s0", .1), ("s2", .1), ("s1", 0.)])
def test_disabled_or_other_stage_is_bit_identical(setup, bank, stage, fraction):
    baseline, augmented = replay(setup), replay(setup)
    configure(augmented, bank, fraction)
    a, b = sample(baseline, stage), sample(augmented, stage)
    for key in ("observations", "actions", "rewards", "next_observations", "dones"):
        assert torch.equal(getattr(a, key), getattr(b, key))
    assert augmented.last_s1_expert_count == 0


def test_missing_level_never_reuses_old_experts(setup, bank, capsys):
    buffer = replay(setup, level=1)
    configure(buffer, bank)
    batch = sample(buffer, level=1)
    assert not (batch.observations[:, 0] == 99).any()
    assert buffer.last_s1_expert_count == 0
    assert "no L1 bank" in capsys.readouterr().out


def test_checkpoint_restores_expert_sampling_rng(setup, bank, tmp_path):
    buffer = replay(setup)
    configure(buffer, bank)
    sample(buffer)
    path = tmp_path / "test-only.pt"
    torch.save(buffer.state_dict(), path)
    expected = sample(buffer)
    restored = replay(setup)
    restored.load_state_dict(torch.load(path, weights_only=False))
    configure(restored, bank)  # Re-specifying the same bank must not reset RNG.
    actual = sample(restored)
    for key in ("observations", "actions", "rewards", "dones", "obstacle_features"):
        assert torch.equal(getattr(expected, key), getattr(actual, key))


def test_old_checkpoint_still_loads_without_expert_fields(setup, bank):
    old = replay(setup).state_dict()
    old.pop("s1_experts")
    buffer = replay(setup)
    configure(buffer, bank)
    buffer.load_state_dict(old)
    assert not buffer.s1_expert_partitions
    assert buffer.s1_expert_fraction == 0


def test_training_helper_is_opt_in_and_inherits_bank(setup, bank, tmp_path):
    config = setup[0]
    buffer = replay(setup)
    assert not configure_s1_expert_replay(buffer, config)["enabled"]
    path = tmp_path / "synthetic-test-only.pt"
    torch.save(bank, path)
    manifest = configure_s1_expert_replay(buffer, config, bank_path=path)
    assert manifest["enabled"] and manifest["fraction"] == .1
    assert manifest["episodes"] == 10
    assert not manifest["behavior_cloning"]
    assert configure_s1_expert_replay(buffer, config)["fraction"] == .1
    assert not configure_s1_expert_replay(buffer, config, fraction=0.)["enabled"]


def test_admission_checks_both_safety_weights():
    candidate = dict(cause="success", physical_substep_margin_passed=True,
                     discounted_return=4., discounted_self_penalty=0.)
    baseline = dict(discounted_return=1., discounted_self_penalty=4.)
    assert candidate_rejection(candidate, [baseline], 2.) == "return_advantage"
    candidate["discounted_return"] = 8.
    assert candidate_rejection(candidate, [baseline], 2.) is None
    candidate["physical_substep_margin_passed"] = False
    assert candidate_rejection(candidate, [baseline], 2.) == "safety_margin"


def test_read_only_observer_does_not_change_physics_or_reward(setup):
    config = copy.deepcopy(setup[0])
    config["thesis"]["obstacles"]["path_sampling"]["probabilities"] = [1., 0., 0., 0.]
    env = ThesisHomotopyEnv(config)
    try:
        env.configure_episode("none", xi=1., strict=True, position_tolerance=.1,
                              orientation_tolerance=.3)
        env.reset(seed=701)
        state = env.episode_state_dict()
        a = env.step(np.zeros(6))
        env.restore_episode_state(state)
        observed = []
        b = env.step(np.zeros(6), substep_observer=lambda: observed.append(env.robot.joint_state_exact()[0]))
        assert len(observed) == b[-1]["executed_substeps"] == 12
        assert a[1:5] == b[1:5]
        np.testing.assert_array_equal(a[0], b[0])
    finally:
        env.close()
