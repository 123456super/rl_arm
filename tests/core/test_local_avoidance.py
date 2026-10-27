from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from rl_risk_sac.algorithms.dual_sac import build_thesis_sac
from rl_risk_sac.algorithms.local_replay import LocalReplayBuffer
from rl_risk_sac.envs.local_avoidance_env import LocalAvoidanceEnv
from rl_risk_sac.envs.obstacle_scene import MovingObstacle
from rl_risk_sac.tasks.local_avoidance import (
    LOCAL_OBSERVATION_DIM, PROTOCOL, actor_digest, arm_vector, contract_digest,
    local_context, local_reward, passage_candidate, passage_progress, validate_local_config,
    obstacle_count_probabilities,
)
from rl_risk_sac.tasks.local_samples import validate_bank
from rl_risk_sac.tasks.obstacle_observation import ObstacleObservation
from rl_risk_sac.utils.config import load_config
from scripts.core.train_thesis_homotopy import apply_stage_profile, load_reference_actor


@pytest.fixture
def config(tmp_path):
    config = load_config("configs/experiments/paper_local_avoidance_geometry.yaml")
    # Reconstruct the trajectory task explicitly for compatibility tests.
    config["local_avoidance"].pop("reference_mode")
    config["local_avoidance"].pop("geometry")
    config["device"] = "cpu"
    config["sac"]["hidden_dims"] = [16, 16]
    config["sac"]["actor_update_interval"] = 1
    apply_stage_profile(config, "s1")
    scale = tmp_path / "scale.pt"
    torch.save({"format_version": 1, "accepted": True,
                "feature_scale": np.ones(37, np.float32)}, scale)
    config["thesis"]["obstacles"]["predictor_checkpoint"] = str(scale)
    return config


def sample_for_env(env, *, progress=0.):
    env.base.configure_episode("none", xi=1., strict=True, position_tolerance=.1,
                               orientation_tolerance=.3, terminate_on_success=False)
    env.base.reset(seed=1234)
    state = copy.deepcopy(env.base.episode_state_dict())
    obstacle = MovingObstacle(0, "sphere", [.04], "small", [-.6, -.6, 1.],
                              [0., 0., 0., 1.], [1., 0., 0.], "constant", .1, 1., 0., -1., 1.)
    state["obstacles"] = [obstacle.metadata()]
    vector = arm_vector(env.base)
    direction = np.ones(24) / np.sqrt(24)
    return dict(sample_id="unit-local-start", state=state,
                entry_arm=vector - progress * .2 * direction, direction=direction,
                length=.2, reference_exit_arm=vector + (1. - progress) * .2 * direction)


def test_retreat_or_clearance_alone_cannot_succeed(config):
    local = config["local_avoidance"]
    assert not passage_candidate(-2., 2., local)
    assert not passage_candidate(.9, 2., local)
    assert not passage_candidate(1.1, .001, local)
    assert passage_candidate(1., local["release_distance_m"], local)


def test_progress_and_context_expose_local_reference_and_time():
    direction = np.ones(24) / np.sqrt(24)
    sample = dict(entry_arm=np.zeros(24), direction=direction, length=.5)
    assert passage_progress(-direction, sample) == pytest.approx(-2.)
    context = local_context(.25 * direction, sample, 5, 10)
    assert context.shape == (27,)
    assert context[-3:] == pytest.approx([.25, .5, .5])


def test_local_reward_has_no_final_pose_bonus_and_cycle_progress_cancels(config):
    reward = config["local_avoidance"]["reward"]
    zero = local_reward(0., 0., 0., 0., 0., False, False, False, reward)
    assert zero == -reward["time_cost"]
    assert local_reward(0., 0., 0., 0., 0., True, False, False, reward) == zero + reward["success_bonus"]
    forward = local_reward(0., 10., 0., 0., 0., False, False, False, reward)
    backward = local_reward(10., 0., 0., 0., 0., False, False, False, reward)
    assert forward + backward == pytest.approx(-2 * reward["time_cost"])


def test_local_sac_keeps_241d_reaching_and_independent_268d_set_policy(config, tmp_path):
    agent = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    assert agent.reaching.actor_obs_dim == 241
    assert agent.avoidance.actor_obs_dim == LOCAL_OBSERVATION_DIM
    assert agent.avoidance.set_obstacles
    assert agent.avoidance.critic_obs_dim == LOCAL_OBSERVATION_DIM - 54
    old = build_thesis_sac(241, 6, config)
    path = tmp_path / "old_actor.pt"
    torch.save(old.avoidance.actor.state_dict(), path)
    initialized = load_reference_actor(agent.avoidance, path)
    assert torch.count_nonzero(initialized.backbone[0].weight[:, 241:]) == 0
    observation = torch.randn(2, LOCAL_OBSERVATION_DIM)
    torch.testing.assert_close(initialized.deterministic(observation),
                               old.avoidance.actor.deterministic(observation[:, :241]))
    state = copy.deepcopy(agent.state_dict())
    restored = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    restored.load_state_dict(state)
    assert actor_digest(restored.reaching.actor) == actor_digest(agent.reaching.actor)


def test_global_target_hit_is_not_local_success(config):
    agent = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    env = LocalAvoidanceEnv(config, agent.reaching)
    try:
        sample = sample_for_env(env)
        position, quaternion = env.base.robot.end_effector_pose()
        sample["state"]["goal_position"] = position
        sample["state"]["goal_quaternion"] = quaternion
        env.reset(options={"sample": sample})
        env.base.success_hold_steps = 1
        _, reward, terminated, truncated, info = env.step(np.zeros(6))
        assert info["position_error_m"] < .001
        assert not info["local_success"]
        assert not terminated and not truncated
        assert reward <= 0.
    finally:
        env.close()


@pytest.mark.parametrize("recovery", [False, True])
def test_local_success_needs_verified_handoff_not_final_goal(config, monkeypatch, recovery):
    agent = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    env = LocalAvoidanceEnv(config, agent.reaching)
    try:
        sample = sample_for_env(env, progress=1.1)
        sample["state"]["goal_position"] = np.asarray([.7, .7, 1.])
        env.reset(options={"sample": sample})
        monkeypatch.setattr(env, "_recovery", lambda: recovery)
        _, _, terminated, _, info = env.step(np.zeros(6))
        assert info["position_error_m"] > .1
        assert info["passage_candidate"]
        assert info["local_success"] is recovery
        assert terminated is recovery
    finally:
        env.close()


def test_collision_overrides_passage_success(config, monkeypatch):
    agent = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    env = LocalAvoidanceEnv(config, agent.reaching)
    try:
        sample = sample_for_env(env, progress=1.1)
        env.reset(options={"sample": sample})
        monkeypatch.setattr(env.base, "_collision_events", lambda: dict(
            obstacle_collision=False, self_collision=True, environment_collision=False))
        monkeypatch.setattr(env, "_recovery", lambda: pytest.fail("collision cannot hand off"))
        _, _, terminated, _, info = env.step(np.zeros(6))
        assert terminated and info["local_failure"] and not info["local_success"]
    finally:
        env.close()


def test_recovery_rollout_does_not_mutate_live_physics(config):
    agent = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    env = LocalAvoidanceEnv(config, agent.reaching)
    try:
        env.reset(options={"sample": sample_for_env(env)})
        before = env.base.episode_state_dict()
        env._recovery()
        after = env.base.episode_state_dict()
        for key in ("q", "qdot", "goal_position", "goal_quaternion"):
            np.testing.assert_array_equal(before[key], after[key])
        for key in ("step_count", "physics_step_count", "success_hold_count", "obstacle_contact_seen"):
            assert before[key] == after[key]
    finally:
        env.close()


def test_local_timeout_state_and_replay_roundtrip(config):
    config["local_avoidance"]["horizon"] = 1
    agent = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    env = LocalAvoidanceEnv(config, agent.reaching)
    try:
        observation, _ = env.reset(options={"sample": sample_for_env(env)})
        nxt, reward, terminated, truncated, _ = env.step(np.zeros(6))
        assert not terminated and truncated and nxt[-1] == 0.
        replay = LocalReplayBuffer(8, "cpu", 1)
        replay.add(observation, np.zeros(6), reward, nxt, terminated or truncated)
        restored = LocalReplayBuffer(8, "cpu", 999)
        restored.load_state_dict(copy.deepcopy(replay.state_dict()))
        a, b = replay.sample(4), restored.sample(4)
        torch.testing.assert_close(a.observations, b.observations)
        torch.testing.assert_close(a.obstacle_features, b.obstacle_features)
        assert a.dones.all()
        metrics = agent.update(a, reference_observations=a)
        assert all(np.isfinite(value) for value in metrics.values())
        with pytest.raises(ValueError, match="passage context"):
            replay.add(observation[:241], np.zeros(6), reward, nxt, True)
    finally:
        env.close()


def test_bank_rejects_global_replay_wrong_actor_and_hidden_reference(config):
    agent = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    with pytest.raises(ValueError, match="global-reaching"):
        validate_bank({}, config, agent.reaching)
    bank = dict(protocol=PROTOCOL, format_version=1, contract_sha256=contract_digest(config),
                policy_sha256="another-policy")
    with pytest.raises(ValueError, match="frozen reaching"):
        validate_bank(bank, config, agent.reaching)


def test_local_protocol_rejects_dynamic_scenes(config):
    config["thesis"]["obstacles"]["dynamic"] = True
    with pytest.raises(ValueError, match="static obstacle"):
        validate_local_config(config)


def test_multi_obstacle_encoder_is_permutation_invariant_and_updates(config):
    config["local_avoidance"]["obstacle_count_probabilities"] = [0., .5, .3, .2]
    validate_local_config(config)
    agent = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    replay = LocalReplayBuffer(8, "cpu", 12)
    state = np.random.default_rng(4).normal(size=LOCAL_OBSERVATION_DIM).astype(np.float32)
    for count in (1, 2, 3):
        features = np.random.default_rng(count).normal(size=(count, 37)).astype(np.float32)
        current = ObstacleObservation(state, features)
        reverse = ObstacleObservation(state, features[::-1])
        np.testing.assert_allclose(agent.select_action(current, True),
                                   agent.select_action(reverse, True), atol=1e-6)
        replay.add(current, np.zeros(6), 0., reverse, False)
    sample = replay.sample(16)
    assert set(sample.obstacle_mask.sum(dim=1).tolist()) == {1, 2, 3}
    assert sample.obstacle_features.shape == (16, 3, 37)
    metrics = agent.update(sample, reference_observations=sample)
    assert all(np.isfinite(value) for value in metrics.values())


def test_second_obstacle_participates_in_collision_and_features(config):
    config["local_avoidance"]["obstacle_count_probabilities"] = [0., 0., 1.]
    agent = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    env = LocalAvoidanceEnv(config, agent.reaching)
    try:
        sample = sample_for_env(env)
        second = copy.deepcopy(sample["state"]["obstacles"][0])
        second["identity"] = 1
        second["initial_position"] = [-.6, .6, 1.]
        sample["state"]["obstacles"].append(second)
        observation, _ = env.reset(options={"sample": sample})
        assert observation.obstacle_features.shape == (2, 37)
        # Place the second object on the actual end effector; the first stays far.
        position, _ = env.base.robot.end_effector_pose()
        env.base.obstacles[1].initial_position = np.asarray(position).tolist()
        nxt, _, terminated, _, info = env.step(np.zeros(6))
        assert nxt.obstacle_features.shape == (2, 37)
        assert env.base._static_clearance(env.base.obstacles[0]) > .1
        assert terminated and info["local_failure"]
        assert not info["local_success"]
    finally:
        env.close()


@pytest.mark.parametrize("probabilities", [[.1, .9], [0., .4, .4], [0., -1., 2.], [0., float('nan'), 1.]])
def test_invalid_multi_count_distribution_fails(config, probabilities):
    config["local_avoidance"]["obstacle_count_probabilities"] = probabilities
    with pytest.raises(ValueError, match="count probabilities"):
        obstacle_count_probabilities(config)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_lightweight_recovery_preserves_s0_observations_actions_and_contacts(config, count):
    from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
    from rl_risk_sac.tasks.local_samples import simulation_config
    agent = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    env = LocalAvoidanceEnv(config, agent.reaching)
    reference = ThesisHomotopyEnv(simulation_config(config))
    try:
        sample = sample_for_env(env)
        for identity in range(1, count):
            obstacle = copy.deepcopy(sample["state"]["obstacles"][0])
            obstacle["identity"] = identity
            obstacle["initial_position"] = [-.6 + .2 * identity, .6, 1.]
            sample["state"]["obstacles"].append(obstacle)
        env.reset(options={"sample": sample})
        state = copy.deepcopy(env.base.episode_state_dict())
        state["contract"].terminate_on_success = False
        left = reference.restore_episode_state(copy.deepcopy(state))
        right = env.checker.restore_episode_state(copy.deepcopy(state))
        assert len(left.obstacle_features) == count
        assert len(right.obstacle_features) == 0
        assert len(env.checker.obstacles) == count
        for _ in range(3):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
            a = agent.reaching.select_action(left, deterministic=True)
            b = agent.reaching.select_action(right, deterministic=True)
            np.testing.assert_array_equal(a, b)
            left, _, _, lt, lx, li = reference.step(a)
            right, _, _, rt, rx, ri = env.checker.step(b)
            assert (lt, lx, li["collision"], li["joint_limit"]) == (rt, rx, ri["collision"], ri["joint_limit"])
        # Disabling features must not disable physical obstacle contact.
        position, _ = env.checker.robot.end_effector_pose()
        for client in (reference, env.checker):
            client.obstacles[-1].initial_position = np.asarray(position).tolist()
        _, _, _, lt, _, li = reference.step(np.zeros(6))
        _, _, _, rt, _, ri = env.checker.step(np.zeros(6))
        assert lt and rt and li["obstacle_collision"] and ri["obstacle_collision"]
    finally:
        reference.close()
        env.close()


def test_query_cache_preserves_substeps_and_invalidates_after_state_changes(config, monkeypatch):
    import pybullet as bullet
    from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
    from rl_risk_sac.tasks.local_avoidance import MeshClearance
    from rl_risk_sac.tasks.local_samples import simulation_config
    agent = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    env = LocalAvoidanceEnv(config, agent.reaching)
    reference = ThesisHomotopyEnv(simulation_config(config))
    try:
        env.reset(options={"sample": sample_for_env(env)})
        reference.restore_episode_state(copy.deepcopy(env.base.episode_state_dict()))
        counts = {reference.client_id: 0, env.base.client_id: 0}
        original = bullet.getClosestPoints
        def query(*args, **kwargs):
            client = kwargs["physicsClientId"]
            if client in counts:
                counts[client] += 1
            return original(*args, **kwargs)
        monkeypatch.setattr(bullet, "getClosestPoints", query)
        results = []
        for client in (reference, env.base):
            margins = MeshClearance(client)
            out = client.step(np.full(6, .1, np.float32), substep_observer=margins)
            results.append((out, margins))
        left, right = results
        np.testing.assert_array_equal(np.asarray(left[0][0]), np.asarray(right[0][0]))
        assert left[0][1:5] == right[0][1:5]
        assert (left[1].external, left[1].self_distance, left[1].checks) == (
            right[1].external, right[1].self_distance, right[1].checks)
        assert right[1].checks == config["thesis"]["control_dt"] / config["thesis"]["physics_dt"] + 1
        assert counts[env.base.client_id] < counts[reference.client_id]
        assert env.base._geometry_queries is None
        with env.base._geometry_query_scope():
            env.base._self_geometry()
            previous = counts[env.base.client_id]
            env.base._self_geometry()
            assert counts[env.base.client_id] == previous
            q, qdot = env.base.robot.joint_state()
            env.base._set_joint_state(q + .01, qdot)
            env.base._self_geometry()
            assert counts[env.base.client_id] > previous
    finally:
        reference.close()
        env.close()


def test_recovery_stops_when_cumulative_margin_makes_success_impossible(config, monkeypatch):
    import rl_risk_sac.envs.local_avoidance_env as module
    agent = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    env = LocalAvoidanceEnv(config, agent.reaching)
    try:
        env.reset(options={"sample": sample_for_env(env)})
        class Margins:
            external = .1
            self_distance = .01
            collision = False
            def __init__(self, checker):
                pass
        monkeypatch.setattr(module, "MeshClearance", Margins)
        steps = []
        def step(action, *, substep_observer):
            steps.append(1)
            substep_observer.external = .02  # Safe from collision, but below 2.5 cm.
            return np.zeros(241), 0., 0., False, False, {"joint_limit": False}
        monkeypatch.setattr(env.checker, "step", step)
        assert not env._recovery()
        assert len(steps) == 1
    finally:
        env.close()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_packed_replay_matches_ragged_reference_after_growth_wraparound_and_resume(device):
    from rl_risk_sac.tasks.obstacle_observation import pad_obstacle_sets
    replay = LocalReplayBuffer(6, device, 21)
    for i in range(11):
        current = ObstacleObservation(np.full(268, i), np.full((i % 4, 37), i))
        following = ObstacleObservation(np.full(268, i + 1), np.full(((i + 2) % 4, 37), i + 1))
        replay.add(current, np.full(6, i), float(i), following, bool(i % 2))
    state = copy.deepcopy(replay.state_dict())
    restored = LocalReplayBuffer(6, device, 999)
    restored.load_state_dict(state)
    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(state["rng"])
    index = rng.integers(replay.size, size=32)
    features, mask = pad_obstacle_sets([state["sets"][i] for i in index], device)
    next_features, next_mask = pad_obstacle_sets([state["next_sets"][i] for i in index], device)
    for batch in (replay.sample(32), restored.sample(32)):
        for key in ("observations", "actions", "rewards", "next_observations", "dones"):
            assert torch.equal(getattr(batch, key), torch.as_tensor(state[key][index], device=device))
            assert getattr(batch, key).is_contiguous()
        assert torch.equal(batch.obstacle_features, features)
        assert torch.equal(batch.obstacle_mask, mask)
        assert torch.equal(batch.next_obstacle_features, next_features)
        assert torch.equal(batch.next_obstacle_mask, next_mask)


def test_packed_diagnostics_preserve_python_ema_precision():
    from rl_risk_sac.algorithms.thesis_sac import _scalar_metrics
    ema = 1. + 2. ** -50
    result = _scalar_metrics({"ema": ema, "loss": torch.tensor(.25), "updated": True})
    assert result == {"ema": ema, "loss": .25, "updated": 1.}


def test_query_cache_invalidates_projection_finite_difference_states(config):
    from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
    from rl_risk_sac.tasks.local_samples import simulation_config
    agent = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    env = LocalAvoidanceEnv(config, agent.reaching)
    reference = ThesisHomotopyEnv(simulation_config(config))
    try:
        env.reset(options={"sample": sample_for_env(env)})
        reference.restore_episode_state(copy.deepcopy(env.base.episode_state_dict()))
        outputs = []
        for client in (reference, env.base):
            client.self_projection_enabled = True
            client.self_projection_activation_distance = 1.
            with client._geometry_query_scope():
                command, info = client._project_self_safe_command(np.full(6, .3, np.float32))
            outputs.append((command, info))
        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
        for key in ("self_projection_active_constraints", "self_projection_intervened",
                    "self_projection_infeasible", "self_projection_correction_norm", "self_projection_max_slack"):
            assert outputs[0][1][key] == outputs[1][1][key]
    finally:
        reference.close()
        env.close()


def test_parallel_local_workers_preserve_ragged_observations_physics_and_recovery_policy(config):
    from rl_risk_sac.envs.parallel_local_avoidance_env import (
        FrozenCPUReachingPolicy, ParallelLocalAvoidanceEnvPool,
    )
    agent = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    env = LocalAvoidanceEnv(config, agent.reaching)
    pool = ParallelLocalAvoidanceEnvPool(config, [101, 102], agent.reaching.actor)
    try:
        info = pool.policy_info()
        assert all(value == dict(device="cpu", actor_sha256=actor_digest(agent.reaching.actor),
                                 trainable=False, cuda_initialized=False) for value in info.values())
        sample = sample_for_env(env)
        multi = copy.deepcopy(sample)
        for identity in (1, 2):
            obstacle = copy.deepcopy(sample["state"]["obstacles"][0])
            obstacle.update(identity=identity, initial_position=[-.6 + .2 * identity, .6, 1.])
            multi["state"]["obstacles"].append(obstacle)
        observations = pool.reset_many({0: sample, 1: multi})
        assert observations[0][0].obstacle_features.shape == (1, 37)
        assert observations[1][0].obstacle_features.shape == (3, 37)
        single, _ = env.reset(options={"sample": sample})
        np.testing.assert_array_equal(single, observations[0][0])
        np.testing.assert_array_equal(single.obstacle_features, observations[0][0].obstacle_features)
        frozen = FrozenCPUReachingPolicy(copy.deepcopy(agent.reaching.actor))
        np.testing.assert_array_equal(frozen.select_action(single),
                                      agent.reaching.select_action(single, deterministic=True))
        expected = env.step(np.full(6, .1, np.float32))
        pool.send_steps({0: np.full(6, .1, np.float32), 1: np.zeros(6)})
        with pytest.raises(RuntimeError, match="pending"):
            pool.states()
        actual = pool.recv_steps()[0]
        np.testing.assert_array_equal(expected[0], actual[0])
        np.testing.assert_array_equal(expected[0].obstacle_features, actual[0].obstacle_features)
        assert expected[1:] == actual[1:]
        assert actual[-1]["physics_substep_checks"] == 13  # 20 Hz control, 240 Hz physics.
        snapshots = pool.states()
        restored = pool.restore_many(snapshots)
        np.testing.assert_array_equal(restored[0], actual[0])
        np.testing.assert_array_equal(restored[0].obstacle_features, actual[0].obstacle_features)
        assert restored[1].obstacle_features.shape == (3, 37)
        assert snapshots[0]["torch_rng"].device.type == "cpu"
    finally:
        env.close()
        pool.close()
    assert all(not process.is_alive() for process in pool.processes)


def test_parallel_local_error_drains_replies_and_cannot_save_partial_state(config):
    from rl_risk_sac.envs.parallel_local_avoidance_env import ParallelLocalAvoidanceEnvPool
    from rl_risk_sac.envs.parallel_thesis_env import ParallelEnvWorkerError
    agent = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    env = LocalAvoidanceEnv(config, agent.reaching)
    pool = ParallelLocalAvoidanceEnvPool(config, [101, 102], agent.reaching.actor)
    try:
        sample = sample_for_env(env)
        with pytest.raises(ParallelEnvWorkerError, match="reset"):
            pool.reset_many({0: {}, 1: sample})
        assert pool.states([1])[1]["environment"]["sample"]["sample_id"] == sample["sample_id"]
        with pytest.raises(ParallelEnvWorkerError):
            pool.states()
    finally:
        env.close()
        pool.close()
    assert all(not process.is_alive() for process in pool.processes)


@pytest.mark.parametrize("rate", [1., .25])
def test_parallel_local_cli_exact_budgets_updates_resume_and_serial_migration(config, tmp_path, rate):
    import csv
    import json
    import subprocess
    import sys
    import yaml

    torch.set_num_threads(1)
    config["local_avoidance"]["horizon"] = 3
    config["train"]["progress_interval"] = 4
    config["train"]["logging"]["updates"] = False
    config["sac"]["updates_per_transition"] = rate
    agent = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    source = tmp_path / "reaching.pt"
    torch.save(agent.reaching.actor.state_dict(), source)
    env = LocalAvoidanceEnv(config, agent.reaching)
    try:
        sample = sample_for_env(env)
    finally:
        env.close()
    sample["provenance"] = dict(source="frozen_s0_policy", policy_sha256=actor_digest(agent.reaching.actor),
                                planner_used=False, reaching_actions_used_as_demonstrations=False,
                                nominal_min_distance_m=-.01, approach_steps=3,
                                nominal_collision_step=5, nominal_exit_step=10, seed=1234)
    bank = tmp_path / "bank.pt"
    torch.save(dict(protocol=PROTOCOL, format_version=1, contract_sha256=contract_digest(config),
                    policy_sha256=actor_digest(agent.reaching.actor), split="train", samples=[sample]), bank)
    configuration = tmp_path / "config.yaml"
    configuration.write_text(yaml.safe_dump(config))

    def run(name, steps, *extra, succeeds=True):
        command = [sys.executable, "scripts/core/train_local_avoidance.py", "--config", str(configuration),
                   "--bank", str(bank), "--steps", str(steps), "--save-interval", "7",
                   "--validation-mode", "--device", "cpu", "--run-name", name,
                   "--output-dir", str(tmp_path / "runs"), *map(str, extra)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        assert (result.returncode == 0) == succeeds, result.stdout + result.stderr
        path = tmp_path / "runs" / name
        return path, result

    path, result = run("parallel", 17, "--num-envs", 3, "--initialize-reaching-from", source)
    assert "环境=3" in result.stdout and "在线成功率" in result.stdout
    with (path / "progress.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["steps"]) for row in rows] == [4, 8, 12, 16, 17]
    checkpoint = path / "checkpoints" / "step_0000017.pt"
    state = torch.load(checkpoint, weights_only=False)
    assert state["counters"]["steps"] == state["replay"]["size"] == 17
    assert state["counters"]["updates"] == int(10 * rate)
    assert state["counters"]["update_credit"] == (10 * rate) % 1
    assert state["training_schedule"]["updates_per_transition"] == rate
    assert state["parallel_updates_due"] == 0
    assert len(state["parallel_workers"]) == 3
    assert all(worker is not None for worker in state["parallel_workers"])
    assert len(state["recent_successes"]) == state["counters"]["episodes"] > 0
    reaching = state["agent"]["dual_sac"]["policies"]["reaching"]
    for key, tensor in agent.reaching.actor.state_dict().items():
        assert torch.equal(reaching["actor"][key], tensor)
    assert sorted(file.name for file in (path / "checkpoints").iterdir()) == [
        "avoidance_actor_step_0000007.pt", "avoidance_actor_step_0000014.pt",
        "avoidance_actor_step_0000017.pt", "step_0000007.pt", "step_0000014.pt", "step_0000017.pt"]
    assert not (path / "updates.csv").exists()
    resumed, _ = run("resumed", 5, "--resume", checkpoint)
    restored = torch.load(resumed / "checkpoints" / "step_0000022.pt", weights_only=False)
    assert restored["parallel_num_envs"] == 3  # Uses the saved count rather than config's 8.
    assert restored["counters"]["updates"] == int(15 * rate)
    assert restored["counters"]["update_credit"] == (15 * rate) % 1
    assert restored["replay"]["size"] == 22
    for key in ("observations", "next_observations", "actions", "rewards", "dones"):
        np.testing.assert_array_equal(restored["replay"][key][:17], state["replay"][key])
    lengths = 0
    for directory in (path, resumed):
        with (directory / "episodes.csv").open() as handle:
            lengths += sum(int(row["length"]) for row in csv.DictReader(handle))
    lengths += sum(worker["episode_steps"] for worker in restored["parallel_workers"] if worker["active"])
    assert lengths == 22  # Resume must retain unfinished episodes and their accumulators.
    rejected, failure = run("wrong_count", 1, "--resume", checkpoint, "--num-envs", 1, succeeds=False)
    assert "keep the worker count" in failure.stderr and not rejected.exists()

    # Produce the old effective UTD=1, then remove new schedule/credit metadata
    # to emulate a genuine pre-migration checkpoint with its trained state.
    config["sac"]["updates_per_transition"] = 1.
    configuration.write_text(yaml.safe_dump(config))
    serial, _ = run("serial", 9, "--num-envs", 1, "--initialize-reaching-from", source)
    legacy = torch.load(serial / "checkpoints" / "step_0000009.pt", weights_only=False)
    legacy.pop("parallel_num_envs")  # The pre-migration checkpoint schema.
    legacy.pop("training_schedule")
    legacy.pop("training_schedule_migrations")
    legacy["counters"].pop("update_credit")
    legacy["config"]["sac"]["updates_per_transition"] = rate
    legacy_path = tmp_path / "legacy.pt"
    torch.save(legacy, legacy_path)
    config["sac"]["updates_per_transition"] = rate
    configuration.write_text(yaml.safe_dump(config))
    migrated, result = run("migrated", 5, "--resume", legacy_path, "--num-envs", 3)
    assert "migrated serial checkpoint: step=9" in result.stdout
    state = torch.load(migrated / "checkpoints" / "step_0000014.pt", weights_only=False)
    assert state["counters"]["updates"] == 2 + int(5 * rate) and state["replay"]["size"] == 14
    assert state["counters"]["update_credit"] == (5 * rate) % 1
    migration = state["training_schedule_migrations"][-1]
    assert migration["step"] == 9
    assert migration["previous"]["updates_per_transition"] == 1.
    assert migration["current"]["updates_per_transition"] == rate
    assert state["parallel_updates_due"] == 0
    assert json.loads((migrated / "summary.json").read_text())["frozen_reaching_actor_unchanged"]


def test_successful_recovery_remains_successful_with_lightweight_checker(config):
    from types import SimpleNamespace
    from rl_risk_sac.tasks.local_avoidance import MeshClearance
    action = np.array([.05, 0., 0., 0., 0., 0.], np.float32)
    reaching = SimpleNamespace(select_action=lambda observation, deterministic: action.copy())
    env = LocalAvoidanceEnv(config, reaching)
    try:
        env.reset(options={"sample": sample_for_env(env)})
        checker = env.checker
        checker.compute_external_features = True
        checker.cache_geometry_queries = False
        state = copy.deepcopy(env.base.episode_state_dict())
        state["contract"].terminate_on_success = False
        checker.restore_episode_state(state)
        start = arm_vector(checker)
        margins = MeshClearance(checker)
        for _ in range(config["local_avoidance"]["recovery_steps"]):
            _, _, _, terminated, truncated, info = checker.step(action, substep_observer=margins)
            assert not terminated and not truncated and not info["joint_limit"]
        delta = arm_vector(checker) - start
        length = float(np.linalg.norm(delta))
        assert length > .001
        assert margins.external >= config["thesis"]["obstacles"]["risk"]["d_safe"]
        assert margins.self_distance >= checker.d_self_safe
        env.sample["direction"] = delta / length
        env.sample["length"] = max(.03, length)
        assert env._recovery()  # Full feature computation, original ten-step rule.
        checker.compute_external_features = False
        checker.cache_geometry_queries = True
        assert env._recovery()  # Same positive result with all optimizations.
    finally:
        env.close()
