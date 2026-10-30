from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from rl_risk_sac.algorithms.dual_sac import ReachingSACAgent
from rl_risk_sac.algorithms.replay_buffer import Batch
from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.envs.parallel_joint_handoff_env import (
    FrozenJointPolicy, JointHandoffEnvironment, ParallelJointHandoffEnvPool,
)
from rl_risk_sac.control.joint_policy import JointPolicyController
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.envs.obstacle_scene import MovingObstacle
from rl_risk_sac.tasks.local_avoidance import LOCAL_OBSERVATION_DIM, actor_digest
from rl_risk_sac.tasks.local_avoidance import arm_vector
from rl_risk_sac.tasks.local_samples import nominal_episode_contract, simulation_config
from rl_risk_sac.tasks.reaching_handoff import (
    JOINT_TRAIN_PROTOCOL, JointHandoffReplay, ReachingDecisionReturn,
    ObstacleAwareJointHandoffReplay, joint_handoff_state, recovery_config,
)
from rl_risk_sac.utils.config import load_config
from tests.core.test_reaching_handoff import sample
from scripts.core.train_thesis_homotopy import apply_stage_profile


@pytest.fixture
def config():
    cfg = load_config("configs/experiments/paper_local_avoidance_geometry.yaml")
    apply_stage_profile(cfg, "s1")
    cfg["device"] = "cpu"
    cfg["sac"]["hidden_dims"] = [16, 16]
    return cfg


def test_joint_handoff_preserves_obstacles_history_motion_and_clock():
    source = sample()["state"]
    restored = joint_handoff_state(source)
    for name in ("q", "qdot", "goal_position", "goal_quaternion"):
        np.testing.assert_array_equal(restored[name], source[name])
    assert restored["physics_step_count"] == source["physics_step_count"]
    assert restored["obstacles"] == source["obstacles"]
    assert restored["predictor_history"] == source["predictor_history"]
    assert restored["step_count"] == restored["success_hold_count"] == 0
    assert restored["contract"].terminate_on_success and restored["contract"].strict_obstacle_collision
    restored["obstacles"][0]["identity"] = 99
    assert source["obstacles"][0]["identity"] == 1
    assert source["step_count"] == 73


def test_reaching_return_spans_avoidance_without_adding_avoidance_actions():
    replay = JointHandoffReplay(8, "cpu", 1)
    decision = ReachingDecisionReturn(np.zeros(241), np.ones(6) * .2)
    # One reaching reward, followed by two deterministic avoidance rewards.
    for reward in [1., 2., 3.]:
        decision.advance(reward, .9)
    decision.store(replay, np.ones(241), False, .9)
    assert replay.size == 1
    batch = replay.sample(1)
    assert batch.rewards.item() == pytest.approx(1 + .9 * 2 + .9**2 * 3)
    assert batch.bootstrap_discounts.item() == pytest.approx(.9**3)
    assert not batch.dones.item()
    torch.testing.assert_close(batch.actions, torch.full((1, 6), .2))
    # Failure during the frozen segment is credited to the preceding decision.
    decision.advance(-20., .9)
    decision.store(replay, np.ones(241), True, .9)
    assert replay.rewards[1, 0] == pytest.approx(1 + .9 * 2 + .9**2 * 3 - .9**3 * 20)
    assert replay.dones[1, 0] == 1
    with pytest.raises(ValueError, match="without an executed"):
        ReachingDecisionReturn(np.zeros(241), np.zeros(6)).store(replay, np.ones(241), False, .9)


def test_joint_replay_roundtrip_cannot_load_old_obstacle_free_replay():
    replay = JointHandoffReplay(8, "cpu", 1)
    for i in range(3):
        replay.add(np.full(241, i), np.full(6, i), i, np.full(241, i + 1), i == 2, .99**(i + 1))
    state = copy.deepcopy(replay.state_dict())
    assert state["protocol"] == JOINT_TRAIN_PROTOCOL
    restored = JointHandoffReplay(8, "cpu", 42)
    restored.load_state_dict(state)
    a, b = replay.sample(5), restored.sample(5)
    for name in ("observations", "actions", "rewards", "dones", "bootstrap_discounts"):
        torch.testing.assert_close(getattr(a, name), getattr(b, name))
    with pytest.raises(ValueError, match="obstacle-free replay"):
        restored.load_state_dict(dict(state, protocol="reaching_handoff_finetune_v1"))


def test_sac_target_uses_elapsed_step_discount_and_terminal_mask(config, monkeypatch):
    agent = ReachingSACAgent(241, 6, recovery_config(config))
    monkeypatch.setattr(agent.actor, "sample", lambda obs, *args: (torch.zeros(len(obs), 6), torch.zeros(len(obs), 1)))
    for critic in (agent.target_q1, agent.target_q2):
        monkeypatch.setattr(critic, "forward", lambda obs, *args: torch.full((len(obs), 1), 10.))
    batch = Batch(torch.zeros(2, 241), torch.zeros(2, 6), torch.ones(2, 1),
                  torch.zeros(2, 1), torch.zeros(2, 241), torch.tensor([[0.], [1.]]),
                  bootstrap_discounts=torch.tensor([[.9**3], [.9**5]]))
    result = agent.update(batch, update_actor=False)
    assert result["target_mean"] == pytest.approx(((1 + .9**3 * 10) + 1) / 2)


def policies(config):
    reaching = ReachingSACAgent(241, 6, recovery_config(config)).actor
    avoidance = ThesisSACAgent(LOCAL_OBSERVATION_DIM, 6, config).actor
    return reaching, avoidance


def test_parallel_snapshots_are_shared_and_avoidance_stays_frozen(config):
    reaching, avoidance = policies(config)
    old_avoidance = actor_digest(avoidance)
    pool = ParallelJointHandoffEnvPool(config, [123, 456], reaching, avoidance, max_steps=10)
    try:
        observations = pool.reset_many({i: nominal_episode_contract(config) for i in range(2)}, seeds={0: 123, 1: 456})
        before = pool.policy_info()
        with torch.no_grad():
            next(reaching.parameters()).add_(.01)
        pool.sync_reaching(reaching)
        after = pool.policy_info()
        assert all(before[i]["reaching"] != after[i]["reaching"] == actor_digest(reaching) for i in range(2))
        assert all(row["avoidance"] == old_avoidance and not row["cuda_initialized"] for row in after.values())
        prepared = pool.prepare_many([0, 1])
        assert all(row["role"] == "reaching" and row["failure"] is None for row in prepared.values())
        pool.send_steps({i: np.zeros(6) for i in range(2)})
        with pytest.raises(RuntimeError, match="during simulation"):
            pool.sync_reaching(reaching)
        results = pool.recv_steps()
        assert all(row[-1]["action_policy"] == "reaching" for row in results.values())
        states = pool.states()
        restored = pool.restore_many(states)
        for i in range(2):
            np.testing.assert_allclose(restored[i], results[i][0], atol=1e-6)
            assert states[i]["controller"]["steps"] == 1
    finally:
        pool.close()


def test_deterministic_training_step_matches_joint_evaluation_and_resume(config):
    reaching, avoidance = policies(config)
    policy = FrozenJointPolicy(copy.deepcopy(reaching), copy.deepcopy(avoidance))
    training = JointHandoffEnvironment(config, copy.deepcopy(reaching), copy.deepcopy(avoidance), 20)
    evaluation = ThesisHomotopyEnv(simulation_config(config), cache_geometry_queries=True)
    controller = JointPolicyController(config, policy)
    try:
        observation, _ = training.reset(nominal_episode_contract(config), 123)
        original = training.base.episode_state_dict()
        original["obstacles"] = [MovingObstacle(
            0, "sphere", [.04], "small", [-.6, -.6, 1.], [0., 0., 0., 1.],
            [1., 0., 0.], "constant", .1, 1., 0., -1., 1.).metadata()]
        training.restore(copy.deepcopy(original))
        assert len(training.base.obstacles) == 1
        eval_observation = evaluation.restore_episode_state(copy.deepcopy(original))
        evaluation.horizon = 20
        for _ in range(3):
            decision = training.prepare()
            action = policy.reaching.select_action(decision["observation"], True)
            actual = training.step(action)
            eval_observation, terminated, truncated, info = controller.step(evaluation, eval_observation)
            np.testing.assert_allclose(actual[0], eval_observation, atol=1e-6)
            assert actual[3:5] == (terminated, truncated)
            assert actual[-1]["action_policy"] == info["action_policy"]
        saved = copy.deepcopy(training.state())
        decision = training.prepare()
        action = policy.reaching.select_action(decision["observation"], True)
        expected = training.step(action)
        training.restore(saved)
        assert len(training.base.obstacles) == 1
        training.prepare()
        resumed = training.step(action)
        np.testing.assert_allclose(expected[0], resumed[0], atol=1e-6)
        assert expected[1:5] == pytest.approx(resumed[1:5])
        assert training.controller.steps == saved["controller"]["steps"] + 1
    finally:
        training.close()
        controller.close()
        evaluation.close()


def test_safety_checks_sampled_proposal_and_retains_rejected_task_decision(config, monkeypatch):
    reaching, avoidance = policies(config)
    training = JointHandoffEnvironment(config, reaching, avoidance, 20)
    try:
        training.reset(nominal_episode_contract(config), 123)
        state = training.base.episode_state_dict()
        state["obstacles"] = [MovingObstacle(
            0, "sphere", [.04], "small", [-.6, -.6, 1.], [0., 0., 0., 1.],
            [1., 0., 0.], "constant", .1, 1., 0., -1., 1.).metadata()]
        observation = training.restore(state)
        proposal = np.full(6, .7, np.float32)
        sample = dict(entry_arm=arm_vector(training.base), direction=np.ones(24) / np.sqrt(24), length=.2)
        from types import SimpleNamespace
        monkeypatch.setattr(training.base, "_static_clearance", lambda obj: .08)
        monkeypatch.setattr(training.controller.references, "build", lambda *args, **kwargs:
                            SimpleNamespace(status="blocked", sample=sample))
        from rl_risk_sac.control import joint_policy
        monkeypatch.setattr(joint_policy, "passage_candidate", lambda *args: False)
        decision = training.prepare(proposal)
        assert decision["task_decision"] and decision["role"] == "avoidance"
        np.testing.assert_array_equal(training.prepared[1].action, proposal)
        pending = ReachingDecisionReturn(observation, proposal)
        following, reward, _, terminated, truncated, info = training.step(None)
        assert info["action_policy"] == "avoidance"  # Proposal was replaced.
        pending.advance(reward, .99)
        # The task is not queried as a new decision while avoidance continues.
        decision = training.prepare(-proposal)
        assert not decision["task_decision"] and decision["role"] == "avoidance"
        following, reward, _, terminated, truncated, info = training.step(None)
        pending.advance(reward, .99)
        replay = ObstacleAwareJointHandoffReplay(4, "cpu", 1)
        pending.store(replay, following, terminated or truncated, .99)
        batch = replay.sample(1)
        np.testing.assert_array_equal(batch.actions[0].numpy(), proposal)
        assert batch.bootstrap_discounts.item() == pytest.approx(.99**2)
        assert batch.obstacle_mask.sum() == batch.next_obstacle_mask.sum() == 1
    finally:
        training.close()
