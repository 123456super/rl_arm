from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from rl_risk_sac.algorithms.paper_sac import PaperGaussianActor, PaperReplayBuffer, PaperSACAgent
from rl_risk_sac.algorithms.replay_buffer import Batch
from rl_risk_sac.envs.paper_reaching_env import PaperReachingEnv
from rl_risk_sac.envs.parallel_paper_env import ParallelPaperEnvPool
from rl_risk_sac.tasks.paper_reaching import (
    centerline_distances, paper_reward, pose_errors, sample_goal_position, sample_obstacle_center,
)
from rl_risk_sac.tasks.paper_task_space_planner import SimulationTaskSpacePlanner
from rl_risk_sac.utils.paper_config import experiment_signature, load_paper_config, validate_paper_config


@pytest.fixture
def config():
    cfg = load_paper_config("configs/experiments/paper_single_sac.yaml")
    cfg["device"] = "cpu"
    cfg["sac"]["hidden_dims"] = [16, 16]
    torch.set_num_threads(1)
    return cfg


def test_reward_matches_paper_and_public_error_weighting(config):
    reward_cfg = config["paper"]["reward"]
    components, distances = [0.06, .04], [.04, .08, .2]
    error = .5 * .06 + .25 * .04
    expected = -.001 * error**2 - np.log(error**2 + .0001) - .1 * .5 - 1
    reward, parts = paper_reward(components, distances, True, reward_cfg)
    assert reward == pytest.approx(expected)
    assert parts["combined_error"] == pytest.approx(error)
    assert parts["collision_reward"] == -1


def test_centerline_distance_is_per_obstacle_and_clips_to_finite_segment():
    distances = centerline_distances([[2, 0, 0], [.5, 1, 0], [0, 0, 0]],
                                    [[0, 0, 0], [0, 2, 0]], [[1, 0, 0], [0, 2, 0]])
    np.testing.assert_allclose(distances, [1, 1, 0])


def test_quaternion_sign_does_not_change_pose_error():
    assert pose_errors([0, 0, 0], [0, 0, 0, 1], [0, 0, 0], [0, 0, 0, -1]) == (0., 0.)


def test_reference_workspace_and_quarter_shell_distributions(config):
    rng = np.random.default_rng(15)
    for _ in range(200):
        point = sample_goal_position(rng, config["paper"]["goal"], [0, 0, 0])
        assert np.linalg.norm(point) <= .85
        assert np.linalg.norm(point[:2]) >= .30
        assert point[1] >= 0 and point[2] >= .05
        center = sample_obstacle_center(rng, point, [0, 0, 0], config["paper"]["obstacles"])
        offset = center - point
        assert .15 - 1e-10 <= np.linalg.norm(offset) <= .60 + 1e-10
        assert offset[2] <= 0 and np.dot(offset[:2], -point[:2]) >= 0


def test_aew_density_matches_variance_reduction_and_tanh_change_of_variables():
    torch.manual_seed(5)
    actor = PaperGaussianActor(hidden=[8, 8])
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
    observations = torch.zeros(10000, 25)
    actions, actual = actor.sample(observations, torch.full((10000, 1), 9.))
    raw = torch.atanh(actions)
    expected = (torch.distributions.Normal(0., 1 / 3).log_prob(raw)
                - torch.log1p(-actions.square())).sum(-1, keepdim=True)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert raw.var().item() == pytest.approx(1 / 9, rel=.025)
    loss = actions.square().mean() + .1 * actual.mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())


def test_sac_terminal_mask_bootstraps_time_limits(config, monkeypatch):
    agent = PaperSACAgent(config)
    agent.actor_update_interval = 2  # isolate the Bellman-mask check from actor optimization
    monkeypatch.setattr(agent.actor, "sample", lambda obs, counts: (torch.zeros(len(obs), 6), torch.zeros(len(obs), 1)))
    for critic in (agent.target_q1, agent.target_q2):
        monkeypatch.setattr(critic, "forward", lambda obs, actions: torch.full((len(obs), 1), 10.))
    batch = Batch(torch.zeros(2, 25), torch.zeros(2, 6), torch.ones(2, 1), torch.zeros(2, 1),
                  torch.zeros(2, 25), torch.tensor([[0.], [1.]]))
    metrics = agent.update(batch, collect_diagnostics=True)
    assert metrics["target_mean"] == pytest.approx(((1 + .99 * 10) + 1) / 2)


def test_full_agent_resume_preserves_ensemble_rng_optimizers_and_next_update(config):
    config["sac"]["action_ensemble"]["enabled"] = True
    agent = PaperSACAgent(config)
    agent.set_episode_progress(12000)
    replay = PaperReplayBuffer(32, "cpu", 5)
    replay.add_many(np.zeros((8, 25)), np.zeros((8, 6)), np.ones(8), np.ones((8, 25)), np.zeros(8))
    batch = replay.sample(8)
    agent.update(batch)
    state, torch_rng = copy.deepcopy(agent.state_dict()), torch.get_rng_state().clone()
    expected = agent.update(batch, collect_diagnostics=True)
    expected_actor = copy.deepcopy(agent.actor.state_dict())
    restored = PaperSACAgent(config)
    restored.load_state_dict(state)
    torch.set_rng_state(torch_rng)
    actual = restored.update(batch, collect_diagnostics=True)
    assert actual == expected
    for key, value in expected_actor.items():
        torch.testing.assert_close(value, restored.actor.state_dict()[key], atol=0, rtol=0)


def test_replay_wrap_and_rng_roundtrip():
    replay = PaperReplayBuffer(5, "cpu", 11)
    for offset in (0, 3):
        values = np.arange(offset, offset + 3)
        replay.add_many(np.repeat(values[:, None], 25, axis=1), np.zeros((3, 6)), values,
                        np.repeat((values + 1)[:, None], 25, axis=1), [False, False, True])
    restored = PaperReplayBuffer(5, "cpu", 100)
    restored.load_state_dict(copy.deepcopy(replay.state_dict()))
    a, b = replay.sample(20), restored.sample(20)
    for key in ("observations", "actions", "rewards", "next_observations", "dones"):
        torch.testing.assert_close(getattr(a, key), getattr(b, key))
    assert replay.size == 5 and replay.ptr == 1
    assert set(replay.arrays["rewards"][:, 0]) == {1, 2, 3, 4, 5}


def test_task_signature_excludes_device_and_budget_but_includes_reward(config):
    signature = experiment_signature(config)
    config["device"] = "cuda:0"
    config["train"]["total_steps"] += 10
    assert experiment_signature(config) == signature
    config["paper"]["gravity"] -= 1
    assert experiment_signature(config) != signature
    config["paper"]["gravity"] += 1
    config["paper"]["reward"]["obstacle_weight"] *= 2
    assert experiment_signature(config) != signature
    config["paper"]["obstacles"]["count"] = 2
    with pytest.raises(ValueError, match="six joints and three"):
        validate_paper_config(config)


def test_environment_reset_and_restore_use_the_same_observation_and_action(config):
    env = PaperReachingEnv(config)
    try:
        obs, _ = env.reset(seed=17)
        assert obs.shape == (25,)
        saved = copy.deepcopy(env.episode_state_dict())
        env._set_q(saved["scene"]["goal_q"])
        # Accepted targets have an actual, collision-free full-pose witness
        # within the training threshold, not merely a coarse IK solution.
        env._move_obstacles(saved["scene"]["obstacle_centers"])
        import pybullet as p
        p.performCollisionDetection(physicsClientId=env.client_id)
        assert not any(env._contacts())
        assert env._observe()[1]["final_combined_error"] <= config["paper"]["success"]["combined_error_tolerance"]
        assert np.all(saved["scene"]["goal_q"] >= env.control_lower)
        assert np.all(saved["scene"]["goal_q"] <= env.control_upper)
        env.restore_episode_state(saved)
        action = np.array([.01, -.02, .01, 0, .02, 0])
        expected = env.step(action)
        restored_obs = env.restore_episode_state(saved)
        np.testing.assert_allclose(obs, restored_obs, atol=1e-7)
        restored = env.step(action)
        np.testing.assert_allclose(expected[0], restored[0], atol=2e-6)
        assert expected[1] == pytest.approx(restored[1], abs=1e-5)
        assert expected[3:5] == restored[3:5]
    finally:
        env.close()


def test_collision_history_prevents_later_reaching_from_being_safe_success(config):
    env = PaperReachingEnv(config)
    try:
        env.reset(seed=17)
        centers = env.obstacle_centers.copy()
        centers[0] = env.robot.end_effector_pose()[0]
        env.obstacle_centers = centers
        env._move_obstacles(centers)
        first = env.step(np.zeros(6))
        assert first[-1]["collision_seen"] and first[2] == 1.
        env.obstacle_centers = np.array([[5, 5, 5], [6, 5, 5], [7, 5, 5]], float)
        env._move_obstacles(env.obstacle_centers)
        env.goal_position, env.goal_quaternion = env.robot.end_effector_pose()
        env.set_reference(env.goal_position, env.goal_quaternion)
        env.task["success"]["combined_error_tolerance"] = 10.
        final = env.step(np.zeros(6))
        assert final[-1]["reached"] and not final[-1]["safe_success"]
    finally:
        env.close()


def test_parallel_pool_agrees_with_serial_and_propagates_worker_errors(config):
    serial = PaperReachingEnv(config)
    pool = ParallelPaperEnvPool(config, [22, 23])
    try:
        expected = serial.reset(seed=17)[0]
        rows = pool.reset_many({0: {}, 1: {}}, seeds={0: 17, 1: 17})
        np.testing.assert_allclose(expected, rows[0][0], atol=1e-7)
        np.testing.assert_array_equal(rows[0][0], rows[1][0])
        action = np.full(6, .005)
        pool.send_steps({0: action, 1: action})
        with pytest.raises(RuntimeError, match="in-flight"):
            pool.set_references({0: ([0, 0, .5], [0, 0, 0, 1])})
        expected = serial.step(action)
        actual = pool.recv_steps()
        np.testing.assert_allclose(expected[0], actual[0][0], atol=1e-7)
        states = pool.states()
        restored = pool.restore_many(states)
        np.testing.assert_allclose(restored[0], actual[0][0], atol=1e-7)
        with pytest.raises(RuntimeError, match="six finite"):
            pool.step_many({0: np.zeros(2)})
    finally:
        serial.close()
        pool.close()


def test_sac_updates_actor_and_critic_each_batch_without_ensemble_or_clipping(config, monkeypatch):
    def unexpected_clip(*args, **kwargs):
        raise AssertionError("standard SAC must not silently clip gradients")
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", unexpected_clip)
    agent = PaperSACAgent(config)
    before_actor = copy.deepcopy(agent.actor.state_dict())
    before_q = copy.deepcopy(agent.q1.state_dict())
    batch = Batch(torch.randn(16, 25), torch.rand(16, 6) * 2 - 1, torch.ones(16, 1),
                  torch.zeros(16, 1), torch.randn(16, 25), torch.zeros(16, 1))
    metrics = agent.update(batch, collect_diagnostics=True)
    assert agent._counts(16) is None
    assert metrics["actor_updated"] == 1 and agent.update_steps == 1
    assert any(not torch.equal(before_actor[k], v) for k, v in agent.actor.state_dict().items())
    assert any(not torch.equal(before_q[k], v) for k, v in agent.q1.state_dict().items())
    assert metrics["critic_gradient_norm"] > 0 and metrics["actor_gradient_norm"] > 0
    assert all(np.isfinite(value) for value in metrics.values())


def test_action_targets_use_fixed_initial_pose_and_urdf_effort_limits(config):
    import pybullet as p
    env = PaperReachingEnv(config)
    try:
        env.reset(seed=17)
        action = np.array([.01, -.02, .01, 0, .02, 0])
        expected = np.clip(env.initial_q + action * config["paper"]["action_limit"],
                           env.robot.joint_lower_limits, env.robot.joint_upper_limits)
        env.step(action)
        np.testing.assert_array_equal(env.desired_q, expected)
        env._set_q(env.initial_q + .05)
        env.step(action)
        np.testing.assert_array_equal(env.desired_q, expected)
        assert env.motor_forces == [p.getJointInfo(env.robot_id, joint, physicsClientId=env.client_id)[10]
                                    for joint in env.robot.joint_ids]
        scene = copy.deepcopy(env.scene)
        scene["start_q"] = scene["goal_q"]
        env.reset(options={"scene": scene})
        goal_action = (scene["goal_q"] - env.initial_q) / config["paper"]["action_limit"]
        assert np.max(np.abs(goal_action)) <= 1
        info = env.step(goal_action)[-1]
        assert info["safe_success"]
    finally:
        env.close()


def test_mid_episode_pose_hit_is_latched_restored_and_invalidated_by_later_collision(config):
    env = PaperReachingEnv(config)
    try:
        _, initial = env.reset(seed=17)
        scene = copy.deepcopy(initial["scene"])
        scene["goal_position"] = initial["ee_position"] + np.array([.007, 0, 0])
        scene["goal_quaternion"] = initial["ee_quaternion"]
        scene["obstacle_centers"] = np.array([[5, 5, 5], [6, 5, 5], [7, 5, 5]])
        env.reset(options={"scene": scene})
        first = env.step(np.zeros(6))
        assert not first[3] and first[-1]["report_success"] and first[-1]["final_report_success"]
        departed = env.step(np.array([.05, 0, 0, 0, 0, 0]))[-1]
        assert departed["report_success"] and not departed["final_report_success"]
        assert departed["min_position_error_m"] < .01 < departed["position_error_m"]
        state = env.episode_state_dict()
        env.reset(seed=23)
        env.restore_episode_state(state)
        restored = env.step(np.array([.05, 0, 0, 0, 0, 0]))[-1]
        assert restored["report_success"] and not restored["final_report_success"]
        assert restored["pose_thresholds_seen"][2]
        env.obstacle_centers[0] = env.robot.end_effector_pose()[0]
        env._move_obstacles(env.obstacle_centers)
        collided = env.step(np.array([.05, 0, 0, 0, 0, 0]))[-1]
        assert collided["collision_seen"] and not collided["report_success"]
    finally:
        env.close()


def test_aew_ablation_changes_only_ensemble_and_v1_config_is_rejected(config):
    standard = load_paper_config("configs/experiments/paper_single_sac.yaml")
    ensemble = load_paper_config("configs/experiments/paper_single_sac_aew.yaml")
    assert not standard["sac"]["action_ensemble"]["enabled"]
    assert ensemble["sac"]["action_ensemble"]["enabled"]
    assert experiment_signature(standard) != experiment_signature(ensemble)
    ensemble["sac"]["action_ensemble"]["enabled"] = False
    assert standard == ensemble
    config["protocol"] = "paper_single_sac_v1"
    with pytest.raises(ValueError, match="v1 has different"):
        validate_paper_config(config)


def test_task_space_plan_detours_and_checks_spline(config):
    planner = SimulationTaskSpacePlanner(config["planner"], [[0, 0, .4]], .025)
    route = planner.plan([-.3, 0, .4], [.3, 0, .4])
    np.testing.assert_allclose(route.positions[0], [-.3, 0, .4], atol=1e-8)
    np.testing.assert_allclose(route.positions[-1], [.3, 0, .4], atol=1e-8)
    assert route.expanded > 0 and route.expanded <= config["planner"]["max_expansions"]
    assert planner.free(route.positions)
    assert np.linalg.norm(np.diff(route.positions, axis=0), axis=1).sum() > .6
    with pytest.raises(RuntimeError, match="endpoint"):
        planner.plan([0, 0, .4], [.3, 0, .4])
