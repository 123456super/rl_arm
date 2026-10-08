from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest
import torch

from rl_risk_sac.algorithms.homotopy_curriculum import HomotopyCurriculum
from rl_risk_sac.algorithms.homotopy_replay import (
    HomotopyReplayBuffer,
    _vectorized_replay_rewards,
)
from rl_risk_sac.algorithms.replay_buffer import Batch
from rl_risk_sac.algorithms.thesis_sac import (
    ThesisSACAgent,
    diagonal_gaussian_kl_old_to_current,
)
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool
from rl_risk_sac.tasks.thesis_reaching import (
    THESIS_OBSERVATION_DIM_HYBRID_KEYPOINT_JACOBIAN,
    SelfCollisionGeometry,
    ThesisGeometry,
    build_thesis_observation,
    compute_self_collision_geometry,
    damped_least_squares_velocity,
    homotopy_reward,
    keypoint_position_jacobian,
    pose_keypoint_errors,
    pose_keypoint_offsets_world,
    thesis_observation_dim,
)
from rl_risk_sac.utils.config import load_config
import scripts.core.train_thesis_homotopy as thesis_trainer


def test_bad_state_reset_preserves_clock_contract_and_local_neighborhood():
    config = load_config(
        Path(__file__).resolve().parents[2]
        / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml"
    )
    env = ThesisHomotopyEnv(config)
    contract = {
        "scene": "none", "xi": 1.0, "strict": True, "goal_scale": 1.0,
        "orientation_scale": 0.0, "position_tolerance": .1,
        "orientation_tolerance": .3, "lambda_self": .2,
    }
    try:
        env.configure_episode(**contract)
        env.reset(seed=842)
        env.step_count = 90
        observation, info, *_ = env._observe()
        center = {
            "state": env.episode_state_dict(), "observation": observation,
            "rho_R": info["orientation_error_norm"],
            "rho_p": info["goal_error_norm"],
            "preserve_goal": True,
        }
        goal_position = env.goal_position.copy()
        goal_quaternion = env.goal_quaternion.copy()
        first, _ = env.reset_bad_state(center, contract)
        assert env.step_count == 90
        assert env.contract.lambda_self == .2
        np.testing.assert_array_equal(env.goal_position, goal_position)
        np.testing.assert_array_equal(env.goal_quaternion, goal_quaternion)
        assert not env._collision_events()["self_collision"]
        assert not np.array_equal(first, observation)
        second, _ = env.reset_bad_state(center, contract)
        assert env.step_count == 90
        assert not np.array_equal(first, second)
        assert np.sqrt(np.mean((second[21:75] - observation[21:75]) ** 2)) <= .18
        terminal_state = copy.deepcopy(center["state"])
        terminal_state["step_count"] = env.horizon
        fresh_episode_center = {
            **center,
            "state": terminal_state,
            "step": env.horizon,
            "reset_episode_clock": True,
        }
        env.reset_bad_state(fresh_episode_center, contract)
        assert env.step_count == 0
        np.testing.assert_array_equal(env.goal_position, goal_position)
        np.testing.assert_array_equal(env.goal_quaternion, goal_quaternion)
    finally:
        env.close()


def test_l6_far_timeout_restart_settings_and_balanced_bank():
    config = load_config(
        Path(__file__).resolve().parents[2]
        / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml"
    )
    settings = thesis_trainer.far_timeout_restart_settings(config)
    assert settings == {
        "level_index": 6,
        "minimum_position_bin": 7,
        "target_transition_fraction": .15,
        "capacity_per_cell": 8,
    }

    state = {}
    for index in range(10):
        thesis_trainer.add_far_timeout_center(
            state,
            {"position_bin": 7, "orientation_bin": 2, "index": index},
            settings,
        )
    thesis_trainer.add_far_timeout_center(
        state,
        {"position_bin": 8, "orientation_bin": 4, "index": 10},
        settings,
    )
    assert state["bank_size"] == 9
    assert state["nonempty_cells"] == 2
    assert [center["index"] for center in state["cells"]["7:2"]] == list(
        range(2, 10)
    )
    sampled = thesis_trainer.sample_far_timeout_center(
        state, np.random.default_rng(11001),
    )
    assert sampled is not None
    assert sampled[1] in {"7:2", "8:4"}


def test_damped_least_squares_velocity_tracks_twist_and_is_bounded():
    jacobian = np.vstack((np.eye(3), np.eye(3)))
    velocity = damped_least_squares_velocity(
        jacobian, np.asarray([2.0, 0.0, 0.0]), np.zeros(3),
        position_gain=1.0, orientation_gain=1.0, damping=0.01,
        max_linear_speed=0.2, max_angular_speed=0.5, max_joint_speed=0.1,
    )
    assert velocity.shape == (3,)
    assert np.max(np.abs(velocity)) <= 0.1 + 1e-7
    assert velocity[0] > 0.0


def test_damped_least_squares_velocity_rejects_nonfinite_jacobian():
    with pytest.raises(ValueError, match="finite"):
        damped_least_squares_velocity(
            np.full((6, 6), np.nan), np.zeros(3), np.zeros(3),
            position_gain=1.0, orientation_gain=1.0, damping=0.1,
            max_linear_speed=0.2, max_angular_speed=0.5, max_joint_speed=0.7,
        )


def _transition_info(contact_before=False, contact_after=False, reached=False, orientation_scale=1.0):
    return {
        "rho_position": .4, "next_rho_position": .3,
        "rho_orientation": .2, "next_rho_orientation": .1,
        "smooth_velocity": .5, "velocity_magnitude": .25,
        "orientation_scale": orientation_scale,
        "task_reached": reached, "hard_failure": False,
        "obstacle_collision": contact_after and not contact_before,
        "self_collision": False, "environment_collision": False, "joint_limit": False,
        "control_max_risk": .6, "control_min_distance": .1,
        "control_self_max_risk": 0.0, "control_self_min_distance": .25,
        "control_self_max_approach": 0.0, "control_self_min_ttc": 3.0,
        "self_clearance_violation": 0.0,
        "obstacle_contact_seen_before": contact_before, "obstacle_contact_seen": contact_after,
    }


def test_actor_can_be_frozen_while_fresh_critics_warm_up():
    config = load_config(
        Path(__file__).resolve().parents[2]
        / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml"
    )
    config["device"] = "cpu"
    config["sac"]["hidden_dims"] = [8, 8]
    config["sac"]["actor_update_interval"] = 1
    agent = ThesisSACAgent(4, 2, config)
    actor_before = {
        key: value.detach().clone() for key, value in agent.actor.state_dict().items()
    }
    batch = Batch(
        observations=torch.randn(16, 4),
        actions=torch.tanh(torch.randn(16, 2)),
        rewards=torch.randn(16, 1),
        costs=torch.zeros(16, 1),
        next_observations=torch.randn(16, 4),
        dones=torch.zeros(16, 1),
    )

    metrics = agent.update(batch, update_actor=False)

    assert metrics["actor_updated"] == 0.0
    assert all(
        torch.equal(actor_before[key], value)
        for key, value in agent.actor.state_dict().items()
    )
    assert metrics["critic_gradient_norm"] > 0.0


def test_update_can_skip_diagnostics_without_skipping_critic_update(monkeypatch):
    config = load_config(
        Path(__file__).resolve().parents[2]
        / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml"
    )
    config["device"] = "cpu"
    config["sac"]["hidden_dims"] = [8, 8]
    agent = ThesisSACAgent(4, 2, config)
    critic_before = {
        key: value.detach().clone() for key, value in agent.q1.state_dict().items()
    }
    monkeypatch.setattr(
        agent.actor,
        "deterministic",
        lambda *_args, **_kwargs: pytest.fail(
            "Critic-only update must not run a diagnostic Actor forward"
        ),
    )
    batch = Batch(
        observations=torch.randn(16, 4),
        actions=torch.tanh(torch.randn(16, 2)),
        rewards=torch.randn(16, 1),
        costs=torch.zeros(16, 1),
        next_observations=torch.randn(16, 4),
        dones=torch.zeros(16, 1),
    )

    metrics = agent.update(
        batch, update_actor=False, collect_diagnostics=False,
    )

    assert metrics == {}
    assert any(
        not torch.equal(critic_before[key], value)
        for key, value in agent.q1.state_dict().items()
    )


def test_chain_pcr_uses_separate_reference_states_and_one_update_lag():
    config = load_config(
        Path(__file__).resolve().parents[2]
        / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml"
    )
    config["device"] = "cpu"
    config["sac"].update({
        "hidden_dims": [8, 8], "actor_update_interval": 1,
        "chain_pcr_enabled": True, "chain_pcr_coefficient": 5e-4,
    })
    agent = ThesisSACAgent(4, 2, config)
    batch = Batch(
        observations=torch.randn(16, 4),
        actions=torch.tanh(torch.randn(16, 2)),
        rewards=torch.randn(16, 1), costs=torch.zeros(16, 1),
        next_observations=torch.randn(16, 4), dones=torch.zeros(16, 1),
    )
    reference = torch.randn(16, 4)
    first = agent.update(batch, reference_observations=reference)
    second = agent.update(batch, reference_observations=reference)
    assert first["policy_churn_kl"] == pytest.approx(0.0, abs=1e-7)
    assert second["policy_churn_kl"] > 0.0
    assert second["actor_loss"] == pytest.approx(
        second["actor_sac_loss"] + second["policy_churn_penalty"]
    )
    assert second["policy_churn_to_sac_ratio"] == pytest.approx(
        second["policy_churn_penalty"] / max(abs(second["actor_sac_loss"]), 1e-8)
    )


def test_chain_pcr_gaussian_kl_direction_and_identity():
    zero = torch.zeros(3, 2)
    assert diagonal_gaussian_kl_old_to_current(zero, zero, zero, zero) == 0.0
    value = diagonal_gaussian_kl_old_to_current(
        zero, zero, torch.ones(3, 2), zero,
    )
    assert value.item() == pytest.approx(1.0)


def test_chain_pcr_loss_ratio_cap_limits_effective_coefficient():
    config = load_config(
        Path(__file__).resolve().parents[2]
        / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml"
    )
    config["device"] = "cpu"
    config["sac"].update({
        "hidden_dims": [8, 8],
        "actor_update_interval": 1,
        "chain_pcr_enabled": True,
        "chain_pcr_coefficient": 1e6,
        "chain_pcr_max_loss_ratio": .01,
    })
    agent = ThesisSACAgent(4, 2, config)
    batch = Batch(
        observations=torch.randn(16, 4),
        actions=torch.tanh(torch.randn(16, 2)),
        rewards=torch.randn(16, 1), costs=torch.zeros(16, 1),
        next_observations=torch.randn(16, 4), dones=torch.zeros(16, 1),
    )
    reference = torch.randn(16, 4)
    agent.update(batch, reference_observations=reference)
    metrics = agent.update(batch, reference_observations=reference)
    assert metrics["policy_churn_kl"] > 0.0
    assert metrics["policy_churn_to_sac_ratio"] <= .010001
    assert metrics["chain_pcr_effective_coefficient"] < 1e6


def test_auto_chain_pcr_tracks_relative_loss_scale_and_restores_ema():
    config = load_config(
        Path(__file__).resolve().parents[2]
        / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml"
    )
    config["device"] = "cpu"
    config["sac"].update({
        "hidden_dims": [8, 8], "actor_update_interval": 1,
        "chain_pcr_enabled": True, "chain_pcr_coefficient": 1.0,
        "chain_pcr_auto_enabled": True, "chain_pcr_target_ratio": .003,
        "chain_pcr_ema_decay": .99, "chain_pcr_auto_min_coefficient": 1e-4,
        "chain_pcr_auto_max_coefficient": 100.0,
        "chain_pcr_max_loss_ratio": .10,
    })
    agent = ThesisSACAgent(4, 2, config)
    batch = Batch(
        observations=torch.randn(32, 4), actions=torch.tanh(torch.randn(32, 2)),
        rewards=torch.randn(32, 1), costs=torch.zeros(32, 1),
        next_observations=torch.randn(32, 4), dones=torch.zeros(32, 1),
    )
    reference = torch.randn(32, 4)
    agent.update(batch, reference_observations=reference)
    metrics = agent.update(batch, reference_observations=reference)
    assert metrics["policy_churn_kl"] > 0.0
    assert 0.0 < metrics["policy_churn_to_sac_ratio"] <= .10
    assert metrics["chain_pcr_sac_loss_ema"] > 0.0
    assert metrics["chain_pcr_loss_ema"] > 0.0
    restored = ThesisSACAgent(4, 2, config)
    restored.load_state_dict(agent.state_dict())
    assert restored.chain_pcr_sac_loss_ema == pytest.approx(
        agent.chain_pcr_sac_loss_ema
    )
    assert restored.chain_pcr_loss_ema == pytest.approx(agent.chain_pcr_loss_ema)


def test_chain_reference_can_be_reset_after_actor_only_initialization():
    config = load_config(
        Path(__file__).resolve().parents[2]
        / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml"
    )
    config["device"] = "cpu"
    config["sac"].update({
        "hidden_dims": [8, 8],
        "chain_pcr_enabled": True,
        "chain_pcr_coefficient": 5e-4,
    })
    agent = ThesisSACAgent(4, 2, config)
    with torch.no_grad():
        next(agent.actor.parameters()).add_(1.0)

    agent.reset_chain_reference_actor()

    assert agent.chain_reference_actor is not None
    assert all(
        torch.equal(actor_value, reference_value)
        for actor_value, reference_value in zip(
            agent.actor.state_dict().values(),
            agent.chain_reference_actor.state_dict().values(),
            strict=True,
        )
    )


def test_chain_reference_is_initialized_when_resuming_pre_chain_agent():
    config = load_config(
        Path(__file__).resolve().parents[2]
        / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml"
    )
    config["device"] = "cpu"
    config["sac"]["hidden_dims"] = [8, 8]
    baseline = ThesisSACAgent(4, 2, config)
    legacy_state = baseline.state_dict()

    config["sac"].update({
        "chain_pcr_enabled": True,
        "chain_pcr_coefficient": 5e-4,
    })
    resumed = ThesisSACAgent(4, 2, config)
    resumed.load_state_dict(legacy_state)

    assert resumed.chain_reference_actor is not None
    assert all(
        torch.equal(actor_value, reference_value)
        for actor_value, reference_value in zip(
            resumed.actor.state_dict().values(),
            resumed.chain_reference_actor.state_dict().values(),
            strict=True,
        )
    )












def test_hybrid_keypoint_auto_chain_contract_is_162d():
    root = Path(__file__).resolve().parents[2]
    config = load_config(
        root / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml"
    )
    assert config["thesis"]["observation"]["hybrid_explicit_pose_error"] is True
    assert config["thesis"]["reward"]["keypoint_tracking_scale"] == .05
    assert config["thesis"]["reward"]["keypoint_progress_scale"] == 20.0
    assert config["thesis"]["reward"]["keypoint_precision_reward_scale"] == .05
    assert config["thesis"]["reward"]["joint_bottleneck_shaping_scale"] == 5.0
    assert config["thesis"]["reward"]["joint_bottleneck_temperature"] == 2.0
    assert config["thesis"]["reward"]["hold_reward_scale"] == .15
    assert config["sac"]["chain_pcr_auto_enabled"] is True
    assert config["sac"]["chain_pcr_target_ratio"] == .003
    env = ThesisHomotopyEnv(config)
    try:
        observation, info = env.reset(seed=31001)
        assert observation.shape == (
            THESIS_OBSERVATION_DIM_HYBRID_KEYPOINT_JACOBIAN,
        )
        np.testing.assert_allclose(
            observation[75:78],
            np.clip((info["goal_position"] - info["ee_position"]) / .05, -1., 1.),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            observation[78:81],
            np.clip(info["orientation_error_vector"] / .50, -1., 1.),
            atol=1e-6,
        )
        assert observation[81] == pytest.approx(
            np.clip(info["goal_error_norm"] / .05, 0., 1.), abs=1e-6,
        )
        assert observation[82] == pytest.approx(
            np.clip(info["orientation_error_norm"] / .50, 0., 1.), abs=1e-6,
        )
    finally:
        env.close()


def test_retired_observation_and_training_modes_are_rejected():
    root = Path(__file__).resolve().parents[2]
    config = load_config(
        root / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml"
    )
    assert thesis_observation_dim() == 162
    for flags in (
        {"keypoint_jacobian_pose": False},
        {"hybrid_explicit_pose_error": False},
        {"include_orientation_error_vector": True},
    ):
        with pytest.raises(ValueError, match="only Hybrid Keypoint"):
            thesis_observation_dim(**flags)
        old_config = copy.deepcopy(config)
        old_config["thesis"]["observation"].update(flags)
        with pytest.raises(ValueError, match="only Hybrid Keypoint"):
            ThesisHomotopyEnv(old_config)
    for section, key, value in (
        ("reward", "pose_objective", "split_pose"),
        ("reward", "keypoint_pose_reward", False),
        ("sac", "chain_pcr_auto_enabled", False),
    ):
        old_config = copy.deepcopy(config)
        target = old_config["sac"] if section == "sac" else old_config["thesis"][section]
        target[key] = value
        with pytest.raises(ValueError, match="only Hybrid Keypoint"):
            thesis_trainer.validate_training_architecture(old_config)


def test_three_keypoints_encode_translation_and_orientation():
    identity = np.asarray([0.0, 0.0, 0.0, 1.0])
    translation = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
    translated = pose_keypoint_errors(
        translation, identity, np.zeros(3), identity,
    )
    np.testing.assert_allclose(
        translated, np.repeat(translation[None, :], 3, axis=0), atol=1e-7,
    )
    half_turn_z = np.asarray([0.0, 0.0, 1.0, 0.0])
    rotated = pose_keypoint_errors(
        np.zeros(3), half_turn_z, np.zeros(3), identity,
    )
    assert rotated.shape == (3, 3)
    assert np.linalg.matrix_rank(rotated) >= 2


def test_keypoint_jacobian_maps_joint_velocity_to_each_point_velocity():
    rng = np.random.default_rng(41)
    geometric = rng.normal(size=(6, 6))
    qdot = rng.normal(size=6)
    quaternion = np.asarray([0.2, -0.1, 0.3, 0.9])
    actual = keypoint_position_jacobian(
        geometric, quaternion, cube_side_m=.5,
    ) @ qdot
    twist = geometric @ qdot
    expected = np.concatenate([
        twist[:3] + np.cross(twist[3:], offset)
        for offset in pose_keypoint_offsets_world(
            quaternion, cube_side_m=.5,
        )
    ])
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_pybullet_keypoint_jacobian_matches_central_finite_difference():
    root = Path(__file__).resolve().parents[2]
    config = load_config(
        root / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml"
    )
    env = ThesisHomotopyEnv(config)
    try:
        env.reset(seed=31001)
        q, _ = env.robot.joint_state_exact()
        direction = np.asarray([.2, -.4, .1, .7, -.3, .45], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        delta = 1e-3
        _, quaternion = env.robot.end_effector_pose()
        predicted = keypoint_position_jacobian(
            env.robot.end_effector_jacobian(), quaternion,
            cube_side_m=env.keypoint_cube_side_m,
        ) @ direction

        points = []
        for displaced in (q - .5 * delta * direction, q + .5 * delta * direction):
            env._set_joint_state(displaced)
            position, orientation = env.robot.end_effector_pose()
            points.append(
                np.asarray(position) + pose_keypoint_offsets_world(
                    orientation, cube_side_m=env.keypoint_cube_side_m,
                )
            )
        finite_difference = ((points[1] - points[0]) / delta).reshape(-1)
        relative_error = np.linalg.norm(predicted - finite_difference) / np.linalg.norm(
            finite_difference
        )
        assert relative_error < 2e-3
    finally:
        env.close()


def test_keypoint_reward_replaces_all_split_pose_shaping():
    common = dict(
        rho_position=.4, next_rho_position=.3,
        rho_orientation=.8, next_rho_orientation=.7,
        smooth_velocity=.2, velocity_magnitude=.1,
        orientation_scale=1.0, task_reached=False, hard_failure=False,
        obstacle_collision=False, risk_max=0.0, distance_min=.8, xi=1.0,
        keypoint_pose_reward=True, keypoint_tracking_quality=.6,
        keypoint_distance=.20, next_keypoint_distance=.19,
        keypoint_tracking_scale=.2,
        keypoint_progress_scale=10.0,
        precision_stop_cost_weight=0.0,
    )
    reward_a, fields_a = homotopy_reward(**common)
    reward_b, fields_b = homotopy_reward(
        **common, position_progress_scale=999.0,
        orientation_progress_scale=999.0,
    )
    assert reward_a == pytest.approx(reward_b)
    assert fields_a["r_goal"] == pytest.approx(fields_b["r_goal"])
    expected = .2 * .6 + 10.0 * .01 - .04 * .1 - .01 * .2
    assert fields_a["r_goal"] == pytest.approx(expected)


def test_keypoint_near_goal_precision_requires_position_and_orientation():
    common = dict(
        rho_position=.20, next_rho_position=.05,
        rho_orientation=.60, next_rho_orientation=.15,
        smooth_velocity=0.0, velocity_magnitude=0.0,
        orientation_scale=1.0, task_reached=False, hard_failure=False,
        obstacle_collision=False, risk_max=0.0, distance_min=.8, xi=1.0,
        keypoint_pose_reward=True, keypoint_tracking_quality=0.0,
        keypoint_distance=.3, next_keypoint_distance=.3,
        keypoint_tracking_scale=0.0, keypoint_progress_scale=0.0,
        joint_position_tolerance=.10, joint_orientation_tolerance=.30,
    )
    baseline, _ = homotopy_reward(
        **common, keypoint_precision_reward_scale=0.0,
    )
    rewarded, fields = homotopy_reward(
        **common, keypoint_precision_reward_scale=.20,
    )
    expected_quality = np.exp(-.5)
    assert fields["keypoint_precision_quality"] == pytest.approx(expected_quality)
    assert fields["keypoint_precision_reward"] == pytest.approx(.20 * expected_quality)
    assert rewarded - baseline == pytest.approx(.20 * expected_quality)


def test_success_adds_only_the_fixed_bonus():
    common = dict(
        rho_position=.01, next_rho_position=.009,
        rho_orientation=.09, next_rho_orientation=.08,
        smooth_velocity=0.0, velocity_magnitude=0.0,
        orientation_scale=1.0, hard_failure=False,
        obstacle_collision=False, risk_max=0.0, distance_min=.8, xi=1.0,
        keypoint_pose_reward=True, keypoint_tracking_quality=.8,
        keypoint_distance=.03, next_keypoint_distance=.025,
        success_bonus=20.0,
    )
    running, _ = homotopy_reward(
        **common, task_reached=False, joint_bottleneck_shaping_scale=0.0,
    )
    succeeded, fields = homotopy_reward(
        **common, task_reached=True, joint_bottleneck_shaping_scale=0.0,
    )
    assert succeeded - running == pytest.approx(20.0)
    assert "terminal_occupancy_compensation" not in fields


def test_joint_bottleneck_potential_shaping_distinguishes_progress_and_stasis():
    common = dict(
        rho_position=.01, rho_orientation=.10,
        smooth_velocity=0.0, velocity_magnitude=0.0,
        orientation_scale=1.0, hard_failure=False,
        obstacle_collision=False, risk_max=0.0, distance_min=.8, xi=1.0,
        keypoint_pose_reward=True, keypoint_tracking_quality=0.0,
        keypoint_distance=.2, next_keypoint_distance=.2,
        keypoint_tracking_scale=0.0, keypoint_progress_scale=0.0,
        keypoint_precision_reward_scale=0.0, hold_reward_scale=0.0,
        velocity_cost_weight=0.0, smooth_cost_weight=0.0,
        precision_stop_cost_weight=0.0,
        joint_position_tolerance=.01, joint_orientation_tolerance=.10,
        joint_bottleneck_shaping_scale=5.0,
        joint_bottleneck_temperature=2.0,
        gamma=.99,
    )
    stationary, stationary_fields = homotopy_reward(
        **common, next_rho_position=.01, next_rho_orientation=.10,
        task_reached=False,
    )
    improving, _ = homotopy_reward(
        **common, next_rho_position=.009, next_rho_orientation=.09,
        task_reached=False,
    )
    worsening, _ = homotopy_reward(
        **common, next_rho_position=.011, next_rho_orientation=.11,
        task_reached=False,
    )
    terminal, terminal_fields = homotopy_reward(
        **common, next_rho_position=.009, next_rho_orientation=.09,
        task_reached=True, success_bonus=0.0,
    )

    expected_potential = 5.0 * np.exp(-.5)
    assert stationary == pytest.approx(-.01 * expected_potential)
    assert improving > 0.0
    assert worsening < stationary
    assert stationary_fields["joint_bottleneck_ratio"] == pytest.approx(1.0)
    assert terminal_fields["next_joint_bottleneck_potential"] == 0.0
    assert terminal == pytest.approx(-expected_potential)


def test_keypoint_replay_reward_matches_scalar_and_uses_29_fields():
    parameters = {
        "keypoint_pose_reward": True,
        "keypoint_tracking_scale": .2,
        "keypoint_progress_scale": 10.0,
        "keypoint_precision_reward_scale": .2,
        "joint_bottleneck_shaping_scale": 5.0,
        "joint_bottleneck_temperature": 2.0,
    }
    replay = HomotopyReplayBuffer(
        162, 6, "cpu", capacities={"none": 8, "static": 8, "dynamic": 8},
        s0_anchor_capacity=8, s0_current_capacity=8,
        s0_history_capacity_per_level=4, s0_success_capacity=8,
        reward_parameters=parameters,
    )
    assert replay.s0_current.raw.shape[1] == 29

    raw = np.zeros((1, 29), dtype=np.float32)
    raw[0, 12] = .8
    raw[0, 18] = .25
    raw[0, 24] = .05
    raw[0, 25] = .10
    raw[0, 26] = .20
    raw[0, 27] = .19
    raw[0, 28] = .6
    vectorized, _ = _vectorized_replay_rewards(
        raw, np.zeros((1, 1), dtype=np.float32), np.ones(1),
        0.0, .99, parameters,
    )
    scalar, _ = homotopy_reward(
        rho_position=0.0, next_rho_position=0.0,
        rho_orientation=0.0, next_rho_orientation=0.0,
        smooth_velocity=0.0, velocity_magnitude=0.0,
        orientation_scale=1.0, task_reached=False, hard_failure=False,
        obstacle_collision=False, risk_max=0.0, distance_min=.8, xi=1.0,
        keypoint_pose_reward=True, keypoint_tracking_quality=.6,
        keypoint_distance=.20, next_keypoint_distance=.19,
        keypoint_tracking_scale=.2,
        keypoint_progress_scale=10.0,
        keypoint_precision_reward_scale=.2,
        joint_bottleneck_shaping_scale=5.0,
        joint_bottleneck_temperature=2.0,
    )
    assert vectorized[0] == pytest.approx(scalar, abs=1e-7)


def test_keypoint_reward_statistics_report_density_and_signed_progress():
    counters = {}
    for quality, progress in ((.1, .02), (.3, -.01), (.2, 0.0)):
        thesis_trainer.record_keypoint_reward_statistics(
            counters,
            {"keypoint_tracking_quality": quality, "keypoint_progress": progress},
            enabled=True,
        )
    metrics = thesis_trainer.keypoint_reward_statistics(counters)
    assert metrics["keypoint_tracking_quality_mean"] == pytest.approx(.2)
    assert metrics["keypoint_progress_mean"] == pytest.approx(.01 / 3)
    assert metrics["keypoint_progress_positive_ratio"] == pytest.approx(1 / 3)
    assert metrics["keypoint_progress_mean_when_positive"] == pytest.approx(.02)
    assert metrics["jacobian_clip_ratio_mean"] == pytest.approx(0.0)







def test_obstacle_observation_exposes_each_link_approach_velocity_and_risk():
    geometry = ThesisGeometry(
        relative_vectors=np.arange(18, dtype=np.float32).reshape(6, 3) / 100.0,
        distances=np.linspace(.05, .10, 6, dtype=np.float32),
        ttc=np.linspace(.1, .6, 6, dtype=np.float32),
        approach_velocities=np.linspace(.1, .6, 6, dtype=np.float32),
        risk_per_link=np.linspace(.2, .7, 6, dtype=np.float32),
        risk_max=.7,
        distance_min=.05,
    )
    observation = build_thesis_observation(
        q=np.zeros(6), qdot=np.zeros(6), joint_lower=-np.pi * np.ones(6),
        joint_upper=np.pi * np.ones(6), joint_velocity_scale=.7 * np.ones(6),
        ee_position=np.zeros(3), ee_quaternion=np.asarray([0., 0., 0., 1.]),
        ee_linear_velocity=np.zeros(3), ee_angular_velocity=np.zeros(3),
        ee_linear_velocity_scale=1.0, ee_angular_velocity_scale=4.2,
        goal_position=np.zeros(3), goal_quaternion=np.asarray([0., 0., 0., 1.]),
        position_error=np.zeros(3), orientation_error=np.zeros(3),
        obstacle_present=True,
        obstacle_position=np.asarray([.4, 0., .4]),
        obstacle_velocity=np.asarray([.1, 0., 0.]), geometry=geometry,
        ee_jacobian=np.eye(6),
    )
    # Approach and risk are the final two
    # six-link vectors immediately before the presence flag.
    np.testing.assert_allclose(observation[125:131], geometry.approach_velocities)
    np.testing.assert_allclose(observation[131:137], geometry.risk_per_link)
    assert observation[141] == 1.0


def test_self_collision_observation_appends_six_link_continuous_features():
    self_geometry = SelfCollisionGeometry(
        distances=np.linspace(.005, .25, 6, dtype=np.float32),
        ttc=np.linspace(0.0, 3.0, 6, dtype=np.float32),
        approach_velocities=np.linspace(0.0, .5, 6, dtype=np.float32),
        risk_per_link=np.linspace(1.0, 0.0, 6, dtype=np.float32),
        risk_max=1.0, distance_min=.005, ttc_min=0.0, approach_max=.5,
        closest_pair=(0, 2),
    )
    observation = build_thesis_observation(
        q=np.zeros(6), qdot=np.zeros(6), joint_lower=-np.pi * np.ones(6),
        joint_upper=np.pi * np.ones(6), joint_velocity_scale=.7 * np.ones(6),
        ee_position=np.zeros(3), ee_quaternion=np.asarray([0., 0., 0., 1.]),
        ee_linear_velocity=np.zeros(3), ee_angular_velocity=np.zeros(3),
        ee_linear_velocity_scale=1.0, ee_angular_velocity_scale=4.2,
        goal_position=np.zeros(3), goal_quaternion=np.asarray([0., 0., 0., 1.]),
        position_error=np.zeros(3), orientation_error=np.zeros(3),
        obstacle_present=False,
        obstacle_position=np.zeros(3), obstacle_velocity=np.zeros(3), geometry=None,
        self_geometry=self_geometry,
        ee_jacobian=np.eye(6),
    )
    assert observation.shape == (THESIS_OBSERVATION_DIM_HYBRID_KEYPOINT_JACOBIAN,)
    expected_distance = 2.0 * (self_geometry.distances + .02) / .27 - 1.0
    np.testing.assert_allclose(observation[138:144], expected_distance)
    np.testing.assert_allclose(observation[144:150], self_geometry.ttc / 3.0)
    np.testing.assert_allclose(observation[150:156], self_geometry.approach_velocities)
    np.testing.assert_allclose(observation[156:162], self_geometry.risk_per_link)
    assert np.all(np.abs(observation) <= 1.0)


def test_self_geometry_ignores_static_structural_neighbours_and_predicts_approach():
    links = [(-1, 1), (2, 4)]
    static = compute_self_collision_geometry(
        np.asarray([.009, .012]), np.zeros(2), links, d_safe=.005,
    )
    approaching = compute_self_collision_geometry(
        np.asarray([.009, .012]), np.asarray([.02, 0.0]), links, d_safe=.005,
    )
    closer = compute_self_collision_geometry(
        np.asarray([.006, .012]), np.asarray([.02, 0.0]), links, d_safe=.005,
    )
    contact = compute_self_collision_geometry(
        np.asarray([-.001, .012]), np.zeros(2), links, d_safe=.005,
    )
    assert static.risk_max == 0.0
    assert 0.0 < approaching.risk_max < closer.risk_max < 1.0
    assert contact.risk_max == 1.0
    assert contact.distance_min == np.float32(-.001)


def test_low_xi_keeps_goal_reward_and_scales_only_safety_terms():
    common = dict(rho_position=.4, next_rho_position=.3, rho_orientation=.2,
                  next_rho_orientation=.1, smooth_velocity=.5, velocity_magnitude=.25,
                  orientation_scale=1.0, task_reached=True,
                  hard_failure=False, obstacle_collision=True, risk_max=.6, distance_min=.1)
    low, low_fields = homotopy_reward(**common, xi=.02)
    strict, strict_fields = homotopy_reward(**common, xi=1.0)
    assert low_fields["r_goal"] == strict_fields["r_goal"]
    expected_gap = .05 * .98 * (
        2 * .6 + 8 * strict_fields["clearance_violation"]
    ) / 10
    assert np.isclose(low - strict, expected_gap)


def test_self_weight_is_independent_of_scene_weight_and_reward_is_reconstructable():
    common = dict(
        rho_position=.4, next_rho_position=.3, rho_orientation=.2,
        next_rho_orientation=.1, smooth_velocity=.5, velocity_magnitude=.25,
        orientation_scale=1.0, task_reached=False, hard_failure=False,
        obstacle_collision=False, risk_max=.4, distance_min=.1,
        self_risk_max=.6, self_distance_min=.002,
    )
    low, low_fields = homotopy_reward(**common, xi=.02, lambda_self=.02)
    self_strict, strict_fields = homotopy_reward(
        **common, xi=.02, lambda_self=1.0,
    )
    scene_strict, scene_fields = homotopy_reward(
        **common, xi=1.0, lambda_self=.02,
    )
    assert self_strict < low and scene_strict < low
    assert strict_fields["external_safety_penalty"] == low_fields["external_safety_penalty"]
    assert scene_fields["self_safety_penalty"] == low_fields["self_safety_penalty"]
    assert np.isclose(
        low,
        low_fields["r_goal"] - low_fields["hard_penalty"]
        - low_fields["external_safety_penalty"] - low_fields["self_safety_penalty"],
    )


def test_dense_safety_groups_are_bounded_before_curriculum_scaling():
    _, fields = homotopy_reward(
        rho_position=.4, next_rho_position=.4,
        rho_orientation=.8, next_rho_orientation=.8,
        smooth_velocity=0.0, velocity_magnitude=0.0,
        orientation_scale=1.0, task_reached=False, hard_failure=False,
        obstacle_collision=False, risk_max=1.0, distance_min=-1.0, xi=1.0,
        self_risk_max=1.0, self_distance_min=-1.0, lambda_self=1.0,
    )
    assert fields["external_safety_cost"] == 1.0
    assert fields["self_safety_cost"] == 1.0
    assert fields["external_safety_penalty"] == .05
    assert fields["self_safety_penalty"] == .05






def test_eta_is_observable_difficulty_only_and_does_not_scale_v12_reward():
    common = dict(
        rho_position=.3, next_rho_position=.2,
        rho_orientation=1.2, next_rho_orientation=.8,
        smooth_velocity=.1, velocity_magnitude=.2, task_reached=False,
        hard_failure=False, obstacle_collision=False, risk_max=0.0,
        distance_min=.8, xi=1.0,
    )
    first, first_fields = homotopy_reward(**common, orientation_scale=0.0)
    final, final_fields = homotopy_reward(**common, orientation_scale=1.0)
    assert np.isclose(first, final)
    assert first_fields["orientation_progress"] > 0.0
    assert first_fields["orientation_reward_scale"] == final_fields["orientation_reward_scale"]


def test_orientation_reward_is_not_gated_by_position_quality():
    common = dict(
        rho_orientation=1.0, next_rho_orientation=.9,
        smooth_velocity=0.0, velocity_magnitude=0.0,
        orientation_scale=1.0, task_reached=False, hard_failure=False,
        obstacle_collision=False, risk_max=0.0, distance_min=.8, xi=1.0,
    )
    _, far = homotopy_reward(
        **common, rho_position=.26, next_rho_position=.25,
    )
    _, middle = homotopy_reward(
        **common, rho_position=.15, next_rho_position=.14,
    )
    _, near = homotopy_reward(
        **common, rho_position=.07, next_rho_position=.06,
    )
    assert far["orientation_reward_scale"] == 2.0
    assert far["orientation_reward_scale"] == middle["orientation_reward_scale"]
    assert middle["orientation_reward_scale"] == near["orientation_reward_scale"]
    assert far["orientation_reward_gate"] == 1.0




def test_progress_is_positive_and_terminal_penalties_are_bounded():
    progress, progress_fields = homotopy_reward(
        rho_position=.42, next_rho_position=.41,
        rho_orientation=2.32, next_rho_orientation=2.32,
        smooth_velocity=0.0, velocity_magnitude=0.0, orientation_scale=1.0,
        task_reached=False, hard_failure=False,
        obstacle_collision=False, risk_max=0.0, distance_min=.8, xi=1.0,
    )
    assert progress_fields["r_goal"] == progress
    assert progress_fields["position_progress"] > 0.0
    assert progress_fields["state_cost"] > 0.0

    common = dict(
        rho_position=.5, next_rho_position=.5, rho_orientation=1.2,
        next_rho_orientation=1.2, smooth_velocity=0.0, velocity_magnitude=0.0,
        orientation_scale=1.0, task_reached=False,
        risk_max=0.0, distance_min=.8, xi=1.0, gamma=.99, horizon=240,
        keypoint_precision_reward_scale=0.0,
    )
    wait_step, wait_fields = homotopy_reward(
        **common, hard_failure=False, obstacle_collision=False,
    )
    hard, hard_fields = homotopy_reward(
        **common, hard_failure=True, obstacle_collision=False,
    )
    strict_collision, strict_fields = homotopy_reward(
        **{**common, "risk_max": 1.0, "distance_min": 0.0},
        hard_failure=False, obstacle_collision=True,
        terminal_obstacle_collision=True,
    )
    tolerant_collision, tolerant_fields = homotopy_reward(
        **{**common, "risk_max": 1.0, "distance_min": 0.0, "xi": .02},
        hard_failure=False, obstacle_collision=True,
        terminal_obstacle_collision=False,
    )
    assert wait_step == 0.0
    assert hard >= -20.1
    assert strict_collision >= -20.1
    assert hard_fields["terminal_guard_penalty"] == 0
    assert strict_fields["terminal_guard_penalty"] == 20.0
    assert tolerant_fields["terminal_guard_penalty"] == 0
    expected_tolerant_penalty = .001
    assert tolerant_collision == wait_fields["r_goal"] - expected_tolerant_penalty
    assert min(hard, strict_collision, tolerant_collision) >= -20.1


def test_timeout_has_an_explicit_terminal_penalty():
    common = dict(
        rho_position=.3, next_rho_position=.3,
        rho_orientation=1.0, next_rho_orientation=1.0,
        smooth_velocity=0.0, velocity_magnitude=0.0, orientation_scale=0.0,
        task_reached=False, hard_failure=False, obstacle_collision=False,
        risk_max=0.0, distance_min=.8, xi=1.0,
    )
    running, _ = homotopy_reward(**common, timeout=False)
    timed_out, fields = homotopy_reward(**common, timeout=True)
    assert np.isclose(running - timed_out, 2.0)
    assert fields["timeout_penalty"] == 2.0



@pytest.mark.parametrize(
    ("orientation_tolerance", "expected_sigma"),
    [(.30, .15), (.24, .12), (.192, .096), (.1536, .0768), (.10, .06)],
)
def test_local_orientation_sigma_tracks_episode_tolerance(
    orientation_tolerance, expected_sigma,
):
    _, fields = homotopy_reward(
        rho_position=.20, next_rho_position=.20,
        rho_orientation=.40, next_rho_orientation=.30,
        orientation_tolerance=orientation_tolerance,
        smooth_velocity=0.0, velocity_magnitude=0.0,
        orientation_scale=1.0, task_reached=False, hard_failure=False,
        obstacle_collision=False, risk_max=0.0, distance_min=.8, xi=1.0,
    )
    assert fields["local_orientation_sigma"] == pytest.approx(expected_sigma)


def test_pose_probe_summary_reports_orientation_bands_and_rebound_metrics():
    values = np.zeros((100, 10), dtype=np.float64)
    values[:, 6] = .20
    values[:, 9] = np.arange(100)
    orientation_bins = np.arange(100) % 10
    values[:, 0] = orientation_bins < 7
    values[:, 3] = orientation_bins >= 7
    values[:, 5] = np.linspace(.01, 1.0, 100)

    metrics = thesis_trainer.summarize_pose_probe_outcomes(
        values, has_bins=True, orientation_bins=10,
        large_angle_steps=10, orientation_improvements=7,
        orientation_rebound_sum=.2,
    )

    assert metrics["success_rate_o0_o4"] == 1.0
    assert metrics["success_rate_o5_o6"] == 1.0
    assert metrics["success_rate_o7_o9"] == 0.0
    assert metrics["timeout_rate_o7_o9"] == 1.0
    assert metrics["weak_bin_count_below_0_6"] == 30
    assert metrics["large_angle_orientation_improvement_ratio"] == .7
    assert metrics["large_angle_orientation_mean_rebound_rad"] == pytest.approx(.02)
    assert metrics["p95_orientation_error_rad"] == pytest.approx(.9505)


def test_stop_cost_is_stronger_near_the_goal():
    common = dict(
        rho_position=.01, rho_orientation=.03,
        smooth_velocity=0.0, velocity_magnitude=.25,
        orientation_scale=0.0, task_reached=False, hard_failure=False,
        obstacle_collision=False, risk_max=0.0, distance_min=.8, xi=1.0,
    )
    near_reward, near = homotopy_reward(
        **common, next_rho_position=.01, next_rho_orientation=.03,
    )
    far_reward, far = homotopy_reward(
        **common, next_rho_position=.20, next_rho_orientation=.80,
    )
    near_without_stop, _ = homotopy_reward(
        **common, next_rho_position=.01, next_rho_orientation=.03,
        precision_stop_cost_weight=0.0,
    )
    far_without_stop, _ = homotopy_reward(
        **common, next_rho_position=.20, next_rho_orientation=.80,
        precision_stop_cost_weight=0.0,
    )
    assert near["precision_proximity"] > far["precision_proximity"]
    assert near["precision_stop_cost"] > far["precision_stop_cost"]
    assert near_without_stop - near_reward == pytest.approx(
        near["precision_stop_cost"]
    )
    assert far_without_stop - far_reward == pytest.approx(
        far["precision_stop_cost"]
    )





def test_success_requires_consecutive_strict_pose_steps():
    env = object.__new__(ThesisHomotopyEnv)
    env.success_hold_steps = 3
    env.success_hold_count = 0
    assert not env._advance_success_hold(True)
    assert not env._advance_success_hold(True)
    assert env.success_hold_count == 2
    assert not env._advance_success_hold(False)
    assert env.success_hold_count == 0
    assert not env._advance_success_hold(True)
    assert not env._advance_success_hold(True)
    assert env._advance_success_hold(True)
    assert env.success_hold_count == 3


def test_clearance_penalty_is_continuous_at_safe_distance():
    common = dict(
        rho_position=.2, next_rho_position=.2, rho_orientation=.3,
        next_rho_orientation=.3, smooth_velocity=0.0, velocity_magnitude=0.0,
        orientation_scale=1.0, task_reached=False,
        hard_failure=False, obstacle_collision=False, risk_max=.5, xi=1.0,
    )
    below, _ = homotopy_reward(**common, distance_min=.12 - 1e-7)
    above, _ = homotopy_reward(**common, distance_min=.12 + 1e-7)
    assert abs(below - above) < 1e-5


def test_curriculum_waits_for_full_window_and_pauses_below_floor():
    curriculum = HomotopyCurriculum("s1", seed=1, ramp_steps=50000)
    for _ in range(99): curriculum.finish_episode("static", True, True, 10)
    assert curriculum.states["static"].eligible_steps == 0
    curriculum.finish_episode("static", True, True, 10)
    assert curriculum.states["static"].eligible_steps == 10
    for _ in range(20): curriculum.finish_episode("static", False, False, 10)
    frozen = curriculum.states["static"].eligible_steps
    curriculum.finish_episode("static", False, False, 10)
    assert curriculum.states["static"].eligible_steps == frozen


def test_goal_curriculum_is_success_gated_and_reaches_full_scale():
    curriculum = HomotopyCurriculum(
        "s0", seed=1, goal_start_scale=.03, goal_end_scale=1.0,
        goal_levels=[.03, .10, 1.0], goal_min_transitions_per_level=10,
        goal_success_window=4, goal_success_floor=.75,
    )
    def finish(reached, *, eligible=True):
        for _ in range(5):
            curriculum.record_transition(
                curriculum.goal_scale, curriculum.orientation_scale,
                curriculum_eligible=eligible,
            )
        curriculum.finish_episode(
            "none", reached, reached, 5, curriculum_eligible=eligible,
        )

    for reached in (True, True, False):
        finish(reached)
    assert curriculum.goal_scale == .03
    finish(True)
    assert curriculum.goal_scale == .10
    assert len(curriculum.goal.outcomes) == 0
    for reached in (False, False, True, True):
        finish(reached)
    assert curriculum.goal_scale == .10
    finish(True)
    assert curriculum.goal_scale == 1.0


def test_warmup_transitions_and_episodes_cannot_advance_goal_curriculum():
    curriculum = HomotopyCurriculum(
        "s0", goal_levels=[.03, 1.0], goal_min_transitions_per_level=1,
        goal_success_window=1, goal_success_floor=1.0,
    )
    curriculum.record_transition(.03, 0.0, curriculum_eligible=False)
    curriculum.finish_episode(
        "none", True, True, 1, curriculum_eligible=False,
    )
    assert curriculum.goal_scale == .03
    assert curriculum.goal.current_level_steps == 0
    assert len(curriculum.goal.outcomes) == 0


def test_s0_gate_requires_full_scale_consolidation_transitions():
    curriculum = HomotopyCurriculum(
        "s0", seed=1, goal_start_scale=.03, goal_end_scale=1.0,
        goal_levels=[.03, 1.0], goal_min_transitions_per_level=1,
        goal_success_window=1, goal_success_floor=1.0,
        goal_full_scale_min_steps=3, orientation_success_window=1,
        orientation_success_floor=1.0,
        orientation_levels=[0.0, 1.0], orientation_min_transitions_per_level=3,
        orientation_anchor_window=1, orientation_anchor_floor=1.0,
        orientation_full_scale_min_steps=3,
        self_ramp_steps=3, self_full_weight_min_steps=1,
    )
    curriculum.record_transition(.03, 0.0)
    curriculum.finish_episode("none", True, True, 1)
    assert curriculum.goal_at_full_scale
    assert not curriculum.s0_goal_gate_eligible(3, 3)
    curriculum.record_transition(1.0, 0.0)
    curriculum.record_transition(1.0, 0.0)
    assert not curriculum.position_phase_complete
    curriculum.record_transition(1.0, 0.0)
    assert curriculum.position_phase_complete
    curriculum.finish_episode("none", True, True, 3)
    assert not curriculum.orientation_at_full_scale
    for _ in range(3):
        curriculum.record_transition(1.0, 0.0)
    curriculum.finish_episode("none", True, True, 3)
    assert curriculum.orientation_at_full_scale
    assert not curriculum.s0_goal_gate_eligible(3, 3)
    for _ in range(3):
        curriculum.record_transition(1.0, 1.0)
    curriculum.finish_episode("none", True, True, 3)
    curriculum.finish_episode("none", True, True, 1, orientation_anchor=True)
    assert curriculum.pose_phase_complete
    for _ in range(3):
        curriculum.record_transition(1.0, 1.0)
    curriculum.finish_episode("none", True, True, 3)
    assert curriculum.lambda_self == 1.0
    curriculum.record_transition(1.0, 1.0)
    curriculum.finish_episode("none", True, True, 1)
    assert curriculum.s0_goal_gate_eligible(3, 3)
    full_goal_steps = curriculum.goal.full_scale_steps
    curriculum.record_transition(.99, 1.0)
    assert curriculum.goal.full_scale_steps == full_goal_steps


def test_orientation_curriculum_is_discrete_and_requires_anchor_retention():
    curriculum = HomotopyCurriculum(
        "s0", goal_levels=[.03, 1.0], goal_min_transitions_per_level=1,
        goal_success_window=1, goal_success_floor=1.0, goal_full_scale_min_steps=1,
        orientation_levels=[0.0, .5, 1.0], orientation_min_transitions_per_level=4,
        orientation_success_window=2, orientation_success_floor=1.0,
        orientation_anchor_window=2, orientation_anchor_floor=1.0,
        orientation_anchor_probability=.25,
    )
    curriculum.record_transition(.03, 0.0)
    curriculum.finish_episode("none", True, True, 1)
    curriculum.record_transition(1.0, 0.0)
    curriculum.finish_episode("none", True, True, 1)
    for _ in range(2):
        for _ in range(2):
            curriculum.record_transition(1.0, 0.0)
        curriculum.finish_episode("none", True, True, 2)
    assert curriculum.orientation_scale == .5
    assert len(curriculum.orientation.outcomes) == 0
    curriculum.finish_episode(
        "none", False, False, 1, orientation_anchor=True,
    )
    for _ in range(2):
        for _ in range(2):
            curriculum.record_transition(1.0, .5)
        curriculum.finish_episode("none", True, True, 2)
    assert curriculum.orientation_scale == .5
    assert curriculum.orientation_anchor_success_rate == 0.0
    curriculum.finish_episode(
        "none", True, True, 1, orientation_anchor=True,
    )
    assert curriculum.orientation_scale == .5
    curriculum.finish_episode(
        "none", True, True, 1, orientation_anchor=True,
    )
    curriculum.finish_episode("none", True, True, 1)
    assert curriculum.orientation_scale == 1.0


def test_only_previous_level_probe_failure_enables_recovery_mix():
    curriculum = HomotopyCurriculum(
        "s0", goal_levels=[.03, 1.0], goal_min_transitions_per_level=1,
        goal_success_window=1, goal_success_floor=1.0, goal_full_scale_min_steps=1,
        orientation_levels=[0.0, .5, 1.0], orientation_min_transitions_per_level=2,
        orientation_success_window=2, orientation_success_floor=1.0,
        orientation_anchor_window=2, orientation_anchor_floor=.5,
        orientation_retention_target=1.0, orientation_retention_min_observations=1,
        orientation_deterministic_probe_required=True,
    )
    curriculum.record_transition(.03, 0.0)
    curriculum.finish_episode("none", True, True, 1)
    curriculum.record_transition(1.0, 0.0)
    curriculum.finish_episode("none", True, True, 1)
    for _ in range(2):
        curriculum.record_transition(1.0, 0.0)
        curriculum.finish_episode("none", True, True, 1)
    assert curriculum.orientation_scale == 0.0
    assert curriculum.orientation_probe_due(1)
    curriculum.record_orientation_probe(
        success_rate=.8, collision_rate=.1, joint_limit_rate=0.0, passed=False,
        previous_success_rate=1.0, previous_collision_rate=0.0,
        previous_joint_limit_rate=0.0, retention_passed=True,
    )
    assert curriculum.orientation_retention_mode == "normal"
    np.testing.assert_allclose(curriculum.orientation_replay_mix, (.25, .50, .25))
    curriculum.record_orientation_probe(
        success_rate=1.0, collision_rate=0.0, joint_limit_rate=0.0,
        previous_success_rate=.9, previous_collision_rate=0.0,
        previous_joint_limit_rate=0.0, passed=False, retention_passed=False,
    )
    assert curriculum.orientation_retention_mode == "recovery"
    assert curriculum.orientation_anchor_probability == .5
    np.testing.assert_allclose(curriculum.orientation_replay_mix, (.5, .35, .15))
    curriculum.record_orientation_probe(
        success_rate=1.0, collision_rate=0.0, joint_limit_rate=0.0,
        previous_success_rate=1.0, previous_collision_rate=0.0,
        previous_joint_limit_rate=0.0, passed=True, retention_passed=True,
    )
    assert curriculum.advance_orientation_if_ready()
    assert curriculum.orientation_scale == .5
    assert curriculum.orientation_retention_mode == "normal"


def test_bounded_recovery_does_not_turn_a_promotion_miss_into_recovery():
    curriculum = HomotopyCurriculum(
        "s0", goal_levels=[.03, 1.0], orientation_levels=[0.0, 1.0],
        orientation_min_transitions_per_level=1,
    )
    curriculum.record_orientation_probe(
        success_rate=.90, collision_rate=.04, joint_limit_rate=0.0,
        previous_success_rate=.94, previous_collision_rate=.01,
        previous_joint_limit_rate=0.0, passed=False,
        retention_passed=False, recovery_required=False,
    )
    assert curriculum.orientation_retention_mode == "intermediate"
    np.testing.assert_allclose(curriculum.orientation_replay_mix, (.40, .40, .20))
    curriculum.record_orientation_probe(
        success_rate=.90, collision_rate=.04, joint_limit_rate=0.0,
        previous_success_rate=.89, previous_collision_rate=.01,
        previous_joint_limit_rate=0.0, passed=False,
        retention_passed=False, recovery_required=True,
    )
    assert curriculum.orientation_retention_mode == "recovery"


def test_v12_joint_pose_curriculum_advances_both_task_tolerances_together():
    levels = [
        {"goal_scale": .03, "position_tolerance_m": .12,
         "orientation_tolerance_rad": 1.5},
        {"goal_scale": .25, "position_tolerance_m": .08,
         "orientation_tolerance_rad": .7},
        {"goal_scale": 1.0, "position_tolerance_m": .055,
         "orientation_tolerance_rad": .1},
    ]
    curriculum = HomotopyCurriculum(
        "s0", goal_levels=[.03, .25, 1.0], goal_min_transitions_per_level=1,
        goal_success_window=1, goal_success_floor=1.0,
        orientation_levels=[0.0, .5, 1.0],
        orientation_min_transitions_per_level=1,
        orientation_success_window=1, orientation_success_floor=1.0,
        orientation_anchor_window=1, orientation_anchor_floor=1.0,
        joint_pose_levels=levels,
    )
    assert curriculum.pose_contract() == {
        "goal_scale": .03, "orientation_scale": 0.0,
        "position_tolerance": .12, "orientation_tolerance": 1.5,
    }
    curriculum.record_transition(.03, 0.0)
    curriculum.finish_episode("none", True, True, 1)
    assert curriculum.advance_orientation_if_ready()
    assert curriculum.pose_contract() == {
        "goal_scale": .25, "orientation_scale": .5,
        "position_tolerance": .08, "orientation_tolerance": .7,
    }
    assert curriculum.pose_contract(previous=True)["orientation_scale"] == 0.0


def test_manual_pose_promotion_can_auditably_skip_intermediate_level():
    levels = [
        {"goal_scale": .03, "position_tolerance_m": .01,
         "orientation_tolerance_rad": .03},
        {"goal_scale": .04, "position_tolerance_m": .01,
         "orientation_tolerance_rad": .03},
        {"goal_scale": .05, "position_tolerance_m": .01,
         "orientation_tolerance_rad": .03},
    ]
    curriculum = HomotopyCurriculum(
        "s0", goal_end_scale=.05, goal_levels=[.03, .04, .05],
        orientation_levels=[0.0, .5, 1.0],
        orientation_min_transitions_per_level=100,
        joint_pose_levels=levels,
    )
    curriculum.orientation.current_level_steps = 100
    curriculum.orientation.outcomes.extend([True, False])

    curriculum.manually_advance_orientation_to(2)

    assert curriculum.orientation.level_index == 2
    assert curriculum.goal.level_index == 2
    assert curriculum.goal.scale == .05
    assert curriculum.orientation.current_level_steps == 0
    assert curriculum.orientation.eligible_steps == 100
    assert not curriculum.orientation.outcomes
    with pytest.raises(ValueError, match="must be above current"):
        curriculum.manually_advance_orientation_to(1)


def test_manual_pose_promotion_requires_explicit_early_override():
    curriculum = HomotopyCurriculum(
        "s0", goal_levels=[.03, 1.0],
        orientation_levels=[0.0, .5, 1.0],
        orientation_min_transitions_per_level=100,
    )
    curriculum.orientation.current_level_steps = 25
    with pytest.raises(ValueError, match="full per-level transition budget"):
        curriculum.manually_advance_orientation_to(2)

    curriculum.manually_advance_orientation_to(2, allow_early=True)
    assert curriculum.orientation.level_index == 2
    assert curriculum.orientation.eligible_steps == 25


def test_v13_5_serial_config_is_self_contained_and_precision_only():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    thesis = config["thesis"]
    levels = thesis["joint_pose_curriculum"]["levels"]
    assert thesis["protocol"] == thesis_trainer.HYBRID_KEYPOINT_AUTO_CHAIN_PROTOCOL
    assert config["sac"]["batch_size"] == 1024
    assert config["sac"]["updates_per_transition"] == .25
    assert config["train"]["actor_initialization_critic_warmup_transitions"] == 5000
    assert thesis["precision_control"] == {
        "enabled": True, "min_action_scale": .05,
        "strict_action_scale": .10,
    }
    assert thesis["success_hold_steps"] == 5
    assert THESIS_OBSERVATION_DIM_HYBRID_KEYPOINT_JACOBIAN == 162
    assert thesis["reward"]["joint_precision_progress_scale"] == 2.0
    assert thesis["reward"]["joint_position_tolerance_m"] == .05
    assert thesis["reward"]["joint_orientation_tolerance_rad"] == .10
    assert thesis["reward"]["leave_tolerance_multiplier"] == 1.0
    assert thesis["reward"]["leave_joint_tolerance_penalty"] == .20
    assert thesis["reward"]["hold_reward_scale"] == .15
    assert thesis["reward"]["joint_precision_temperature"] == 2.0
    assert "orientation_sigma_rad" not in thesis["reward"]
    assert "nominal_controller" not in thesis
    assert thesis["orientation_curriculum"]["deterministic_probe_collision_ceiling"] == .05
    assert thesis["orientation_curriculum"]["deterministic_previous_collision_ceiling"] == .05
    assert thesis["self_collision"]["curriculum"]["probe_collision_ceiling"] == .01
    assert len(levels) == len(thesis["orientation_curriculum"]["levels"])
    settings = thesis["orientation_curriculum"]
    assert settings["replay_success_schedule"] == {
        "low_start_episodes": 5,
        "medium_start_episodes": 20,
        "full_start_episodes": 50,
        "low_fraction": .05,
        "medium_fraction": .15,
    }
    assert settings["replay_success_fraction"] == .20
    assert settings["replay_current_fraction"] == .80
    assert settings["replay_current_intermediate"] == .80
    assert settings["replay_current_recovery"] == .80
    assert thesis["reward"]["precision_stop_cost_weight"] == .10
    assert thesis["precision_boundary_replay"] == {
        "enabled": True,
        "fraction": .20,
        "position_boundary_low_ratio": .8,
        "position_boundary_high_ratio": 1.5,
    }
    assert settings["min_transitions_per_level"] == 100000
    assert settings["deterministic_probe_interval_transitions"] == 100000
    assert settings["manual_promotion"] is True
    assert settings["manual_probe_interval_transitions"] == 25000
    assert config["train"]["evaluate_during_training"] is False
    assert config["train"]["total_steps"] == 200000
    assert settings["deterministic_probe_episodes"] == 1000
    assert settings["deterministic_probe_position_bins"] == 10
    assert settings["deterministic_probe_orientation_bins"] == 10
    assert settings["anchor_probability"] == 0.0
    assert config["train"]["max_stage_steps"] >= len(levels) * 100000
    for previous, current in zip(levels, levels[1:]):
        assert current["target_distance_max_m"] == previous["target_distance_max_m"]
        assert current["target_orientation_max_rad"] == previous["target_orientation_max_rad"]
    assert [level["position_tolerance_m"] for level in levels] == [
        .10, .08, .064, .0512, .05, .025, .01,
    ]
    assert [level["orientation_tolerance_rad"] for level in levels] == [
        .30, .24, .192, .1536, .10, .10, .10,
    ]
    assert levels[-1]["target_distance_max_m"] == .50
    assert np.isclose(levels[-1]["target_orientation_max_rad"], np.pi, atol=1e-6)


def test_v13_5_precision_curriculum_keeps_restricted_range_and_tightens_only_precision():
    root = Path(__file__).resolve().parents[2]
    config = load_config(
        root / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml"
    )
    thesis = config["thesis"]
    goal = thesis["goal_curriculum"]
    orientation = thesis["orientation_curriculum"]
    levels = thesis["joint_pose_curriculum"]["levels"]
    curriculum = HomotopyCurriculum(
        "s0",
        goal_start_scale=goal["start_scale"],
        goal_end_scale=goal["end_scale"],
        goal_levels=goal["levels"],
        orientation_start_scale=orientation["start_scale"],
        orientation_end_scale=orientation["end_scale"],
        orientation_levels=orientation["levels"],
        orientation_tolerance_start=orientation["tolerance_start"],
        orientation_tolerance_end=orientation["tolerance_end"],
        joint_pose_levels=levels,
    )

    assert thesis["protocol"] == thesis_trainer.HYBRID_KEYPOINT_AUTO_CHAIN_PROTOCOL
    assert thesis["success_hold_steps"] == 5
    assert curriculum.precision_only_task_space
    assert curriculum.orientation_scale == 0.0
    assert curriculum.pose_contract() == {
        "goal_scale": 1.0,
        "orientation_scale": 0.0,
        "position_tolerance": .10,
        "orientation_tolerance": .30,
        "target_distance_min_m": .03,
        "target_distance_max_m": .50,
        "target_orientation_min_rad": .03,
        "target_orientation_max_rad": 3.141592,
    }
    bounds = {
        (
            level["target_distance_min_m"], level["target_distance_max_m"],
            level["target_orientation_min_rad"],
            level["target_orientation_max_rad"],
        )
        for level in levels
    }
    assert len(bounds) == 1
    assert [level["position_tolerance_m"] for level in levels] == [
        .10, .08, .064, .0512, .05, .025, .01,
    ]
    assert [level["orientation_tolerance_rad"] for level in levels] == [
        .30, .24, .192, .1536, .10, .10, .10,
    ]
    assert thesis["reward"]["use_episode_tolerance_for_joint_precision"] is True
    assert thesis["reward"]["partial_precision_reward_scale"] == .10
    assert orientation["cross_level_probe_enabled"] is True
    adaptive = thesis["joint_pose_curriculum"]["task_space_sampling"]["adaptive"]
    assert adaptive["enabled"] is False
    assert adaptive["beta"] == .60
    mixture = thesis["joint_pose_curriculum"]["task_space_sampling"][
        "fixed_orientation_mixture"
    ]
    assert mixture == {
        "enabled": True,
        "uniform_fraction": .50,
        "orientation_groups": [
            {"fraction": .25, "bins": [5, 6, 7, 8, 9]},
            {"fraction": .25, "bins": [7, 8, 9]},
        ],
    }


def test_fixed_orientation_mixture_keeps_full_coverage_and_weights_hard_half():
    config = {
        "thesis": {
            "joint_pose_curriculum": {
                "task_space_sampling": {
                    "position_bins": 2,
                    "orientation_bins": 4,
                    "fixed_orientation_mixture": {
                        "enabled": True,
                        "uniform_fraction": .50,
                        "orientation_groups": [
                            {"fraction": .25, "bins": [2, 3]},
                            {"fraction": .25, "bins": [3]},
                        ],
                    },
                }
            }
        }
    }
    probabilities = np.asarray(
        thesis_trainer.fixed_task_space_bin_probabilities(config)
    ).reshape(2, 4)
    np.testing.assert_allclose(probabilities[:, :2], .50 / 8)
    np.testing.assert_allclose(probabilities[:, 2], .50 / 8 + .25 / 4)
    np.testing.assert_allclose(
        probabilities[:, 3], .50 / 8 + .25 / 4 + .25 / 2,
    )
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities > 0.0)


def test_v13_5_cross_level_probe_enforces_previous_precision_gate(monkeypatch):
    levels = [
        {
            "goal_scale": 1.0,
            "target_distance_min_m": .03,
            "target_distance_max_m": .70,
            "target_orientation_min_rad": .03,
            "target_orientation_max_rad": 3.14,
            "position_tolerance_m": position_tolerance,
            "orientation_tolerance_rad": orientation_tolerance,
        }
        for position_tolerance, orientation_tolerance in (
            (.10, .30), (.08, .24), (.064, .192), (.0512, .1536), (.05, .10),
        )
    ]
    config = {
        "thesis": {
            "joint_pose_curriculum": {"levels": levels},
            "orientation_curriculum": {"levels": [0.0, .25, .50, .75, 1.0]},
        }
    }
    current_level = 2
    curriculum = SimpleNamespace(
        orientation=SimpleNamespace(level_index=current_level),
        pose_contract=lambda previous=False: {
            "goal_scale": 1.0,
            "orientation_scale": [0.0, .25, .50, .75, 1.0][
                current_level - int(previous)
            ],
            "position_tolerance": levels[current_level - int(previous)][
                "position_tolerance_m"
            ],
            "orientation_tolerance": levels[current_level - int(previous)][
                "orientation_tolerance_rad"
            ],
            "target_distance_min_m": .03,
            "target_distance_max_m": .70,
            "target_orientation_min_rad": .03,
            "target_orientation_max_rad": 3.14,
        },
    )
    settings = {
        "deterministic_probe_episodes": 100,
        "deterministic_probe_seed": 31001,
        "deterministic_probe_success_floor": .85,
        "deterministic_previous_success_floor": .80,
        "deterministic_probe_collision_ceiling": .05,
        "deterministic_previous_collision_ceiling": .05,
        "deterministic_confirmation_episodes": 100,
        "deterministic_confirmation_seed": 51001,
        "cross_level_probe_enabled": True,
    }

    def probe(*args, **kwargs):
        tolerance = float(kwargs["position_tolerance"])
        success = .70 if np.isclose(tolerance, .08) else .90
        return {
            "success_rate": success,
            "collision_rate": 0.0,
            "joint_limit_rate": 0.0,
        }

    monkeypatch.setattr(thesis_trainer, "run_deterministic_pose_probe", probe)
    result = thesis_trainer.run_pose_promotion_probe(
        object(), config, curriculum, settings,
    )

    assert set(result["cross_level"]) == {"P1", "P2", "P3", "P4", "P5"}
    assert result["current"]["success_rate"] == pytest.approx(.90)
    assert result["previous"] is result["cross_level"]["P2"]
    assert result["previous"]["success_rate"] == pytest.approx(.70)
    assert not result["retention_passed"]
    assert not result["passed"]


def test_adaptive_task_space_sampling_mixes_uniform_coverage_with_probe_scores():
    config = {
        "thesis": {
            "joint_pose_curriculum": {
                "task_space_sampling": {
                    "position_bins": 2,
                    "orientation_bins": 2,
                    "adaptive": {
                        "enabled": True,
                        "beta": .60,
                        "improvement_weight": .50,
                        "score_floor": .01,
                    },
                }
            }
        }
    }
    counters = {}
    rates = np.asarray([.90, .50, .20, .00])
    probabilities = thesis_trainer.update_adaptive_task_space_sampling(
        config, counters, {"current": {"bin_success_rates": rates.tolist()}},
    )

    scores = 1.0 - rates + .01
    expected = .40 / 4 + .60 * scores / scores.sum()
    np.testing.assert_allclose(probabilities, expected)
    assert np.isclose(sum(probabilities), 1.0)
    assert min(probabilities) >= .40 / 4
    assert counters["adaptive_sampling_updates"] == 1

    improved = np.asarray([.90, .70, .20, .00])
    second = thesis_trainer.update_adaptive_task_space_sampling(
        config, counters, {"current": {"bin_success_rates": improved.tolist()}},
    )
    second_scores = 1.0 - improved + .50 * np.maximum(improved - rates, 0.0) + .01
    second_expected = .40 / 4 + .60 * second_scores / second_scores.sum()
    np.testing.assert_allclose(second, second_expected)
    assert counters["adaptive_sampling_updates"] == 2


def test_actor_migration_warmup_is_relative_to_inherited_stage_total():
    counters = {
        "stage_total_step": 800_000,
        "actor_initialization": {
            "critic_warmup_transitions": 5_000,
            "warmup_start_stage_total_step": 800_000,
        },
    }
    assert not thesis_trainer.actor_updates_enabled(counters)
    counters["stage_total_step"] = 804_999
    assert not thesis_trainer.actor_updates_enabled(counters)
    counters["stage_total_step"] = 805_000
    assert thesis_trainer.actor_updates_enabled(counters)


def test_v13_5_previous_level_probe_blocks_confirmation_and_promotion(monkeypatch):
    curriculum = SimpleNamespace(
        orientation=SimpleNamespace(level_index=1),
        pose_contract=lambda previous=False: {"previous": previous},
    )
    settings = {
        "deterministic_probe_episodes": 50,
        "deterministic_probe_seed": 31001,
        "deterministic_probe_success_floor": .80,
        "deterministic_previous_success_floor": .95,
        "deterministic_probe_collision_ceiling": .02,
        "deterministic_previous_collision_ceiling": .02,
        "deterministic_confirmation_episodes": 200,
        "deterministic_confirmation_seed": 51001,
    }

    def metrics(success):
        return {
            "success_rate": success, "collision_rate": 0.0,
            "joint_limit_rate": 0.0,
        }

    calls = iter((metrics(.90), metrics(.90)))
    monkeypatch.setattr(
        thesis_trainer, "run_deterministic_pose_probe",
        lambda *args, **kwargs: next(calls),
    )
    failed = thesis_trainer.run_pose_promotion_probe(
        None, {}, curriculum, settings,
    )
    assert not failed["screen_passed"]
    assert failed["confirmation_current"] is None
    assert not failed["passed"]
    assert not failed["retention_passed"]

    calls = iter((metrics(.90), metrics(1.0), metrics(.85), metrics(.95)))
    monkeypatch.setattr(
        thesis_trainer, "run_deterministic_pose_probe",
        lambda *args, **kwargs: next(calls),
    )
    passed = thesis_trainer.run_pose_promotion_probe(
        None, {}, curriculum, settings,
    )
    assert passed["screen_passed"]
    assert passed["confirmation_passed"]
    assert passed["retention_passed"]
    assert passed["passed"]


def test_v13_5_pose_probe_allows_residual_collision_at_configured_ceiling(monkeypatch):
    curriculum = SimpleNamespace(
        orientation=SimpleNamespace(level_index=1),
        pose_contract=lambda previous=False: {"previous": previous},
    )
    settings = {
        "deterministic_probe_episodes": 50,
        "deterministic_probe_seed": 31001,
        "deterministic_probe_success_floor": .80,
        "deterministic_previous_success_floor": .95,
        "deterministic_probe_collision_ceiling": .02,
        "deterministic_previous_collision_ceiling": .02,
        "deterministic_confirmation_episodes": 200,
        "deterministic_confirmation_seed": 51001,
    }

    def metrics(success, collision):
        return {
            "success_rate": success, "collision_rate": collision,
            "joint_limit_rate": 0.0,
        }

    calls = iter((
        metrics(.98, .02), metrics(1.0, .0),
        metrics(.98, .02), metrics(.995, .005),
    ))
    monkeypatch.setattr(
        thesis_trainer, "run_deterministic_pose_probe",
        lambda *args, **kwargs: next(calls),
    )
    result = thesis_trainer.run_pose_promotion_probe(
        None, {}, curriculum, settings,
    )
    assert result["passed"]


def test_manual_checkpoint_probe_records_result_without_mutating_curriculum(monkeypatch):
    metrics = {
        "success_rate": .91, "collision_rate": .01, "joint_limit_rate": 0.0,
        "timeout_rate": .08, "mean_position_error_m": .009,
        "mean_orientation_error_rad": .025,
        "mean_minimum_self_distance_m": .12, "self_violation_rate": 0.0,
        "minimum_bin_success_rate": .82,
    }
    result = {
        "current": metrics, "previous": None, "screen_passed": True,
        "confirmation_current": None, "confirmation_previous": None,
        "confirmation_passed": True, "retention_passed": True,
        "recovery_required": False, "passed": True,
    }
    monkeypatch.setattr(
        thesis_trainer, "run_pose_promotion_probe",
        lambda *args, **kwargs: result,
    )
    orientation = SimpleNamespace(
        level_index=3, scale=.05, current_level_steps=100000,
    )
    curriculum = SimpleNamespace(
        orientation=orientation, s0_phase="joint_pose", lambda_self=0.0,
        precision_only_task_space=False, orientation_at_full_scale=False,
    )
    counters = {"global_step": 400000, "stage_step": 25000,
                "stage_total_step": 400000}
    settings = {
        "deterministic_probe_episodes": 1000,
        "deterministic_probe_seed": 31001,
        "deterministic_confirmation_episodes": 1000,
        "deterministic_confirmation_seed": 51001,
    }

    returned, row = thesis_trainer.run_manual_pose_checkpoint_probe(
        object(), {}, curriculum, settings, counters,
    )

    assert returned is result
    assert row["probe_track"] == "manual_checkpoint"
    assert row["orientation_level_index"] == 3
    assert counters["last_manual_pose_probe"]["metrics"] == metrics
    assert orientation.level_index == 3
    assert orientation.current_level_steps == 100000
    assert result["retention_passed"]


def test_v12_5_probe_separates_retention_warning_from_recovery(monkeypatch):
    curriculum = SimpleNamespace(
        orientation=SimpleNamespace(level_index=1),
        pose_contract=lambda previous=False: {"previous": previous},
    )
    settings = {
        "deterministic_probe_episodes": 50,
        "deterministic_probe_seed": 31001,
        "deterministic_probe_success_floor": .80,
        "deterministic_previous_success_floor": .95,
        "deterministic_probe_collision_ceiling": .05,
        "deterministic_previous_collision_ceiling": .05,
        "deterministic_confirmation_episodes": 200,
        "deterministic_confirmation_seed": 51001,
        "retention_recovery_success_floor": .90,
        "retention_recovery_collision_ceiling": .05,
    }

    def metrics(success, collision=0.0):
        return {
            "success_rate": success, "collision_rate": collision,
            "joint_limit_rate": 0.0,
        }

    calls = iter((metrics(.90), metrics(.94)))
    monkeypatch.setattr(
        thesis_trainer, "run_deterministic_pose_probe",
        lambda *args, **kwargs: next(calls),
    )
    warning = thesis_trainer.run_pose_promotion_probe(
        None, {}, curriculum, settings,
    )
    assert not warning["retention_passed"]
    assert not warning["recovery_required"]

    calls = iter((metrics(.90), metrics(.88)))
    severe = thesis_trainer.run_pose_promotion_probe(
        None, {}, curriculum, settings,
    )
    assert not severe["retention_passed"]
    assert severe["recovery_required"]


def test_v13_5_best_probe_prioritizes_gate_feasibility_over_raw_success():
    settings = {
        "deterministic_probe_success_floor": .80,
        "deterministic_previous_success_floor": .95,
        "deterministic_probe_collision_ceiling": .02,
        "deterministic_previous_collision_ceiling": .02,
    }

    def metrics(success, collision, *, joint_limit=0.0):
        return {
            "success_rate": success,
            "collision_rate": collision,
            "joint_limit_rate": joint_limit,
            "timeout_rate": 1.0 - success - collision - joint_limit,
            "mean_position_error_m": .05,
            "mean_orientation_error_rad": .25,
            "mean_minimum_self_distance_m": .01,
            "self_violation_rate": collision,
        }

    safe = {
        "current": metrics(.88, .0), "previous": metrics(.96, .0),
        "confirmation_current": None, "confirmation_previous": None,
        "passed": False,
    }
    unsafe = {
        "current": metrics(.90, .04), "previous": metrics(.96, .0),
        "confirmation_current": None, "confirmation_previous": None,
        "passed": False,
    }
    safe_score = thesis_trainer.s0_probe_score(
        "level_15_pose", safe, settings, self_collision_ceiling=.01,
    )
    unsafe_score = thesis_trainer.s0_probe_score(
        "level_15_pose", unsafe, settings, self_collision_ceiling=.01,
    )
    assert safe_score > unsafe_score

    confirmed = {
        "current": metrics(1.0, .0), "previous": metrics(1.0, .0),
        "confirmation_current": metrics(.81, .02),
        "confirmation_previous": metrics(.95, .02),
        "passed": True,
    }
    screen_only = {
        "current": metrics(.99, .0), "previous": metrics(1.0, .0),
        "confirmation_current": None, "confirmation_previous": None,
        "passed": False,
    }
    confirmed_score = thesis_trainer.s0_probe_score(
        "level_15_pose", confirmed, settings, self_collision_ceiling=.01,
    )
    screen_score = thesis_trainer.s0_probe_score(
        "level_15_pose", screen_only, settings, self_collision_ceiling=.01,
    )
    assert confirmed_score > screen_score
    assert confirmed_score[5] == .81
    assert confirmed_score[7] == -.02


def test_v13_5_self_safety_best_uses_stricter_final_collision_ceiling():
    settings = {
        "deterministic_probe_success_floor": .80,
        "deterministic_previous_success_floor": .95,
        "deterministic_probe_collision_ceiling": .02,
        "deterministic_previous_collision_ceiling": .02,
    }
    metrics = {
        "success_rate": .95, "collision_rate": .015, "joint_limit_rate": 0.0,
        "timeout_rate": .035, "mean_position_error_m": .04,
        "mean_orientation_error_rad": .15,
        "mean_minimum_self_distance_m": .012, "self_violation_rate": .02,
    }
    result = {
        "current": metrics, "previous": None,
        "confirmation_current": metrics, "confirmation_previous": None,
        "passed": True,
    }
    pose_score = thesis_trainer.s0_probe_score(
        "level_24_pose", result, settings, self_collision_ceiling=.01,
    )
    self_score = thesis_trainer.s0_probe_score(
        "level_24_self_full", result, settings, self_collision_ceiling=.01,
    )
    assert pose_score[0] == 1.0
    assert pose_score[2] == 1.0
    assert self_score[0] == 0.0
    assert self_score[2] == 0.0


def test_legacy_probe_scores_are_not_reused_by_serial_protocol():
    settings = {
        "deterministic_probe_success_floor": .80,
        "deterministic_previous_success_floor": .95,
        "deterministic_probe_collision_ceiling": .02,
        "deterministic_previous_collision_ceiling": .02,
    }
    metrics = {
        "success_rate": .88, "collision_rate": 0.0, "joint_limit_rate": 0.0,
        "timeout_rate": .12, "mean_position_error_m": .06,
        "mean_orientation_error_rad": .35,
        "mean_minimum_self_distance_m": .01, "self_violation_rate": 0.0,
    }
    result = {
        "current": metrics, "previous": {
            **metrics, "success_rate": .96,
        },
        "confirmation_current": None, "confirmation_previous": None,
        "passed": False,
    }
    score = thesis_trainer.s0_probe_score(
        "level_15_pose", result, settings, self_collision_ceiling=.01,
    )
    legacy_unsafe = {
        "score": (.90, -.04, -.06, -.058, -.332),
        "success_rate": .90, "path": "best_level_15_pose.pt",
    }
    assert not thesis_trainer.s0_probe_score_is_better(
        "level_15_pose", score, legacy_unsafe, settings,
        self_collision_ceiling=.01,
    )

    legacy_safe = {
        "score": (.90, 0.0, -.10, -.055, -.30),
        "success_rate": .90, "path": "best_level_15_pose.pt",
    }
    assert not thesis_trainer.s0_probe_score_is_better(
        "level_15_pose", score, legacy_safe, settings,
        self_collision_ceiling=.01,
    )


def test_s0_replay_reserves_anchor_current_and_historical_pose_samples():
    replay = HomotopyReplayBuffer(
        2, 1, "cpu", capacities={"none": 600, "static": 1, "dynamic": 1},
        seed=7, s0_anchor_fraction=.25, s0_current_fraction=.50,
        s0_anchor_capacity=200, s0_current_capacity=200,
        s0_history_capacity_per_level=100,
    )
    episode = 0
    for scale, count in ((0.0, 200), (.2, 125), (.5, 125), (.8, 150)):
        for _ in range(count):
            replay.add(
                "none", np.zeros(2), np.zeros(1), np.ones(2), False,
                _transition_info(orientation_scale=scale), episode, 0,
            )
            episode += 1
    replay.sample(
        "s0", {"static": .02, "dynamic": .02}, 256, orientation_scale=.8,
    )
    assert replay.last_orientation_sample_counts == {
        "anchor": 64, "current": 128, "historical": 64,
    }
    replay.sample(
        "s0", {"static": .02, "dynamic": .02}, 256, orientation_scale=.8,
        s0_anchor_fraction=.50, s0_current_fraction=.35,
    )
    assert replay.last_orientation_sample_counts == {
        "anchor": 128, "current": 90, "historical": 38,
    }
    assert replay.orientation_storage_counts() == {
        "anchor": 200,
        "current": 150,
        "current_scale": .8,
        "historical": 200,
        "historical_levels": {"0.2": 100, "0.5": 100},
    }


def test_s0_replay_history_survives_many_current_level_overwrites_and_restore():
    replay = HomotopyReplayBuffer(
        2, 1, "cpu", capacities={"none": 20, "static": 1, "dynamic": 1},
        seed=17, s0_anchor_capacity=10, s0_current_capacity=20,
        s0_history_capacity_per_level=6,
    )
    episode = 0
    for scale, count in ((0.0, 30), (.2, 15), (.5, 15), (.75, 200)):
        for _ in range(count):
            replay.add(
                "none", np.zeros(2), np.zeros(1), np.ones(2), False,
                _transition_info(orientation_scale=scale), episode, 0,
                orientation_anchor=scale == 0.0,
            )
            episode += 1
    assert replay.orientation_storage_counts() == {
        "anchor": 10,
        "current": 20,
        "current_scale": .75,
        "historical": 12,
        "historical_levels": {"0.2": 6, "0.5": 6},
    }
    replay.sample(
        "s0", {"static": .02, "dynamic": .02}, 16,
        orientation_scale=.75, s0_anchor_fraction=.25,
        s0_current_fraction=.50,
    )
    assert replay.last_orientation_sample_counts == {
        "anchor": 4, "current": 8, "historical": 4,
    }
    restored = HomotopyReplayBuffer(
        2, 1, "cpu", capacities={"none": 1, "static": 1, "dynamic": 1},
        s0_anchor_capacity=1, s0_current_capacity=1,
        s0_history_capacity_per_level=1,
    )
    restored.load_state_dict(replay.state_dict())
    assert restored.orientation_storage_counts() == replay.orientation_storage_counts()
    assert restored.partition_sha256("none") == replay.partition_sha256("none")


def test_s0_replay_level_advance_before_anchor_episode_archives_current_level():
    replay = HomotopyReplayBuffer(
        2, 1, "cpu", capacities={"none": 20, "static": 1, "dynamic": 1},
        seed=23, s0_anchor_capacity=20, s0_current_capacity=20,
        s0_history_capacity_per_level=6,
    )
    for episode in range(16):
        replay.add(
            "none", np.zeros(2), np.zeros(1), np.ones(2), False,
            _transition_info(orientation_scale=0.3), episode, 0,
        )
    for episode in range(16, 36):
        replay.add(
            "none", np.zeros(2), np.zeros(1), np.ones(2), False,
            _transition_info(orientation_scale=0.0), episode, 0,
            orientation_anchor=True,
        )

    replay.begin_orientation_level(0.4)
    assert replay.orientation_storage_counts() == {
        "anchor": 20,
        "current": 0,
        "current_scale": 0.4,
        "historical": 6,
        "historical_levels": {"0.3": 6},
    }

    replay.sample(
        "s0", {"static": .02, "dynamic": .02}, 16,
        orientation_scale=.4, s0_anchor_fraction=.25,
        s0_current_fraction=.50,
    )
    assert replay.last_orientation_sample_counts == {
        "anchor": 10, "current": 0, "historical": 6,
    }


def test_v12_3_replay_reserves_success_recent_and_previous_samples():
    replay = HomotopyReplayBuffer(
        2, 1, "cpu", capacities={"none": 400, "static": 1, "dynamic": 1},
        seed=29, s0_joint_pose=True, s0_success_fraction=.30,
        s0_anchor_fraction=.35, s0_current_fraction=.35,
        s0_current_capacity=300, s0_success_capacity=100,
        s0_history_capacity_per_level=100,
    )
    episode = 0
    for _ in range(40):
        replay.add(
            "none", np.zeros(2), np.zeros(1), np.ones(2), True,
            _transition_info(reached=True, orientation_scale=0.0), episode, 0,
        )
        episode += 1
    for reached, count in ((True, 40), (False, 80)):
        for _ in range(count):
            replay.add(
                "none", np.zeros(2), np.zeros(1), np.ones(2), True,
                _transition_info(reached=reached, orientation_scale=.2), episode, 0,
            )
            episode += 1
    for _ in range(10):
        replay.add(
            "none", np.zeros(2), np.zeros(1), np.ones(2), True,
            _transition_info(reached=True, orientation_scale=0.0), episode, 0,
            orientation_anchor=True,
        )
        episode += 1
    assert replay.orientation_storage_counts() == {
        "anchor": 10, "current": 120, "current_scale": .2,
        "historical": 40, "historical_levels": {"0.0": 40},
        "current_success": 40,
    }
    batch = replay.sample(
        "s0", {"static": .02, "dynamic": .02}, 100, orientation_scale=.2,
    )
    assert replay.last_orientation_sample_counts == {
        "anchor": 30, "current": 35, "historical": 35,
        "current_success": 30, "current_recent": 35, "previous": 35,
    }
    assert 30 < int(batch.protected_mask.sum()) <= 65
    batch = replay.sample(
        "s0", {"static": .02, "dynamic": .02}, 100,
        orientation_scale=.2, s0_anchor_fraction=.50,
        s0_current_fraction=.20,
    )
    assert replay.last_orientation_sample_counts == {
        "anchor": 30, "current": 20, "historical": 50,
        "current_success": 30, "current_recent": 20, "previous": 50,
    }
    assert 30 < int(batch.protected_mask.sum()) <= 80
    rows = replay._sample_joint_pose_rows(100, .2)
    successful = set(
        int(value) for value in np.unique(
            replay.s0_current_success.episode[
                replay.s0_current_success.chronological_indices()
            ]
        )
    )
    recent_episodes = {
        int(part.episode[index])
        for part, index, category in rows if category == "current_recent"
    }
    assert recent_episodes.isdisjoint(successful)

    part = replay.s0_current
    rows = replay._sample_episode_balanced_rows(part, 100, "current_recent")
    sampled_episodes = [int(row[0].episode[row[1]]) for row in rows]
    assert len(set(sampled_episodes)) == 100


@pytest.mark.parametrize(
    ("successful_episodes", "expected_fraction", "expected_samples"),
    [(4, 0.0, 0), (5, 0.05, 5), (20, 0.15, 15), (50, 0.30, 30)],
)
def test_v13_2_success_replay_warms_up_by_independent_episode_count(
    successful_episodes, expected_fraction, expected_samples,
):
    replay = HomotopyReplayBuffer(
        2, 1, "cpu", capacities={"none": 400, "static": 1, "dynamic": 1},
        seed=37, s0_joint_pose=True, s0_success_fraction=.30,
        s0_success_schedule={
            "low_start_episodes": 5,
            "medium_start_episodes": 20,
            "full_start_episodes": 50,
            "low_fraction": .05,
            "medium_fraction": .15,
        },
        s0_anchor_fraction=0.0, s0_current_fraction=.70,
        s0_current_capacity=400, s0_success_capacity=100,
    )
    episode = 0
    for _ in range(120):
        replay.add(
            "none", np.zeros(2), np.zeros(1), np.ones(2), True,
            _transition_info(reached=False, orientation_scale=.2), episode, 0,
        )
        episode += 1
    for _ in range(successful_episodes):
        replay.add(
            "none", np.zeros(2), np.zeros(1), np.ones(2), True,
            _transition_info(reached=True, orientation_scale=.2), episode, 0,
        )
        episode += 1

    assert replay.current_success_episode_count == successful_episodes
    assert replay.effective_s0_success_fraction == expected_fraction
    replay.sample(
        "s0", {"static": .02, "dynamic": .02}, 100,
        orientation_scale=.2,
    )
    assert replay.last_orientation_sample_counts["current_success"] == expected_samples
    assert replay.last_orientation_sample_counts["current_recent"] == 100 - expected_samples


def test_precision_boundary_replay_balances_live_error_strata():
    replay = HomotopyReplayBuffer(
        2, 1, "cpu", capacities={"none": 400, "static": 1, "dynamic": 1},
        seed=39, s0_joint_pose=True, s0_success_fraction=0.0,
        s0_anchor_fraction=0.0, s0_current_fraction=1.0,
        s0_current_capacity=400, s0_success_capacity=10,
        precision_replay_fraction=.40,
        precision_position_boundary_low=.8,
        precision_position_boundary_high=1.5,
    )
    episode = 0

    def add_transition(rho_p, next_rho_p, rho_r, next_rho_r):
        nonlocal episode
        info = _transition_info(orientation_scale=.2)
        info.update({
            "rho_position": rho_p,
            "next_rho_position": next_rho_p,
            "rho_orientation": rho_r,
            "next_rho_orientation": next_rho_r,
            "position_tolerance": .01,
            "orientation_tolerance": .10,
        })
        replay.add(
            "none", np.full(2, episode), np.zeros(1),
            np.full(2, episode + 1), False, info, episode, 0,
        )
        episode += 1

    for _ in range(20):
        add_transition(.009, .011, .09, .09)   # strict exit
        add_transition(.012, .009, .11, .09)   # strict entry/hold
        add_transition(.020, .009, .20, .20)   # position only inside
        add_transition(.020, .020, .11, .09)   # orientation only inside
        add_transition(.020, .012, .20, .20)   # position boundary
    for _ in range(60):
        add_transition(.30, .29, .50, .49)     # ordinary recent

    rows = replay._sample_joint_pose_rows(100, .2)
    assert replay.last_orientation_sample_counts["precision_boundary"] == 40
    assert replay.last_orientation_sample_counts["current_recent"] == 60
    assert replay.last_orientation_sample_counts["current"] == 100
    assert replay.last_precision_sample_counts == {
        "strict_exit": 8,
        "inside_hold": 8,
        "position_only_inside": 8,
        "orientation_only_inside": 8,
        "position_boundary": 8,
    }
    current_rows = [index for part, index, _ in rows if part is replay.s0_current]
    assert len(current_rows) == len(set(current_rows)) == 100

    restored = HomotopyReplayBuffer(
        2, 1, "cpu", capacities={"none": 1, "static": 1, "dynamic": 1},
        seed=40, s0_joint_pose=True,
    )
    restored.load_state_dict(replay.state_dict())
    assert restored.precision_replay_fraction == .40
    assert restored.precision_position_boundary_low == .8
    assert restored.precision_position_boundary_high == 1.5
    assert restored.last_precision_sample_counts == replay.last_precision_sample_counts


def test_semantic_long_term_replay_balances_bins_and_survives_restore():
    replay = HomotopyReplayBuffer(
        2, 1, "cpu", capacities={"none": 200, "static": 1, "dynamic": 1},
        seed=41, s0_joint_pose=True, s0_success_fraction=0.0,
        s0_anchor_fraction=0.0, s0_current_fraction=1.0,
        s0_current_capacity=200, s0_success_capacity=10,
    )
    settings = dict(
        enabled=True, fraction=.50, position_bins=2, orientation_bins=2,
        episodes_per_bin=2, transitions_per_episode=2,
        distance_min=.10, distance_max=.30,
        orientation_min=.20, orientation_max=.60, seed=1234,
    )
    replay.configure_semantic_long_term(**settings)
    episode = 0
    for distance in (.15, .25):
        for orientation in (.30, .50):
            for _ in range(3):
                for step in range(3):
                    info = _transition_info(orientation_scale=.5)
                    info.update({
                        "sampled_target_distance_m": distance,
                        "sampled_target_orientation_rad": orientation,
                    })
                    replay.add(
                        "none", np.full(2, episode), np.zeros(1),
                        np.full(2, episode + 1), step == 2, info, episode, step,
                    )
                episode += 1

    storage = replay.orientation_storage_counts()
    assert storage["semantic_long_term_nonempty_bins"] == 4
    assert storage["semantic_long_term_episodes"] == 8
    assert storage["semantic_long_term_transitions"] == 16
    replay.sample(
        "s0", {"static": .02, "dynamic": .02}, 16,
        orientation_scale=.5,
    )
    assert replay.last_orientation_sample_counts["semantic_long_term"] == 8
    assert replay.last_orientation_sample_counts["current_recent"] == 8

    restored = HomotopyReplayBuffer(
        2, 1, "cpu", capacities={"none": 1, "static": 1, "dynamic": 1},
        s0_joint_pose=True, s0_success_fraction=0.0,
        s0_anchor_fraction=0.0, s0_current_fraction=1.0,
        s0_current_capacity=1, s0_success_capacity=1,
    )
    restored.load_state_dict(replay.state_dict())
    restored.configure_semantic_long_term(**settings)
    assert restored.orientation_storage_counts() == replay.orientation_storage_counts()
    restored.sample(
        "s0", {"static": .02, "dynamic": .02}, 16,
        orientation_scale=.5,
    )
    assert restored.last_orientation_sample_counts["semantic_long_term"] == 8

    promoted = HomotopyReplayBuffer(
        2, 1, "cpu", capacities={"none": 1, "static": 1, "dynamic": 1},
        s0_joint_pose=True, s0_success_fraction=0.0,
        s0_anchor_fraction=0.0, s0_current_fraction=1.0,
        s0_current_capacity=1, s0_success_capacity=1,
    )
    promoted.load_state_dict(replay.state_dict())
    promoted.begin_orientation_level(.75)
    promoted.configure_semantic_long_term(
        **{
            **settings,
            "distance_min": .20,
            "distance_max": .50,
            "orientation_min": .40,
            "orientation_max": 1.20,
        },
        reset_on_contract_change=True,
    )
    promoted_storage = promoted.orientation_storage_counts()
    assert promoted_storage["semantic_long_term_episodes"] == 0
    assert promoted_storage["semantic_long_term_seen_episodes"] == 0


def test_semantic_long_term_replay_survives_precision_only_level_change():
    replay = HomotopyReplayBuffer(
        2, 1, "cpu", capacities={"none": 20, "static": 1, "dynamic": 1},
        seed=59, s0_joint_pose=True, s0_success_fraction=0.0,
        s0_anchor_fraction=0.0, s0_current_fraction=1.0,
        s0_current_capacity=20, s0_success_capacity=4,
    )
    replay.configure_semantic_long_term(
        enabled=True, fraction=.20, position_bins=2, orientation_bins=2,
        episodes_per_bin=2, transitions_per_episode=2,
        distance_min=.03, distance_max=.70,
        orientation_min=.03, orientation_max=3.14, seed=991,
    )
    for step in range(3):
        info = _transition_info(orientation_scale=0.0)
        info.update({
            "sampled_target_distance_m": .20,
            "sampled_target_orientation_rad": .80,
        })
        replay.add(
            "none", np.full(2, step), np.zeros(1), np.full(2, step + 1),
            step == 2, info, 7, step,
        )
    before = replay.orientation_storage_counts()

    replay.begin_orientation_level(.25)
    replay.update_semantic_bounds(
        distance_min=.03, distance_max=.70,
        orientation_min=.03, orientation_max=3.14,
    )
    after = replay.orientation_storage_counts()

    assert before["semantic_long_term_episodes"] == 1
    assert after["semantic_long_term_episodes"] == 1
    assert after["semantic_long_term_transitions"] == before[
        "semantic_long_term_transitions"
    ]
    assert after["semantic_long_term_seen_episodes"] == before[
        "semantic_long_term_seen_episodes"
    ]


def test_semantic_long_term_replay_keeps_lowest_final_pose_error():
    replay = HomotopyReplayBuffer(
        2, 1, "cpu", capacities={"none": 20, "static": 1, "dynamic": 1},
        seed=73, s0_joint_pose=True, s0_success_fraction=0.0,
        s0_anchor_fraction=0.0, s0_current_fraction=1.0,
        s0_current_capacity=20, s0_success_capacity=4,
    )
    replay.configure_semantic_long_term(
        enabled=True, fraction=.20, position_bins=1, orientation_bins=1,
        episodes_per_bin=2, transitions_per_episode=1,
        distance_min=.03, distance_max=.70,
        orientation_min=.03, orientation_max=3.14, seed=991,
    )
    for episode, final_position_error in enumerate((.50, .05, .25)):
        info = _transition_info(orientation_scale=.5)
        info.update({
            "rho_position": final_position_error + .01,
            "next_rho_position": final_position_error,
            "rho_orientation": final_position_error + .01,
            "next_rho_orientation": final_position_error,
            "sampled_target_distance_m": .20,
            "sampled_target_orientation_rad": .80,
        })
        replay.add(
            "none", np.zeros(2), np.zeros(1), np.ones(2), True,
            info, episode, 0,
        )
    slot_qualities = sorted(
        replay._snapshot_quality(snapshot)
        for snapshot in replay.semantic_slots[0]
    )
    np.testing.assert_allclose(slot_qualities, [1.0, 5.0], atol=1e-6)


def test_replay_preserves_self_geometry_and_relabels_current_self_weight():
    replay = HomotopyReplayBuffer(
        2, 1, "cpu", capacities={"none": 1, "static": 1, "dynamic": 1}, seed=11,
    )
    info = _transition_info()
    info.update({
        "control_self_max_risk": .6,
        "control_self_min_distance": .002,
        "self_clearance_violation": .6,
        "control_self_max_approach": .3,
        "control_self_min_ttc": .01,
    })
    replay.add("none", np.zeros(2), np.zeros(1), np.ones(2), False, info, 1, 0)
    low = replay.sample(
        "s0", {"static": .02, "dynamic": .02}, batch_size=1, lambda_self=.02,
    )
    strict = replay.sample(
        "s0", {"static": .02, "dynamic": .02}, batch_size=1, lambda_self=1.0,
    )
    assert replay.s0_current.raw.shape[1] == 29
    np.testing.assert_allclose(replay.s0_current.raw[0, 17:22], [.6, .002, .6, .3, .01])
    assert strict.rewards.item() < low.rewards.item()


@pytest.mark.parametrize("parameters", [
    {"keypoint_precision_reward_scale": .20},
    {"keypoint_precision_reward_scale": .20, "hold_reward_scale": .05,
     "leave_joint_tolerance_penalty": .20},
    {"keypoint_tracking_scale": .35, "keypoint_progress_scale": 7.0,
     "keypoint_precision_reward_scale": .10, "joint_precision_temperature": .75},
])
def test_vectorized_replay_reward_matches_scalar_contract(parameters):
    rng = np.random.default_rng(20260927)
    count = 128
    raw = rng.uniform(0.0, 1.0, size=(count, 29)).astype(np.float32)
    raw[:, 5:11] = rng.integers(0, 2, size=(count, 6))
    raw[:, 11] = rng.uniform(-0.2, 1.2, size=count)
    raw[:, 12] = rng.uniform(-0.02, 0.8, size=count)
    raw[:, 15] = rng.uniform(-0.2, 1.2, size=count)
    raw[:, 17] = rng.uniform(-0.2, 1.2, size=count)
    raw[:, 18] = rng.uniform(-0.02, 0.25, size=count)
    raw[:, 22] = rng.integers(0, 2, size=count)
    raw[:, 24] = rng.choice([.05, .064, .08, .10], size=count)
    raw[:, 25] = rng.choice([.10, .1536, .192, .24, .30], size=count)
    dones = rng.integers(0, 2, size=(count, 1)).astype(np.float32)
    xi = rng.uniform(0.02, 1.0, size=count)
    rewards, proximity = _vectorized_replay_rewards(
        raw, dones, xi, .37, .99, parameters,
    )
    expected_rewards = []
    expected_proximity = []
    for index, row in enumerate(raw):
        reward, fields = homotopy_reward(
            rho_position=row[0], next_rho_position=row[1],
            rho_orientation=row[2], next_rho_orientation=row[3],
            smooth_velocity=row[4], task_reached=bool(row[5]),
            hard_failure=bool(row[6]), timeout=bool(row[22]),
            velocity_magnitude=row[15], orientation_scale=row[16],
            obstacle_collision=bool(row[7]),
            terminal_obstacle_collision=bool(
                row[7] and dones[index, 0] and not row[5] and not row[6]
            ),
            risk_max=row[11], distance_min=row[12],
            self_risk_max=row[17], self_distance_min=row[18],
            lambda_self=.37, xi=xi[index], gamma=.99,
            joint_position_tolerance=row[24],
            joint_orientation_tolerance=row[25],
            orientation_tolerance=row[25],
            keypoint_distance=row[26], next_keypoint_distance=row[27],
            keypoint_tracking_quality=row[28],
            **parameters,
        )
        expected_rewards.append(reward)
        expected_proximity.append(fields["c_proximity"])
    # NumPy evaluates array expressions in a slightly different order than
    # the scalar path; differences are limited to a float32 rounding unit.
    np.testing.assert_allclose(
        rewards.astype(np.float32), np.asarray(expected_rewards, np.float32),
        rtol=1e-6, atol=1e-7,
    )
    np.testing.assert_allclose(
        proximity.astype(np.float32), np.asarray(expected_proximity, np.float32),
        rtol=1e-6, atol=1e-7,
    )


def test_replay_preserves_and_recomputes_timeout_penalty():
    replay = HomotopyReplayBuffer(
        2, 1, "cpu", capacities={"none": 1, "static": 1, "dynamic": 1}, seed=31,
    )
    info = _transition_info(orientation_scale=0.0)
    info["timeout"] = True
    replay.add("none", np.zeros(2), np.zeros(1), np.ones(2), True, info, 1, 239)
    batch = replay.sample(
        "s0", {"static": .02, "dynamic": .02}, batch_size=1,
        orientation_scale=0.0,
    )
    expected, fields = homotopy_reward(
        rho_position=.4, next_rho_position=.3,
        rho_orientation=.2, next_rho_orientation=.1,
        smooth_velocity=.5, velocity_magnitude=.25,
        orientation_scale=0.0, task_reached=False, hard_failure=False,
        timeout=True, obstacle_collision=False, risk_max=.6,
        distance_min=.1, xi=1.0,
    )
    assert replay.s0_anchor.raw[0, 22] == 1.0
    assert fields["timeout_penalty"] == 2.0
    assert np.isclose(batch.rewards.item(), expected)


def test_replay_reward_relabel_uses_each_transition_curriculum_tolerances():
    reward_parameters = {
        "keypoint_precision_reward_scale": .20,
        "joint_position_tolerance": .05,
        "joint_orientation_tolerance": .10,
    }
    replay = HomotopyReplayBuffer(
        2, 1, "cpu", capacities={"none": 2, "static": 1, "dynamic": 1},
        seed=47, s0_joint_pose=True, s0_anchor_fraction=0.0,
        s0_current_fraction=1.0, s0_success_fraction=0.0,
        reward_parameters=reward_parameters,
    )
    info = _transition_info(orientation_scale=.50)
    info.update({
        "rho_position": .11,
        "next_rho_position": .09,
        "rho_orientation": .31,
        "next_rho_orientation": .21,
        "position_tolerance": .10,
        "orientation_tolerance": .30,
    })
    replay.add(
        "none", np.zeros(2), np.zeros(1), np.ones(2), False,
        info, 1, 0, curriculum_level=2,
    )

    batch = replay.sample(
        "s0", {"static": .02, "dynamic": .02}, batch_size=1,
        orientation_scale=.50, curriculum_level=2,
    )
    expected, _ = homotopy_reward(
        rho_position=.11, next_rho_position=.09,
        rho_orientation=.31, next_rho_orientation=.21,
        smooth_velocity=.5, velocity_magnitude=.25,
        orientation_scale=.50, task_reached=False, hard_failure=False,
        obstacle_collision=False, risk_max=.6, distance_min=.1, xi=1.0,
        **{
            **reward_parameters,
            "joint_position_tolerance": .10,
            "joint_orientation_tolerance": .30,
            "orientation_tolerance": .30,
        },
    )
    fixed_p5, _ = homotopy_reward(
        rho_position=.11, next_rho_position=.09,
        rho_orientation=.31, next_rho_orientation=.21,
        smooth_velocity=.5, velocity_magnitude=.25,
        orientation_scale=.50, task_reached=False, hard_failure=False,
        obstacle_collision=False, risk_max=.6, distance_min=.1, xi=1.0,
        orientation_tolerance=.10, **reward_parameters,
    )

    np.testing.assert_allclose(replay.s0_current.raw[0, 23:26], [2.0, .10, .30])
    assert batch.rewards.item() == pytest.approx(expected)
    assert not np.isclose(batch.rewards.item(), fixed_p5)


def test_self_safety_ramp_starts_after_pose_phase_and_restores_from_state():
    kwargs = dict(
        goal_levels=[.03, 1.0], goal_min_transitions_per_level=1,
        goal_success_window=1, goal_success_floor=1.0,
        goal_full_scale_min_steps=1,
        orientation_levels=[0.0, 1.0], orientation_min_transitions_per_level=1,
        orientation_success_window=1, orientation_success_floor=1.0,
        orientation_anchor_window=1, orientation_anchor_floor=1.0,
        orientation_full_scale_min_steps=1,
        self_start_weight=0.0, self_end_weight=1.0, self_ramp_steps=10,
        self_full_weight_min_steps=1,
    )
    curriculum = HomotopyCurriculum("s0", **kwargs)
    curriculum.record_transition(.03, 0.0)
    curriculum.finish_episode("none", True, True, 1)
    curriculum.record_transition(1.0, 0.0)
    curriculum.finish_episode("none", True, True, 1)
    assert curriculum.position_phase_complete
    assert not curriculum.pose_phase_complete
    assert curriculum.lambda_self == 0.0
    curriculum.record_transition(1.0, 0.0)
    curriculum.finish_episode("none", True, True, 1)
    assert curriculum.orientation_at_full_scale
    curriculum.record_transition(1.0, 1.0)
    curriculum.finish_episode("none", True, True, 1)
    curriculum.finish_episode("none", True, True, 1, orientation_anchor=True)
    assert curriculum.pose_phase_complete
    for _ in range(5):
        curriculum.record_transition(1.0, 1.0)
    curriculum.finish_episode("none", True, True, 5)
    assert np.isclose(curriculum.lambda_self, .5)
    restored = HomotopyCurriculum("s0", **kwargs)
    restored.load_state_dict(curriculum.state_dict())
    assert restored.self_safety == curriculum.self_safety
    for _ in range(5):
        restored.record_transition(1.0, 1.0)
    restored.finish_episode("none", True, True, 5)
    assert restored.lambda_self == 1.0
    assert HomotopyCurriculum("s1", **kwargs).lambda_self == 1.0


def test_self_safety_completion_uses_nonzero_collision_ceiling():
    curriculum = HomotopyCurriculum(
        "s0", orientation_deterministic_probe_required=True,
        self_full_weight_min_steps=25,
        self_probe_collision_ceiling=.01,
    )
    curriculum.self_safety.weight = curriculum.self_safety.end
    curriculum.self_safety.full_weight_steps = 25
    curriculum.orientation.deterministic_probe_passed = True
    curriculum.orientation.deterministic_probe_collision_rate = .02
    assert not curriculum.self_safety_phase_complete
    curriculum.orientation.deterministic_probe_collision_rate = .01
    assert curriculum.self_safety_phase_complete


def test_parallel_episode_uses_safety_phase_state_captured_at_reset():
    levels = [
        {"goal_scale": .03, "position_tolerance_m": .08,
         "orientation_tolerance_rad": .5},
        {"goal_scale": 1.0, "position_tolerance_m": .055,
         "orientation_tolerance_rad": .1},
    ]
    curriculum = HomotopyCurriculum(
        "s0", goal_levels=[.03, 1.0], orientation_levels=[0.0, 1.0],
        joint_pose_levels=levels, orientation_success_window=1,
        orientation_anchor_window=1, orientation_full_scale_min_steps=1,
        self_ramp_steps=10, self_full_weight_min_steps=1,
    )
    curriculum.orientation.level_index = 1
    curriculum.orientation.scale = 1.0
    curriculum.goal.level_index = 1
    curriculum.goal.scale = 1.0
    curriculum.goal.full_scale_steps = 10
    curriculum.orientation.full_scale_steps = 10
    curriculum.orientation.outcomes.append(True)
    curriculum.orientation.anchor_outcomes.append(True)
    assert curriculum.pose_phase_complete

    curriculum.finish_episode(
        "none", True, True, 5,
        pose_phase_complete_at_start=False,
        self_weight_full_at_start=False,
    )
    assert curriculum.self_safety.eligible_steps == 0
    curriculum.finish_episode(
        "none", True, True, 10,
        pose_phase_complete_at_start=True,
        self_weight_full_at_start=False,
    )
    assert curriculum.lambda_self == 1.0
    curriculum.finish_episode(
        "none", True, True, 5,
        pose_phase_complete_at_start=True,
        self_weight_full_at_start=False,
    )
    assert curriculum.self_safety.full_weight_steps == 0
    curriculum.finish_episode(
        "none", True, True, 5,
        pose_phase_complete_at_start=True,
        self_weight_full_at_start=True,
    )
    assert curriculum.self_safety.full_weight_steps == 5


def test_stage_transition_inherits_task_and_completed_static_state_by_copy():
    source = HomotopyCurriculum("s0")
    source.goal.scale = source.goal.end
    source.orientation.scale = source.orientation.end
    source.self_safety.weight = source.self_safety.end
    source.states["static"].strict = True
    source.states["static"].replay_strictified = True
    source.states["static"].strict_steps = 25000

    s1 = HomotopyCurriculum("s1")
    s1.inherit_task_state(source.state_dict())
    assert s1.goal.scale == 1.0
    assert s1.orientation.scale == 1.0
    assert s1.lambda_self == 1.0
    assert not s1.states["static"].strict
    source.goal.scale = .03
    assert s1.goal.scale == 1.0

    s2 = HomotopyCurriculum("s2")
    s2.inherit_task_state(s1.state_dict())
    s2.inherit_scene_state(source.state_dict(), "static")
    assert s2.states["static"].strict
    assert s2.states["static"].replay_strictified
    assert s2.states["static"].strict_steps == 25000


def test_self_safety_ramp_freezes_when_full_pose_retention_drops():
    curriculum = HomotopyCurriculum(
        "s0", goal_levels=[.03, 1.0], goal_min_transitions_per_level=1,
        goal_success_window=1, goal_success_floor=1.0,
        goal_full_scale_min_steps=1,
        orientation_levels=[0.0, 1.0], orientation_min_transitions_per_level=1,
        orientation_success_window=2, orientation_success_floor=.5,
        orientation_anchor_window=1, orientation_anchor_floor=1.0,
        orientation_full_scale_min_steps=1,
        self_start_weight=0.0, self_end_weight=1.0, self_ramp_steps=10,
        self_full_weight_min_steps=2,
    )
    curriculum.record_transition(.03, 0.0)
    curriculum.finish_episode("none", True, True, 1)
    curriculum.record_transition(1.0, 0.0)
    curriculum.finish_episode("none", True, True, 1)
    for _ in range(2):
        curriculum.record_transition(1.0, 0.0)
        curriculum.finish_episode("none", True, True, 1)
    assert curriculum.orientation_at_full_scale
    for _ in range(2):
        curriculum.record_transition(1.0, 1.0)
        curriculum.finish_episode("none", True, True, 1)
    curriculum.finish_episode("none", True, True, 1, orientation_anchor=True)
    assert curriculum.pose_phase_complete

    for _ in range(5):
        curriculum.record_transition(1.0, 1.0)
    curriculum.finish_episode("none", True, True, 5)
    assert curriculum.lambda_self == .5

    for _ in range(2):
        curriculum.record_transition(1.0, 1.0)
        curriculum.finish_episode("none", True, False, 1)
    assert not curriculum.pose_phase_complete
    assert curriculum.lambda_self == .7
    frozen_steps = curriculum.self_safety.eligible_steps
    curriculum.record_transition(1.0, 1.0)
    curriculum.finish_episode("none", True, False, 1)
    assert curriculum.self_safety.eligible_steps == frozen_steps
    assert curriculum.lambda_self == .7
    assert curriculum.s0_phase == "self_safety_ramp"


def test_archived_pose_levels_are_sampled_round_robin():
    replay = HomotopyReplayBuffer(
        2, 1, "cpu", capacities={"none": 600, "static": 1, "dynamic": 1},
        seed=19, s0_anchor_capacity=100, s0_current_capacity=120,
        s0_history_capacity_per_level=20,
    )
    episode = 0
    for scale in (0.0, .1, .2, .3, .4, .5):
        for _ in range(30):
            replay.add(
                "none", np.zeros(2), np.zeros(1), np.ones(2), False,
                _transition_info(orientation_scale=scale), episode, 0,
                orientation_anchor=scale == 0.0,
            )
            episode += 1
    rows = replay._stratified_historical_rows(20)
    sampled_scales = [round(float(part.raw[index, 16]), 6) for part, index, _ in rows]
    counts = {scale: sampled_scales.count(scale) for scale in (.1, .2, .3, .4)}
    assert counts == {.1: 5, .2: 5, .3: 5, .4: 5}


def test_filtered_self_pair_query_matches_whole_body_query():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    env = ThesisHomotopyEnv(config)
    try:
        env.configure_episode(
            "none", xi=1.0, strict=True, goal_scale=1.0,
            orientation_scale=0.0, position_tolerance=.10,
            orientation_tolerance=.30,
            target_distance_min_m=.03, target_distance_max_m=.70,
            target_orientation_min_rad=.03,
            target_orientation_max_rad=np.pi,
        )
        observation, _ = env.reset(seed=31001)
        del observation
        pair_lookup = {
            pair: index for index, pair in enumerate(env._self_pair_link_ids)
        }

        def whole_body_reference():
            distances = np.full(
                len(env._self_pair_link_ids), env.self_query_distance,
                dtype=np.float32,
            )
            approaches = np.zeros(len(env._self_pair_link_ids), dtype=np.float32)
            points = p.getClosestPoints(
                bodyA=env.robot_id, bodyB=env.robot_id,
                distance=env.self_query_distance,
                physicsClientId=env.client_id,
            )
            for point in points:
                first, second = int(point[3]), int(point[4])
                if first == second:
                    continue
                if first < second:
                    pair = (first, second)
                    point_first = np.asarray(point[5], dtype=np.float32)
                    point_second = np.asarray(point[6], dtype=np.float32)
                else:
                    pair = (second, first)
                    point_first = np.asarray(point[6], dtype=np.float32)
                    point_second = np.asarray(point[5], dtype=np.float32)
                index = pair_lookup.get(pair)
                if index is None or float(point[8]) >= distances[index]:
                    continue
                vector = point_second - point_first
                norm = float(np.linalg.norm(vector))
                normal = (
                    vector / norm if norm > 1e-8
                    else -np.asarray(point[7], dtype=np.float32)
                )
                first_velocity = env._link_world_point_velocity(first, point_first)
                second_velocity = env._link_world_point_velocity(second, point_second)
                distances[index] = float(point[8])
                approaches[index] = max(
                    -float(np.dot(normal, second_velocity - first_velocity)), 0.0,
                )
            return distances, approaches

        rng = np.random.default_rng(7)
        for _ in range(20):
            env.step(rng.uniform(-1.0, 1.0, size=6).astype(np.float32))
            expected_distance, expected_approach = whole_body_reference()
            distance, approach = env._self_pair_geometry()
            np.testing.assert_array_equal(distance, expected_distance)
            np.testing.assert_array_equal(approach, expected_approach)
    finally:
        env.close()


def test_strictification_keeps_first_contact_and_drops_post_contact():
    replay = HomotopyReplayBuffer(2, 1, "cpu", capacities={"none": 10, "static": 10, "dynamic": 10})
    for step, (before, after) in enumerate(((False, False), (False, True), (True, True))):
        replay.add("static", np.zeros(2), np.zeros(1), np.ones(2), False,
                   _transition_info(before, after, reached=step == 2), 7, step)
    report = replay.strictify("static"); part = replay.parts["static"]
    assert report["removed"] == 1 and part.size == 2
    assert part.done[1, 0] == 1.0
    assert part.raw[1, 5] == 0.0
    assert report["sha256_before"] != report["sha256_after"]


def test_strictified_replay_applies_bounded_collision_penalty_to_first_contact():
    replay = HomotopyReplayBuffer(
        2, 1, "cpu", capacities={"none": 1, "static": 1, "dynamic": 1}, seed=3,
    )
    info = _transition_info(contact_before=False, contact_after=True)
    replay.add("static", np.zeros(2), np.zeros(1), np.ones(2), False, info, 1, 0)
    replay.strictify("static")
    batch = replay.sample("s1", {"static": 1.0, "dynamic": .02}, batch_size=1)
    strict_reward, fields = homotopy_reward(
        rho_position=.4, next_rho_position=.3, rho_orientation=.2,
        next_rho_orientation=.1, smooth_velocity=.5, velocity_magnitude=.25,
        orientation_scale=1.0, task_reached=False,
        hard_failure=False, obstacle_collision=True, terminal_obstacle_collision=True,
        risk_max=.6, distance_min=.1, xi=1.0,
    )
    assert fields["terminal_guard_penalty"] == 20.0
    assert strict_reward >= -20.1
    assert batch.dones.item() == 1.0
    assert np.isclose(batch.rewards.item(), strict_reward)


def test_s1_sampling_uses_64_192_partition_contract():
    replay = HomotopyReplayBuffer(2, 1, "cpu", capacities={"none": 300, "static": 300, "dynamic": 300})
    for scene in ("none", "static"):
        for episode in range(220):
            replay.add(scene, np.zeros(2), np.zeros(1), np.ones(2), False,
                       _transition_info(), episode, 0)
    batch = replay.sample("s1", {"static": .02, "dynamic": .02}, 256)
    assert batch.observations.shape[0] == 256
    assert replay.last_sample_counts == {"none": 64, "static": 192, "dynamic": 0}


def test_sampled_goal_passes_pose_ik_fk_round_trip():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    env = ThesisHomotopyEnv(config)
    try:
        l6 = config["thesis"]["joint_pose_curriculum"]["levels"][-1]
        env.configure_episode(
            "none", xi=1.0, strict=True,
            goal_scale=float(l6["goal_scale"]),
            position_tolerance=float(l6["position_tolerance_m"]),
            orientation_tolerance=float(l6["orientation_tolerance_rad"]),
            target_distance_min_m=float(l6["target_distance_min_m"]),
            target_distance_max_m=float(l6["target_distance_max_m"]),
            target_orientation_min_rad=float(l6["target_orientation_min_rad"]),
            target_orientation_max_rad=float(l6["target_orientation_max_rad"]),
        )
        for seed in range(20260915, 20260925):
            _, info = env.reset(seed=seed)
            validation = info["goal_ik_validation"]
            assert validation["reachable"]
            assert validation["finite"]
            assert validation["within_limits"]
            assert validation["position_error_m"] <= env.contract.position_tolerance + 1e-12
            assert validation["orientation_error_rad"] <= env.contract.orientation_tolerance + 1e-12
            assert info["goal_sample_attempts"] >= 1
            assert info["self_d_min"] >= env.d_self_safe
    finally:
        env.close()


def test_repeated_reset_reuses_robot_and_obstacle_bodies():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    env = ThesisHomotopyEnv(config)
    try:
        robot_id = env.robot_id
        obstacle_body_id = env._obstacle_body_id
        body_count = p.getNumBodies(physicsClientId=env.client_id)
        for scene, seed in (("none", 71), ("static", 72), ("dynamic", 73), ("none", 74)):
            env.configure_episode(scene, xi=1.0, strict=True)
            env.reset(seed=seed)
            assert env.robot_id == robot_id
            assert env._obstacle_body_id == obstacle_body_id
            assert p.getNumBodies(physicsClientId=env.client_id) == body_count
            assert (env.obstacle_id is None) == (scene == "none")
    finally:
        env.close()


def test_reused_environment_reset_matches_fresh_environment_for_fixed_seed():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    reused = ThesisHomotopyEnv(config)
    fresh = ThesisHomotopyEnv(config)
    try:
        for env in (reused, fresh):
            env.configure_episode(
                "static", xi=.25, strict=False, goal_scale=.25,
                orientation_scale=.4, position_tolerance=.08,
                orientation_tolerance=.7, lambda_self=.1,
            )
        reused.reset(seed=80)
        reused_observation, reused_info = reused.reset(seed=81)
        fresh_observation, fresh_info = fresh.reset(seed=81)
        np.testing.assert_allclose(reused_observation, fresh_observation, atol=1e-7)
        np.testing.assert_allclose(
            reused_info["goal_joint_positions"],
            fresh_info["goal_joint_positions"],
            atol=1e-7,
        )
        np.testing.assert_allclose(reused.obstacle_position, fresh.obstacle_position, atol=1e-7)
        np.testing.assert_allclose(reused.obstacle_velocity, fresh.obstacle_velocity, atol=1e-7)
    finally:
        reused.close()
        fresh.close()


def test_parallel_environment_pool_resets_and_steps_independent_clients():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    contract = {
        "scene": "none", "xi": 1.0, "strict": True,
        "goal_scale": .03, "orientation_scale": 0.0,
        "position_tolerance": .12, "orientation_tolerance": 1.5,
        "lambda_self": 0.0,
    }
    with ParallelThesisEnvPool(config, [101, 102]) as pool:
        assert len(pool) == 2
        resets = pool.reset_many({0: contract, 1: contract})
        assert resets[0][0].shape == (THESIS_OBSERVATION_DIM_HYBRID_KEYPOINT_JACOBIAN,)
        assert resets[1][0].shape == (THESIS_OBSERVATION_DIM_HYBRID_KEYPOINT_JACOBIAN,)
        diagnostics = pool.pose_crossing_diagnostics_many([0, 1])
        assert set(diagnostics) == {0, 1}
        for values in diagnostics.values():
            assert values["step"] == 0
            assert values["remaining_steps"] > 0
            assert values["rho_position"] >= 0.0
            assert values["rho_orientation"] >= 0.0
            assert values["keypoint_distance"] >= 0.0
            assert values["joint_limit_margin_min_rad"] > 0.0
            assert values["ee_jacobian_sigma_min"] >= 0.0
            assert values["keypoint_jacobian_sigma_min"] >= 0.0
            assert np.isfinite(values["ee_log_manipulability"])
            assert np.isfinite(values["keypoint_log_manipulability"])
            assert values["self_distance"] >= 0.0
        actions = {
            0: np.zeros(6, dtype=np.float32),
            1: np.zeros(6, dtype=np.float32),
        }
        pool.send_steps(actions)
        with pytest.raises(RuntimeError, match="already in flight"):
            pool.send_steps(actions)
        with pytest.raises(RuntimeError, match="step is pending"):
            pool.pose_crossing_diagnostics_many([0])
        results = pool.recv_steps()
        assert set(results) == {0, 1}
        assert all(result[0].shape == (THESIS_OBSERVATION_DIM_HYBRID_KEYPOINT_JACOBIAN,) for result in results.values())
        with pytest.raises(RuntimeError, match="no parallel environment step"):
            pool.recv_steps()
        wrapped_results = pool.step_many(actions)
        assert set(wrapped_results) == {0, 1}
        states = pool.states()
        restored = pool.restore_many(states)
        np.testing.assert_allclose(restored[0], wrapped_results[0][0], atol=1e-7)
        np.testing.assert_allclose(restored[1], wrapped_results[1][0], atol=1e-7)


def test_parallel_environment_pool_can_filter_step_info_for_evaluation():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    contract = {
        "scene": "none", "xi": 1.0, "strict": True,
        "goal_scale": .03, "orientation_scale": 0.0,
        "position_tolerance": .12, "orientation_tolerance": 1.5,
        "lambda_self": 0.0,
    }
    keys = ("task_reached", "next_rho_position")
    with ParallelThesisEnvPool(config, [103], step_info_keys=keys) as pool:
        pool.reset_many({0: contract}, seeds={0: 77})
        result = pool.step_many({0: np.zeros(6, dtype=np.float32)})[0]
        assert tuple(result[-1]) == keys


def test_parallel_promotion_probe_restores_training_worker_state():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    contract = {
        "scene": "none", "xi": 1.0, "strict": True,
        "goal_scale": .03, "orientation_scale": 0.0,
        "position_tolerance": .08, "orientation_tolerance": .5,
        "lambda_self": 0.0,
    }

    class ZeroAgent:
        @staticmethod
        def select_actions(observations, deterministic=False):
            assert deterministic
            return np.zeros((len(observations), 6), dtype=np.float32)

    curriculum = SimpleNamespace(
        orientation=SimpleNamespace(level_index=0),
        pose_contract=lambda previous=False: {
            "goal_scale": .03, "orientation_scale": 0.0,
            "position_tolerance": .08, "orientation_tolerance": .5,
        },
    )
    settings = {
        "deterministic_probe_episodes": 2,
        "deterministic_probe_seed": 31001,
        "deterministic_probe_success_floor": 0.0,
        "deterministic_previous_success_floor": .95,
        "deterministic_probe_collision_ceiling": .02,
        "deterministic_previous_collision_ceiling": .02,
        "deterministic_confirmation_episodes": 2,
        "deterministic_confirmation_seed": 51001,
    }
    with ParallelThesisEnvPool(config, [201, 202]) as pool:
        pool.reset_many({0: contract, 1: contract})
        pool.step_many({
            0: np.zeros(6, dtype=np.float32),
            1: np.zeros(6, dtype=np.float32),
        })
        before = pool.states()
        result = thesis_trainer.run_pose_promotion_probe(
            ZeroAgent(), config, curriculum, settings, pool=pool,
        )
        after = pool.states()
        assert result["confirmation_passed"]
        for index in range(2):
            assert after[index]["step_count"] == before[index]["step_count"]
            assert after[index]["rng_state"] == before[index]["rng_state"]
            np.testing.assert_allclose(after[index]["q"], before[index]["q"], atol=1e-8)
            np.testing.assert_allclose(
                after[index]["qdot"], before[index]["qdot"], atol=1e-8,
            )


def test_pybullet_self_contact_has_nonpositive_mesh_clearance_and_unit_risk():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    env = ThesisHomotopyEnv(config)
    rng = np.random.default_rng(20260918)
    try:
        env.configure_episode("none", xi=1.0, strict=True)
        env.reset(seed=20260918)
        found = False
        for _ in range(500):
            q = rng.uniform(
                env.robot.joint_lower_limits + .1,
                env.robot.joint_upper_limits - .1,
            )
            env._set_joint_state(q)
            p.performCollisionDetection(physicsClientId=env.client_id)
            if env._collision_events()["self_collision"]:
                geometry = env._self_geometry()
                assert geometry.distance_min <= 0.0
                assert geometry.risk_max == 1.0
                _, _, _, terminated, _, info = env.step(np.zeros(6, dtype=np.float32))
                assert terminated
                assert info["self_collision"] and info["hard_failure"]
                assert info["hard_penalty"] == 20.0
                found = True
                break
        assert found
    finally:
        env.close()


def test_environment_smoothness_uses_per_joint_mean_not_six_joint_sum():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    env = ThesisHomotopyEnv(config)
    try:
        env.configure_episode("none", xi=1.0, strict=True)
        env.reset(seed=20260917)
        _, qdot_before = env.robot.joint_state()
        _, _, _, _, _, info = env.step(np.asarray([.8, -.7, .6, -.5, .4, -.3]))
        expected = float(np.mean(np.square(
            (np.asarray(info["qdot_measured"]) - qdot_before) / env.action_scale
        )))
        assert np.isclose(info["smooth_velocity"], expected)
        assert np.isclose(
            info["smooth_cost"],
            env.reward_parameters["smooth_cost_weight"] * np.clip(expected, 0.0, 1.0),
        )
    finally:
        env.close()


def test_direct_sac_action_is_scaled_only_by_precision_envelope():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    assert config["thesis"]["self_safety_projection"]["enabled"] is False
    env = ThesisHomotopyEnv(config)
    try:
        env.configure_episode("none", xi=1.0, strict=True)
        env.reset(seed=20260921)
        action = np.asarray([.8, -.7, .6, -.5, .4, -.3], dtype=np.float32)
        _, _, _, _, _, info = env.step(action)
        expected = action * env.action_scale
        np.testing.assert_allclose(info["policy_joint_velocity"], expected, atol=1e-7)
        scaled = expected * float(info["precision_action_scale"])
        np.testing.assert_allclose(info["scaled_policy_joint_velocity"], scaled, atol=1e-7)
        np.testing.assert_allclose(info["commanded_joint_velocity"], scaled, atol=1e-7)
        np.testing.assert_allclose(info["qdot_cmd"], scaled, atol=1e-7)
        assert "nominal_joint_velocity" not in info
        assert "residual_joint_velocity" not in info
    finally:
        env.close()


def test_precision_control_reduces_velocity_near_goal():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    env = ThesisHomotopyEnv(config)
    try:
        env.configure_episode("none", xi=1.0, strict=True)
        env.reset(seed=20260922)
        env.previous_rho_position = .01
        env.previous_rho_orientation = .03
        _, _, _, _, _, info = env.step(np.ones(6, dtype=np.float32))
        z = max(.01 / env.contract.position_tolerance, .03 / env.contract.orientation_tolerance)
        expected_scale = .05 + .05 * z
        assert np.isclose(info["precision_action_scale"], expected_scale)
        np.testing.assert_allclose(
            info["commanded_joint_velocity"], .7 * expected_scale * np.ones(6), atol=1e-7,
        )
    finally:
        env.close()


@pytest.mark.parametrize("level", [0, 4, 5, 6])
@pytest.mark.parametrize("z, expected", [(0.0, .05), (.5, .075), (1., .10), (1.5, .325), (2., 1.), (3., 1.)])
def test_precision_control_uses_episode_tolerances(level, z, expected):
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    env = ThesisHomotopyEnv(config)
    try:
        thresholds = config["thesis"]["joint_pose_curriculum"]["levels"][level]
        env.configure_episode(
            "none", xi=1.0, strict=True,
            position_tolerance=thresholds["position_tolerance_m"],
            orientation_tolerance=thresholds["orientation_tolerance_rad"],
        )
        env.reset(seed=20260922)
        env.previous_rho_position = z * env.contract.position_tolerance
        env.previous_rho_orientation = z * env.contract.orientation_tolerance
        _, _, _, _, _, info = env.step(np.ones(6, dtype=np.float32))
        assert info["precision_action_scale"] == pytest.approx(expected)
        np.testing.assert_allclose(info["commanded_joint_velocity"], .7 * expected, atol=1e-7)
    finally:
        env.close()


@pytest.mark.parametrize("before, after, expected", [(2., .5, 0.), (.5, .5, .05), (.5, 2., -.2), (2., 2., 0.)])
@pytest.mark.parametrize("success", [False, True])
def test_pose_hold_and_exit_rewards_match_replay(before, after, expected, success):
    raw = np.zeros((1, 29), dtype=np.float32)
    raw[0, 0:4] = [before * .05, after * .05, before * .10, after * .10]
    raw[0, 5] = success
    raw[0, 12] = .8
    raw[0, 18] = .25
    raw[0, 24:26] = [.05, .10]
    parameters = dict(
        hold_reward_scale=.05,
        leave_joint_tolerance_penalty=.20,
        keypoint_tracking_scale=0.0,
        keypoint_precision_reward_scale=0.0,
    )
    reward, fields = homotopy_reward(
        rho_position=float(raw[0, 0]), next_rho_position=float(raw[0, 1]),
        rho_orientation=float(raw[0, 2]), next_rho_orientation=float(raw[0, 3]),
        joint_position_tolerance=float(raw[0, 24]), joint_orientation_tolerance=float(raw[0, 25]),
        smooth_velocity=0., velocity_magnitude=0., orientation_scale=1., task_reached=success,
        hard_failure=False, obstacle_collision=False, risk_max=0., distance_min=.8, xi=1.,
        **parameters,
    )
    assert fields["r_goal"] == pytest.approx(expected + 20. * success)
    vectorized, _ = _vectorized_replay_rewards(raw, np.zeros((1, 1)), np.ones(1), 0., .99, parameters)
    assert vectorized[0] == pytest.approx(reward)


def test_joint_precision_quality_uses_sum_and_temperature():
    _, fields = homotopy_reward(
        rho_position=.05, next_rho_position=.025,
        rho_orientation=.1, next_rho_orientation=.1,
        joint_position_tolerance=.05, joint_orientation_tolerance=.1,
        joint_precision_temperature=2.,
        smooth_velocity=0., velocity_magnitude=0., orientation_scale=1., task_reached=False,
        hard_failure=False, obstacle_collision=False, risk_max=0., distance_min=.8, xi=1.,
    )
    assert fields["joint_precision_quality"] == pytest.approx(np.exp(-1.))
    assert fields["next_joint_precision_quality"] == pytest.approx(np.exp(-.75))
    assert fields["keypoint_precision_reward"] == pytest.approx(
        .05 * np.exp(-.75)
    )


def test_initial_goal_curriculum_samples_locally():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    env = ThesisHomotopyEnv(config)
    try:
        scale = float(config["thesis"]["goal_curriculum"]["start_scale"])
        env.configure_episode("none", xi=1.0, strict=True, goal_scale=scale)
        _, info = env.reset(seed=20260916)
        initial_q, _ = env.robot.joint_state()
        maximum_possible_delta = scale * float(
            np.max(env.robot.joint_upper_limits - env.robot.joint_lower_limits)
        )
        assert np.max(np.abs(info["goal_joint_positions"] - initial_q)) <= maximum_possible_delta
        assert info["goal_scale"] == scale
    finally:
        env.close()


def test_goal_scale_sampling_is_split_between_history_and_frontier():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    env = ThesisHomotopyEnv(config)
    try:
        sampled = []
        for seed in range(200):
            env.configure_episode(
                "none", xi=1.0, strict=True, goal_scale=.35,
                goal_scale_history_min=.03, goal_scale_frontier_min=.30,
                goal_scale_history_probability=.5,
            )
            _, info = env.reset(seed=seed)
            sampled.append(float(info["sampled_goal_scale"]))
        history_count = sum(.03 <= scale <= .30 for scale in sampled)
        frontier_count = sum(.30 < scale <= .35 for scale in sampled)
        assert history_count + frontier_count == len(sampled)
        assert 70 <= history_count <= 130
        assert 70 <= frontier_count <= 130
    finally:
        env.close()


def test_v13_task_space_sampling_balances_distance_orientation_cells():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    sampling = config["thesis"]["joint_pose_curriculum"]["task_space_sampling"]
    sampling["balanced_bins"] = True
    sampling["position_bins"] = 2
    sampling["orientation_bins"] = 2
    env = ThesisHomotopyEnv(config)
    counts = np.zeros((2, 2), dtype=np.int64)
    try:
        for episode in range(200):
            env.configure_episode(
                "none", xi=1.0, strict=True, goal_scale=.04,
                position_tolerance=.01, orientation_tolerance=.03,
                target_distance_min_m=.03, target_distance_max_m=.125,
                target_orientation_min_rad=.03,
                target_orientation_max_rad=.274650,
                target_history_distance_max_m=.10,
                target_history_orientation_max_rad=.15,
                target_history_probability=.5,
            )
            _, info = env.reset(seed=10000 if episode == 0 else None)
            distance = float(info["sampled_target_distance_m"])
            angle = float(info["sampled_target_orientation_rad"])
            assert np.isclose(info["goal_error_norm"], distance, atol=1e-5)
            assert np.isclose(info["orientation_error_norm"], angle, atol=1e-5)
            position_bin = int(info["sampled_target_position_bin"])
            orientation_bin = int(info["sampled_target_orientation_bin"])
            assert position_bin == min(int((distance - .03) / ((.125 - .03) / 2)), 1)
            assert orientation_bin == min(
                int((angle - .03) / ((.274650 - .03) / 2)), 1
            )
            counts[position_bin, orientation_bin] += 1
        assert counts.sum() == 200
        np.testing.assert_array_equal(counts, np.full((2, 2), 50))
    finally:
        env.close()


def test_real_pybullet_contacts_are_covered_by_capsule_distance_and_risk():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    env = ThesisHomotopyEnv(config)
    try:
        env.configure_episode("static", xi=1.0, strict=True)
        env.reset(seed=20260917)
        for capsule in env.robot.capsules():
            center = .5 * (capsule.start + capsule.end)
            axis = capsule.end - capsule.start
            if np.linalg.norm(axis) <= 1e-8:
                perpendicular = np.asarray([1.0, 0.0, 0.0])
            else:
                axis = axis / np.linalg.norm(axis)
                basis = np.eye(3)[int(np.argmin(np.abs(axis)))]
                perpendicular = np.cross(axis, basis)
                perpendicular = perpendicular / np.linalg.norm(perpendicular)

            def move_obstacle(position):
                env.obstacle_position = np.asarray(position, dtype=np.float32)
                env.obstacle_velocity.fill(0.0)
                p.resetBasePositionAndOrientation(
                    env.obstacle_id, position.tolist(), [0.0, 0.0, 0.0, 1.0],
                    physicsClientId=env.client_id,
                )
                p.performCollisionDetection(physicsClientId=env.client_id)
                return env._collision_events()["obstacle_collision"]

            candidates = [
                perpendicular, -perpendicular,
                np.asarray([1.0, 0.0, 0.0]), np.asarray([-1.0, 0.0, 0.0]),
                np.asarray([0.0, 1.0, 0.0]), np.asarray([0.0, -1.0, 0.0]),
                np.asarray([0.0, 0.0, 1.0]), np.asarray([0.0, 0.0, -1.0]),
            ]
            directions = []
            for direction in candidates:
                if any(np.allclose(direction, selected) for selected in directions):
                    continue
                if not move_obstacle(center + .6 * direction):
                    directions.append(direction)
                if len(directions) == 2:
                    break
            assert len(directions) == 2, capsule.name
            for direction in directions:
                assert move_obstacle(center), capsule.name
                low, high = 0.0, 0.6
                for _ in range(24):
                    middle = .5 * (low + high)
                    if move_obstacle(center + middle * direction):
                        low = middle
                    else:
                        high = middle
                assert move_obstacle(center + low * direction), capsule.name
                geometry = env._geometry()
                assert geometry is not None
                assert geometry.distance_min < env.d_safe, capsule.name
                assert geometry.risk_max >= .79, capsule.name
    finally:
        env.close()
