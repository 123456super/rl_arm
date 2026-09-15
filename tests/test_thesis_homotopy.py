from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_risk_sac.algorithms.homotopy_curriculum import HomotopyCurriculum
from rl_risk_sac.algorithms.homotopy_replay import HomotopyReplayBuffer
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.tasks.thesis_reaching import build_thesis_observation, homotopy_reward
from rl_risk_sac.utils.config import load_config


def _transition_info(contact_before=False, contact_after=False, reached=False):
    return {
        "rho_position": .4, "next_rho_position": .3,
        "rho_orientation": .2, "next_rho_orientation": .1,
        "smooth_velocity": .5, "task_reached": reached, "hard_failure": False,
        "obstacle_collision": contact_after and not contact_before,
        "self_collision": False, "environment_collision": False, "joint_limit": False,
        "control_max_risk": .6, "control_min_distance": .1,
        "obstacle_contact_seen_before": contact_before, "obstacle_contact_seen": contact_after,
    }


def test_no_obstacle_observation_contract_is_55_and_bounded():
    observation = build_thesis_observation(
        q=np.zeros(6), qdot=np.zeros(6), joint_lower=-np.pi * np.ones(6),
        joint_upper=np.pi * np.ones(6), joint_velocity_limits=np.pi * np.ones(6),
        position_error=np.asarray([.2, -.1, .3]), orientation_error=np.asarray([.1, .2, -.3]),
        obstacle_present=False, obstacle_position=np.zeros(3), obstacle_velocity=np.zeros(3), geometry=None,
    )
    assert observation.shape == (55,)
    assert np.all(np.abs(observation) <= 1.0)
    assert observation[-1] == 0.0


def test_low_xi_keeps_goal_reward_and_scales_only_safety_terms():
    common = dict(rho_position=.4, next_rho_position=.3, rho_orientation=.2,
                  next_rho_orientation=.1, smooth_velocity=.5, task_reached=True,
                  hard_failure=False, obstacle_collision=True, risk_max=.6, distance_min=.1)
    low, low_fields = homotopy_reward(**common, xi=.02)
    strict, strict_fields = homotopy_reward(**common, xi=1.0)
    assert low_fields["r_goal"] == strict_fields["r_goal"]
    expected_gap = .98 * (4 * strict_fields["c_proximity"] + 34)
    assert np.isclose(low - strict, expected_gap)


def test_terminal_absorbing_guard_removes_early_failure_shortcut():
    common = dict(
        rho_position=.5, next_rho_position=.5, rho_orientation=1.2,
        next_rho_orientation=1.2, smooth_velocity=0.0, task_reached=False,
        risk_max=0.0, distance_min=.8, xi=1.0, gamma=.99, horizon=240,
    )
    wait_step, wait_fields = homotopy_reward(
        **common, hard_failure=False, obstacle_collision=False,
    )
    timeout = sum(.99**step * wait_step for step in range(240))
    hard, hard_fields = homotopy_reward(
        **common, hard_failure=True, obstacle_collision=False,
    )
    strict_collision, strict_fields = homotopy_reward(
        **common, hard_failure=False, obstacle_collision=True,
        terminal_obstacle_collision=True,
    )
    tolerant_collision, tolerant_fields = homotopy_reward(
        **{**common, "xi": .02}, hard_failure=False, obstacle_collision=True,
        terminal_obstacle_collision=False,
    )
    assert hard < timeout and strict_collision < timeout
    assert hard_fields["terminal_guard_penalty"] > 0
    assert strict_fields["terminal_guard_penalty"] > 0
    assert tolerant_fields["terminal_guard_penalty"] == 0
    assert tolerant_collision == wait_fields["r_goal"] - .02 * 34.0


def test_curriculum_waits_for_full_window_and_pauses_below_floor():
    curriculum = HomotopyCurriculum("s1", seed=1, ramp_steps=50000)
    for _ in range(99): curriculum.finish_episode("static", True, 10)
    assert curriculum.states["static"].eligible_steps == 0
    curriculum.finish_episode("static", True, 10)
    assert curriculum.states["static"].eligible_steps == 10
    for _ in range(20): curriculum.finish_episode("static", False, 10)
    frozen = curriculum.states["static"].eligible_steps
    curriculum.finish_episode("static", False, 10)
    assert curriculum.states["static"].eligible_steps == frozen


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


def test_strictified_replay_applies_terminal_guard_to_first_contact():
    replay = HomotopyReplayBuffer(
        2, 1, "cpu", capacities={"none": 1, "static": 1, "dynamic": 1}, seed=3,
    )
    info = _transition_info(contact_before=False, contact_after=True)
    replay.add("static", np.zeros(2), np.zeros(1), np.ones(2), False, info, 1, 0)
    replay.strictify("static")
    batch = replay.sample("s1", {"static": 1.0, "dynamic": .02}, batch_size=1)
    strict_reward, fields = homotopy_reward(
        rho_position=.4, next_rho_position=.3, rho_orientation=.2,
        next_rho_orientation=.1, smooth_velocity=.5, task_reached=False,
        hard_failure=False, obstacle_collision=True, terminal_obstacle_collision=True,
        risk_max=.6, distance_min=.1, xi=1.0,
    )
    assert fields["terminal_guard_penalty"] > 0
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
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/experiments/thesis_homotopy.yaml")
    env = ThesisHomotopyEnv(config)
    try:
        env.configure_episode("none", xi=1.0, strict=True)
        _, info = env.reset(seed=20260915)
        validation = info["goal_ik_validation"]
        assert validation["reachable"]
        assert validation["within_limits"]
        assert validation["position_error_m"] <= 0.01
        assert validation["orientation_error_rad"] <= 0.05
        assert info["goal_sample_attempts"] >= 1
    finally:
        env.close()
