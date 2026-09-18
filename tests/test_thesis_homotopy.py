from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet as p

from rl_risk_sac.algorithms.homotopy_curriculum import HomotopyCurriculum
from rl_risk_sac.algorithms.homotopy_replay import HomotopyReplayBuffer
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.tasks.thesis_reaching import (
    THESIS_OBSERVATION_DIM,
    SelfCollisionGeometry,
    ThesisGeometry,
    build_thesis_observation,
    compute_self_collision_geometry,
    homotopy_reward,
    quaternion_rotation_6d,
)
from rl_risk_sac.utils.config import load_config


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


def test_no_obstacle_observation_contract_contains_continuous_pose_twist_and_is_bounded():
    ee_quaternion = np.asarray([0.0, 0.0, 0.0, -1.0])
    goal_quaternion = np.asarray([0.0, 0.0, 0.5, np.sqrt(0.75)])
    observation = build_thesis_observation(
        q=np.zeros(6), qdot=.35 * np.ones(6), joint_lower=-np.pi * np.ones(6),
        joint_upper=np.pi * np.ones(6), joint_velocity_scale=.7 * np.ones(6),
        ee_position=np.asarray([.4, -.2, .3]), ee_quaternion=ee_quaternion,
        ee_linear_velocity=np.asarray([.1, -.2, .3]),
        ee_angular_velocity=np.asarray([.4, -.8, 1.2]),
        ee_linear_velocity_scale=1.0, ee_angular_velocity_scale=4.0,
        goal_position=np.asarray([.6, -.3, .4]), goal_quaternion=goal_quaternion,
        position_error=np.asarray([.2, -.1, .3]), orientation_error=np.asarray([.1, .2, -.3]),
        orientation_scale=.25, remaining_time_fraction=.75,
        obstacle_present=False, obstacle_position=np.zeros(3), obstacle_velocity=np.zeros(3), geometry=None,
    )
    assert observation.shape == (THESIS_OBSERVATION_DIM,)
    assert np.all(np.abs(observation) <= 1.0)
    np.testing.assert_allclose(observation[6:12], .5)
    np.testing.assert_allclose(observation[12:15], [.4, -.2, .3])
    np.testing.assert_allclose(observation[15:21], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(observation[21:24], [.6, -.3, .4])
    np.testing.assert_allclose(observation[24:30], quaternion_rotation_6d(goal_quaternion))
    np.testing.assert_allclose(observation[41:44], [.1, -.2, .3])
    np.testing.assert_allclose(observation[44:47], [.1, -.2, .3])
    assert observation[47] == -.5
    assert observation[48] == .5
    assert observation[97] == 0.0
    np.testing.assert_allclose(observation[98:110], 1.0)
    np.testing.assert_allclose(observation[110:122], 0.0)


def test_continuous_rotation_representation_is_quaternion_sign_invariant():
    quaternion = np.asarray([.2, -.3, .4, .5], dtype=np.float32)
    np.testing.assert_allclose(
        quaternion_rotation_6d(quaternion), quaternion_rotation_6d(-quaternion), atol=1e-7,
    )


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
        position_error=np.zeros(3), orientation_error=np.zeros(3), orientation_scale=1.0,
        remaining_time_fraction=1.0, obstacle_present=True,
        obstacle_position=np.asarray([.4, 0., .4]),
        obstacle_velocity=np.asarray([.1, 0., 0.]), geometry=geometry,
    )
    # Base state occupies [0,49); approach and risk are the final two
    # six-link vectors immediately before the presence flag.
    np.testing.assert_allclose(observation[85:91], geometry.approach_velocities)
    np.testing.assert_allclose(observation[91:97], geometry.risk_per_link)
    assert observation[97] == 1.0


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
        position_error=np.zeros(3), orientation_error=np.zeros(3), orientation_scale=1.0,
        remaining_time_fraction=1.0, obstacle_present=False,
        obstacle_position=np.zeros(3), obstacle_velocity=np.zeros(3), geometry=None,
        self_geometry=self_geometry,
    )
    assert observation.shape == (122,)
    expected_distance = 2.0 * (self_geometry.distances + .02) / .27 - 1.0
    np.testing.assert_allclose(observation[98:104], expected_distance)
    np.testing.assert_allclose(observation[104:110], self_geometry.ttc / 3.0)
    np.testing.assert_allclose(observation[110:116], self_geometry.approach_velocities)
    np.testing.assert_allclose(observation[116:122], self_geometry.risk_per_link)
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
    expected_gap = .98 * (
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
    assert fields["external_safety_penalty"] == 1.0
    assert fields["self_safety_penalty"] == 1.0


def test_position_phase_ignores_orientation_reward_but_preserves_position_reward():
    common = dict(
        rho_position=.20, next_rho_position=.18, smooth_velocity=0.0,
        velocity_magnitude=0.0, orientation_scale=0.0, task_reached=False,
        hard_failure=False, obstacle_collision=False, risk_max=0.0,
        distance_min=.8, xi=1.0,
    )
    improved, improved_fields = homotopy_reward(
        **common, rho_orientation=2.0, next_rho_orientation=1.0,
    )
    worsened, worsened_fields = homotopy_reward(
        **common, rho_orientation=1.0, next_rho_orientation=2.0,
    )
    assert np.isclose(improved, worsened)
    assert improved_fields["orientation_state_cost"] == 0.0
    assert improved_fields["position_progress"] > 0.0
    assert improved_fields["r_goal"] > 0.0
    assert worsened_fields["orientation_scale"] == 0.0


def test_pose_reward_priority_multiplicatively_rises_near_the_position_target():
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
    assert 0.0 < far["orientation_reward_scale"]
    assert far["orientation_reward_scale"] < middle["orientation_reward_scale"]
    assert middle["orientation_reward_scale"] < near["orientation_reward_scale"] < 1.0


def test_pose_potential_keeps_position_primary_when_orientation_improves():
    reward, fields = homotopy_reward(
        rho_position=.06, next_rho_position=.16,
        rho_orientation=1.8, next_rho_orientation=.1,
        smooth_velocity=0.0, velocity_magnitude=0.0,
        orientation_scale=1.0, task_reached=False, hard_failure=False,
        obstacle_collision=False, risk_max=0.0, distance_min=.8, xi=1.0,
    )
    assert fields["orientation_progress"] > 0.0
    assert fields["position_progress"] < 0.0
    assert fields["pose_potential_progress"] < 0.0
    assert reward < 0.0


def test_discounted_pose_potential_does_not_reward_a_closed_cycle():
    forward, _ = homotopy_reward(
        rho_position=.20, next_rho_position=.10,
        rho_orientation=.8, next_rho_orientation=.4,
        smooth_velocity=0.0, velocity_magnitude=0.0,
        orientation_scale=1.0, task_reached=False, hard_failure=False,
        obstacle_collision=False, risk_max=0.0, distance_min=.8, xi=0.0,
    )
    backward, _ = homotopy_reward(
        rho_position=.10, next_rho_position=.20,
        rho_orientation=.4, next_rho_orientation=.8,
        smooth_velocity=0.0, velocity_magnitude=0.0,
        orientation_scale=1.0, task_reached=False, hard_failure=False,
        obstacle_collision=False, risk_max=0.0, distance_min=.8, xi=0.0,
    )
    # A two-step return must use the same gamma as SAC.
    assert forward + .99 * backward < 0.0


def test_progress_is_positive_and_terminal_penalties_are_bounded():
    progress, progress_fields = homotopy_reward(
        rho_position=.42, next_rho_position=.41,
        rho_orientation=2.32, next_rho_orientation=2.32,
        smooth_velocity=0.0, velocity_magnitude=0.0, orientation_scale=1.0,
        task_reached=False, hard_failure=False,
        obstacle_collision=False, risk_max=0.0, distance_min=.8, xi=1.0,
    )
    assert progress > 0.0
    assert progress_fields["position_progress"] > 0.0

    common = dict(
        rho_position=.5, next_rho_position=.5, rho_orientation=1.2,
        next_rho_orientation=1.2, smooth_velocity=0.0, velocity_magnitude=0.0,
        orientation_scale=1.0, task_reached=False,
        risk_max=0.0, distance_min=.8, xi=1.0, gamma=.99, horizon=240,
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
    assert wait_step < 0.0
    assert hard >= -10.1
    assert strict_collision >= -11.1
    assert hard_fields["terminal_guard_penalty"] == 0
    assert strict_fields["terminal_guard_penalty"] == 10.0
    assert tolerant_fields["terminal_guard_penalty"] == 0
    expected_tolerant_penalty = .02
    assert tolerant_collision == wait_fields["r_goal"] - expected_tolerant_penalty
    assert min(hard, strict_collision, tolerant_collision) >= -11.1


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


def test_retention_probe_blocks_advancement_and_failure_enables_recovery_mix():
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
    )
    assert curriculum.orientation_retention_mode == "recovery"
    assert curriculum.orientation_anchor_probability == .5
    np.testing.assert_allclose(curriculum.orientation_replay_mix, (.5, .35, .15))
    curriculum.record_orientation_probe(
        success_rate=1.0, collision_rate=0.0, joint_limit_rate=0.0, passed=True,
    )
    assert curriculum.advance_orientation_if_ready()
    assert curriculum.orientation_scale == .5
    assert curriculum.orientation_retention_mode == "normal"


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
    assert replay.s0_current.raw.shape[1] == 22
    np.testing.assert_allclose(replay.s0_current.raw[0, 17:22], [.6, .002, .6, .3, .01])
    assert strict.rewards.item() < low.rewards.item()


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
    assert fields["terminal_guard_penalty"] == 10.0
    assert strict_reward >= -20.0
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
        assert info["self_d_min"] >= env.d_self_safe
    finally:
        env.close()


def test_pybullet_self_contact_has_nonpositive_mesh_clearance_and_unit_risk():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/experiments/thesis_homotopy.yaml")
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
                assert info["hard_penalty"] == 10.0
                found = True
                break
        assert found
    finally:
        env.close()


def test_environment_smoothness_uses_per_joint_mean_not_six_joint_sum():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/experiments/thesis_homotopy.yaml")
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


def test_initial_goal_curriculum_samples_locally():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/experiments/thesis_homotopy.yaml")
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


def test_real_pybullet_contacts_are_covered_by_capsule_distance_and_risk():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/experiments/thesis_homotopy.yaml")
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
