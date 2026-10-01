from __future__ import annotations

from pathlib import Path

from rl_risk_sac.algorithms.homotopy_curriculum import HomotopyCurriculum
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.utils.config import load_config


ROOT = Path(__file__).resolve().parents[2]


def _s0_ready() -> HomotopyCurriculum:
    levels = [
        {
            "goal_scale": 0.03,
            "target_distance_min_m": 0.03,
            "target_distance_max_m": 0.70,
            "target_orientation_min_rad": 0.03,
            "target_orientation_max_rad": 3.141592,
            "position_tolerance_m": 0.10,
            "orientation_tolerance_rad": 0.30,
        },
        {
            "goal_scale": 1.0,
            "target_distance_min_m": 0.03,
            "target_distance_max_m": 0.70,
            "target_orientation_min_rad": 0.03,
            "target_orientation_max_rad": 3.141592,
            "position_tolerance_m": 0.05,
            "orientation_tolerance_rad": 0.10,
        },
    ]
    curriculum = HomotopyCurriculum(
        "s0", goal_levels=[0.03, 1.0], orientation_levels=[0.0, 1.0],
        joint_pose_levels=levels, orientation_success_window=1,
        orientation_min_transitions_per_level=1,
        orientation_full_scale_min_steps=1,
        goal_full_scale_min_steps=1, self_full_weight_min_steps=1,
        self_ramp_steps=1,
    )
    curriculum.orientation.level_index = 1
    curriculum.orientation.scale = 1.0
    curriculum.goal.level_index = 1
    curriculum.goal.scale = 1.0
    curriculum.goal.full_scale_steps = 1
    curriculum.orientation.full_scale_steps = 1
    curriculum.orientation.outcomes.append(True)
    curriculum.orientation.deterministic_probe_previous_success_rate = 1.0
    curriculum.orientation.deterministic_probe_previous_collision_rate = 0.0
    curriculum.orientation.deterministic_probe_previous_joint_limit_rate = 0.0
    curriculum.orientation.deterministic_probe_passed = True
    curriculum.self_safety.weight = 1.0
    curriculum.self_safety.full_weight_steps = 1
    return curriculum


def test_s0_minimal_gate_and_s0_to_s1_handoff():
    config = load_config(ROOT / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    curriculum = _s0_ready()
    assert curriculum.stage_complete(1, 1, 1)

    s1 = HomotopyCurriculum("s1")
    s1.inherit_task_state(curriculum.state_dict())
    s1.states["static"].strict = True
    s1.states["static"].replay_strictified = True
    s1.states["static"].strict_steps = int(config["thesis"]["strict_min_transitions"])
    assert s1.stage_complete(1, 1, config["thesis"]["strict_min_transitions"])


def test_s1_to_s2_handoff_requires_dynamic_strict_replay():
    config = load_config(ROOT / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    s2 = HomotopyCurriculum("s2")
    s2.states["dynamic"].strict = True
    s2.states["dynamic"].replay_strictified = True
    s2.states["dynamic"].strict_steps = int(config["thesis"]["strict_min_transitions"])
    assert s2.stage_complete(1, 1, config["thesis"]["strict_min_transitions"])
    s2.states["dynamic"].replay_strictified = False
    assert not s2.stage_complete(1, 1, config["thesis"]["strict_min_transitions"])


def test_serial_config_has_no_historical_experiment_include():
    config = load_config(ROOT / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    assert config["thesis"]["serial_stages"]["s0"]["purpose"] == "reaching"
    assert config["thesis"]["serial_stages"]["s1"]["purpose"] == "static_obstacle_avoidance"
    assert config["thesis"]["serial_stages"]["s2"]["purpose"] == "dynamic_obstacle_avoidance"


def test_task_space_goal_acceptance_stats_are_recorded_per_cell():
    config = load_config(ROOT / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    config["thesis"]["joint_pose_curriculum"]["task_space_sampling"].update({
        "position_bins": 2, "orientation_bins": 2,
    })
    env = ThesisHomotopyEnv(config)
    try:
        env.configure_episode(
            "none", xi=1.0, strict=True, goal_scale=1.0,
            orientation_scale=1.0, position_tolerance=.05,
            orientation_tolerance=.10, target_distance_min_m=.03,
            target_distance_max_m=.20, target_orientation_min_rad=.03,
            target_orientation_max_rad=.80,
        )
        for seed in range(12):
            env.reset(seed=seed)
        assert env.goal_proposal_counts.shape == (4,)
        assert int(env.goal_proposal_counts.sum()) >= 12
        assert int(env.goal_accept_counts.sum()) == 12
        assert float(env._contract_info()["goal_acceptance_rates"].max()) <= 1.0
    finally:
        env.close()
