from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import pytest

from rl_risk_sac.algorithms.homotopy_curriculum import HomotopyCurriculum
from rl_risk_sac.algorithms.homotopy_replay import HomotopyReplayBuffer
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.utils.config import load_config as _load_config


def load_config(path):
    config = _load_config(path)
    config["thesis"]["obstacles"]["enabled"] = False
    return config


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("stage", ["s0", "s1", "s2", "s3"])
def test_full_curriculum_stages_are_supported(stage):
    curriculum = HomotopyCurriculum(stage)
    assert curriculum.stage == stage


@pytest.mark.parametrize("scene", ["static", "dynamic"])
def test_removed_scenes_cannot_enter_training_or_restore(scene):
    curriculum = HomotopyCurriculum("s0")
    with pytest.raises(ValueError, match="only reaching scene none"):
        curriculum.contract(scene)
    with pytest.raises(ValueError, match="only reaching scene none"):
        curriculum.finish_episode(scene, True, True, 1)
    replay = HomotopyReplayBuffer(2, 1, "cpu", s0_anchor_capacity=1,
                                  s0_current_capacity=1, s0_success_capacity=1)
    with pytest.raises(ValueError, match="only reaching scene none"):
        replay.add(scene, None, None, None, False, {}, 0, 0)
    env = ThesisHomotopyEnv(load_config(ROOT / "configs/experiments/paper_full.yaml"))
    try:
        with pytest.raises(ValueError, match="only reaching scene none"):
            env.configure_episode(scene, xi=1.0, strict=True)
        with pytest.raises(ValueError, match="only reaching scene none"):
            env.restore_episode_state({"contract": replace(env.contract, scene=scene)})
        assert env.contract.scene == "none"
    finally:
        env.close()


def test_legacy_reaching_state_cannot_restore_obstacle_sampling():
    original = HomotopyCurriculum("s0", seed=31)
    state = original.state_dict()
    state["probabilities"] = [0.0, 0.5, 0.5]
    state["states"] = {"static": object(), "dynamic": object()}
    restored = HomotopyCurriculum("s0", seed=32)
    restored.load_state_dict(state)
    assert restored.states == {}
    assert restored.probabilities == [1.0]
    assert restored.xi_map() == {}
    assert {restored.choose_scene() for _ in range(100)} == {"none"}


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


def test_reaching_completion_gate():
    assert _s0_ready().stage_complete(1, 1)


def test_full_config_contains_only_full_avoidance():
    config = load_config(ROOT / "configs/experiments/paper_full.yaml")
    assert set(config["thesis"]["serial_stages"]) == {"s0", "s1", "s2", "s3"}
    assert [config["thesis"]["serial_stages"][stage]["purpose"] for stage in ("s0", "s1", "s2", "s3")] == [
        "predictive_obstacle_avoidance", "static_obstacle_reaching",
        "simple_dynamic_obstacle_reaching", "full_dynamic_obstacle_avoidance",
    ]


def test_task_space_goal_acceptance_stats_are_recorded_per_cell():
    config = load_config(ROOT / "configs/experiments/paper_full.yaml")
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
