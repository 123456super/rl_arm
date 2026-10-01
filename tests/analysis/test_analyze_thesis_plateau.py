from copy import deepcopy

from scripts.analysis.analyze_thesis_plateau import decide_s0_stop
from scripts.core.train_thesis_homotopy import (
    accrue_update_credit,
    resolve_num_envs,
    resolve_stage_block_steps,
    s0_probe_track_key,
)
from rl_risk_sac.algorithms.homotopy_curriculum import HomotopyCurriculum


CRITERIA = {
    "min_current_episodes_per_block": 100,
    "min_anchor_episodes_per_block": 50,
    "safe_success_delta": 0.02,
    "self_collision_delta": 0.01,
    "timeout_delta": 0.02,
    "anchor_success_delta": 0.01,
    "probe_success_delta": 0.02,
    "probe_collision_delta": 0.01,
    "probe_timeout_delta": 0.02,
    "position_error_delta_m": 0.005,
    "orientation_error_delta_rad": 0.03,
    "probe_position_error_delta_m": 0.005,
    "probe_orientation_error_delta_rad": 0.03,
    "q_abs_limit": 1000.0,
    "critic_loss_limit": 1000.0,
    "divergence_ratio": 5.0,
}


def _snapshot():
    return {
        "run": "block",
        "stage_total_step": 200000,
        "orientation_level_index": 5,
        "orientation_scale": 0.5,
        "s0_phase": "joint_pose",
        "lambda_self": 0.0,
        "self_safety_eligible_steps": 0,
        "self_safety_full_weight_steps": 0,
        "s0_goal_gate_eligible": False,
        "pose_scales_seen": [0.5],
        "current_episodes": 200,
        "anchor_episodes": 100,
        "safe_success_rate": 0.70,
        "self_collision_rate": 0.10,
        "timeout_rate": 0.20,
        "mean_position_error_m": 0.10,
        "mean_orientation_error_rad": 0.50,
        "anchor_success_rate": 0.96,
        "probe": {
            "success_rate": 0.90, "collision_rate": 0.04,
            "timeout_rate": 0.06, "passed": False,
            "mean_position_error_m": 0.08,
            "mean_orientation_error_rad": 0.40,
        },
        "updates": {
            "finite": True, "q_abs_p95": 30.0, "critic_loss_p95": 5.0,
            "updates": 100000, "tail_updates": 10000,
            "alpha_min": 0.05, "alpha_max": 0.2,
        },
    }


def test_stage_budget_caps_final_block_and_rejects_exhausted_stage():
    assert resolve_stage_block_steps(0, 100000, 600000) == 100000
    assert resolve_stage_block_steps(575000, 100000, 600000) == 25000
    try:
        resolve_stage_block_steps(600000, 100000, 600000)
    except ValueError as error:
        assert "hard budget exhausted" in str(error)
    else:
        raise AssertionError("exhausted stage budget must be rejected")


def test_environment_defaults_are_parallel_only_for_s0():
    assert resolve_num_envs("s0", None, 8, False) == 8
    assert resolve_num_envs("s1", None, 8, False) == 1
    assert resolve_num_envs("s2", None, 8, False) == 1
    assert resolve_num_envs("s0", None, 8, True) == 1
    assert resolve_num_envs("s1", 2, 8, False) == 2


def test_fractional_utd_accumulates_one_update_for_four_transitions():
    counters = {}
    assert [accrue_update_credit(counters, .25) for _ in range(8)] == [
        0, 0, 0, 1, 0, 0, 0, 1,
    ]
    assert counters["update_credit"] == 0.0


def test_final_pose_probe_track_is_separate_from_self_safety_tracks():
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
        orientation_deterministic_probe_required=True,
        self_ramp_steps=10, self_full_weight_min_steps=1,
    )
    curriculum.orientation.level_index = 1
    curriculum.orientation.scale = 1.0
    curriculum.goal.level_index = 1
    curriculum.goal.scale = 1.0
    curriculum.goal.full_scale_steps = 1
    curriculum.orientation.full_scale_steps = 1
    curriculum.orientation.outcomes.append(True)
    curriculum.orientation.anchor_outcomes.append(True)
    assert s0_probe_track_key(curriculum) == "level_01_pose"

    curriculum.orientation.deterministic_probe_passed = True
    assert s0_probe_track_key(curriculum) == "level_01_self_ramp_00"
    curriculum.self_safety.weight = 1.0
    assert s0_probe_track_key(curriculum) == "level_01_self_full"


def test_two_block_plateau_requires_no_material_improvement():
    previous = _snapshot()
    current = deepcopy(previous)
    current["run"] = "block_next"
    current["stage_total_step"] = 300000
    result = decide_s0_stop(previous, current, CRITERIA, max_stage_steps=600000)
    assert result["decision"] == "plateau_stop"
    assert result["comparison_data_sufficient"]
    assert result["plateau_detected"]

    current["safe_success_rate"] += 0.02
    result = decide_s0_stop(previous, current, CRITERIA, max_stage_steps=600000)
    assert result["decision"] == "continue"
    assert result["substantial_improvements"]["safe_success"]
    assert not result["plateau_detected"]


def test_l0_plateau_does_not_require_anchor_and_self_progress_blocks_stop():
    previous = _snapshot()
    current = deepcopy(previous)
    previous["orientation_level_index"] = current["orientation_level_index"] = 0
    previous["orientation_scale"] = current["orientation_scale"] = 0.0
    previous["pose_scales_seen"] = current["pose_scales_seen"] = [0.0]
    previous["anchor_episodes"] = current["anchor_episodes"] = 0
    previous["anchor_success_rate"] = current["anchor_success_rate"] = None
    current["stage_total_step"] = 300000
    result = decide_s0_stop(previous, current, CRITERIA, 600000)
    assert result["decision"] == "plateau_stop"
    assert not result["anchor_data_required"]

    current["self_safety_eligible_steps"] = 1000
    result = decide_s0_stop(previous, current, CRITERIA, 600000)
    assert result["decision"] == "continue"
    assert result["self_curriculum_progress"]

    current["self_safety_eligible_steps"] = 0
    current["s0_phase"] = "self_safety_ramp"
    result = decide_s0_stop(previous, current, CRITERIA, 600000)
    assert result["decision"] == "continue"
    assert not result["same_s0_phase"]


def test_stop_priority_is_numerical_then_validation_then_budget():
    previous = _snapshot()
    previous["stage_total_step"] = 500000
    current = deepcopy(previous)
    current["stage_total_step"] = 600000
    current["s0_goal_gate_eligible"] = True
    assert decide_s0_stop(previous, current, CRITERIA, 600000)["decision"] == "run_validation"

    current["updates"]["finite"] = False
    assert decide_s0_stop(previous, current, CRITERIA, 600000)["decision"] == "numerical_failure"

    current["updates"]["finite"] = True
    current["s0_goal_gate_eligible"] = False
    assert decide_s0_stop(previous, current, CRITERIA, 600000)["decision"] == "hard_budget_stop"


def test_plateau_comparison_rejects_nonconsecutive_blocks():
    previous = _snapshot()
    current = deepcopy(previous)
    current["stage_total_step"] = 400000
    try:
        decide_s0_stop(previous, current, CRITERIA, 600000)
    except ValueError as error:
        assert "consecutive block endpoints" in str(error)
    else:
        raise AssertionError("nonconsecutive block endpoints must be rejected")
