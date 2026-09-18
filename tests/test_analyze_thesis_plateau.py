from copy import deepcopy

from scripts.analyze_thesis_plateau import decide_s0_stop
from scripts.train_thesis_homotopy import resolve_stage_block_steps


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
        "s0_goal_gate_eligible": False,
        "pose_scales_seen": [0.5],
        "current_episodes": 200,
        "anchor_episodes": 100,
        "safe_success_rate": 0.70,
        "self_collision_rate": 0.10,
        "timeout_rate": 0.20,
        "anchor_success_rate": 0.96,
        "probe": {
            "success_rate": 0.90, "collision_rate": 0.04,
            "timeout_rate": 0.06, "passed": False,
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
