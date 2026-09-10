from __future__ import annotations

import pandas as pd

from scripts.check_restart_r2e_capacity_gate import (
    best_diagnostic,
    build_capacity_table,
    json_record,
)


def _matrix() -> dict:
    return {
        "variants": {
            "step60": {
                "checkpoint_step": 60000,
                "checkpoints": {11: "actor60.pt"},
            },
            "step70": {
                "checkpoint_step": 70000,
                "checkpoints": {11: "actor70.pt"},
            },
        },
        "diagnostic_gate": {"minimum": 0.80},
    }


def test_capacity_gate_is_any_checkpoint_at_preregistered_threshold() -> None:
    episodes = pd.DataFrame(
        [
            {"variant": "step60", "scenario": "no_obstacle", "success": value,
             "collision": 0, "final_position_error": 0.1, "min_position_error": 0.08,
             "completion_time": 3.0}
            for value in [1, 1, 1, 1, 0]
        ]
        + [
            {"variant": "step70", "scenario": "no_obstacle", "success": value,
             "collision": 0, "final_position_error": 0.2, "min_position_error": 0.18,
             "completion_time": 4.0}
            for value in [1, 1, 1, 0, 0]
        ]
    )

    table = build_capacity_table(episodes, _matrix())

    assert table.set_index("variant").loc["step60", "capacity_pass"]
    assert not table.set_index("variant").loc["step70", "capacity_pass"]
    assert best_diagnostic(table)["variant"] == "step60"


def test_capacity_best_tie_breaks_on_error_then_earlier_step() -> None:
    episodes = pd.DataFrame(
        [
            {"variant": variant, "scenario": "no_obstacle", "success": 1,
             "collision": 0, "final_position_error": error, "min_position_error": error,
             "completion_time": 3.0}
            for variant, error in [("step60", 0.03), ("step70", 0.02)]
        ]
    )

    assert best_diagnostic(build_capacity_table(episodes, _matrix()))["variant"] == "step70"


def test_capacity_json_record_replaces_missing_outcome_metrics() -> None:
    episodes = pd.DataFrame(
        [
            {"variant": variant, "scenario": "no_obstacle", "success": 1,
             "collision": 0, "final_position_error": 0.02, "min_position_error": 0.02,
             "completion_time": 3.0}
            for variant in ("step60", "step70")
        ]
    )

    record = json_record(best_diagnostic(build_capacity_table(episodes, _matrix())))

    assert record["timeout_mean_final_position_error"] is None
