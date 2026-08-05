import numpy as np
import pytest

from rl_risk_sac.utils.config import load_config
from scripts.compare_vaps_v0_v1 import _load_checkpoint, _validate_g2_config, compare_step


def _info() -> dict[str, object]:
    return {
        "qdot_requested": np.asarray([0.1, -0.1]),
        "collision_capsule_overlap": False,
        "collision_pybullet_contact": False,
        "collision_any": False,
        "termination_collision": False,
        "termination_reason": "",
        "safety_filter_status": "filtered",
        "safety_filter_safe_stop": False,
    }


def test_compare_step_accepts_v1_diagnostic_fields_without_execution_change() -> None:
    v0 = _info()
    v1 = {
        **_info(),
        "viability_status": "certified_viable",
        "viability_horizon_s": 0.15,
        "viability_min_h_m": 0.1,
        "viability_strict_feasible": True,
        "viability_solver_status": "one_step_strict_sufficient",
        "viability_model_assumptions_valid": True,
    }

    result = compare_step(
        np.asarray([1.0, 2.0]),
        np.asarray([1.0, 2.0]),
        np.asarray([0.2, -0.2]),
        np.asarray([0.2, -0.2]),
        np.asarray([0.1, -0.1]),
        np.asarray([0.1, -0.1]),
        v0,
        v1,
        False,
        False,
        False,
        False,
        1.0e-7,
    )

    assert result["passed"] is True


def test_compare_step_rejects_command_or_termination_difference() -> None:
    v0 = _info()
    v1 = {
        **_info(),
        "viability_status": "certified_viable",
        "viability_horizon_s": 0.15,
        "viability_min_h_m": 0.1,
        "viability_strict_feasible": True,
        "viability_solver_status": "one_step_strict_sufficient",
        "viability_model_assumptions_valid": True,
    }
    v1["termination_reason"] = "physical_contact"

    result = compare_step(
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        np.asarray([0.0, 1.0e-3]),
        v0,
        v1,
        False,
        True,
        False,
        False,
        1.0e-7,
    )

    assert result["passed"] is False
    assert "termination_reason" in result["field_mismatches"]
    assert "terminated" in result["field_mismatches"]


def test_checkpoint_manifest_requires_selected_record_to_match_actor(tmp_path) -> None:
    checkpoint = tmp_path / "actor.pt"
    checkpoint.write_bytes(b"actor")
    selection = tmp_path / "selected_checkpoint.csv"
    selection.write_text("step,checkpoint,selected\n50000,other.pt,1\n", encoding="utf-8")
    manifest = tmp_path / "actors.json"
    manifest.write_text(
        """{
  "protocol": "vaps_g2_strict_execution_v2",
  "method": "link_fixed",
  "actors": [{
    "train_seed": 4108,
    "checkpoint_step": 50000,
    "checkpoint": "%s",
    "selection_record": "%s"
  }]
}
""" % (checkpoint, selection),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="selection record checkpoint"):
        _load_checkpoint(manifest, 4108)


def test_g2_config_rejects_non_deterministic_wall_clock_budget() -> None:
    config = load_config("configs/experiments/vaps/v2_g2_strict_execution.yaml")
    _validate_g2_config(config)
    config["env"]["safety_filter"]["max_filter_compute_time_s"] = 0.30

    with pytest.raises(ValueError, match="max_filter_compute_time_s"):
        _validate_g2_config(config)
