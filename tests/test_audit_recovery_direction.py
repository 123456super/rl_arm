import csv
from pathlib import Path

from scripts.audit_recovery_direction import audit_trace_dir


def test_audit_recovery_direction_reports_margin_delta(tmp_path: Path) -> None:
    trace = tmp_path / "episode_0000.csv"
    fields = [
        "seed", "step", "recovery_active", "recovery_trigger_reason", "recovery_success",
        "recovery_command_norm", "predictive_h_min_m", "predictive_h_by_link_m",
        "safety_jacobian_command_by_link_mps", "safety_drift_by_link_mps",
        "collision_pybullet_contact", "collision_contact_link_names", "collision_min_contact_distance",
    ]
    with open(trace, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"seed": "1", "step": "10", "recovery_active": "1", "recovery_trigger_reason": "margin", "recovery_success": "0", "recovery_command_norm": "0.5", "predictive_h_min_m": "-0.1", "predictive_h_by_link_m": "-0.1|0.2", "safety_jacobian_command_by_link_mps": "0.1|-0.2", "safety_drift_by_link_mps": "-0.1|0.0", "collision_pybullet_contact": "0", "collision_contact_link_names": "", "collision_min_contact_distance": "nan"})
        writer.writerow({"seed": "1", "step": "11", "recovery_active": "1", "recovery_trigger_reason": "margin", "recovery_success": "1", "recovery_command_norm": "0.4", "predictive_h_min_m": "-0.05", "predictive_h_by_link_m": "-0.05|0.3", "safety_jacobian_command_by_link_mps": "0.2|-0.1", "safety_drift_by_link_mps": "-0.1|0.0", "collision_pybullet_contact": "1", "collision_contact_link_names": "forearm_link", "collision_min_contact_distance": "-0.001"})

    cases = audit_trace_dir(tmp_path)

    assert len(cases) == 1
    assert cases[0]["h_min_delta_m"] == 0.05
    assert cases[0]["physical_contact"] is True
