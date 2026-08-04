import csv
from pathlib import Path

from scripts.audit_paired_physical_contacts import paired_contact_audit


def write_trace(path: Path, contacts: set[int]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["step", "collision_pybullet_contact", "predictive_h_min_m"],
        )
        writer.writeheader()
        for step in range(4):
            writer.writerow(
                {
                    "step": step,
                    "collision_pybullet_contact": int(step in contacts),
                    "predictive_h_min_m": float(step),
                }
            )


def test_paired_contact_audit_keeps_only_shared_contact_episodes(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    ablation = tmp_path / "ablation"
    baseline.mkdir()
    ablation.mkdir()
    write_trace(baseline / "episode_0000.csv", {3})
    write_trace(ablation / "episode_0000.csv", {2})
    write_trace(baseline / "episode_0001.csv", {3})
    write_trace(ablation / "episode_0001.csv", set())

    result = paired_contact_audit([baseline], [ablation], context_steps=1)

    assert result["baseline_physical_contact_episodes"] == 2
    assert result["ablation_physical_contact_episodes"] == 1
    assert result["shared_physical_contact_episodes"] == 1
    case = result["cases"][0]
    assert case["baseline_contact_step"] == 3
    assert case["ablation_contact_step"] == 2
    assert len(case["baseline_context"]) == 2
