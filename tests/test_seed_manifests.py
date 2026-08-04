import json
from pathlib import Path

import pytest

from scripts.build_seed_manifests import feasible_seeds
from scripts.seed_manifest import load_seed_manifest


def test_feasible_seeds_uses_strictly_positive_initial_margin() -> None:
    audit = {
        "conditions": [
            {
                "episodes_detail": [
                    {"seed": 1, "initial_h_min_m": 0.01, "initially_unsafe": False},
                    {"seed": 2, "initial_h_min_m": 0.0, "initially_unsafe": True},
                    {"seed": 3, "initial_h_min_m": -0.01, "initially_unsafe": True},
                ]
            }
        ]
    }

    seeds, _ = feasible_seeds(audit)

    assert seeds == [1]


def test_seed_manifest_rejects_duplicate_or_non_integer_seeds(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps({"seeds": [1, 1]}), encoding="utf-8")
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"seeds": [1, "2"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        load_seed_manifest(duplicate)
    with pytest.raises(ValueError, match="integers"):
        load_seed_manifest(invalid)


def test_b5_feasible_manifests_are_disjoint_and_cover_audit_feasible_seeds() -> None:
    audit_path = Path("outputs/p3_postfix_dev/initial_feasibility_b5_seed7101_n200.json")
    with open(audit_path, encoding="utf-8") as file:
        audit = json.load(file)
    expected, _ = feasible_seeds(audit)
    validation_path = Path("configs/experiments/p3_manifests/b5_robust_feasible_validation.json")
    final_path = Path("configs/experiments/p3_manifests/b5_robust_feasible_final.json")
    validation = load_seed_manifest(validation_path)
    final = load_seed_manifest(final_path)

    assert len(validation) == 20
    assert len(final) == 144
    assert not set(validation).intersection(final)
    assert validation + final == expected
