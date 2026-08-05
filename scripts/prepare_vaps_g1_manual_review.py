"""Prepare a deterministic, reset-trace manual-review package for VAPS G1."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rl_risk_sac.utils.safety_filter import ViabilityStatus


REVIEW_PER_OBSERVED_STATUS = 20
EXPECTED_LINK_NAMES = ["shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3"]
ZERO_COUNT_EVIDENCE = {
    ViabilityStatus.MODEL_INFEASIBLE_DYNAMIC.value: [
        "tests/test_safety_filter_integration.py::test_safe_stop_drift_classifier_matches_audit_threshold",
        "tests/test_safety_filter.py::test_viability_classifies_strict_infeasibility_by_safe_stop_drift",
    ],
    ViabilityStatus.INVALID_OR_STALE_OBSERVATION.value: [
        "tests/test_safety_filter_integration.py::test_invalid_perception_estimate_forces_end_to_end_safe_stop",
        "tests/test_safety_filter_integration.py::test_stale_perception_estimate_forces_end_to_end_safe_stop",
    ],
    ViabilityStatus.UNKNOWN_COMPUTE_BUDGET.value: [
        "tests/test_safety_filter_integration.py::test_filter_compute_budget_forces_zero_velocity_after_an_overrun",
        "tests/test_safety_filter.py::test_projection_residual_is_reported_separately_from_confirmed_infeasibility",
    ],
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the VAPS G1 manual-review sample package.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _validate_input(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("protocol") != "vaps_g1_reset_only_coverage_v1":
        raise ValueError("input is not a VAPS G1 reset-only coverage result")
    if payload.get("episodes") != 1000:
        raise ValueError("manual review requires the complete 1000-reset result")
    if payload.get("actor_loaded") or payload.get("training_performed"):
        raise ValueError("manual review input must contain no actor or training")
    if payload.get("no_nonfinite_valid_margins") is not True or payload.get("no_nonfinite_timing") is not True:
        raise ValueError("manual review input has a non-finite margin or timing check failure")
    manifest = payload.get("seed_manifest")
    if not isinstance(manifest, dict) or manifest.get("count") != 1000:
        raise ValueError("manual review input must contain the fixed 1000-seed manifest")
    rows = payload.get("episodes_detail")
    if not isinstance(rows, list) or len(rows) != 1000:
        raise ValueError("manual review input must contain 1000 episode details")
    expected_seeds = list(range(10001, 11001))
    observed_seeds = sorted(row.get("seed") for row in rows)
    if observed_seeds != expected_seeds:
        raise ValueError("episode details do not exactly cover seeds 10001--11000")
    valid_statuses = {status.value for status in ViabilityStatus}
    for row in rows:
        if row.get("viability_status") not in valid_statuses:
            raise ValueError(f"unknown viability status in seed {row.get('seed')}")
        if row.get("link_names") != EXPECTED_LINK_NAMES:
            raise ValueError(f"unexpected link names in seed {row.get('seed')}")
        if len(row.get("initial_h_by_link_m") or []) != len(EXPECTED_LINK_NAMES):
            raise ValueError(f"initial h/link length mismatch in seed {row.get('seed')}")
        if len(row.get("qdot_requested_radps") or []) != len(row.get("qdot_cmd_radps") or []):
            raise ValueError(f"command length mismatch in seed {row.get('seed')}")
        if not np.isfinite(float(row["obstacle_speed_mps"])):
            raise ValueError(f"non-finite obstacle speed in seed {row.get('seed')}")
    return rows


def _select_even_quantiles(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"need {count} samples, found {len(rows)}")
    ordered = sorted(rows, key=lambda row: (float(row["viability_min_h_m"]), int(row["seed"])))
    indices = np.rint(np.linspace(0, len(ordered) - 1, count)).astype(int).tolist()
    return [ordered[index] for index in indices]


def prepare_manual_review(payload: dict[str, Any]) -> dict[str, Any]:
    rows = _validate_input(payload)
    status_counts = payload["viability_status_counts"]
    review_by_status: dict[str, dict[str, Any]] = {}
    for status in (item.value for item in ViabilityStatus):
        candidates = [row for row in rows if row["viability_status"] == status]
        selected = _select_even_quantiles(candidates, REVIEW_PER_OBSERVED_STATUS) if candidates else []
        review_by_status[status] = {
            "natural_count": len(candidates),
            "requested_sample_count": REVIEW_PER_OBSERVED_STATUS,
            "selected_count": len(selected),
            "selection": "h_min_sorted_even_quantiles_then_seed" if selected else "no_natural_samples",
            "g0_evidence": ZERO_COUNT_EVIDENCE.get(status, []) if not candidates else [],
            "samples": selected,
        }
    return {
        "protocol": "vaps_g1_manual_review_v1",
        "source_coverage_result": payload.get("config"),
        "source_seed_manifest": payload["seed_manifest"],
        "review_scope": "reset-only traces; no actor, training, or performance conclusion",
        "review_checklist": [
            "verify h_min and per-link h values use metres",
            "verify obstacle_velocity_mps and obstacle_speed_mps use metres per second",
            "verify qdot_*_radps use radians per second",
            "verify six link_names match the fixed frame-corrected capsule order",
            "verify viability_status agrees with safety_filter_status and solver status",
            "verify qdot_cmd_radps is zero for every safe-stop sample",
        ],
        "status_counts": status_counts,
        "review_by_status": review_by_status,
        "zero_count_statuses_require_g0_evidence": [
            status for status, detail in review_by_status.items() if detail["natural_count"] == 0
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    with open(args.input, encoding="utf-8") as file:
        payload = json.load(file)
    result = prepare_manual_review(payload)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status_counts": result["status_counts"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
