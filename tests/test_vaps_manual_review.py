from scripts.prepare_vaps_g1_manual_review import prepare_manual_review
from rl_risk_sac.utils.safety_filter import ViabilityStatus


def _row(seed: int, status: str, h_min: float) -> dict[str, object]:
    return {
        "seed": seed,
        "viability_status": status,
        "viability_min_h_m": h_min,
        "link_names": ["shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3"],
        "initial_h_by_link_m": [h_min] * 6,
        "qdot_requested_radps": [0.0] * 6,
        "qdot_cmd_radps": [0.0] * 6,
        "obstacle_speed_mps": 0.05,
    }


def test_manual_review_selects_even_samples_and_records_zero_status_evidence() -> None:
    statuses = [ViabilityStatus.CERTIFIED_VIABLE.value] * 20
    rows = [_row(seed, status, (seed - 10001) / 100.0) for seed, status in zip(range(10001, 10021), statuses)]
    rows.extend(_row(seed, ViabilityStatus.MODEL_INFEASIBLE_STATIC_OR_SLOW.value, -0.1) for seed in range(10021, 10041))
    rows.extend(_row(seed, ViabilityStatus.CERTIFIED_VIABLE.value, 0.2) for seed in range(10041, 11001))
    payload = {
        "protocol": "vaps_g1_reset_only_coverage_v1",
        "episodes": 1000,
        "actor_loaded": False,
        "training_performed": False,
        "no_nonfinite_valid_margins": True,
        "no_nonfinite_timing": True,
        "seed_manifest": {"count": 1000},
        "status_counts": {},
        "episodes_detail": rows,
        "config": "configs/experiments/vaps/v1_g1_coverage.yaml",
    }
    payload["seed_manifest"]["seeds"] = list(range(10001, 11001))
    payload["viability_status_counts"] = {
        status.value: sum(row["viability_status"] == status.value for row in rows) for status in ViabilityStatus
    }

    result = prepare_manual_review(payload)

    cert = result["review_by_status"][ViabilityStatus.CERTIFIED_VIABLE.value]
    dynamic = result["review_by_status"][ViabilityStatus.MODEL_INFEASIBLE_DYNAMIC.value]
    assert cert["selected_count"] == 20
    assert dynamic["natural_count"] == 0
    assert dynamic["g0_evidence"]
