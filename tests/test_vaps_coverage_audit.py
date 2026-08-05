import json

from scripts.audit_vaps_coverage import (
    DEFAULT_EPISODES,
    DEFAULT_SEED_START,
    audit_vaps_coverage,
    coverage_shard,
    coverage_seeds,
)
from scripts.merge_vaps_coverage import merge_vaps_coverage


def test_coverage_seeds_are_the_fixed_g1_manifest() -> None:
    seeds = coverage_seeds()

    assert len(seeds) == DEFAULT_EPISODES
    assert seeds[0] == DEFAULT_SEED_START
    assert seeds[-1] == DEFAULT_SEED_START + DEFAULT_EPISODES - 1


def test_coverage_shards_partition_the_fixed_g1_manifest() -> None:
    shards = [coverage_shard(index, 4) for index in range(4)]

    assert sorted(seed for shard in shards for seed in shard) == coverage_seeds()
    assert {len(shard) for shard in shards} == {250}


def test_coverage_audit_is_reset_only_and_records_viability_fields(monkeypatch) -> None:
    import scripts.audit_vaps_coverage as audit_module

    fixed_seeds = [DEFAULT_SEED_START, DEFAULT_SEED_START + 1]
    monkeypatch.setattr(audit_module, "coverage_seeds", lambda: fixed_seeds)

    result = audit_vaps_coverage("configs/experiments/vaps/v1_g1_coverage.yaml", fixed_seeds)

    assert result["actor_loaded"] is False
    assert result["training_performed"] is False
    assert result["episodes"] == 2
    assert sum(result["viability_status_counts"].values()) == 2
    assert result["no_nonfinite_valid_margins"] is True
    assert result["no_nonfinite_timing"] is True
    assert result["compute_budget_safe_stop_count"] == 0
    assert result["strict_projection_failure_count"] == 0
    for row in result["episodes_detail"]:
        assert row["viability_status"] in result["viability_status_counts"]
        assert len(row["qdot_requested_radps"]) == len(row["qdot_cmd_radps"])
        assert row["link_names"]
        assert row["safety_filter_solve_time_s"] >= 0.0


def test_merge_requires_complete_non_overlapping_shards(tmp_path) -> None:
    seeds = [DEFAULT_SEED_START, DEFAULT_SEED_START + 1]
    common = {
        "protocol": "vaps_g1_reset_only_coverage_v1",
        "config": "configs/experiments/vaps/v1_g1_coverage.yaml",
        "method": "link_fixed",
        "actor_loaded": False,
        "training_performed": False,
        "seed_manifest": {"seeds": seeds},
    }
    paths = []
    for index, seed in enumerate(seeds):
        payload = {
            **common,
            "shard": {"index": index, "count": 2, "executed_seeds": [seed]},
            "episodes_detail": [
                {
                    "seed": seed,
                    "viability_status": "certified_viable",
                    "viability_min_h_m": 0.01,
                    "safety_filter_status": "passthrough",
                    "safety_filter_solve_time_s": 0.01,
                    "safety_filter_predictive_risk_time_s": 0.002,
                    "safety_filter_jacobian_workspace_time_s": 0.003,
                    "safety_filter_projection_time_s": 0.004,
                }
            ],
        }
        path = tmp_path / f"shard_{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    result = merge_vaps_coverage(paths)

    assert result["episodes"] == 2
    assert result["certified_viable_coverage_rate"] == 1.0
    assert [row["seed"] for row in result["episodes_detail"]] == seeds


def test_g1_iteration_cap_resolves_seed_10965_as_strict_infeasible(monkeypatch) -> None:
    import scripts.audit_vaps_coverage as audit_module

    monkeypatch.setattr(audit_module, "coverage_seeds", lambda: [10965])

    result = audit_vaps_coverage("configs/experiments/vaps/v1_g1_coverage.yaml", [10965])

    row = result["episodes_detail"][0]
    assert row["safety_filter_status"] == "safe_stop_infeasible"
    assert row["viability_status"] == "model_infeasible_static_or_slow"
    assert row["safety_filter_qp_solver_status"] == "primal infeasible"
