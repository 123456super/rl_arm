from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METHODS = (
    "b1_ee_current",
    "b2_link_current",
    "b3_link_predictive",
    "b4_predictive_nonrobust_filter",
    "b5_robust_predictive_filter",
)
TRAIN_SEEDS = (4108, 4109, 4110)
MAIN_METRICS = (
    "success",
    "collision_capsule_overlap",
    "collision_pybullet_contact",
    "safety_violation_rate",
    "safety_filter_intervention_rate",
    "safety_filter_safe_stop_rate",
    "safety_filter_infeasible_rate",
    "safety_filter_compute_budget_stop_rate",
    "reward",
    "final_position_error",
    "mean_safety_filter_solve_time_s",
    "max_safety_filter_solve_time_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an auditable P3 freeze-decision package.")
    parser.add_argument("--output", default="outputs/p3_postfix_dev_100k/p3_freeze_decision.json")
    return parser.parse_args()


def run_dir(method: str, train_seed: int) -> Path:
    latest = Path(f"outputs/p3_postfix_dev_100k/{method}/seed_{train_seed}/latest_run.txt")
    if not latest.is_file():
        raise FileNotFoundError(f"Missing latest run pointer: {latest}")
    return Path(latest.read_text(encoding="utf-8").strip())


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def per_seed_means(rows: list[dict[str, str]]) -> dict[str, float]:
    return {metric: mean(float(row[metric]) for row in rows) for metric in MAIN_METRICS}


def method_summary(method: str) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    for train_seed in TRAIN_SEEDS:
        directory = run_dir(method, train_seed)
        metrics_path = directory / "final_eval_b5_robust_feasible_144.csv"
        rows = read_csv(metrics_path)
        if len(rows) != 144 or len({row["seed"] for row in rows}) != 144:
            raise ValueError(f"Expected 144 distinct final-manifest rows: {metrics_path}")
        per_seed.append(
            {
                "train_seed": train_seed,
                "run_dir": str(directory),
                "eval_csv": str(metrics_path),
                "episodes": len(rows),
                "means": per_seed_means(rows),
            }
        )

    aggregate: dict[str, dict[str, float]] = {}
    for metric in MAIN_METRICS:
        values = [
            entry["means"][metric]
            for entry in per_seed
            if math.isfinite(entry["means"][metric])
        ]
        aggregate[metric] = (
            {"mean": mean(values), "sample_sd": stdev(values)}
            if len(values) >= 2
            else {"mean": None, "sample_sd": None}
        )
    return {"per_train_seed": per_seed, "aggregate_over_train_seeds": aggregate}


def b4_recovery_failure_summary() -> dict[str, Any]:
    strict_total = {"episodes": 0, "physical_contact": 0, "success": 0}
    recovery_total = {
        "episodes": 0,
        "physical_contact": 0,
        "success": 0,
        "recovery_triggered": 0,
        "recovery_success": 0,
        "infeasible_episode": 0,
        "compute_budget_episode": 0,
    }
    evidence: list[str] = []
    for train_seed in TRAIN_SEEDS:
        directory = run_dir("b4_predictive_nonrobust_filter", train_seed)
        strict_rows = read_csv(directory / "budget300_eval_b5_robust_feasible_144.csv")
        recovery_path = directory / "worst_link_escape_eval_b5_robust_feasible_144.csv"
        recovery_rows = read_csv(recovery_path)
        if {row["seed"] for row in strict_rows} != {row["seed"] for row in recovery_rows}:
            raise ValueError(f"Baseline/recovery seeds differ for train seed {train_seed}")
        strict_total["episodes"] += len(strict_rows)
        strict_total["physical_contact"] += sum(int(row["collision_pybullet_contact"]) for row in strict_rows)
        strict_total["success"] += sum(int(row["success"]) for row in strict_rows)
        recovery_total["episodes"] += len(recovery_rows)
        recovery_total["physical_contact"] += sum(
            int(row["collision_pybullet_contact"]) for row in recovery_rows
        )
        recovery_total["success"] += sum(int(row["success"]) for row in recovery_rows)
        recovery_total["recovery_triggered"] += sum(int(row["recovery_triggered"]) for row in recovery_rows)
        recovery_total["recovery_success"] += sum(int(row["recovery_success"]) for row in recovery_rows)
        recovery_total["infeasible_episode"] += sum(
            float(row["safety_filter_infeasible_rate"]) > 0.0 for row in recovery_rows
        )
        recovery_total["compute_budget_episode"] += sum(
            float(row["safety_filter_compute_budget_stop_rate"]) > 0.0 for row in recovery_rows
        )
        evidence.append(str(recovery_path))
    return {
        "comparison": "strict B4 300 ms baseline vs frozen-actor worst-link predictive-barrier relaxation",
        "strict_baseline": strict_total,
        "worst_link_recovery": recovery_total,
        "decision": "reject_worst_link_recovery",
        "reason": "It suppresses strict-QP infeasibility but increases physical contact on the shared final seed set.",
        "evidence_csvs": evidence,
    }


def reset_coverage() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in ("b4", "b5"):
        path = Path(f"outputs/p3_postfix_dev/initial_feasibility_{method}_seed7101_n200.json")
        with open(path, encoding="utf-8") as file:
            payload = json.load(file)
        condition = payload["conditions"][0]
        result[method] = {
            "audit": str(path),
            "episodes": condition["episodes"],
            "initially_unsafe_count": condition["initially_unsafe_count"],
            "initially_unsafe_rate": condition["initially_unsafe_rate"],
        }
    return result


def main() -> None:
    args = parse_args()
    main_comparison = {method: method_summary(method) for method in METHODS}
    payload = {
        "decision": "do_not_freeze_p3_or_expand_recovery",
        "protocol": {
            "final_manifest": "configs/experiments/p3_manifests/b5_robust_feasible_final.json",
            "episodes_per_train_seed": 144,
            "train_seeds": list(TRAIN_SEEDS),
            "timing_gate_ms": 300,
            "timing_note": "The simulator-side budget is a post-return check, not a hard deadline.",
        },
        "main_comparison": main_comparison,
        "b4_recovery_failure": b4_recovery_failure_summary(),
        "unconditional_reset_coverage": reset_coverage(),
        "next_action": "Keep strict B4 as the diagnostic baseline; do not tune or expand predictive-barrier relaxation recovery. Any successor requires a separately specified safety objective and evaluation protocol.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
