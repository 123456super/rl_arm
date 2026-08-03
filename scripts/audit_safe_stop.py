from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit predictive-margin drift after strict safe-stop.")
    parser.add_argument("--trace-dir", action="append", required=True)
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument(
        "--dynamic-drift-threshold-mps",
        type=float,
        default=0.05,
        help="Classify an infeasible stop as dynamic when h drift is below -threshold.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def audit_trace_dir(
    trace_dir: Path, control_dt: float, dynamic_drift_threshold_mps: float = 0.05
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trace_path in sorted(trace_dir.glob("episode_*.csv")):
        with trace_path.open(newline="", encoding="utf-8") as file:
            trace = list(csv.DictReader(file))
        infeasible = [row for row in trace if row.get("safety_filter_status") == "safe_stop_infeasible"]
        if not infeasible:
            continue
        first = infeasible[0]
        last = infeasible[-1]
        first_step = int(first["step"])
        last_step = int(last["step"])
        duration_s = (last_step - first_step) * control_dt
        h_first = float(first["predictive_h_min_m"])
        h_last = float(last["predictive_h_min_m"])
        drift_mps = (h_last - h_first) / duration_s if duration_s > 0.0 else 0.0
        collision_any_rows = [
            row for row in trace if row.get("collision_any", row.get("collision")) == "1"
        ]
        capsule_overlap_rows = [row for row in trace if row.get("collision_capsule_overlap") == "1"]
        physical_contact_rows = [row for row in trace if row.get("collision_pybullet_contact") == "1"]
        termination_collision_rows = [
            row for row in trace if row.get("termination_collision", row.get("collision")) == "1"
        ]
        collision_step = int(collision_any_rows[0]["step"]) if collision_any_rows else None
        rows.append(
            {
                "trace_dir": str(trace_dir),
                "episode": trace_path.stem,
                "first_infeasible_step": first_step,
                "last_infeasible_step": last_step,
                "infeasible_steps": len(infeasible),
                "initial_h_min_m": h_first,
                "final_h_min_m": h_last,
                "h_drift_mps": drift_mps,
                "drift_class": (
                    "dynamic_drift" if drift_mps < -dynamic_drift_threshold_mps else "static_or_slow"
                ),
                "collision": bool(collision_any_rows),
                "collision_any": bool(collision_any_rows),
                "collision_capsule_overlap": bool(capsule_overlap_rows),
                "collision_pybullet_contact": bool(physical_contact_rows),
                "termination_collision": bool(termination_collision_rows),
                "collision_step": collision_step,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.control_dt <= 0.0:
        raise ValueError("--control-dt must be positive")
    if args.dynamic_drift_threshold_mps < 0.0:
        raise ValueError("--dynamic-drift-threshold-mps must be non-negative")
    rows: list[dict[str, object]] = []
    for trace_dir in args.trace_dir:
        rows.extend(
            audit_trace_dir(Path(trace_dir), args.control_dt, args.dynamic_drift_threshold_mps)
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=True, indent=2)
    dynamic = [row for row in rows if row["drift_class"] == "dynamic_drift"]
    print(
        json.dumps(
            {
                "episodes_with_infeasible_stop": len(rows),
                "dynamic_drift_episodes": len(dynamic),
                "static_or_slow_episodes": len(rows) - len(dynamic),
                "collision_any_episodes": sum(row["collision_any"] for row in rows),
                "capsule_overlap_episodes": sum(row["collision_capsule_overlap"] for row in rows),
                "physical_contact_episodes": sum(row["collision_pybullet_contact"] for row in rows),
                "termination_collision_episodes": sum(row["termination_collision"] for row in rows),
                "dynamic_drift_collision_any_episodes": sum(row["collision_any"] for row in dynamic),
                "output": str(output),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
