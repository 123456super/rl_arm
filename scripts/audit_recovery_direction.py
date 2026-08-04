from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit bounded-recovery direction and margin evolution.")
    parser.add_argument("--trace-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def vector(value: str) -> list[float]:
    return [float(item) for item in value.split("|") if item]


def audit_trace_dir(trace_dir: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for trace_path in sorted(trace_dir.glob("episode_*.csv")):
        with open(trace_path, newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        recovery_rows = [row for row in rows if row.get("recovery_active") == "1"]
        if not recovery_rows:
            continue
        first = recovery_rows[0]
        last = recovery_rows[-1]
        h_values = [float(row["predictive_h_min_m"]) for row in recovery_rows]
        command_norms = [float(row["recovery_command_norm"]) for row in recovery_rows]
        final_rows = [row for row in rows if row.get("collision_pybullet_contact") == "1"]
        collision = final_rows[0] if final_rows else None
        per_link_start = vector(first.get("predictive_h_by_link_m", ""))
        per_link_end = vector(last.get("predictive_h_by_link_m", ""))
        per_link_delta = [end - start for start, end in zip(per_link_start, per_link_end, strict=True)]
        jacobian_command = vector(first.get("safety_jacobian_command_by_link_mps", ""))
        drift = vector(first.get("safety_drift_by_link_mps", ""))
        cases.append(
            {
                "trace_dir": str(trace_dir),
                "episode": trace_path.stem,
                "eval_seed": int(first["seed"]) if first.get("seed") else None,
                "recovery_first_step": int(first["step"]),
                "recovery_last_step": int(last["step"]),
                "recovery_steps": len(recovery_rows),
                "recovery_trigger_reason": first.get("recovery_trigger_reason", ""),
                "recovery_success": bool(int(last.get("recovery_success", "0"))),
                "recovery_command_norm_max": max(command_norms),
                "h_min_start_m": h_values[0],
                "h_min_end_m": h_values[-1],
                "h_min_best_m": max(h_values),
                "h_min_delta_m": h_values[-1] - h_values[0],
                "per_link_h_start_m": per_link_start,
                "per_link_h_end_m": per_link_end,
                "per_link_h_delta_m": per_link_delta,
                "first_jacobian_command_by_link_mps": jacobian_command,
                "first_safety_drift_by_link_mps": drift,
                "physical_contact": collision is not None,
                "contact_step": int(collision["step"]) if collision else None,
                "contact_link_names": collision.get("collision_contact_link_names", "") if collision else "",
                "contact_min_distance_m": (
                    float(collision["collision_min_contact_distance"]) if collision else None
                ),
            }
        )
    return cases


def main() -> None:
    args = parse_args()
    cases: list[dict[str, Any]] = []
    for trace_dir in args.trace_dir:
        cases.extend(audit_trace_dir(Path(trace_dir)))
    result = {
        "trace_dirs": args.trace_dir,
        "recovery_episodes": len(cases),
        "recovery_successes": sum(case["recovery_success"] for case in cases),
        "physical_contact_recovery_episodes": sum(case["physical_contact"] for case in cases),
        "cases": cases,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
