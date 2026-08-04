from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reproducible validation and final seed manifests from a feasibility audit."
    )
    parser.add_argument(
        "--audit",
        default="outputs/p3_postfix_dev/initial_feasibility_b5_seed7101_n200.json",
    )
    parser.add_argument(
        "--output-dir",
        default="configs/experiments/p3_manifests",
    )
    parser.add_argument("--validation-count", type=int, default=20)
    return parser.parse_args()


def feasible_seeds(audit: dict[str, Any]) -> tuple[list[int], dict[str, Any]]:
    conditions = audit.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != 1:
        raise ValueError("The feasibility audit must contain exactly one condition")
    condition = conditions[0]
    details = condition.get("episodes_detail")
    if not isinstance(details, list) or not details:
        raise ValueError("The feasibility audit has no episode details")

    feasible: list[int] = []
    for detail in details:
        if not isinstance(detail, dict):
            raise ValueError("Each feasibility audit episode must be an object")
        seed = detail.get("seed")
        initial_h = detail.get("initial_h_min_m")
        initially_unsafe = detail.get("initially_unsafe")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Each feasibility audit episode must define an integer seed")
        if not isinstance(initial_h, (int, float)) or isinstance(initial_h, bool):
            raise ValueError("Each feasibility audit episode must define initial_h_min_m")
        if bool(initially_unsafe) != (float(initial_h) <= 0.0):
            raise ValueError(f"Inconsistent initial feasibility flags for seed {seed}")
        if not bool(initially_unsafe) and float(initial_h) > 0.0:
            feasible.append(int(seed))

    if len(set(feasible)) != len(feasible):
        raise ValueError("The feasibility audit contains duplicate feasible seeds")
    return feasible, condition


def write_manifest(
    path: Path,
    *,
    seeds: list[int],
    role: str,
    audit_path: Path,
    audit: dict[str, Any],
    condition: dict[str, Any],
    total_feasible: int,
) -> None:
    payload = {
        "manifest_version": 1,
        "role": role,
        "source_audit": str(audit_path),
        "source_config": condition.get("config"),
        "method": condition.get("method"),
        "criterion": "initial_h_min_m > 0",
        "source_seed": audit.get("seed"),
        "source_episodes": audit.get("episodes_per_config"),
        "total_feasible": total_feasible,
        "selected_count": len(seeds),
        "seeds": seeds,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> None:
    args = parse_args()
    if args.validation_count <= 0:
        raise ValueError("--validation-count must be positive")
    audit_path = Path(args.audit)
    with open(audit_path, "r", encoding="utf-8") as file:
        audit = json.load(file)
    feasible, condition = feasible_seeds(audit)
    if len(feasible) <= args.validation_count:
        raise ValueError(
            f"Need seeds for both manifests; found {len(feasible)} feasible seeds and "
            f"validation count {args.validation_count}"
        )

    validation = feasible[: args.validation_count]
    final = feasible[args.validation_count :]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        output_dir / "b5_robust_feasible_validation.json",
        seeds=validation,
        role="validation",
        audit_path=audit_path,
        audit=audit,
        condition=condition,
        total_feasible=len(feasible),
    )
    write_manifest(
        output_dir / "b5_robust_feasible_final.json",
        seeds=final,
        role="final",
        audit_path=audit_path,
        audit=audit,
        condition=condition,
        total_feasible=len(feasible),
    )
    print(f"feasible seeds: {len(feasible)}")
    print(f"validation seeds: {len(validation)} -> {output_dir / 'b5_robust_feasible_validation.json'}")
    print(f"final seeds: {len(final)} -> {output_dir / 'b5_robust_feasible_final.json'}")


if __name__ == "__main__":
    main()
