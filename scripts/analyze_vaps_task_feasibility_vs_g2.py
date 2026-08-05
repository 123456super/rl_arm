"""Group frozen G2 execution events by the offline task-feasibility labels.

This is an explanatory audit only.  G2 lock-step traces do not contain the
environment's success field, so this script deliberately reports collision,
safe-stop and episode-end events rather than inventing a success rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any


TRACE_NAME_RE = re.compile(r"train_seed_(?P<train_seed>\d+)_(?P<role>final|validation)_trace\.csv$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group frozen VAPS G2 trace events by task-feasibility labels."
    )
    parser.add_argument("--feasibility", required=True, help="Merged task-feasibility JSON.")
    parser.add_argument(
        "--trace",
        action="append",
        required=True,
        metavar="TRAIN_SEED=TRACE_CSV",
        help="One frozen G2 trace per actor, for example 4108=outputs/...csv.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-episode-steps", type=int, default=240)
    return parser.parse_args(argv)


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _parse_trace_arg(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise ValueError(f"trace must use TRAIN_SEED=TRACE_CSV: {value}")
    seed_text, path_text = value.split("=", 1)
    try:
        train_seed = int(seed_text)
    except ValueError as error:
        raise ValueError(f"invalid train seed in --trace: {value}") from error
    path = Path(path_text)
    match = TRACE_NAME_RE.search(path.name)
    if match is not None and int(match.group("train_seed")) != train_seed:
        raise ValueError(f"trace filename train seed disagrees with --trace: {value}")
    return train_seed, path


def _as_bool(row: dict[str, str], field: str) -> bool:
    return str(row.get(field, "0")).strip().lower() in {"1", "true", "yes"}


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _episode_from_rows(train_seed: int, seed: int, rows: list[dict[str, str]], max_steps: int) -> dict[str, Any]:
    rows.sort(key=lambda row: int(row["step"]))
    termination_reasons = [str(row.get("termination_reason", "")) for row in rows if row.get("termination_reason", "")]
    physical_contact = any(_as_bool(row, "collision_pybullet_contact") for row in rows)
    capsule_overlap = any(_as_bool(row, "collision_capsule_overlap") for row in rows)
    any_collision = any(_as_bool(row, "collision_any") for row in rows)
    termination_collision = any(_as_bool(row, "termination_collision") for row in rows)
    safe_stop = any(_as_bool(row, "safety_filter_safe_stop") for row in rows)
    projection_failure = any(row.get("safety_filter_status") == "safe_stop_projection_failed" for row in rows)
    infeasible_stop = any(row.get("safety_filter_status") == "safe_stop_infeasible" for row in rows)
    steps = max(int(row["step"]) for row in rows) + 1
    if physical_contact or termination_collision or "pybullet_contact" in termination_reasons:
        end_type = "physical_contact"
    elif steps >= max_steps:
        end_type = "max_steps_or_unknown"
    else:
        end_type = "early_non_collision_end"
    return {
        "train_seed": train_seed,
        "seed": seed,
        "steps": steps,
        "end_type": end_type,
        "termination_reason": termination_reasons[-1] if termination_reasons else "",
        "physical_contact": physical_contact,
        "capsule_overlap": capsule_overlap,
        "any_collision": any_collision,
        "termination_collision": termination_collision,
        "safe_stop": safe_stop,
        "projection_failure": projection_failure,
        "infeasible_stop": infeasible_stop,
        "status_step_counts": dict(Counter(row.get("safety_filter_status", "") for row in rows)),
        "viability_step_counts": dict(Counter(row.get("viability_status", "") for row in rows)),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if not total:
        return {"episodes": 0}
    end_types = Counter(row["end_type"] for row in rows)
    status_counts: Counter[str] = Counter()
    viability_counts: Counter[str] = Counter()
    for row in rows:
        status_counts.update(row["status_step_counts"])
        viability_counts.update(row["viability_step_counts"])
    return {
        "episodes": total,
        "physical_contact_episodes": sum(row["physical_contact"] for row in rows),
        "physical_contact_episode_rate": _rate(sum(row["physical_contact"] for row in rows), total),
        "capsule_overlap_episodes": sum(row["capsule_overlap"] for row in rows),
        "capsule_overlap_episode_rate": _rate(sum(row["capsule_overlap"] for row in rows), total),
        "any_collision_episodes": sum(row["any_collision"] for row in rows),
        "any_collision_episode_rate": _rate(sum(row["any_collision"] for row in rows), total),
        "termination_collision_episodes": sum(row["termination_collision"] for row in rows),
        "safe_stop_episodes": sum(row["safe_stop"] for row in rows),
        "safe_stop_episode_rate": _rate(sum(row["safe_stop"] for row in rows), total),
        "projection_failure_episodes": sum(row["projection_failure"] for row in rows),
        "projection_failure_episode_rate": _rate(sum(row["projection_failure"] for row in rows), total),
        "infeasible_stop_episodes": sum(row["infeasible_stop"] for row in rows),
        "infeasible_stop_episode_rate": _rate(sum(row["infeasible_stop"] for row in rows), total),
        "mean_steps": float(sum(row["steps"] for row in rows) / total),
        "end_type_counts": dict(sorted(end_types.items())),
        "end_type_rates": {key: _rate(value, total) for key, value in sorted(end_types.items())},
        "status_step_counts": dict(sorted(status_counts.items())),
        "viability_step_counts": dict(sorted(viability_counts.items())),
    }


def analyze(
    feasibility_path: str | Path,
    trace_specs: Sequence[str],
    *,
    max_episode_steps: int,
) -> dict[str, Any]:
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    feasibility = _load_json(feasibility_path)
    episodes = feasibility.get("episodes_detail")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("feasibility JSON has no episodes_detail")
    labels_by_seed: dict[int, dict[str, Any]] = {}
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("feasibility episode must be an object")
        seed = int(episode["seed"])
        if seed in labels_by_seed:
            raise ValueError(f"duplicate feasibility seed: {seed}")
        labels_by_seed[seed] = episode

    parsed_specs = [_parse_trace_arg(spec) for spec in trace_specs]
    if len({train_seed for train_seed, _ in parsed_specs}) != len(parsed_specs):
        raise ValueError("duplicate train seed in --trace")
    all_events: list[dict[str, Any]] = []
    per_train_seed: dict[str, dict[str, Any]] = {}
    expected_seeds = set(labels_by_seed)
    for train_seed, trace_path in parsed_specs:
        if not trace_path.is_file():
            raise FileNotFoundError(trace_path)
        grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
        with open(trace_path, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                grouped[int(row["seed"])].append(row)
        if set(grouped) != expected_seeds:
            raise ValueError(
                f"trace {trace_path} does not cover feasibility seeds; "
                f"missing={sorted(expected_seeds - set(grouped))}, extra={sorted(set(grouped) - expected_seeds)}"
            )
        events = []
        for seed in sorted(expected_seeds):
            event = _episode_from_rows(train_seed, seed, grouped[seed], max_episode_steps)
            label = labels_by_seed[seed]
            event.update(
                {
                    "ik_status": label["ik"]["status"],
                    "no_obstacle_task_status": label["no_obstacle_task"]["status"],
                    "dynamic_obstacle_path_status": label["dynamic_obstacle_path"]["status"],
                }
            )
            events.append(event)
            all_events.append(event)
        per_train_seed[str(train_seed)] = {
            "trace": str(trace_path),
            "episodes": len(events),
            "groups": _group_events(events),
        }
    return {
        "protocol": "vaps_task_feasibility_vs_g2_execution_audit_v1",
        "feasibility": str(feasibility_path),
        "trace_train_seeds": [train_seed for train_seed, _ in parsed_specs],
        "trace_count": len(parsed_specs),
        "max_episode_steps": max_episode_steps,
        "episode_count_per_train_seed": len(expected_seeds),
        "success_rate_available": False,
        "success_rate_note": "G2 lock-step traces do not record environment success; end_type is not a success label.",
        "candidate_search_is_not_completeness_proof": bool(
            feasibility.get("candidate_search_is_not_completeness_proof", False)
        ),
        "groups_pooled": _group_events(all_events),
        "per_train_seed": per_train_seed,
        "episodes": sorted(all_events, key=lambda row: (row["train_seed"], row["seed"])),
    }


def _group_events(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    dimensions = (
        "ik_status",
        "no_obstacle_task_status",
        "dynamic_obstacle_path_status",
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    for dimension in dimensions:
        for value in sorted({str(event[dimension]) for event in events}):
            groups[f"{dimension}={value}"] = [event for event in events if str(event[dimension]) == value]
    for event in events:
        key = "|".join(f"{dimension}={event[dimension]}" for dimension in dimensions)
        groups.setdefault(key, []).append(event)
    return {key: _summarize(groups[key]) for key in sorted(groups)}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = analyze(
        args.feasibility,
        args.trace,
        max_episode_steps=args.max_episode_steps,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "episodes": len(payload["episodes"]),
                "trace_count": payload["trace_count"],
                "success_rate_available": payload["success_rate_available"],
            }
        )
    )


if __name__ == "__main__":
    main()
