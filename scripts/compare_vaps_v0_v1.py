"""Strict, lock-step V0/V1 execution-equivalence check for VAPS G2."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rl_risk_sac.algorithms import SACAgent
from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config
from rl_risk_sac.utils.device import resolve_device
from rl_risk_sac.utils.seeding import set_seed

try:
    from scripts.seed_manifest import load_seed_manifest
except ModuleNotFoundError:
    from seed_manifest import load_seed_manifest


VIABILITY_FIELDS = {
    "viability_status",
    "viability_horizon_s",
    "viability_min_h_m",
    "viability_strict_feasible",
    "viability_solver_status",
    "viability_model_assumptions_valid",
}
PROTOCOL = "vaps_g2_strict_execution_v2"
EQUIVALENCE_FIELDS = (
    "collision_capsule_overlap",
    "collision_pybullet_contact",
    "collision_any",
    "termination_collision",
    "termination_reason",
    "safety_filter_status",
    "safety_filter_safe_stop",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare V0 and V1 strict commands in lock-step.")
    parser.add_argument("--config", default="configs/experiments/vaps/v2_g2_strict_execution.yaml")
    parser.add_argument(
        "--checkpoint-manifest",
        default="configs/experiments/vaps_manifests/v2_b4_selected_checkpoints.json",
    )
    parser.add_argument("--train-seed", required=True, type=int)
    parser.add_argument("--seed-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace-output", required=True)
    parser.add_argument("--atol", type=float, default=1.0e-7)
    return parser.parse_args(argv)


def _load_checkpoint(checkpoint_manifest: str | Path, train_seed: int) -> dict[str, Any]:
    path = Path(checkpoint_manifest)
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("checkpoint manifest has the wrong protocol")
    if payload.get("method") != "link_fixed":
        raise ValueError("checkpoint manifest method must be link_fixed")
    matches = [actor for actor in payload.get("actors", []) if actor.get("train_seed") == train_seed]
    if len(matches) != 1:
        raise ValueError(f"checkpoint manifest must contain exactly one actor for train seed {train_seed}")
    actor = matches[0]
    checkpoint = Path(actor["checkpoint"])
    selection_record = Path(actor["selection_record"])
    if not checkpoint.is_file() or not selection_record.is_file():
        raise FileNotFoundError(f"checkpoint or selection record is missing for train seed {train_seed}")
    with open(selection_record, newline="", encoding="utf-8") as file:
        selected_rows = [
            row
            for row in csv.DictReader(file)
            if str(row.get("selected", "")).strip().lower() in {"1", "true", "yes"}
        ]
    if len(selected_rows) != 1:
        raise ValueError(f"selection record must contain exactly one selected checkpoint for train seed {train_seed}")
    selected = selected_rows[0]
    if int(selected["step"]) != int(actor["checkpoint_step"]):
        raise ValueError(f"selection record step does not match the frozen manifest for train seed {train_seed}")
    if Path(selected["checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError(f"selection record checkpoint does not match the frozen manifest for train seed {train_seed}")
    return actor


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _max_abs_difference(left: Any, right: Any) -> float:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.shape != right_values.shape:
        return float("inf")
    return float(np.max(np.abs(left_values - right_values), initial=0.0))


def compare_step(
    observation_v0: np.ndarray,
    observation_v1: np.ndarray,
    action_v0: np.ndarray,
    action_v1: np.ndarray,
    command_v0: np.ndarray,
    command_v1: np.ndarray,
    info_v0: dict[str, Any],
    info_v1: dict[str, Any],
    terminated_v0: bool,
    terminated_v1: bool,
    truncated_v0: bool,
    truncated_v1: bool,
    atol: float,
) -> dict[str, Any]:
    observation_difference = _max_abs_difference(observation_v0, observation_v1)
    action_difference = _max_abs_difference(action_v0, action_v1)
    command_difference = _max_abs_difference(command_v0, command_v1)
    requested_difference = _max_abs_difference(info_v0["qdot_requested"], info_v1["qdot_requested"])
    field_mismatches = [field for field in EQUIVALENCE_FIELDS if info_v0[field] != info_v1[field]]
    if terminated_v0 != terminated_v1:
        field_mismatches.append("terminated")
    if truncated_v0 != truncated_v1:
        field_mismatches.append("truncated")
    if not VIABILITY_FIELDS.issubset(info_v1):
        field_mismatches.append("v1_viability_fields")
    return {
        "observation_max_abs_diff": observation_difference,
        "action_max_abs_diff": action_difference,
        "qdot_cmd_max_abs_diff": command_difference,
        "qdot_requested_max_abs_diff": requested_difference,
        "field_mismatches": field_mismatches,
        "passed": (
            observation_difference <= atol
            and action_difference <= atol
            and command_difference <= atol
            and requested_difference <= atol
            and not field_mismatches
        ),
    }


def _validate_g2_config(config: dict[str, Any]) -> None:
    filter_cfg = config["env"]["safety_filter"]
    if config["eval"]["method"] != "link_fixed":
        raise ValueError("eval.method must be link_fixed")
    if not bool(filter_cfg.get("enabled", False)) or not bool(filter_cfg.get("viability_monitor_enabled", False)):
        raise ValueError("G2 requires an enabled V1 viability monitor")
    if not bool(filter_cfg.get("use_qp_solver", False)):
        raise ValueError("G2 requires the strict QP solver")
    if filter_cfg.get("max_filter_compute_time_s") is not None:
        raise ValueError("G2 requires max_filter_compute_time_s=null for deterministic lock-step execution")
    for key in (
        "recovery_mode_enabled",
        "recovery_allow_constraint_relaxation",
        "recovery_maximize_min_clearance",
        "risk_speed_scaling_enabled",
    ):
        if bool(filter_cfg.get(key, False)):
            raise ValueError(f"G2 requires env.safety_filter.{key}=false")


def _trace_row(seed: int, step: int, comparison: dict[str, Any], info_v0: dict[str, Any], info_v1: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": seed,
        "step": step,
        **comparison,
        "qdot_cmd_radps": json.dumps(np.asarray(info_v0["qdot_cmd"], dtype=np.float64).tolist()),
        "qdot_requested_radps": json.dumps(np.asarray(info_v0["qdot_requested"], dtype=np.float64).tolist()),
        "termination_reason": info_v0["termination_reason"],
        "collision_capsule_overlap": int(bool(info_v0["collision_capsule_overlap"])),
        "collision_pybullet_contact": int(bool(info_v0["collision_pybullet_contact"])),
        "collision_any": int(bool(info_v0["collision_any"])),
        "termination_collision": int(bool(info_v0["termination_collision"])),
        "safety_filter_status": info_v0["safety_filter_status"],
        "v1_safety_filter_status": info_v1["safety_filter_status"],
        "safety_filter_safe_stop": int(bool(info_v0["safety_filter_safe_stop"])),
        "v1_safety_filter_safe_stop": int(bool(info_v1["safety_filter_safe_stop"])),
        "viability_status": info_v1["viability_status"],
        "viability_solver_status": info_v1["viability_solver_status"],
    }


def compare_v0_v1(
    config_path: str | Path,
    checkpoint: str | Path,
    seeds: Sequence[int],
    *,
    atol: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and non-negative")
    config = load_config(config_path)
    _validate_g2_config(config)
    max_episode_steps = int(config["env"]["max_episode_steps"])
    if max_episode_steps <= 0:
        raise ValueError("env.max_episode_steps must be positive for G2")
    config["device"] = resolve_device(config)
    set_seed(int(config["seed"]))
    v0_config = copy.deepcopy(config)
    v0_config["env"]["safety_filter"]["viability_monitor_enabled"] = False
    v1_config = copy.deepcopy(config)
    v0 = UR5DynamicObstacleEnv(v0_config, method="link_fixed")
    v1 = UR5DynamicObstacleEnv(v1_config, method="link_fixed")
    agent = SACAgent(v0.observation_space.shape[0], v0.action_space.shape[0], config, method="link_fixed")
    agent.load_actor(checkpoint)
    trace_rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    try:
        for seed in seeds:
            observation_v0, _ = v0.reset(seed=int(seed))
            observation_v1, _ = v1.reset(seed=int(seed))
            maximums = {
                "observation_max_abs_diff": _max_abs_difference(observation_v0, observation_v1),
                "action_max_abs_diff": 0.0,
                "qdot_cmd_max_abs_diff": 0.0,
                "qdot_requested_max_abs_diff": 0.0,
            }
            episode_passed = maximums["observation_max_abs_diff"] <= atol
            termination_reason = ""
            for step in range(max_episode_steps):
                action_v0 = agent.select_action(observation_v0, deterministic=True)
                action_v1 = agent.select_action(observation_v1, deterministic=True)
                next_v0, _, _, terminated_v0, truncated_v0, info_v0 = v0.step(action_v0)
                next_v1, _, _, terminated_v1, truncated_v1, info_v1 = v1.step(action_v1)
                comparison = compare_step(
                    next_v0,
                    next_v1,
                    action_v0,
                    action_v1,
                    info_v0["qdot_cmd"],
                    info_v1["qdot_cmd"],
                    info_v0,
                    info_v1,
                    terminated_v0,
                    terminated_v1,
                    truncated_v0,
                    truncated_v1,
                    atol,
                )
                for field in maximums:
                    maximums[field] = max(maximums[field], float(comparison[field]))
                trace_rows.append(_trace_row(int(seed), step, comparison, info_v0, info_v1))
                episode_passed = episode_passed and bool(comparison["passed"])
                termination_reason = str(info_v0["termination_reason"])
                observation_v0, observation_v1 = next_v0, next_v1
                if terminated_v0 or truncated_v0 or terminated_v1 or truncated_v1:
                    break
            episodes.append(
                {
                    "seed": int(seed),
                    "steps": step + 1,
                    "passed": episode_passed,
                    "termination_reason": termination_reason,
                    **maximums,
                }
            )
            if not episode_passed:
                break
    finally:
        v0.close()
        v1.close()
    passed = len(episodes) == len(seeds) and all(bool(episode["passed"]) for episode in episodes)
    return {
        "passed": passed,
        "episodes_requested": len(seeds),
        "episodes_completed": len(episodes),
        "episodes": episodes,
    }, trace_rows


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    actor = _load_checkpoint(args.checkpoint_manifest, args.train_seed)
    seeds = load_seed_manifest(args.seed_manifest)
    result, trace_rows = compare_v0_v1(args.config, actor["checkpoint"], seeds, atol=args.atol)
    if not trace_rows:
        raise RuntimeError("G2 produced no step trace rows")
    payload = {
        "protocol": PROTOCOL,
        "config": str(args.config),
        "checkpoint_manifest": str(args.checkpoint_manifest),
        "train_seed": args.train_seed,
        "checkpoint": actor["checkpoint"],
        "checkpoint_step": actor["checkpoint_step"],
        "checkpoint_sha256": _sha256(actor["checkpoint"]),
        "selection_record": actor["selection_record"],
        "selection_record_sha256": _sha256(actor["selection_record"]),
        "seed_manifest": str(args.seed_manifest),
        "config_sha256": _sha256(args.config),
        "resolved_config_sha256": _canonical_sha256(load_config(args.config)),
        "checkpoint_manifest_sha256": _sha256(args.checkpoint_manifest),
        "seed_manifest_sha256": _sha256(args.seed_manifest),
        "atol": args.atol,
        **result,
    }
    output = Path(args.output)
    trace_output = Path(args.trace_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    trace_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with open(trace_output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(trace_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trace_rows)
    print(json.dumps({"passed": payload["passed"], "episodes": payload["episodes_completed"], "output": str(output)}))
    if not payload["passed"]:
        raise SystemExit("V0/V1 strict execution equivalence failed; see the summary and trace outputs")


if __name__ == "__main__":
    main()
