"""Replay and audit the frozen G2 v2 strict-projection failures without training."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

import rl_risk_sac.envs.ur5_dynamic_obstacle_env as environment_module
from rl_risk_sac.algorithms import SACAgent
from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config
from rl_risk_sac.utils.device import resolve_device
from rl_risk_sac.utils.safety_filter import (
    SafetyFilterInput,
    SafetyFilterResult,
    SafetyFilterStatus,
    _build_constraints,
    _constraint_category,
)
from rl_risk_sac.utils.seeding import set_seed

try:
    from scripts.compare_vaps_v0_v1 import _load_checkpoint
except ModuleNotFoundError:
    from compare_vaps_v0_v1 import _load_checkpoint


PROTOCOL = "vaps_g2_projection_failure_audit_v1"
SOURCE_PROTOCOL = "vaps_g2_strict_execution_v2"
SOURCE_STATUS = SafetyFilterStatus.SAFE_STOP_PROJECTION_FAILED.value
SOURCE_TRACE_PATTERN = re.compile(r"train_seed_(\d+)_(?:validation|final)_trace\.csv$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the frozen G2 v2 strict-projection failures and verify fail-safe semantics."
    )
    parser.add_argument("--config", default="configs/experiments/vaps/v2_g2_strict_execution.yaml")
    parser.add_argument(
        "--checkpoint-manifest",
        default="configs/experiments/vaps_manifests/v2_b4_selected_checkpoints.json",
    )
    parser.add_argument(
        "--target-manifest",
        default="configs/experiments/vaps_manifests/v1_g2_projection_failure_targets.json",
    )
    parser.add_argument(
        "--target-index",
        type=int,
        default=None,
        help="Replay one zero-based frozen target in a dedicated process; omit only when the platform supports all targets.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _target_key(target: dict[str, Any]) -> tuple[int, int, int]:
    values = tuple(target.get(field) for field in ("train_seed", "seed", "step"))
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("every target must define integer train_seed, seed, and step")
    train_seed, seed, step = values
    if train_seed <= 0 or seed <= 0 or step < 0:
        raise ValueError("target train_seed/seed must be positive and step must be non-negative")
    return train_seed, seed, step


def load_target_manifest(path: str | Path) -> tuple[list[dict[str, int]], dict[str, str]]:
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict) or payload.get("protocol") != PROTOCOL:
        raise ValueError("target manifest has the wrong protocol")
    if payload.get("source_protocol") != SOURCE_PROTOCOL:
        raise ValueError("target manifest has the wrong source protocol")
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("target manifest must contain a non-empty targets list")
    targets: list[dict[str, int]] = []
    keys: set[tuple[int, int, int]] = set()
    for raw_target in raw_targets:
        if not isinstance(raw_target, dict):
            raise ValueError("target manifest entries must be objects")
        key = _target_key(raw_target)
        if key in keys:
            raise ValueError("target manifest contains duplicate targets")
        keys.add(key)
        targets.append({"train_seed": key[0], "seed": key[1], "step": key[2]})
    raw_hashes = payload.get("source_trace_sha256")
    if not isinstance(raw_hashes, dict) or not raw_hashes:
        raise ValueError("target manifest must contain source_trace_sha256")
    trace_hashes: dict[str, str] = {}
    for raw_path, digest in raw_hashes.items():
        if not isinstance(raw_path, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("source_trace_sha256 entries must be path/SHA-256 strings")
        trace_hashes[raw_path] = digest
    return targets, trace_hashes


def source_projection_failures(
    targets: Sequence[dict[str, int]],
    trace_hashes: dict[str, str],
) -> dict[tuple[int, int, int], dict[str, Any]]:
    expected_keys = {_target_key(target) for target in targets}
    events: dict[tuple[int, int, int], dict[str, Any]] = {}
    for raw_path, expected_hash in trace_hashes.items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"source G2 trace is missing: {path}")
        if _sha256(path) != expected_hash:
            raise ValueError(f"source G2 trace hash mismatch: {path}")
        match = SOURCE_TRACE_PATTERN.search(path.name)
        if match is None:
            raise ValueError(f"source trace name does not identify a train seed: {path}")
        train_seed = int(match.group(1))
        with open(path, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("safety_filter_status") != SOURCE_STATUS:
                    continue
                if row.get("v1_safety_filter_status") != SOURCE_STATUS:
                    raise ValueError(f"source failure is not V0/V1-equivalent: {path}")
                if row.get("passed") != "True" or row.get("field_mismatches") != "[]":
                    raise ValueError(f"source failure has an execution mismatch: {path}")
                key = (train_seed, int(row["seed"]), int(row["step"]))
                if key in events:
                    raise ValueError(f"duplicate source failure event: {key}")
                events[key] = {"source_trace": str(path), **row}
    if set(events) != expected_keys:
        missing = sorted(expected_keys - set(events))
        unexpected = sorted(set(events) - expected_keys)
        raise ValueError(f"source failures do not exactly match the frozen targets; missing={missing}, unexpected={unexpected}")
    return events


def _vector(values: np.ndarray) -> list[float]:
    return np.asarray(values, dtype=np.float64).tolist()


def _max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        return float("inf")
    return float(np.max(np.abs(left_array - right_array), initial=0.0))


def _constraint_snapshot(filter_input: SafetyFilterInput, env: UR5DynamicObstacleEnv) -> dict[str, Any]:
    lower, upper, requested, rows, bounds, labels, lower_labels, upper_labels = _build_constraints(
        filter_input, env.safety_filter_config
    )
    zero_command = np.zeros_like(requested)
    zero_residual = bounds - rows @ zero_command
    zero_violation = np.maximum(zero_residual, 0.0)
    lower_violation = np.maximum(lower - zero_command, 0.0)
    upper_violation = np.maximum(zero_command - upper, 0.0)
    all_labels = [*labels, *lower_labels, *upper_labels]
    all_violations = np.concatenate((zero_violation, lower_violation, upper_violation))
    max_index = int(np.argmax(all_violations))
    max_label = all_labels[max_index]
    obstacle = env._obstacle_state_estimate()
    risk = filter_input.predictive_risk
    return {
        "joint_positions_rad": _vector(filter_input.joint_positions_rad),
        "previous_command_radps": _vector(filter_input.previous_command_radps),
        "requested_joint_velocity_radps": _vector(requested),
        "joint_lower_radps": _vector(lower),
        "joint_upper_radps": _vector(upper),
        "constraint_labels": list(labels),
        "constraint_lower_bounds": _vector(bounds),
        "zero_command_constraint_residuals": _vector(zero_residual),
        "zero_command_constraint_violations": _vector(zero_violation),
        "zero_command_max_constraint_violation": float(all_violations[max_index]),
        "zero_command_max_constraint_label": max_label,
        "zero_command_max_constraint_category": _constraint_category(max_label),
        "zero_command_satisfies_all_constraints": bool(
            all_violations[max_index] <= env.safety_filter_config.projection_failure_tolerance
        ),
        "predictive_risk_status": risk.status.value,
        "predictive_risk_age_s": float(risk.observation_age_s),
        "predictive_h_by_link_m": _vector(risk.safety_functions_m),
        "predictive_predicted_distance_by_link_m": _vector(risk.predicted_distances_m),
        "predictive_robust_distance_by_link_m": _vector(risk.robust_distances_m),
        "predictive_link_velocity_norms_mps": _vector(np.linalg.norm(risk.link_velocities_mps, axis=1)),
        "obstacle_position_m": _vector(obstacle.position),
        "obstacle_velocity_mps": _vector(obstacle.velocity),
        "capsules": [
            {
                "name": spec.name,
                "start_m": _vector(capsule.start),
                "end_m": _vector(capsule.end),
                "radius_m": float(capsule.radius),
            }
            for spec, capsule in zip(env.capsule_model.specs, env._capsules(), strict=True)
        ],
    }


def _result_snapshot(result: SafetyFilterResult) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "reason": result.reason,
        "command_joint_velocity_radps": _vector(result.command_joint_velocity_radps),
        "intervention_norm_radps": float(result.intervention_norm_radps),
        "active_constraint_count": int(result.active_constraint_count),
        "constraint_count": int(result.constraint_count),
        "active_constraint_categories": list(result.active_constraint_categories),
        "max_constraint_violation": float(result.max_constraint_violation),
        "max_constraint_category": result.max_constraint_category,
        "projection_iterations": int(result.projection_iterations),
        "fallback_used": bool(result.fallback_used),
        "qp_solver_used": bool(result.qp_solver_used),
        "qp_solver_status": result.qp_solver_status,
        "fallback_stage": result.fallback_stage,
        "infeasible_constraint_categories": list(result.infeasible_constraint_categories),
        "infeasibility_diagnostic_status": result.infeasibility_diagnostic_status,
    }


def _replay_target(
    v0: UR5DynamicObstacleEnv,
    v1: UR5DynamicObstacleEnv,
    agent: SACAgent,
    target: dict[str, int],
    source: dict[str, Any],
    *,
    atol: float,
) -> dict[str, Any]:
    captured: dict[str, dict[str, Any]] = {}
    current_step = -1
    active_version = ""
    original_filter = environment_module.filter_joint_velocity

    def capture_filter(filter_input: SafetyFilterInput, filter_config: Any) -> SafetyFilterResult:
        result = original_filter(filter_input, filter_config)
        if current_step == target["step"]:
            if active_version not in {"v0", "v1"} or active_version in captured:
                raise RuntimeError("target control step did not identify exactly one V0/V1 filter call")
            active_env = v0 if active_version == "v0" else v1
            captured[active_version] = {
                "input": filter_input,
                "result": result,
                "constraints": _constraint_snapshot(filter_input, active_env),
            }
        return result

    environment_module.filter_joint_velocity = capture_filter
    try:
        observation_v0, _ = v0.reset(seed=target["seed"])
        observation_v1, _ = v1.reset(seed=target["seed"])
        for current_step in range(target["step"] + 1):
            action_v0 = agent.select_action(observation_v0, deterministic=True)
            action_v1 = agent.select_action(observation_v1, deterministic=True)
            active_version = "v0"
            next_v0, _, _, terminated_v0, truncated_v0, _ = v0.step(action_v0)
            active_version = "v1"
            next_v1, _, _, terminated_v1, truncated_v1, _ = v1.step(action_v1)
            if current_step < target["step"] and (terminated_v0 or truncated_v0 or terminated_v1 or truncated_v1):
                raise RuntimeError(f"episode ended before frozen target {target}")
            observation_v0, observation_v1 = next_v0, next_v1
        if set(captured) != {"v0", "v1"}:
            raise RuntimeError(f"safety-filter inputs were not captured for both versions at frozen target {target}")
    finally:
        environment_module.filter_joint_velocity = original_filter

    v0_input = captured["v0"]["input"]
    v0_result = captured["v0"]["result"]
    v1_input = captured["v1"]["input"]
    v1_result = captured["v1"]["result"]
    constraints = captured["v1"]["constraints"]
    source_requested = np.asarray(json.loads(source["qdot_requested_radps"]), dtype=np.float64)
    source_command = np.asarray(json.loads(source["qdot_cmd_radps"]), dtype=np.float64)
    requested = np.asarray(v1_input.requested_joint_velocity_radps, dtype=np.float64)
    command = np.asarray(v1_result.command_joint_velocity_radps, dtype=np.float64)
    v0_requested = np.asarray(v0_input.requested_joint_velocity_radps, dtype=np.float64)
    v0_command = np.asarray(v0_result.command_joint_velocity_radps, dtype=np.float64)
    command_is_zero = bool(np.max(np.abs(command), initial=0.0) <= atol)
    status_reproduced = v0_result.status.value == SOURCE_STATUS and v1_result.status.value == SOURCE_STATUS
    source_requested_difference = _max_abs_difference(v0_requested, source_requested)
    source_command_difference = _max_abs_difference(v0_command, source_command)
    v0_v1_requested_difference = _max_abs_difference(v0_requested, requested)
    v0_v1_command_difference = _max_abs_difference(v0_command, command)
    source_command_match = source_command_difference <= atol
    versions_match = v0_v1_requested_difference <= atol and v0_v1_command_difference <= atol
    if not status_reproduced or not command_is_zero or not source_command_match or not versions_match:
        raise RuntimeError(
            f"frozen target did not reproduce strict fail-safe semantics: {target}; "
            f"status={v1_result.status.value}, command_is_zero={command_is_zero}, "
            f"source_command_match={source_command_match}, versions_match={versions_match}, "
            f"source_requested_max_abs_diff={source_requested_difference}, "
            f"source_command_max_abs_diff={source_command_difference}, "
            f"v0_v1_requested_max_abs_diff={v0_v1_requested_difference}, "
            f"v0_v1_command_max_abs_diff={v0_v1_command_difference}"
        )
    return {
        **target,
        "source_trace": source["source_trace"],
        "source_viability_status": source["viability_status"],
        "source_viability_solver_status": source["viability_solver_status"],
        "source_requested_max_abs_diff": source_requested_difference,
        "source_command_max_abs_diff": source_command_difference,
        "v0_v1_requested_max_abs_diff": v0_v1_requested_difference,
        "v0_v1_command_max_abs_diff": v0_v1_command_difference,
        "source_requested_exactly_reproduced": bool(source_requested_difference <= atol),
        "source_command_reproduced": source_command_match,
        "v0_v1_execution_reproduced": versions_match,
        "status_reproduced": status_reproduced,
        "command_is_zero": command_is_zero,
        "command_differs_from_requested": bool(_max_abs_difference(command, requested) > atol),
        "filter_result": _result_snapshot(v1_result),
        "constraints": constraints,
    }


def audit_projection_failures(
    config_path: str | Path,
    checkpoint_manifest: str | Path,
    target_manifest: str | Path,
    *,
    atol: float = 1.0e-7,
    target_index: int | None = None,
) -> dict[str, Any]:
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and non-negative")
    all_targets, trace_hashes = load_target_manifest(target_manifest)
    source_events = source_projection_failures(all_targets, trace_hashes)
    if target_index is None:
        targets = all_targets
    else:
        if target_index < 0 or target_index >= len(all_targets):
            raise ValueError(f"target_index must be in [0, {len(all_targets) - 1}]")
        targets = [all_targets[target_index]]
    config = load_config(config_path)
    filter_cfg = config["env"]["safety_filter"]
    if config["eval"]["method"] != "link_fixed" or not bool(filter_cfg["enabled"]):
        raise ValueError("projection-failure audit requires enabled link_fixed strict filtering")
    if not bool(filter_cfg["use_qp_solver"]) or bool(filter_cfg["recovery_mode_enabled"]):
        raise ValueError("projection-failure audit requires strict QP filtering with recovery disabled")
    if filter_cfg.get("max_filter_compute_time_s") is not None:
        raise ValueError("projection-failure audit requires the deterministic G2 timing configuration")
    config["device"] = resolve_device(config)
    set_seed(int(config["seed"]))

    grouped_targets: dict[int, list[dict[str, int]]] = defaultdict(list)
    for target in targets:
        grouped_targets[target["train_seed"]].append(target)
    events: list[dict[str, Any]] = []
    for train_seed in sorted(grouped_targets):
        actor = _load_checkpoint(checkpoint_manifest, train_seed)
        for target in sorted(grouped_targets[train_seed], key=lambda item: (item["seed"], item["step"])):
            v0_config = copy.deepcopy(config)
            v0_config["env"]["safety_filter"]["viability_monitor_enabled"] = False
            v1_config = copy.deepcopy(config)
            v0 = UR5DynamicObstacleEnv(v0_config, method="link_fixed")
            v1 = UR5DynamicObstacleEnv(v1_config, method="link_fixed")
            agent = SACAgent(v0.observation_space.shape[0], v0.action_space.shape[0], config, method="link_fixed")
            agent.load_actor(actor["checkpoint"])
            try:
                key = _target_key(target)
                events.append(_replay_target(v0, v1, agent, target, source_events[key], atol=atol))
            finally:
                v0.close()
                v1.close()

    zero_constraint_feasible_count = sum(
        bool(event["constraints"]["zero_command_satisfies_all_constraints"]) for event in events
    )
    strict_fail_safe_semantics_verified = (
        all(event["status_reproduced"] for event in events)
        and all(event["command_is_zero"] for event in events)
        and all(event["source_command_max_abs_diff"] <= atol for event in events)
        and all(event["v0_v1_execution_reproduced"] for event in events)
    )
    return {
        "protocol": PROTOCOL,
        "source_protocol": SOURCE_PROTOCOL,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "resolved_config_sha256": _canonical_sha256(load_config(config_path)),
        "checkpoint_manifest": str(checkpoint_manifest),
        "checkpoint_manifest_sha256": _sha256(checkpoint_manifest),
        "target_manifest": str(target_manifest),
        "target_manifest_sha256": _sha256(target_manifest),
        "source_trace_sha256": trace_hashes,
        "atol": atol,
        "training_performed": False,
        "actor_checkpoint_selection_performed": False,
        "target_count": len(events),
        "total_frozen_target_count": len(all_targets),
        "target_index": target_index,
        "source_projection_failure_count": len(source_events),
        "all_statuses_reproduced": all(event["status_reproduced"] for event in events),
        "all_commands_zero": all(event["command_is_zero"] for event in events),
        "all_source_commands_reproduced": all(event["source_command_max_abs_diff"] <= atol for event in events),
        "all_v0_v1_execution_reproduced": all(event["v0_v1_execution_reproduced"] for event in events),
        "source_requested_exact_reproduction_count": sum(
            bool(event["source_requested_exactly_reproduced"]) for event in events
        ),
        "max_source_requested_abs_diff": max(
            event["source_requested_max_abs_diff"] for event in events
        ),
        "zero_command_constraint_feasible_count": zero_constraint_feasible_count,
        "strict_fail_safe_semantics_verified": strict_fail_safe_semantics_verified,
        "projection_failures_resolved": False,
        "g3_authorization": "not_granted_by_this_audit",
        "events": events,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = audit_projection_failures(
        args.config,
        args.checkpoint_manifest,
        args.target_manifest,
        atol=1.0e-7,
        target_index=args.target_index,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "targets": payload["target_count"],
                "target_index": payload["target_index"],
                "all_statuses_reproduced": payload["all_statuses_reproduced"],
                "all_commands_zero": payload["all_commands_zero"],
                "strict_fail_safe_semantics_verified": payload["strict_fail_safe_semantics_verified"],
                "projection_failures_resolved": payload["projection_failures_resolved"],
                "zero_command_constraint_feasible_count": payload["zero_command_constraint_feasible_count"],
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()
