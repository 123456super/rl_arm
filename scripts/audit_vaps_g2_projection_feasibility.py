"""Diagnose frozen G2 projection failures on their original strict QP constraints.

This is an offline, target-by-target numerical audit.  It never changes the
environment command, enables recovery, relaxes a constraint, or authorizes G3.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import rl_risk_sac.envs.ur5_dynamic_obstacle_env as environment_module
from rl_risk_sac.algorithms import SACAgent
from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config
from rl_risk_sac.utils.device import resolve_device
from rl_risk_sac.utils.safety_filter import (
    SafetyFilterConfig,
    SafetyFilterInput,
    SafetyFilterResult,
    SafetyFilterStatus,
    _build_constraints,
    _constraint_category,
)
from rl_risk_sac.utils.seeding import set_seed

try:
    from scripts.audit_vaps_g2_projection_failures import (
        SOURCE_PROTOCOL,
        SOURCE_STATUS,
        _canonical_sha256,
        _max_abs_difference,
        _sha256,
        _target_key,
        load_target_manifest,
        source_projection_failures,
    )
    from scripts.compare_vaps_v0_v1 import _load_checkpoint
except ModuleNotFoundError:
    from audit_vaps_g2_projection_failures import (
        SOURCE_PROTOCOL,
        SOURCE_STATUS,
        _canonical_sha256,
        _max_abs_difference,
        _sha256,
        _target_key,
        load_target_manifest,
        source_projection_failures,
    )
    from compare_vaps_v0_v1 import _load_checkpoint


PROTOCOL = "vaps_g2_projection_feasibility_audit_v1"
ITERATION_BUDGETS = (2000, 10000, 50000)
DEFAULT_CONFIG = "configs/experiments/vaps/v2_g2_strict_execution.yaml"
DEFAULT_CHECKPOINT_MANIFEST = "configs/experiments/vaps_manifests/v2_b4_selected_checkpoints.json"
DEFAULT_TARGET_MANIFEST = "configs/experiments/vaps_manifests/v1_g2_projection_failure_targets.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one isolated offline feasibility diagnosis for a frozen G2 projection failure."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-manifest", default=DEFAULT_CHECKPOINT_MANIFEST)
    parser.add_argument("--target-manifest", default=DEFAULT_TARGET_MANIFEST)
    parser.add_argument(
        "--target-index",
        required=True,
        type=int,
        help="Zero-based frozen target index. Run each index in a separate process.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _vector(values: np.ndarray) -> list[float]:
    return np.asarray(values, dtype=np.float64).tolist()


def _validate_config(config: dict[str, Any]) -> None:
    filter_cfg = config["env"]["safety_filter"]
    if config["eval"]["method"] != "link_fixed":
        raise ValueError("projection-feasibility audit requires eval.method=link_fixed")
    if not bool(filter_cfg.get("enabled", False)) or not bool(filter_cfg.get("use_qp_solver", False)):
        raise ValueError("projection-feasibility audit requires enabled strict QP filtering")
    if filter_cfg.get("max_filter_compute_time_s") is not None:
        raise ValueError("projection-feasibility audit requires deterministic G2 timing configuration")
    for key in (
        "recovery_mode_enabled",
        "recovery_allow_constraint_relaxation",
        "recovery_maximize_min_clearance",
        "risk_speed_scaling_enabled",
    ):
        if bool(filter_cfg.get(key, False)):
            raise ValueError(f"projection-feasibility audit requires env.safety_filter.{key}=false")


def constraint_residual_audit(
    candidate: np.ndarray | None,
    lower: np.ndarray,
    upper: np.ndarray,
    rows: np.ndarray,
    bounds: np.ndarray,
    labels: Sequence[str],
    lower_labels: Sequence[str],
    upper_labels: Sequence[str],
    tolerance: float,
) -> dict[str, Any]:
    """Recheck a candidate against every unrelaxed original constraint."""

    if candidate is None:
        return {
            "candidate_available": False,
            "candidate_is_finite": False,
            "max_constraint_violation": None,
            "max_constraint_label": None,
            "max_constraint_category": None,
            "satisfies_all_constraints": False,
        }
    command = np.asarray(candidate, dtype=np.float64)
    if command.shape != lower.shape or not np.isfinite(command).all():
        return {
            "candidate_available": True,
            "candidate_is_finite": False,
            "max_constraint_violation": None,
            "max_constraint_label": None,
            "max_constraint_category": None,
            "satisfies_all_constraints": False,
        }
    violations = np.concatenate(
        (
            np.maximum(bounds - rows @ command, 0.0),
            np.maximum(lower - command, 0.0),
            np.maximum(command - upper, 0.0),
        )
    )
    all_labels = (*labels, *lower_labels, *upper_labels)
    if violations.size != len(all_labels):
        raise ValueError("constraint labels do not match the original constraint dimensions")
    maximum_index = int(np.argmax(violations))
    maximum_label = all_labels[maximum_index]
    maximum = float(violations[maximum_index])
    return {
        "candidate_available": True,
        "candidate_is_finite": True,
        "max_constraint_violation": maximum,
        "max_constraint_label": maximum_label,
        "max_constraint_category": _constraint_category(maximum_label),
        "satisfies_all_constraints": bool(maximum <= tolerance),
    }


def classify_attempt(status: str, residual: dict[str, Any]) -> str:
    """Classify feasibility only from an unrelaxed witness or OSQP certificate."""

    if residual["satisfies_all_constraints"]:
        return "strict_feasible"
    normalized_status = status.lower()
    if "primal infeasible" in normalized_status:
        return "primal_infeasible"
    if "maximum iterations reached" in normalized_status:
        return "indeterminate_max_iterations"
    return "indeterminate_solver_status"


def _run_osqp_projection(
    requested: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rows: np.ndarray,
    bounds: np.ndarray,
    config: SafetyFilterConfig,
) -> tuple[np.ndarray | None, str, int | None]:
    """Solve the same strict OSQP projection while retaining any finite iterate."""

    try:
        import osqp
        from scipy import sparse
    except ImportError as error:
        raise RuntimeError("projection-feasibility audit requires osqp and scipy") from error

    matrix = sparse.csc_matrix(np.vstack((rows, np.eye(requested.size))))
    lower_bound = np.concatenate((bounds, lower))
    upper_bound = np.concatenate((np.full(len(bounds), np.inf), upper))
    problem = osqp.OSQP()
    problem.setup(
        P=sparse.eye(requested.size, format="csc"),
        q=-requested,
        A=matrix,
        l=lower_bound,
        u=upper_bound,
        eps_abs=config.projection_failure_tolerance,
        eps_rel=config.projection_failure_tolerance,
        max_iter=max(400, config.fallback_projection_iterations),
        polish=False,
        verbose=False,
    )
    result = problem.solve()
    status = str(result.info.status).lower()
    iterations = int(result.info.iter) if result.info.iter is not None else None
    candidate = None if result.x is None else np.asarray(result.x, dtype=np.float64)
    if candidate is not None and (candidate.shape != requested.shape or not np.isfinite(candidate).all()):
        candidate = None
    return candidate, status, iterations


def _constraint_snapshot(filter_input: SafetyFilterInput, config: SafetyFilterConfig) -> tuple[dict[str, Any], tuple[Any, ...]]:
    lower, upper, requested, rows, bounds, labels, lower_labels, upper_labels = _build_constraints(filter_input, config)
    snapshot = {
        "requested_joint_velocity_radps": _vector(requested),
        "joint_lower_radps": _vector(lower),
        "joint_upper_radps": _vector(upper),
        "constraint_rows": np.asarray(rows, dtype=np.float64).tolist(),
        "constraint_lower_bounds": _vector(bounds),
        "constraint_labels": list(labels),
        "joint_lower_labels": list(lower_labels),
        "joint_upper_labels": list(upper_labels),
    }
    return snapshot, (lower, upper, requested, rows, bounds, labels, lower_labels, upper_labels)


def _replay_target(
    config: dict[str, Any],
    checkpoint_manifest: str | Path,
    target: dict[str, int],
    source: dict[str, Any],
    *,
    atol: float,
) -> tuple[SafetyFilterInput, SafetyFilterResult, SafetyFilterConfig, float, float]:
    actor = _load_checkpoint(checkpoint_manifest, target["train_seed"])
    environment_config = copy.deepcopy(config)
    environment_config["env"]["safety_filter"]["viability_monitor_enabled"] = True
    environment = UR5DynamicObstacleEnv(environment_config, method="link_fixed")
    agent = SACAgent(environment.observation_space.shape[0], environment.action_space.shape[0], config, method="link_fixed")
    agent.load_actor(actor["checkpoint"])
    captured: dict[str, Any] = {}
    current_step = -1
    original_filter = environment_module.filter_joint_velocity

    def capture_filter(filter_input: SafetyFilterInput, filter_config: SafetyFilterConfig) -> SafetyFilterResult:
        result = original_filter(filter_input, filter_config)
        if current_step == target["step"]:
            if captured:
                raise RuntimeError("frozen target invoked the safety filter more than once")
            captured["input"] = copy.deepcopy(filter_input)
            captured["result"] = result
        return result

    environment_module.filter_joint_velocity = capture_filter
    try:
        observation, _ = environment.reset(seed=target["seed"])
        for current_step in range(target["step"] + 1):
            action = agent.select_action(observation, deterministic=True)
            observation, _, _, terminated, truncated, _ = environment.step(action)
            if current_step < target["step"] and (terminated or truncated):
                raise RuntimeError(f"episode ended before frozen target {target}")
    finally:
        environment_module.filter_joint_velocity = original_filter
        environment.close()

    if set(captured) != {"input", "result"}:
        raise RuntimeError(f"failed to capture frozen safety-filter input for target {target}")
    filter_input = captured["input"]
    result = captured["result"]
    source_requested = np.asarray(json.loads(source["qdot_requested_radps"]), dtype=np.float64)
    source_command = np.asarray(json.loads(source["qdot_cmd_radps"]), dtype=np.float64)
    requested_difference = _max_abs_difference(filter_input.requested_joint_velocity_radps, source_requested)
    command_difference = _max_abs_difference(result.command_joint_velocity_radps, source_command)
    if result.status.value != SOURCE_STATUS or requested_difference > atol or command_difference > atol:
        raise RuntimeError(
            f"frozen strict failure was not reproduced: {target}; status={result.status.value}, "
            f"requested_difference={requested_difference}, command_difference={command_difference}"
        )
    return (
        filter_input,
        result,
        environment.safety_filter_config,
        requested_difference,
        command_difference,
    )


def audit_projection_feasibility(
    config_path: str | Path,
    checkpoint_manifest: str | Path,
    target_manifest: str | Path,
    *,
    target_index: int,
    atol: float = 1.0e-7,
) -> dict[str, Any]:
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and non-negative")
    targets, trace_hashes = load_target_manifest(target_manifest)
    if target_index < 0 or target_index >= len(targets):
        raise ValueError(f"target_index must be in [0, {len(targets) - 1}]")
    source_events = source_projection_failures(targets, trace_hashes)
    config = load_config(config_path)
    _validate_config(config)
    config["device"] = resolve_device(config)
    set_seed(int(config["seed"]))
    target = targets[target_index]
    source = source_events[_target_key(target)]
    filter_input, baseline_result, baseline_config, requested_difference, command_difference = _replay_target(
        config, checkpoint_manifest, target, source, atol=atol
    )
    snapshot, (lower, upper, requested, rows, bounds, labels, lower_labels, upper_labels) = _constraint_snapshot(
        filter_input, baseline_config
    )
    attempts: list[dict[str, Any]] = []
    for maximum_iterations in ITERATION_BUDGETS:
        diagnostic_config = replace(
            baseline_config,
            fallback_projection_iterations=maximum_iterations,
            qp_time_limit_s=None,
        )
        candidate, status, iterations_used = _run_osqp_projection(
            requested, lower, upper, rows, bounds, diagnostic_config
        )
        residual = constraint_residual_audit(
            candidate,
            lower,
            upper,
            rows,
            bounds,
            labels,
            lower_labels,
            upper_labels,
            baseline_config.projection_failure_tolerance,
        )
        attempts.append(
            {
                "maximum_iterations": maximum_iterations,
                "osqp_status": status,
                "osqp_iterations_used": iterations_used,
                "candidate_joint_velocity_radps": None if candidate is None else _vector(candidate),
                "residual_audit": residual,
                "classification": classify_attempt(status, residual),
            }
        )
    event = {
        **target,
        "source_trace": source["source_trace"],
        "baseline_status": baseline_result.status.value,
        "baseline_qp_solver_status": baseline_result.qp_solver_status,
        "baseline_command_joint_velocity_radps": _vector(baseline_result.command_joint_velocity_radps),
        "baseline_command_is_zero": bool(
            np.max(np.abs(baseline_result.command_joint_velocity_radps), initial=0.0) <= atol
        ),
        "source_requested_max_abs_diff": requested_difference,
        "source_command_max_abs_diff": command_difference,
        "constraint_snapshot": snapshot,
        "attempts": attempts,
    }
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
        "projection_failure_tolerance": baseline_config.projection_failure_tolerance,
        "iteration_budgets": list(ITERATION_BUDGETS),
        "training_performed": False,
        "actor_checkpoint_selection_performed": False,
        "recovery_or_relaxation_performed": False,
        "target_count": 1,
        "total_frozen_target_count": len(targets),
        "target_index": target_index,
        "source_projection_failure_count": len(source_events),
        "baseline_failure_reproduced": event["baseline_status"] == SOURCE_STATUS,
        "baseline_zero_command_reproduced": event["baseline_command_is_zero"],
        "source_requested_reproduced": event["source_requested_max_abs_diff"] <= atol,
        "source_command_reproduced": event["source_command_max_abs_diff"] <= atol,
        "strict_feasible_attempt_count": sum(
            attempt["classification"] == "strict_feasible" for attempt in attempts
        ),
        "g3_authorization": "not_granted_by_this_audit",
        "event": event,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = audit_projection_feasibility(
        args.config,
        args.checkpoint_manifest,
        args.target_manifest,
        target_index=args.target_index,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    event = payload["event"]
    print(
        json.dumps(
            {
                "target_index": payload["target_index"],
                "target": {key: event[key] for key in ("train_seed", "seed", "step")},
                "classifications": [attempt["classification"] for attempt in event["attempts"]],
                "g3_authorization": payload["g3_authorization"],
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()
