"""Reset-only G1 coverage audit for the VAPS strict viability monitor."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config
from rl_risk_sac.utils.predictive_risk import PredictiveLinkRisk
from rl_risk_sac.utils.safety_filter import SafetyFilterResult, ViabilityStatus


DEFAULT_CONFIG = "configs/experiments/vaps/v1_g1_coverage.yaml"
DEFAULT_SEED_START = 10001
DEFAULT_EPISODES = 1000


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VAPS G1 reset-only coverage audit without an actor.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def coverage_seeds(seed_start: int = DEFAULT_SEED_START, episodes: int = DEFAULT_EPISODES) -> list[int]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if seed_start != DEFAULT_SEED_START or episodes != DEFAULT_EPISODES:
        raise ValueError(
            f"G1 coverage seeds are fixed to {DEFAULT_SEED_START}--"
            f"{DEFAULT_SEED_START + DEFAULT_EPISODES - 1}"
        )
    return list(range(seed_start, seed_start + episodes))


def coverage_shard(shard_index: int, shard_count: int) -> list[int]:
    if shard_count <= 0 or shard_count > DEFAULT_EPISODES:
        raise ValueError(f"shard_count must be between 1 and {DEFAULT_EPISODES}")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    return coverage_seeds()[shard_index::shard_count]


def _require_close(value: Any, expected: float, field: str) -> None:
    if not np.isclose(float(value), expected, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{field} must be {expected}, got {value}")


def validate_g1_config(config: dict[str, Any]) -> None:
    env_cfg = config["env"]
    filter_cfg = env_cfg["safety_filter"]
    _require_close(env_cfg["control_dt"], 0.05, "env.control_dt")
    _require_close(env_cfg["time_step"], 0.0041666667, "env.time_step")
    _require_close(env_cfg["action_scale"], 1.0, "env.action_scale")
    _require_close(env_cfg["obstacle"]["radius"], 0.07, "env.obstacle.radius")
    speed_range = env_cfg["obstacle"]["speed_range"]
    if len(speed_range) != 2:
        raise ValueError("env.obstacle.speed_range must have two values")
    for speed in speed_range:
        _require_close(speed, 0.05, "env.obstacle.speed_range")
    if env_cfg["collision"]["termination"] != "physical_contact":
        raise ValueError("env.collision.termination must be physical_contact")
    _require_close(filter_cfg["prediction_horizon_s"], 0.15, "env.safety_filter.prediction_horizon_s")
    _require_close(filter_cfg["max_observation_age_s"], 0.10, "env.safety_filter.max_observation_age_s")
    _require_close(filter_cfg["control_delay_s"], 0.05, "env.safety_filter.control_delay_s")
    _require_close(filter_cfg["geometry_margin_m"], 0.03, "env.safety_filter.geometry_margin_m")
    _require_close(filter_cfg["tracking_error_bound_m"], 0.01, "env.safety_filter.tracking_error_bound_m")
    _require_close(
        filter_cfg["joint_acceleration_limit_radps2"], 4.0, "env.safety_filter.joint_acceleration_limit_radps2"
    )
    _require_close(filter_cfg["max_filter_compute_time_s"], 0.30, "env.safety_filter.max_filter_compute_time_s")
    if int(filter_cfg["fallback_projection_iterations"]) != 2000:
        raise ValueError("env.safety_filter.fallback_projection_iterations must be 2000")
    if not bool(filter_cfg["enabled"]):
        raise ValueError("env.safety_filter.enabled must be true")
    if not bool(filter_cfg["viability_monitor_enabled"]):
        raise ValueError("env.safety_filter.viability_monitor_enabled must be true")
    if not bool(filter_cfg["use_qp_solver"]):
        raise ValueError("env.safety_filter.use_qp_solver must be true")
    for key in (
        "recovery_mode_enabled",
        "recovery_allow_constraint_relaxation",
        "recovery_maximize_min_clearance",
        "risk_speed_scaling_enabled",
    ):
        if bool(filter_cfg[key]):
            raise ValueError(f"env.safety_filter.{key} must be false")
    if config["eval"]["method"] != "link_fixed":
        raise ValueError("eval.method must be link_fixed")


def _finite_or_none(value: float) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _initial_risk_detail(predictive_risk: PredictiveLinkRisk | None) -> dict[str, Any]:
    if predictive_risk is None:
        return {
            "predictive_risk_status": "integration_error",
            "predictive_risk_reason": "predictive risk was not produced",
            "initial_h_by_link_m": None,
            "predictive_link_velocity_norms_mps": None,
        }
    usable = predictive_risk.usable
    return {
        "predictive_risk_status": predictive_risk.status.value,
        "predictive_risk_reason": predictive_risk.status_reason,
        "initial_h_by_link_m": (
            np.asarray(predictive_risk.safety_functions_m, dtype=np.float64).tolist() if usable else None
        ),
        "predictive_link_velocity_norms_mps": (
            np.linalg.norm(predictive_risk.link_velocities_mps, axis=1).tolist() if usable else None
        ),
    }


def _audit_row(
    env: UR5DynamicObstacleEnv,
    seed: int,
    filter_result: SafetyFilterResult,
    predictive_risk: PredictiveLinkRisk | None,
    solve_time_s: float,
    phase_times_s: dict[str, float],
) -> dict[str, Any]:
    viability = env._viability_info(filter_result, predictive_risk, "not_applicable")
    risk_detail = _initial_risk_detail(predictive_risk)
    if predictive_risk is not None and predictive_risk.usable:
        h_values = np.asarray(predictive_risk.safety_functions_m, dtype=np.float64)
        if not np.isfinite(h_values).all():
            raise ValueError(f"usable predictive risk contains non-finite h for seed {seed}")
    timing = {
        "safety_filter_solve_time_s": solve_time_s,
        "safety_filter_predictive_risk_time_s": phase_times_s.get("predictive_risk", 0.0),
        "safety_filter_jacobian_workspace_time_s": phase_times_s.get("jacobian_workspace", 0.0),
        "safety_filter_projection_time_s": phase_times_s.get("projection", 0.0),
    }
    if not all(np.isfinite(float(value)) and float(value) >= 0.0 for value in timing.values()):
        raise ValueError(f"safety-filter timing must be finite and non-negative for seed {seed}")
    return {
        "seed": int(seed),
        "viability_status": viability["viability_status"],
        "viability_horizon_s": float(viability["viability_horizon_s"]),
        "viability_min_h_m": _finite_or_none(viability["viability_min_h_m"]),
        "viability_strict_feasible": bool(viability["viability_strict_feasible"]),
        "viability_solver_status": viability["viability_solver_status"],
        "viability_model_assumptions_valid": bool(viability["viability_model_assumptions_valid"]),
        "safety_filter_status": filter_result.status.value,
        "safety_filter_reason": filter_result.reason,
        "safety_filter_qp_solver_status": filter_result.qp_solver_status,
        "safety_filter_active_constraint_categories": list(filter_result.active_constraint_categories),
        "safety_filter_infeasible_constraint_categories": list(filter_result.infeasible_constraint_categories),
        **{name: float(value) for name, value in timing.items()},
        "qdot_requested_radps": np.zeros(env.joint_count, dtype=np.float64).tolist(),
        "qdot_cmd_radps": np.asarray(filter_result.command_joint_velocity_radps, dtype=np.float64).tolist(),
        "goal_m": np.asarray(env.goal, dtype=np.float64).tolist(),
        "obstacle_position_m": np.asarray(env.obstacle_center, dtype=np.float64).tolist(),
        "obstacle_velocity_mps": np.asarray(env.obstacle_velocity, dtype=np.float64).tolist(),
        "obstacle_speed_mps": float(np.linalg.norm(env.obstacle_velocity)),
        "link_names": [spec.name for spec in env.capsule_model.specs],
        **risk_detail,
    }


def _margin_summary(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    margins = np.asarray(
        [row["viability_min_h_m"] for row in rows if row["viability_min_h_m"] is not None], dtype=np.float64
    )
    if margins.size == 0:
        return {"valid_count": 0, "min": None, "mean": None, "p05": None, "p50": None, "p95": None, "max": None}
    return {
        "valid_count": int(margins.size),
        "min": float(np.min(margins)),
        "mean": float(np.mean(margins)),
        "p05": float(np.percentile(margins, 5)),
        "p50": float(np.percentile(margins, 50)),
        "p95": float(np.percentile(margins, 95)),
        "max": float(np.max(margins)),
    }


def _timing_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    timing_fields = (
        "safety_filter_solve_time_s",
        "safety_filter_predictive_risk_time_s",
        "safety_filter_jacobian_workspace_time_s",
        "safety_filter_projection_time_s",
    )
    summaries: dict[str, dict[str, float]] = {}
    for field in timing_fields:
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError(f"{field} contains a non-finite or negative value")
        summaries[field] = {
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values)),
        }
    return summaries


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = {status.value: 0 for status in ViabilityStatus}
    for row in rows:
        status_counts[str(row["viability_status"])] += 1
    episodes = len(rows)
    initially_unsafe_count = sum(
        row["viability_min_h_m"] is not None and float(row["viability_min_h_m"]) <= 0.0 for row in rows
    )
    compute_budget_safe_stop_count = sum(
        row["safety_filter_status"] == "safe_stop_compute_budget" for row in rows
    )
    projection_failure_count = sum(
        row["safety_filter_status"] == "safe_stop_projection_failed" for row in rows
    )
    return {
        "episodes": episodes,
        "viability_status_counts": status_counts,
        "viability_status_rates": {status: count / episodes for status, count in status_counts.items()},
        "certified_viable_coverage_rate": status_counts[ViabilityStatus.CERTIFIED_VIABLE.value] / episodes,
        "initially_unsafe_count": initially_unsafe_count,
        "initially_unsafe_rate": initially_unsafe_count / episodes,
        "initial_h_min_m": _margin_summary(rows),
        "no_nonfinite_valid_margins": True,
        "no_nonfinite_timing": True,
        "safety_filter_timing_s": _timing_summary(rows),
        "compute_budget_safe_stop_count": compute_budget_safe_stop_count,
        "compute_budget_safe_stop_rate": compute_budget_safe_stop_count / episodes,
        "strict_projection_failure_count": projection_failure_count,
        "strict_projection_failure_rate": projection_failure_count / episodes,
    }


def audit_vaps_coverage(
    config_path: str | Path,
    seeds: Sequence[int],
    *,
    full_seed_manifest: Sequence[int] | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("seeds must not be empty")
    normalized_seeds = [int(seed) for seed in seeds]
    if len(normalized_seeds) != len(set(normalized_seeds)):
        raise ValueError("seeds must be unique")
    expected_seeds = list(coverage_seeds() if full_seed_manifest is None else full_seed_manifest)
    if not set(normalized_seeds).issubset(expected_seeds):
        raise ValueError("seeds must belong to the fixed G1 coverage manifest")
    if shard_count <= 0 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("invalid shard index or count")

    config = load_config(config_path)
    validate_g1_config(config)
    env = UR5DynamicObstacleEnv(config, method=str(config["eval"]["method"]))
    try:
        rows: list[dict[str, Any]] = []
        for seed in normalized_seeds:
            env.reset(seed=seed)
            requested = np.zeros(env.joint_count, dtype=np.float64)
            filter_result, predictive_risk, solve_time_s = env._filter_command(requested)
            rows.append(
                _audit_row(
                    env,
                    seed,
                    filter_result,
                    predictive_risk,
                    solve_time_s,
                    dict(env.last_filter_phase_times_s),
                )
            )
    finally:
        env.close()

    return {
        "protocol": "vaps_g1_reset_only_coverage_v1",
        "config": str(config_path),
        "method": str(config["eval"]["method"]),
        "actor_loaded": False,
        "training_performed": False,
        "seed_manifest": {
            "kind": "fixed_contiguous_range",
            "start": DEFAULT_SEED_START,
            "end": DEFAULT_SEED_START + DEFAULT_EPISODES - 1,
            "count": DEFAULT_EPISODES,
            "seeds": expected_seeds,
        },
        "shard": {
            "index": shard_index,
            "count": shard_count,
            "executed_seeds": normalized_seeds,
        },
        **_summarize_rows(rows),
        "episodes_detail": rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    coverage_seeds(args.seed_start, args.episodes)
    seeds = coverage_shard(args.shard_index, args.shard_count)
    result = audit_vaps_coverage(
        args.config,
        seeds,
        full_seed_manifest=coverage_seeds(),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "episodes": result["episodes"],
                "certified_viable_coverage_rate": result["certified_viable_coverage_rate"],
                "initially_unsafe_rate": result["initially_unsafe_rate"],
                "output": str(output),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
