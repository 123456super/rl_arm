"""Locate minimal strict-constraint conflicts for frozen G2 v2 events.

This is an offline audit of the frozen matrix snapshots only. It does not
replay the environment, modify runtime control, enable recovery, or relax any
constraint for execution. Constraint subsets exist solely to attribute why an
already certified primal-infeasible original matrix has no strict command.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.audit_vaps_g2_projection_failures import _target_key
    from scripts.audit_vaps_g2_projection_feasibility import PROTOCOL as SOURCE_PROTOCOL
    from rl_risk_sac.utils.safety_filter import _constraint_category
except ModuleNotFoundError:
    from audit_vaps_g2_projection_failures import _target_key
    from audit_vaps_g2_projection_feasibility import PROTOCOL as SOURCE_PROTOCOL
    from rl_risk_sac.utils.safety_filter import _constraint_category


PROTOCOL = "vaps_g2_constraint_conflict_audit_v1"
DEFAULT_TARGET_MANIFEST = "configs/experiments/vaps_manifests/v2_g2_primal_infeasible_targets.json"


@dataclass(frozen=True)
class ConstraintTerm:
    """One original strict inequality in ``row @ qdot >= bound`` form."""

    label: str
    category: str
    row: np.ndarray
    bound: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit one frozen G2 v2 primal-infeasible matrix for minimal strict constraint conflicts."
    )
    parser.add_argument("--target-manifest", default=DEFAULT_TARGET_MANIFEST)
    parser.add_argument(
        "--target-index",
        required=True,
        type=int,
        help="Zero-based frozen target index. Run each index in a separate process.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def load_target_manifest(path: str | Path) -> tuple[dict[str, Any], list[dict[str, int]]]:
    payload = _load_json(path)
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("target manifest has the wrong protocol")
    if payload.get("source_protocol") != SOURCE_PROTOCOL:
        raise ValueError("target manifest has the wrong source protocol")
    if not isinstance(payload.get("source_audit"), str) or not isinstance(payload.get("source_audit_sha256"), str):
        raise ValueError("target manifest must pin the source audit path and SHA-256")
    if payload.get("required_iteration_budget") != 50000 or payload.get("required_classification") != "primal_infeasible":
        raise ValueError("target manifest must require the frozen 50000-iteration primal-infeasible result")
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("target manifest must contain a non-empty targets list")
    targets: list[dict[str, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for raw_target in raw_targets:
        if not isinstance(raw_target, dict):
            raise ValueError("target manifest entries must be objects")
        key = _target_key(raw_target)
        if key in seen:
            raise ValueError("target manifest contains duplicate targets")
        seen.add(key)
        targets.append({"train_seed": key[0], "seed": key[1], "step": key[2]})
    return payload, targets


def _attempt_at_budget(event: dict[str, Any], budget: int) -> dict[str, Any]:
    attempts = event.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("source audit event is missing attempts")
    matches = [attempt for attempt in attempts if isinstance(attempt, dict) and attempt.get("maximum_iterations") == budget]
    if len(matches) != 1:
        raise ValueError(f"source audit event must contain exactly one {budget}-iteration attempt")
    return matches[0]


def load_frozen_event(
    target_manifest_path: str | Path,
    *,
    target_index: int,
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any], dict[str, Any]]:
    manifest, targets = load_target_manifest(target_manifest_path)
    if target_index < 0 or target_index >= len(targets):
        raise ValueError(f"target_index must be in [0, {len(targets) - 1}]")
    source_path = Path(manifest["source_audit"])
    if not source_path.is_file():
        raise FileNotFoundError(f"frozen source audit is missing: {source_path}")
    if _sha256(source_path) != manifest["source_audit_sha256"]:
        raise ValueError("frozen source audit SHA-256 mismatch")
    source = _load_json(source_path)
    if source.get("protocol") != SOURCE_PROTOCOL:
        raise ValueError("frozen source audit has the wrong protocol")
    if source.get("training_performed") is not False or source.get("recovery_or_relaxation_performed") is not False:
        raise ValueError("frozen source audit does not preserve the offline strict scope")
    raw_events = source.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("frozen source audit is missing events")
    events = {_target_key(event): event for event in raw_events if isinstance(event, dict)}
    if len(events) != len(raw_events):
        raise ValueError("frozen source audit contains duplicate or invalid event targets")
    target = targets[target_index]
    event = events.get(_target_key(target))
    if event is None:
        raise ValueError("target is not present in the frozen source audit")
    if event.get("baseline_status") != "safe_stop_projection_failed" or event.get("baseline_command_is_zero") is not True:
        raise ValueError("target does not preserve the original strict fail-safe result")
    attempt = _attempt_at_budget(event, int(manifest["required_iteration_budget"]))
    if attempt.get("classification") != manifest["required_classification"]:
        raise ValueError("target is not frozen as primal infeasible at 50000 iterations")
    return manifest, target, source, event


def terms_from_snapshot(snapshot: dict[str, Any]) -> tuple[ConstraintTerm, ...]:
    """Convert an immutable source snapshot to labelled strict inequalities."""

    try:
        rows = np.asarray(snapshot["constraint_rows"], dtype=np.float64)
        bounds = np.asarray(snapshot["constraint_lower_bounds"], dtype=np.float64)
        lower = np.asarray(snapshot["joint_lower_radps"], dtype=np.float64)
        upper = np.asarray(snapshot["joint_upper_radps"], dtype=np.float64)
        labels = tuple(snapshot["constraint_labels"])
        lower_labels = tuple(snapshot["joint_lower_labels"])
        upper_labels = tuple(snapshot["joint_upper_labels"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("source event has an invalid constraint snapshot") from error
    if (
        rows.ndim != 2
        or bounds.shape != (rows.shape[0],)
        or lower.ndim != 1
        or upper.shape != lower.shape
        or rows.shape[1] != lower.size
        or len(labels) != rows.shape[0]
        or len(lower_labels) != lower.size
        or len(upper_labels) != lower.size
        or not np.isfinite(rows).all()
        or not np.isfinite(bounds).all()
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
    ):
        raise ValueError("source event has incompatible or non-finite strict constraints")
    identity = np.eye(lower.size, dtype=np.float64)
    terms = [
        ConstraintTerm(str(label), _constraint_category(str(label)), rows[index], float(bounds[index]))
        for index, label in enumerate(labels)
    ]
    terms.extend(
        ConstraintTerm(str(label), _constraint_category(str(label)), identity[index], float(lower[index]))
        for index, label in enumerate(lower_labels)
    )
    terms.extend(
        ConstraintTerm(str(label), _constraint_category(str(label)), -identity[index], float(-upper[index]))
        for index, label in enumerate(upper_labels)
    )
    return tuple(terms)


def solve_strict_feasibility(terms: Sequence[ConstraintTerm], dimension: int, tolerance: float) -> dict[str, Any]:
    """Solve a selected original-constraint subset without changing any value."""

    if not terms:
        return {"classification": "strict_feasible", "solver_status": "unconstrained", "residual_audit": None}
    try:
        from scipy.optimize import linprog
    except ImportError as error:
        raise RuntimeError("constraint-conflict audit requires scipy") from error
    matrix = np.vstack([term.row for term in terms])
    bounds = np.asarray([term.bound for term in terms], dtype=np.float64)
    result = linprog(
        c=np.zeros(dimension, dtype=np.float64),
        A_ub=-matrix,
        b_ub=-bounds,
        bounds=[(None, None)] * dimension,
        method="highs",
    )
    status = str(result.message)
    if result.status == 2:
        return {"classification": "primal_infeasible", "solver_status": status, "residual_audit": None}
    if result.status != 0 or result.x is None:
        return {"classification": "indeterminate_solver_status", "solver_status": status, "residual_audit": None}
    candidate = np.asarray(result.x, dtype=np.float64)
    residuals = bounds - matrix @ candidate
    maximum_index = int(np.argmax(residuals))
    maximum = float(max(0.0, residuals[maximum_index]))
    residual_audit = {
        "candidate_available": True,
        "candidate_is_finite": bool(np.isfinite(candidate).all()),
        "max_constraint_violation": maximum,
        "max_constraint_label": terms[maximum_index].label,
        "max_constraint_category": terms[maximum_index].category,
        "satisfies_all_constraints": bool(np.isfinite(candidate).all() and maximum <= tolerance),
    }
    return {
        "classification": "strict_feasible" if residual_audit["satisfies_all_constraints"] else "indeterminate_solver_status",
        "solver_status": status,
        "candidate_joint_velocity_radps": candidate.tolist(),
        "residual_audit": residual_audit,
    }


def _subset_terms(terms: Sequence[ConstraintTerm], categories: Iterable[str]) -> tuple[ConstraintTerm, ...]:
    selected = frozenset(categories)
    return tuple(term for term in terms if term.category in selected)


def find_minimal_category_conflicts(
    terms: Sequence[ConstraintTerm],
    dimension: int,
    tolerance: float,
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    """Exhaustively find every inclusion-minimal infeasible category combination."""

    categories = tuple(sorted({term.category for term in terms}))
    probes: list[dict[str, Any]] = []
    infeasible: set[tuple[str, ...]] = set()
    for count in range(1, len(categories) + 1):
        for subset in itertools.combinations(categories, count):
            result = solve_strict_feasibility(_subset_terms(terms, subset), dimension, tolerance)
            probe = {
                "categories": list(subset),
                "constraint_labels": [term.label for term in _subset_terms(terms, subset)],
                **result,
            }
            probes.append(probe)
            if result["classification"] == "primal_infeasible":
                infeasible.add(subset)
    if any(probe["classification"].startswith("indeterminate_") for probe in probes):
        raise RuntimeError("category conflict attribution is indeterminate; do not report a minimal conflict")
    if not infeasible:
        raise RuntimeError("full frozen strict matrix was not reproduced as primal infeasible")
    return categories, probes


def irreducible_constraint_core(
    terms: Sequence[ConstraintTerm],
    dimension: int,
    tolerance: float,
) -> list[ConstraintTerm]:
    """Deterministically delete redundant rows to produce one irreducible core."""

    core = list(terms)
    if solve_strict_feasibility(core, dimension, tolerance)["classification"] != "primal_infeasible":
        raise ValueError("irreducible-core input must be primal infeasible")
    index = 0
    while index < len(core):
        trial = core[:index] + core[index + 1 :]
        if solve_strict_feasibility(trial, dimension, tolerance)["classification"] == "primal_infeasible":
            core = trial
        else:
            index += 1
    return core


def audit_constraint_conflict(
    target_manifest_path: str | Path,
    *,
    target_index: int,
) -> dict[str, Any]:
    manifest, target, source, event = load_frozen_event(target_manifest_path, target_index=target_index)
    snapshot = event.get("constraint_snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("source event is missing its constraint snapshot")
    terms = terms_from_snapshot(snapshot)
    dimension = len(snapshot["joint_lower_radps"])
    tolerance = float(source.get("projection_failure_tolerance"))
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("frozen source audit has an invalid projection failure tolerance")
    categories, probes = find_minimal_category_conflicts(terms, dimension, tolerance)
    minimal_probes = [
        probe
        for probe in probes
        if probe["classification"] == "primal_infeasible"
        and not any(
            other["classification"] == "primal_infeasible"
            and set(other["categories"]) < set(probe["categories"])
            for other in probes
        )
    ]
    minimal_cores = []
    for probe in minimal_probes:
        core = irreducible_constraint_core(
            _subset_terms(terms, probe["categories"]), dimension, tolerance
        )
        drop_classifications = [
            solve_strict_feasibility(core[:index] + core[index + 1 :], dimension, tolerance)["classification"]
            for index in range(len(core))
        ]
        if any(classification != "strict_feasible" for classification in drop_classifications):
            raise RuntimeError("irreducible constraint-core verification is indeterminate")
        minimal_cores.append(
            {
                "categories": probe["categories"],
                "irreducible_constraint_labels": [term.label for term in core],
                "irreducible_constraint_categories": [term.category for term in core],
                "irreducible_core_drop_classifications": drop_classifications,
            }
        )
    return {
        "protocol": PROTOCOL,
        "source_protocol": SOURCE_PROTOCOL,
        "target_manifest": str(target_manifest_path),
        "target_manifest_sha256": _sha256(target_manifest_path),
        "source_audit": manifest["source_audit"],
        "source_audit_sha256": manifest["source_audit_sha256"],
        "source_audit_protocol": source["protocol"],
        "required_iteration_budget": manifest["required_iteration_budget"],
        "required_classification": manifest["required_classification"],
        "projection_failure_tolerance": tolerance,
        "training_performed": False,
        "actor_checkpoint_selection_performed": False,
        "recovery_or_relaxation_performed": False,
        "runtime_control_modified": False,
        "target_count": 1,
        "total_frozen_target_count": len(load_target_manifest(target_manifest_path)[1]),
        "target_index": target_index,
        "target": target,
        "source_baseline_status": event["baseline_status"],
        "source_baseline_command_is_zero": event["baseline_command_is_zero"],
        "available_constraint_categories": list(categories),
        "category_subset_probes": probes,
        "minimal_category_conflicts": minimal_cores,
        "g3_authorization": "not_granted_by_this_audit",
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = audit_constraint_conflict(args.target_manifest, target_index=args.target_index)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "target_index": payload["target_index"],
                "target": payload["target"],
                "minimal_category_conflicts": payload["minimal_category_conflicts"],
                "g3_authorization": payload["g3_authorization"],
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()
