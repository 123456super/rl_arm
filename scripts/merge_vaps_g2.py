"""Audit the six required VAPS G2 V0/V1 strict-equivalence result files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rl_risk_sac.utils.config import load_config

try:
    from scripts.seed_manifest import load_seed_manifest
except ModuleNotFoundError:
    from seed_manifest import load_seed_manifest


PROTOCOL = "vaps_g2_strict_execution_v2"
ROLES = frozenset({"validation", "final"})
MAX_FIELDS = (
    "observation_max_abs_diff",
    "action_max_abs_diff",
    "qdot_cmd_max_abs_diff",
    "qdot_requested_max_abs_diff",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the complete VAPS G2 strict-equivalence result set.")
    parser.add_argument("--input", action="append", required=True, help="One G2 result JSON; specify six times.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--checkpoint-manifest",
        default="configs/experiments/vaps_manifests/v2_b4_selected_checkpoints.json",
    )
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


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"result must be a JSON object: {path}")
    return payload


def _expected_actors(path: str | Path) -> dict[int, dict[str, Any]]:
    payload = _load_json(path)
    if payload.get("protocol") != PROTOCOL or payload.get("method") != "link_fixed":
        raise ValueError("checkpoint manifest has the wrong G2 protocol or method")
    actors = payload.get("actors")
    if not isinstance(actors, list) or not actors:
        raise ValueError("checkpoint manifest must contain actors")
    indexed: dict[int, dict[str, Any]] = {}
    for actor in actors:
        if not isinstance(actor, dict) or isinstance(actor.get("train_seed"), bool):
            raise ValueError("checkpoint manifest actor must have an integer train_seed")
        train_seed = actor["train_seed"]
        if not isinstance(train_seed, int) or train_seed in indexed:
            raise ValueError("checkpoint manifest train seeds must be unique integers")
        for field in ("checkpoint", "checkpoint_step", "selection_record"):
            if field not in actor:
                raise ValueError(f"checkpoint manifest actor missing {field}")
        indexed[train_seed] = actor
    return indexed


def _manifest_role_and_seeds(path: str | Path) -> tuple[str, list[int]]:
    payload = _load_json(path)
    role = payload.get("role")
    if payload.get("protocol") != PROTOCOL or role not in ROLES:
        raise ValueError(f"seed manifest is not a G2 validation/final manifest: {path}")
    return str(role), load_seed_manifest(path)


def _require_hash(payload: dict[str, Any], field: str, expected: str) -> None:
    value = payload.get(field)
    if not isinstance(value, str) or value != expected:
        raise ValueError(f"result {field} does not match the frozen input")


def _validate_result(
    result_path: str | Path,
    actors: dict[int, dict[str, Any]],
    checkpoint_manifest: str | Path,
) -> tuple[str, int, dict[str, Any]]:
    payload = _load_json(result_path)
    if payload.get("protocol") != PROTOCOL:
        raise ValueError(f"wrong G2 protocol in {result_path}")
    train_seed = payload.get("train_seed")
    if isinstance(train_seed, bool) or not isinstance(train_seed, int) or train_seed not in actors:
        raise ValueError(f"result has an unknown train seed: {result_path}")
    actor = actors[int(train_seed)]
    if payload.get("checkpoint") != actor["checkpoint"]:
        raise ValueError(f"result checkpoint does not match the frozen manifest: {result_path}")
    if payload.get("checkpoint_step") != actor["checkpoint_step"]:
        raise ValueError(f"result checkpoint step does not match the frozen manifest: {result_path}")
    if payload.get("selection_record") != actor["selection_record"]:
        raise ValueError(f"result selection record does not match the frozen manifest: {result_path}")
    _require_hash(payload, "checkpoint_sha256", _sha256(actor["checkpoint"]))
    _require_hash(payload, "selection_record_sha256", _sha256(actor["selection_record"]))
    if payload.get("checkpoint_manifest") != str(checkpoint_manifest):
        raise ValueError(f"result checkpoint manifest path does not match the audit input: {result_path}")
    _require_hash(payload, "checkpoint_manifest_sha256", _sha256(checkpoint_manifest))

    seed_manifest = payload.get("seed_manifest")
    if not isinstance(seed_manifest, str):
        raise ValueError(f"result seed manifest is missing: {result_path}")
    role, expected_seeds = _manifest_role_and_seeds(seed_manifest)
    _require_hash(payload, "seed_manifest_sha256", _sha256(seed_manifest))
    config = payload.get("config")
    if not isinstance(config, str) or not Path(config).is_file():
        raise ValueError(f"result config is missing: {result_path}")
    _require_hash(payload, "config_sha256", _sha256(config))
    _require_hash(payload, "resolved_config_sha256", _canonical_sha256(load_config(config)))

    atol = payload.get("atol")
    if isinstance(atol, bool) or not isinstance(atol, (int, float)) or not np.isfinite(atol) or atol < 0.0:
        raise ValueError(f"result atol is invalid: {result_path}")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(expected_seeds):
        raise ValueError(f"result episode count does not match its manifest: {result_path}")
    if payload.get("episodes_requested") != len(expected_seeds) or payload.get("episodes_completed") != len(expected_seeds):
        raise ValueError(f"result does not complete its full manifest: {result_path}")
    if payload.get("passed") is not True:
        raise ValueError(f"result reports V0/V1 mismatch: {result_path}")
    for expected_seed, episode in zip(expected_seeds, episodes, strict=True):
        if not isinstance(episode, dict) or episode.get("seed") != expected_seed or episode.get("passed") is not True:
            raise ValueError(f"result episode does not pass its expected seed: {result_path}")
        for field in MAX_FIELDS:
            value = episode.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"result {field} is invalid: {result_path}")
            if float(value) > float(atol):
                raise ValueError(f"result {field} exceeds atol: {result_path}")
    return role, int(train_seed), payload


def audit_results(
    input_paths: Sequence[str | Path],
    checkpoint_manifest: str | Path,
) -> dict[str, Any]:
    if not input_paths:
        raise ValueError("at least one G2 result is required")
    actors = _expected_actors(checkpoint_manifest)
    expected_pairs = {(role, seed) for role in ROLES for seed in actors}
    results: dict[tuple[str, int], dict[str, Any]] = {}
    for result_path in input_paths:
        role, train_seed, payload = _validate_result(result_path, actors, checkpoint_manifest)
        key = (role, train_seed)
        if key in results:
            raise ValueError(f"duplicate G2 result for {role} train seed {train_seed}")
        results[key] = payload
    if set(results) != expected_pairs:
        missing = sorted(expected_pairs - set(results))
        unexpected = sorted(set(results) - expected_pairs)
        raise ValueError(f"incomplete G2 result set; missing={missing}, unexpected={unexpected}")

    first = next(iter(results.values()))
    for field in (
        "config",
        "config_sha256",
        "resolved_config_sha256",
        "checkpoint_manifest",
        "checkpoint_manifest_sha256",
        "atol",
    ):
        if any(payload.get(field) != first.get(field) for payload in results.values()):
            raise ValueError(f"G2 results disagree on {field}")
    return {
        "protocol": PROTOCOL,
        "passed": True,
        "config": first["config"],
        "config_sha256": first["config_sha256"],
        "resolved_config_sha256": first["resolved_config_sha256"],
        "checkpoint_manifest": first["checkpoint_manifest"],
        "checkpoint_manifest_sha256": first["checkpoint_manifest_sha256"],
        "atol": first["atol"],
        "train_seeds": sorted(actors),
        "groups": {
            role: {
                "episodes_per_train_seed": [
                    results[(role, seed)]["episodes_completed"] for seed in sorted(actors)
                ],
                "max_abs_differences": {
                    field: max(
                        episode[field]
                        for seed in sorted(actors)
                        for episode in results[(role, seed)]["episodes"]
                    )
                    for field in MAX_FIELDS
                },
                "result_files": [
                    str(result_path)
                    for result_path in input_paths
                    if _manifest_role_and_seeds(_load_json(result_path)["seed_manifest"])[0] == role
                ],
            }
            for role in sorted(ROLES)
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = audit_results(args.input, args.checkpoint_manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "output": str(output)}))


if __name__ == "__main__":
    main()
