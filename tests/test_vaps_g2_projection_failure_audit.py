import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_vaps_g2_projection_failures import load_target_manifest, source_projection_failures
from scripts.merge_vaps_g2_projection_failure_audits import merge_audits


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_trace(
    path: Path,
    *,
    status: str = "safe_stop_projection_failed",
    events: list[tuple[int, int]] | None = None,
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "seed",
                "step",
                "safety_filter_status",
                "v1_safety_filter_status",
                "passed",
                "field_mismatches",
            ),
        )
        writer.writeheader()
        for seed, step in events or [(8201, 17)]:
            writer.writerow(
                {
                    "seed": seed,
                    "step": step,
                    "safety_filter_status": status,
                    "v1_safety_filter_status": status,
                    "passed": "True",
                    "field_mismatches": "[]",
                }
            )


def test_source_projection_failures_requires_exact_frozen_event_set(tmp_path: Path) -> None:
    trace = tmp_path / "train_seed_4108_validation_trace.csv"
    _write_trace(trace)
    targets = [{"train_seed": 4108, "seed": 8201, "step": 17}]

    events = source_projection_failures(targets, {str(trace): _sha256(trace)})

    assert list(events) == [(4108, 8201, 17)]


def test_source_projection_failures_rejects_tampered_trace_or_unfrozen_event(tmp_path: Path) -> None:
    trace = tmp_path / "train_seed_4108_validation_trace.csv"
    _write_trace(trace)
    expected_hash = _sha256(trace)
    trace.write_text(trace.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        source_projection_failures([{"train_seed": 4108, "seed": 8201, "step": 17}], {str(trace): expected_hash})


def test_target_manifest_rejects_duplicate_targets(tmp_path: Path) -> None:
    manifest = tmp_path / "targets.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "vaps_g2_projection_failure_audit_v1",
                "source_protocol": "vaps_g2_strict_execution_v2",
                "source_trace_sha256": {"trace.csv": "a" * 64},
                "targets": [
                    {"train_seed": 4108, "seed": 8201, "step": 17},
                    {"train_seed": 4108, "seed": 8201, "step": 17},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_target_manifest(manifest)


def test_merge_rejects_incomplete_isolated_audit_set(tmp_path: Path) -> None:
    target_manifest = tmp_path / "targets.json"
    trace = tmp_path / "train_seed_4108_validation_trace.csv"
    _write_trace(trace, events=[(8201, 17), (8202, 18)])
    target_manifest.write_text(
        json.dumps(
            {
                "protocol": "vaps_g2_projection_failure_audit_v1",
                "source_protocol": "vaps_g2_strict_execution_v2",
                "source_trace_sha256": {str(trace): _sha256(trace)},
                "targets": [
                    {"train_seed": 4108, "seed": 8201, "step": 17},
                    {"train_seed": 4108, "seed": 8202, "step": 18},
                ],
            }
        ),
        encoding="utf-8",
    )
    shard = tmp_path / "shard_0.json"
    shard.write_text(
        json.dumps(
            {
                "protocol": "vaps_g2_projection_failure_audit_v1",
                "source_protocol": "vaps_g2_strict_execution_v2",
                "config": "config.yaml",
                "config_sha256": "a" * 64,
                "resolved_config_sha256": "b" * 64,
                "checkpoint_manifest": "actors.json",
                "checkpoint_manifest_sha256": "c" * 64,
                "target_manifest": str(target_manifest),
                "target_manifest_sha256": _sha256(target_manifest),
                "source_trace_sha256": {str(trace): _sha256(trace)},
                "atol": 1.0e-7,
                "training_performed": False,
                "actor_checkpoint_selection_performed": False,
                "target_count": 1,
                "total_frozen_target_count": 2,
                "target_index": 0,
                "events": [
                    {
                        "train_seed": 4108,
                        "seed": 8201,
                        "step": 17,
                        "status_reproduced": True,
                        "command_is_zero": True,
                        "source_command_reproduced": True,
                        "v0_v1_execution_reproduced": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cover every frozen target index"):
        merge_audits([shard], target_manifest)
