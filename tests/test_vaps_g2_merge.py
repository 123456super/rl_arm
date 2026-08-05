import hashlib
import json
from pathlib import Path

import pytest

from scripts.merge_vaps_g2 import _canonical_sha256, audit_results
from rl_risk_sac.utils.config import load_config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_audit_results_requires_each_actor_for_each_g2_group(tmp_path: Path) -> None:
    checkpoint = tmp_path / "actor.pt"
    checkpoint.write_bytes(b"actor")
    selection = tmp_path / "selected_checkpoint.csv"
    selection.write_text("step,checkpoint,selected\n50000,actor.pt,1\n", encoding="utf-8")
    actor_manifest = tmp_path / "actors.json"
    _write_json(
        actor_manifest,
        {
            "protocol": "vaps_g2_strict_execution_v2",
            "method": "link_fixed",
            "actors": [
                {
                    "train_seed": 4108,
                    "checkpoint_step": 50000,
                    "checkpoint": str(checkpoint),
                    "selection_record": str(selection),
                }
            ],
        },
    )
    config = Path("configs/experiments/vaps/v2_g2_strict_execution.yaml")
    validation = tmp_path / "validation.json"
    _write_json(
        validation,
        {"protocol": "vaps_g2_strict_execution_v2", "role": "validation", "seeds": [8201]},
    )
    result = tmp_path / "result.json"
    _write_json(
        result,
        {
            "protocol": "vaps_g2_strict_execution_v2",
            "config": str(config),
            "config_sha256": _sha256(config),
            "resolved_config_sha256": _canonical_sha256(load_config(config)),
            "checkpoint_manifest": str(actor_manifest),
            "checkpoint_manifest_sha256": _sha256(actor_manifest),
            "train_seed": 4108,
            "checkpoint": str(checkpoint),
            "checkpoint_step": 50000,
            "checkpoint_sha256": _sha256(checkpoint),
            "selection_record": str(selection),
            "selection_record_sha256": _sha256(selection),
            "seed_manifest": str(validation),
            "seed_manifest_sha256": _sha256(validation),
            "atol": 1.0e-7,
            "passed": True,
            "episodes_requested": 1,
            "episodes_completed": 1,
            "episodes": [
                {
                    "seed": 8201,
                    "passed": True,
                    "observation_max_abs_diff": 0.0,
                    "action_max_abs_diff": 0.0,
                    "qdot_cmd_max_abs_diff": 0.0,
                    "qdot_requested_max_abs_diff": 0.0,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="incomplete G2 result set"):
        audit_results([result], actor_manifest)
