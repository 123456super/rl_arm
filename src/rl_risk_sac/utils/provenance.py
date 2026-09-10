from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_run_manifest(
    *,
    repo_root: Path,
    config_path: Path,
    config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    """Capture the minimum provenance required by the restarted protocol."""
    commit = _git(repo_root, "rev-parse", "HEAD")
    status = _git(repo_root, "status", "--porcelain")
    urdf = Path(config["robot"]["urdf"])
    if not urdf.is_absolute():
        urdf = repo_root / urdf
    packages = {}
    for name in ("numpy", "scipy", "pybullet", "torch", "gymnasium", "osqp", "PyYAML"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None

    hardware: dict[str, Any] = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        hardware["cuda_available"] = bool(torch.cuda.is_available())
        hardware["cuda_version"] = torch.version.cuda
        hardware["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        hardware["cuda_available"] = False
        hardware["cuda_version"] = None
        hardware["gpu_names"] = []

    return {
        "schema_version": "restart_run_manifest_v1",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "finished_at_utc": None,
        "run_dir": str(run_dir),
        "config_source": str(config_path),
        "config_source_sha256": sha256_file(config_path),
        "resolved_config_sha256": sha256_json(config),
        "urdf": str(urdf),
        "urdf_sha256": sha256_file(urdf),
        "capsules_sha256": sha256_json(config["robot"]["capsules"]),
        "source_files_sha256": _source_hashes(repo_root),
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_status_porcelain": status.splitlines(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "packages": packages,
        "hardware": hardware,
        "seed": int(config["seed"]),
        "method": str(config["train"]["method"]),
        "checkpoint_sha256": {},
    }


def finalize_run_manifest(manifest_path: Path, run_dir: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoints = {}
    for path in sorted(run_dir.glob("*.pt")):
        checkpoints[path.name] = sha256_file(path)
    manifest["checkpoint_sha256"] = checkpoints
    manifest["status"] = "complete"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _source_hashes(repo_root: Path) -> dict[str, str]:
    """Hash executable/configuration inputs so a dirty-tree run remains auditable."""
    patterns = (
        "src/**/*.py",
        "scripts/*.py",
        "configs/**/*.yaml",
        "assets/robots/universal_robots/ur_models/meshes/ur5/collision/*.stl",
    )
    paths = {path for pattern in patterns for path in repo_root.glob(pattern) if path.is_file()}
    return {str(path.relative_to(repo_root)): sha256_file(path) for path in sorted(paths)}
