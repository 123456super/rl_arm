from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_seed_manifest(path: str | Path) -> list[int]:
    manifest_path = Path(path)
    with open(manifest_path, "r", encoding="utf-8") as file:
        payload: Any = json.load(file)

    if isinstance(payload, list):
        seeds = payload
    elif isinstance(payload, dict):
        seeds = payload.get("seeds")
    else:
        seeds = None
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"Seed manifest must contain a non-empty 'seeds' list: {manifest_path}")

    normalized: list[int] = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"Seed manifest entries must be integers: {manifest_path}")
        normalized.append(int(seed))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Seed manifest contains duplicate seeds: {manifest_path}")
    return normalized
