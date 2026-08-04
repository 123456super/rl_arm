from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit predictive-safe-set feasibility at episode reset.")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--seed", type=int, default=7101)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def audit_initial_feasibility(config_path: str | Path, seed: int, episodes: int) -> dict[str, object]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    config = load_config(config_path)
    if not bool(config["env"].get("safety_filter", {}).get("enabled", False)):
        raise ValueError(f"Initial feasibility requires an enabled safety filter: {config_path}")

    method = str(config["eval"]["method"])
    env = UR5DynamicObstacleEnv(config, method=method)
    try:
        rows: list[dict[str, object]] = []
        for offset in range(episodes):
            episode_seed = seed + offset
            _, info = env.reset(seed=episode_seed)
            initial_h_min_m = float(info["recovery_initial_h_min_m"])
            if not np.isfinite(initial_h_min_m):
                raise ValueError(f"Initial predictive margin must be finite for seed {episode_seed}")
            rows.append(
                {
                    "seed": episode_seed,
                    "initial_h_min_m": initial_h_min_m,
                    "initially_unsafe": bool(info["recovery_initially_unsafe"]),
                    "initial_distance_m": float(info["d_min"]),
                    "goal_m": np.asarray(info["goal"], dtype=np.float64).tolist(),
                    "obstacle_position_m": np.asarray(info["obstacle_center"], dtype=np.float64).tolist(),
                    "obstacle_velocity_mps": np.asarray(info["obstacle_velocity"], dtype=np.float64).tolist(),
                }
            )
    finally:
        env.close()

    margins_m = np.asarray([row["initial_h_min_m"] for row in rows], dtype=np.float64)
    unsafe_count = sum(bool(row["initially_unsafe"]) for row in rows)
    return {
        "config": str(config_path),
        "method": method,
        "seed_start": seed,
        "episodes": episodes,
        "initially_unsafe_count": unsafe_count,
        "initially_unsafe_rate": unsafe_count / episodes,
        "initial_h_min_m": {
            "min": float(np.min(margins_m)),
            "mean": float(np.mean(margins_m)),
            "p05": float(np.percentile(margins_m, 5)),
            "p50": float(np.percentile(margins_m, 50)),
            "p95": float(np.percentile(margins_m, 95)),
            "max": float(np.max(margins_m)),
        },
        "episodes_detail": rows,
    }


def main() -> None:
    args = parse_args()
    result = {
        "seed": args.seed,
        "episodes_per_config": args.episodes,
        "conditions": [audit_initial_feasibility(path, args.seed, args.episodes) for path in args.config],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
