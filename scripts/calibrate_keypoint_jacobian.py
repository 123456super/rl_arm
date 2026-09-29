from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pybullet as p


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.tasks.thesis_reaching import keypoint_position_jacobian
from rl_risk_sac.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate 9x6 keypoint-Jacobian observation scaling."
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml",
    )
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=81001)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    config = load_config(ROOT / args.config)
    env = ThesisHomotopyEnv(config)
    rng = np.random.default_rng(args.seed)
    values = np.empty((args.samples, 54), dtype=np.float32)
    accepted = 0
    attempted = 0
    maximum_attempts = 20 * args.samples
    try:
        lower = np.asarray(env.robot.joint_lower_limits, dtype=np.float32)
        upper = np.asarray(env.robot.joint_upper_limits, dtype=np.float32)
        while accepted < args.samples and attempted < maximum_attempts:
            attempted += 1
            q = rng.uniform(lower, upper).astype(np.float32)
            env._set_joint_state(q)
            p.performCollisionDetection(physicsClientId=env.client_id)
            events = env._collision_events()
            if events["self_collision"] or events["environment_collision"]:
                continue
            _, quaternion, _, _ = env.robot.end_effector_kinematics()
            values[accepted] = keypoint_position_jacobian(
                env.robot.end_effector_jacobian(), quaternion,
                cube_side_m=env.keypoint_cube_side_m,
            ).reshape(-1)
            accepted += 1
    finally:
        env.close()
    if accepted != args.samples:
        raise RuntimeError(
            f"accepted only {accepted}/{args.samples} legal states after {attempted} attempts"
        )

    absolute = np.abs(values)
    per_feature = []
    for index in range(values.shape[1]):
        column = values[:, index]
        per_feature.append({
            "index": index,
            "keypoint": index // 18,
            "axis": (index % 18) // 6,
            "joint": index % 6,
            "mean": float(np.mean(column)),
            "std": float(np.std(column)),
            "p01": float(np.quantile(column, .01)),
            "p99": float(np.quantile(column, .99)),
            "max_abs": float(np.max(np.abs(column))),
        })
    report = {
        "config": args.config,
        "samples": args.samples,
        "attempts": attempted,
        "seed": args.seed,
        "cube_side_m": env.keypoint_cube_side_m,
        "shape": [9, 6],
        "overall": {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "p01": float(np.quantile(values, .01)),
            "p99": float(np.quantile(values, .99)),
            "max_abs": float(np.max(absolute)),
            "fraction_abs_gt_1": float(np.mean(absolute > 1.0)),
        },
        "per_feature": per_feature,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
