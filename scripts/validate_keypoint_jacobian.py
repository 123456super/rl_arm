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
from rl_risk_sac.tasks.thesis_reaching import (
    keypoint_position_jacobian,
    pose_keypoint_offsets_world,
)
from rl_risk_sac.utils.config import load_config


def keypoints(env: ThesisHomotopyEnv) -> np.ndarray:
    position, quaternion = env.robot.end_effector_pose()
    return np.asarray(position, dtype=np.float64) + pose_keypoint_offsets_world(
        quaternion, cube_side_m=env.keypoint_cube_side_m,
    ).astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate J_KP(q) qdot against a central finite difference of "
            "the three wrist_3_link keypoints."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml",
    )
    parser.add_argument("--samples", type=int, default=1_000)
    parser.add_argument("--delta-q", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=81002)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.samples < 1 or not 0.0 < args.delta_q <= 1e-2:
        parser.error("--samples must be positive and --delta-q must be in (0, 1e-2]")

    config = load_config(ROOT / args.config)
    env = ThesisHomotopyEnv(config)
    rng = np.random.default_rng(args.seed)
    absolute_errors: list[float] = []
    relative_errors: list[float] = []
    accepted = 0
    attempts = 0
    try:
        lower = np.asarray(env.robot.joint_lower_limits, dtype=np.float64)
        upper = np.asarray(env.robot.joint_upper_limits, dtype=np.float64)
        while accepted < args.samples and attempts < 20 * args.samples:
            attempts += 1
            q = rng.uniform(lower, upper)
            direction = rng.normal(size=env.robot.joint_count)
            direction /= np.linalg.norm(direction)
            half_step = 0.5 * args.delta_q * direction
            if np.any(q - half_step < lower) or np.any(q + half_step > upper):
                continue
            env._set_joint_state(q)
            p.performCollisionDetection(physicsClientId=env.client_id)
            events = env._collision_events()
            if events["self_collision"] or events["environment_collision"]:
                continue

            _, quaternion = env.robot.end_effector_pose()
            predicted = keypoint_position_jacobian(
                env.robot.end_effector_jacobian(), quaternion,
                cube_side_m=env.keypoint_cube_side_m,
            ).astype(np.float64) @ direction
            env._set_joint_state(q - half_step)
            minus = keypoints(env).reshape(-1)
            env._set_joint_state(q + half_step)
            plus = keypoints(env).reshape(-1)
            finite_difference = (plus - minus) / args.delta_q
            error = float(np.linalg.norm(predicted - finite_difference))
            absolute_errors.append(error)
            relative_errors.append(
                error / max(float(np.linalg.norm(finite_difference)), 1e-12)
            )
            accepted += 1

        if accepted != args.samples:
            raise RuntimeError(
                f"accepted only {accepted}/{args.samples} legal states after {attempts} attempts"
            )
        env._set_joint_state(np.zeros(env.robot.joint_count, dtype=np.float64))
        link_state = p.getLinkState(
            env.robot.robot_id, env.robot.tool_link_id,
            computeForwardKinematics=True, physicsClientId=env.client_id,
        )
        local_inertial_position = list(map(float, link_state[2]))
    finally:
        env.close()

    absolute = np.asarray(absolute_errors)
    relative = np.asarray(relative_errors)
    report = {
        "config": args.config,
        "samples": args.samples,
        "attempts": attempts,
        "seed": args.seed,
        "delta_q_rad": args.delta_q,
        "reference_point": "wrist_3_link URDF link-frame origin",
        "coordinate_frame": "PyBullet world/base frame",
        "calculate_jacobian_local_position": [0.0, 0.0, 0.0],
        "link_local_inertial_position_m": local_inertial_position,
        "absolute_l2_error_m_per_rad": {
            "mean": float(np.mean(absolute)),
            "p50": float(np.quantile(absolute, .50)),
            "p95": float(np.quantile(absolute, .95)),
            "p99": float(np.quantile(absolute, .99)),
            "max": float(np.max(absolute)),
        },
        "relative_l2_error": {
            "mean": float(np.mean(relative)),
            "p50": float(np.quantile(relative, .50)),
            "p95": float(np.quantile(relative, .95)),
            "p99": float(np.quantile(relative, .99)),
            "max": float(np.max(relative)),
        },
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
