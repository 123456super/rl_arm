from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pybullet as p

from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config
from rl_risk_sac.utils.risk import closest_point_on_segment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare capsule gaps with PyBullet collision-shape gaps.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--samples-per-link", type=int, default=400)
    parser.add_argument("--seed", type=int, default=71001)
    parser.add_argument("--geometry-margin", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        default="outputs/restart_2026-09-09/r1_geometry/calibration_seed71001",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_link < 20:
        raise ValueError("--samples-per-link must be at least 20")
    config = load_config(args.config)
    geometry_margin = (
        float(config["risk"]["geometry_margin"])
        if args.geometry_margin is None
        else float(args.geometry_margin)
    )
    if geometry_margin < 0.0:
        raise ValueError("--geometry-margin must be non-negative")
    config["device"] = "cpu"
    env = UR5DynamicObstacleEnv(config, method="link_fixed")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty calibration directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    try:
        env.reset(seed=args.seed)
        joint_ranges = []
        for joint_id in env.joint_ids:
            info = p.getJointInfo(env.robot_id, joint_id, physicsClientId=env.physics_client_id)
            low, high = float(info[8]), float(info[9])
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                low, high = -np.pi, np.pi
            joint_ranges.append((max(low, -np.pi), min(high, np.pi)))

        rows: list[dict[str, object]] = []
        for target_index, target_spec in enumerate(env.capsule_model.specs):
            for sample_index in range(args.samples_per_link):
                q = np.asarray([rng.uniform(low, high) for low, high in joint_ranges], dtype=np.float32)
                for joint_id, value in zip(env.joint_ids, q):
                    p.resetJointState(
                        env.robot_id,
                        joint_id,
                        float(value),
                        targetVelocity=0.0,
                        physicsClientId=env.physics_client_id,
                    )
                capsules = env._capsules()
                target = capsules[target_index]
                rho = rng.uniform(0.05, 0.95)
                centerline_point = target.start + rho * (target.end - target.start)
                direction = rng.normal(size=3)
                direction /= max(float(np.linalg.norm(direction)), 1e-12)
                intended_gap = rng.uniform(-0.03, 0.30)
                obstacle_center = centerline_point + direction * (
                    target.radius + env.obstacle_cfg.radius + intended_gap
                )
                p.resetBasePositionAndOrientation(
                    env.obstacle_id,
                    obstacle_center.tolist(),
                    [0.0, 0.0, 0.0, 1.0],
                    physicsClientId=env.physics_client_id,
                )
                p.performCollisionDetection(physicsClientId=env.physics_client_id)

                capsule_gaps = []
                for capsule in capsules:
                    closest, _ = closest_point_on_segment(obstacle_center, capsule.start, capsule.end)
                    capsule_gaps.append(
                        float(np.linalg.norm(obstacle_center - closest))
                        - capsule.radius
                        - env.obstacle_cfg.radius
                    )
                capsule_distance = float(np.min(capsule_gaps))
                capsule_closest = int(np.argmin(capsule_gaps))
                closest_points = p.getClosestPoints(
                    bodyA=env.robot_id,
                    bodyB=env.obstacle_id,
                    distance=2.0,
                    physicsClientId=env.physics_client_id,
                )
                if not closest_points:
                    raise RuntimeError("PyBullet returned no robot-obstacle closest point within 2 m")
                actual = min(closest_points, key=lambda point: float(point[8]))
                bullet_distance = float(actual[8])
                rows.append(
                    {
                        "target_capsule": target_spec.name,
                        "target_capsule_index": target_index,
                        "sample_index": sample_index,
                        "capsule_closest_index": capsule_closest,
                        "bullet_closest_link_index": int(actual[3]),
                        "capsule_distance_m": capsule_distance,
                        "bullet_distance_m": bullet_distance,
                        "capsule_minus_bullet_m": capsule_distance - bullet_distance,
                        "capsule_violation": int(capsule_distance < env.risk_config.d_safe),
                        "bullet_violation": int(bullet_distance < env.risk_config.d_safe),
                        "capsule_collision": int(capsule_distance <= 0.0),
                        "bullet_collision": int(bullet_distance <= 0.0),
                    }
                )

        csv_path = output_dir / "samples.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        errors = np.asarray([float(row["capsule_minus_bullet_m"]) for row in rows])
        bullet_violation = np.asarray([bool(row["bullet_violation"]) for row in rows])
        capsule_violation = np.asarray([bool(row["capsule_violation"]) for row in rows])
        bullet_collision = np.asarray([bool(row["bullet_collision"]) for row in rows])
        capsule_collision = np.asarray([bool(row["capsule_collision"]) for row in rows])
        adjusted_capsule_violation = np.asarray(
            [float(row["capsule_distance_m"]) - geometry_margin < env.risk_config.d_safe for row in rows]
        )
        summary = {
            "schema_version": "capsule_geometry_calibration_v1",
            "config": args.config,
            "seed": args.seed,
            "samples_per_link": args.samples_per_link,
            "sample_count": len(rows),
            "safe_distance_m": env.risk_config.d_safe,
            "frozen_geometry_margin_m": geometry_margin,
            "error_definition": "capsule_distance_minus_pybullet_collision_shape_distance",
            "error_quantiles_m": _quantiles(errors),
            "danger_false_negative_count": int(np.sum(bullet_violation & ~capsule_violation)),
            "danger_false_negative_rate": float(
                np.sum(bullet_violation & ~capsule_violation) / max(np.sum(bullet_violation), 1)
            ),
            "adjusted_danger_false_negative_count": int(
                np.sum(bullet_violation & ~adjusted_capsule_violation)
            ),
            "adjusted_danger_false_negative_rate": float(
                np.sum(bullet_violation & ~adjusted_capsule_violation) / max(np.sum(bullet_violation), 1)
            ),
            "collision_false_negative_count": int(np.sum(bullet_collision & ~capsule_collision)),
            "collision_false_negative_rate": float(
                np.sum(bullet_collision & ~capsule_collision) / max(np.sum(bullet_collision), 1)
            ),
            "per_target_capsule": {},
        }
        for spec in env.capsule_model.specs:
            selected = [row for row in rows if row["target_capsule"] == spec.name]
            selected_errors = np.asarray([float(row["capsule_minus_bullet_m"]) for row in selected])
            selected_bullet_violation = np.asarray([bool(row["bullet_violation"]) for row in selected])
            selected_capsule_violation = np.asarray([bool(row["capsule_violation"]) for row in selected])
            summary["per_target_capsule"][spec.name] = {
                "sample_count": len(selected),
                "error_quantiles_m": _quantiles(selected_errors),
                "danger_false_negative_count": int(
                    np.sum(selected_bullet_violation & ~selected_capsule_violation)
                ),
            }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        env.close()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


if __name__ == "__main__":
    main()
