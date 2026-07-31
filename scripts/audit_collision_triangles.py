from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p

from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config


MESH_TO_CAPSULE = {
    "shoulder_link": "shoulder",
    "upper_arm_link": "upper_arm",
    "forearm_link": "forearm",
    "wrist_1_link": "wrist_1",
    "wrist_2_link": "wrist_2",
    "wrist_3_link": "wrist_3",
}


def read_binary_stl_triangles(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"STL is too short: {path}")
    count = struct.unpack_from("<I", data, 80)[0]
    if 84 + 50 * count != len(data):
        raise ValueError(f"Only binary STL is supported and size is invalid: {path}")
    raw = np.frombuffer(data, dtype=np.uint8, offset=84).reshape(count, 50)
    return np.frombuffer(raw[:, 12:48].tobytes(), dtype="<f4").reshape(count, 3, 3).astype(np.float64)


def transform_points(points: np.ndarray, position: Any, orientation: Any) -> np.ndarray:
    rotation = np.asarray(p.getMatrixFromQuaternion(orientation), dtype=np.float64).reshape(3, 3)
    return np.asarray(position, dtype=np.float64)[None, :] + points @ rotation.T


def to_link_frame(points: np.ndarray, shape: tuple[Any, ...], link_state: tuple[Any, ...]) -> np.ndarray:
    frame_position, frame_orientation = p.multiplyTransforms(
        link_state[0], link_state[1], shape[5], shape[6]
    )
    world = transform_points(points, frame_position, frame_orientation)
    inverse_position, inverse_orientation = p.invertTransform(link_state[4], link_state[5])
    return transform_points(world, inverse_position, inverse_orientation)


def point_segment_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1.0e-14:
        return np.linalg.norm(points - start, axis=1)
    fraction = np.clip(((points - start) @ segment) / length_sq, 0.0, 1.0)
    closest = start[None, :] + fraction[:, None] * segment[None, :]
    return np.linalg.norm(points - closest, axis=1)


def sample_triangles(triangles: np.ndarray, samples_per_triangle: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = [triangles.reshape(-1, 3)]
    if samples_per_triangle:
        weights = rng.random((len(triangles), samples_per_triangle, 2))
        weights.sort(axis=2)
        a = weights[:, :, 0]
        b = weights[:, :, 1] - weights[:, :, 0]
        c = 1.0 - weights[:, :, 1]
        sampled = (
            a[:, :, None] * triangles[:, None, 0]
            + b[:, :, None] * triangles[:, None, 1]
            + c[:, :, None] * triangles[:, None, 2]
        )
        points.append(sampled.reshape(-1, 3))
    return np.concatenate(points, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit collision capsules against sampled STL triangle surfaces.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples-per-triangle", type=int, default=8)
    parser.add_argument("--poses", type=int, default=64)
    parser.add_argument("--seed", type=int, default=6301)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.samples_per_triangle < 0:
        raise ValueError("--samples-per-triangle must be non-negative")
    if args.poses <= 0:
        raise ValueError("--poses must be positive")

    config = load_config(args.config)
    env = UR5DynamicObstacleEnv(config, method=str(config.get("eval", {}).get("method", "link_fixed")))
    try:
        env.reset(seed=args.seed)
        assert env.robot_id is not None
        link_ids = {"base": -1}
        for link_id in range(p.getNumJoints(env.robot_id, physicsClientId=env.physics_client_id)):
            name = p.getJointInfo(env.robot_id, link_id, physicsClientId=env.physics_client_id)[12].decode("utf-8")
            link_ids[name] = link_id
        specs = {spec.name: spec for spec in env.capsule_model.specs}
        result: dict[str, Any] = {
            "config": args.config,
            "seed": args.seed,
            "samples_per_triangle": args.samples_per_triangle,
            "poses": args.poses,
            "links": {},
        }
        rng = np.random.default_rng(args.seed)
        defaults = np.asarray(config["robot"]["reset"]["default_joint_positions"], dtype=np.float64)
        joint_ids = env.joint_ids
        for mesh_link, capsule_name in MESH_TO_CAPSULE.items():
            result["links"][mesh_link] = {
                "capsule": capsule_name,
                "triangle_count": 0,
                "sample_count": 0,
                "configured_radius_m": float(next(spec.radius for spec in env.capsule_model.specs if spec.name == capsule_name)),
                "max_required_radius_m": 0.0,
                "max_radius_excess_m": float("-inf"),
                "worst_pose": -1,
                "coverage_ok_at_samples": False,
            }
        mesh_data = {}
        for mesh_link in MESH_TO_CAPSULE:
            link_id = link_ids[mesh_link]
            shape = p.getCollisionShapeData(env.robot_id, link_id, physicsClientId=env.physics_client_id)[0]
            filename = shape[4].decode("utf-8") if isinstance(shape[4], bytes) else str(shape[4])
            mesh_data[mesh_link] = (shape, read_binary_stl_triangles(Path(filename)))
        for pose in range(args.poses):
            positions = defaults if pose == 0 else defaults + rng.uniform(-0.45, 0.45, size=len(joint_ids))
            for joint_id, position in zip(joint_ids, positions, strict=True):
                p.resetJointState(env.robot_id, joint_id, float(position), targetVelocity=0.0, physicsClientId=env.physics_client_id)
            p.stepSimulation(physicsClientId=env.physics_client_id)
            for mesh_link, capsule_name in MESH_TO_CAPSULE.items():
                link_id = link_ids[mesh_link]
                shape, triangles = mesh_data[mesh_link]
                link_state = p.getLinkState(
                    env.robot_id, link_id, computeForwardKinematics=True, physicsClientId=env.physics_client_id
                )
                link_points = to_link_frame(sample_triangles(triangles, args.samples_per_triangle, args.seed + pose), shape, link_state)
                spec = specs[capsule_name]
                if spec.start_local_position is None or spec.end_local_position is None:
                    raise ValueError(f"{capsule_name} must define local endpoints")
                start = np.asarray(spec.start_local_position, dtype=np.float64)
                end = np.asarray(spec.end_local_position, dtype=np.float64)
                max_required = float(np.max(point_segment_distances(link_points, start, end)))
                row = result["links"][mesh_link]
                row["triangle_count"] = int(len(triangles))
                row["sample_count"] += int(len(link_points))
                excess = max_required - float(spec.radius)
                if excess > row["max_radius_excess_m"]:
                    row["max_required_radius_m"] = max_required
                    row["max_radius_excess_m"] = excess
                    row["worst_pose"] = pose
        for row in result["links"].values():
            row["coverage_ok_at_samples"] = bool(row["max_radius_excess_m"] <= 1.0e-9)
        result["all_sampled_points_covered"] = all(
            row["coverage_ok_at_samples"] for row in result["links"].values()
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"all_sampled_points_covered": result["all_sampled_points_covered"], "output": str(output)}))
    finally:
        env.close()


if __name__ == "__main__":
    main()
