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


def read_binary_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"STL is too short: {path}")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + 50 * triangle_count
    if expected_size != len(data):
        raise ValueError(f"Only binary STL is supported and size is invalid: {path}")
    raw = np.frombuffer(data, dtype=np.uint8, offset=84).reshape(triangle_count, 50)
    vertices = np.frombuffer(raw[:, 12:48].tobytes(), dtype="<f4").reshape(-1, 3)
    return np.unique(vertices.astype(np.float64), axis=0)


def transform_points(points: np.ndarray, position: Any, orientation: Any) -> np.ndarray:
    rotation = np.asarray(p.getMatrixFromQuaternion(orientation), dtype=np.float64).reshape(3, 3)
    return np.asarray(position, dtype=np.float64)[None, :] + points @ rotation.T


def point_segment_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1.0e-14:
        return np.linalg.norm(points - start, axis=1)
    fraction = np.clip(((points - start) @ segment) / length_sq, 0.0, 1.0)
    closest = start[None, :] + fraction[:, None] * segment[None, :]
    return np.linalg.norm(points - closest, axis=1)


def link_local_mesh(points: np.ndarray, shape: tuple[Any, ...], link_state: tuple[Any, ...]) -> np.ndarray:
    # Collision-shape local frame is relative to the inertial frame (state 0/1).
    frame_position, frame_orientation = p.multiplyTransforms(
        link_state[0], link_state[1], shape[5], shape[6]
    )
    world_points = transform_points(points, frame_position, frame_orientation)
    inverse_position, inverse_orientation = p.invertTransform(link_state[4], link_state[5])
    return transform_points(world_points, inverse_position, inverse_orientation)


def fit_capsule(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = points.mean(axis=0)
    _, _, vectors = np.linalg.svd(points - center, full_matrices=False)
    axis = vectors[0]
    projection = (points - center) @ axis
    start = center + axis * float(np.min(projection))
    end = center + axis * float(np.max(projection))
    radius = float(np.max(point_segment_distances(points, start, end)))
    return start, end, radius


def link_id_map(robot_id: int, physics_client_id: int) -> dict[str, int]:
    result = {"base": -1}
    for link_id in range(p.getNumJoints(robot_id, physicsClientId=physics_client_id)):
        name = p.getJointInfo(robot_id, link_id, physicsClientId=physics_client_id)[12].decode("utf-8")
        result[name] = link_id
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit collision meshes in the correct link frame.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=6301)
    parser.add_argument("--margin-m", type=float, default=0.003)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-output", required=True)
    args = parser.parse_args()
    if args.margin_m < 0.0:
        raise ValueError("--margin-m must be non-negative")

    config = load_config(args.config)
    env = UR5DynamicObstacleEnv(config, method=str(config.get("eval", {}).get("method", "link_fixed")))
    try:
        env.reset(seed=args.seed)
        assert env.robot_id is not None
        ids = link_id_map(env.robot_id, env.physics_client_id)
        by_name = {spec.name: spec for spec in env.capsule_model.specs}
        result: dict[str, Any] = {"config": args.config, "seed": args.seed, "margin_m": args.margin_m, "links": {}}
        capsule_specs: list[dict[str, Any]] = []
        for mesh_link, capsule_name in MESH_TO_CAPSULE.items():
            link_id = ids[mesh_link]
            shapes = p.getCollisionShapeData(env.robot_id, link_id, physicsClientId=env.physics_client_id)
            if not shapes:
                raise RuntimeError(f"No collision shape found for {mesh_link}")
            shape = shapes[0]
            filename = shape[4].decode("utf-8") if isinstance(shape[4], bytes) else str(shape[4])
            link_state = p.getLinkState(
                env.robot_id, link_id, computeForwardKinematics=True, physicsClientId=env.physics_client_id
            )
            local_vertices = link_local_mesh(read_binary_stl_vertices(Path(filename)), shape, link_state)
            start, end, radius = fit_capsule(local_vertices)
            radius_with_margin = radius + args.margin_m
            result["links"][mesh_link] = {
                "capsule": capsule_name,
                "start_link_frame": start.tolist(),
                "end_link_frame": end.tolist(),
                "fitted_radius_m": radius,
                "configured_radius_m": radius_with_margin,
                "vertex_count": int(len(local_vertices)),
            }
            old = by_name[capsule_name]
            capsule_specs.append(
                {
                    "name": capsule_name,
                    "parent_link_name": mesh_link,
                    "child_link_name": mesh_link,
                    "start_local_position": [float(x) for x in start],
                    "end_local_position": [float(x) for x in end],
                    "radius": radius_with_margin,
                }
            )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        config_output = Path(args.config_output)
        config_output.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Generated from collision meshes with inertial-to-link frame correction.",
            "# Offline diagnostic only; do not mix with historical B4 results.",
            "includes:",
            "  - b4_osqp_strict_margin30_heldout.yaml",
            "",
            "robot:",
            "  capsules:",
        ]
        for spec in capsule_specs:
            lines.append(f"    - {json.dumps(spec, ensure_ascii=True)}")
        lines += ["", "eval:", "  method: link_fixed", "  episodes: 20", f"  seed: {args.seed}", ""]
        config_output.write_text("\n".join(lines), encoding="utf-8")
        print(json.dumps({"output": str(output), "config_output": str(config_output)}, ensure_ascii=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
