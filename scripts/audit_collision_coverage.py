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
    raw = np.frombuffer(data, dtype=np.uint8, offset=84)
    records = raw.reshape(triangle_count, 50)
    vertices = np.frombuffer(records[:, 12:48].tobytes(), dtype="<f4").reshape(-1, 3)
    return np.unique(vertices.astype(np.float64), axis=0)


def point_segment_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1.0e-14:
        return np.linalg.norm(points - start, axis=1)
    fraction = np.clip(((points - start) @ segment) / length_sq, 0.0, 1.0)
    closest = start[None, :] + fraction[:, None] * segment[None, :]
    return np.linalg.norm(points - closest, axis=1)


def transform_points(
    points: np.ndarray,
    frame_position: Any,
    frame_orientation: Any,
    link_position: Any,
    link_orientation: Any,
) -> np.ndarray:
    local_position, local_orientation = p.multiplyTransforms(
        link_position,
        link_orientation,
        frame_position,
        frame_orientation,
    )
    rotation = np.asarray(p.getMatrixFromQuaternion(local_orientation), dtype=np.float64).reshape(3, 3)
    return np.asarray(local_position, dtype=np.float64)[None, :] + points @ rotation.T


def capsule_from_mesh_vertices(
    vertices: np.ndarray, frame_position: Any, frame_orientation: Any
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a principal-axis segment in the link frame and return its radius."""
    local_vertices = transform_points(
        vertices,
        frame_position,
        frame_orientation,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    )
    center = local_vertices.mean(axis=0)
    _, _, right_singular_vectors = np.linalg.svd(local_vertices - center, full_matrices=False)
    axis = right_singular_vectors[0]
    projection = (local_vertices - center) @ axis
    start = center + axis * float(np.min(projection))
    end = center + axis * float(np.max(projection))
    radius = float(np.max(point_segment_distances(local_vertices, start, end)))
    return start, end, radius


def _link_id_map(robot_id: int, physics_client_id: int) -> dict[str, int]:
    names = {"base": -1}
    for link_id in range(p.getNumJoints(robot_id, physicsClientId=physics_client_id)):
        name = p.getJointInfo(robot_id, link_id, physicsClientId=physics_client_id)[12].decode("utf-8")
        names[name] = link_id
    return names


def _shape_mesh_path(shape: tuple[Any, ...]) -> Path:
    filename = shape[4]
    if isinstance(filename, bytes):
        filename = filename.decode("utf-8")
    return Path(str(filename))


def audit_collision_coverage(config_path: str | Path, samples: int, seed: int) -> dict[str, Any]:
    config = load_config(config_path)
    method = str(config.get("eval", {}).get("method", "link_fixed"))
    env = UR5DynamicObstacleEnv(config, method=method)
    try:
        env.reset(seed=seed)
        assert env.robot_id is not None
        link_ids = _link_id_map(env.robot_id, env.physics_client_id)
        capsule_by_name = {spec.name: (spec, index) for index, spec in enumerate(env.capsule_model.specs)}
        mesh_data: dict[str, tuple[int, np.ndarray, tuple[Any, ...]]] = {}
        for link_name in MESH_TO_CAPSULE:
            link_id = link_ids[link_name]
            shapes = p.getCollisionShapeData(env.robot_id, link_id, physicsClientId=env.physics_client_id)
            if not shapes:
                raise RuntimeError(f"No collision shape found for {link_name}")
            shape = shapes[0]
            mesh_data[link_name] = (link_id, read_binary_stl_vertices(_shape_mesh_path(shape)), shape)

        rng = np.random.default_rng(seed)
        defaults = np.asarray(config["robot"]["reset"]["default_joint_positions"], dtype=np.float64)
        joint_ids = env.joint_ids
        rows: dict[str, dict[str, Any]] = {}
        for mesh_link, capsule_name in MESH_TO_CAPSULE.items():
            spec, capsule_index = capsule_by_name[capsule_name]
            rows[mesh_link] = {
                "capsule": capsule_name,
                "capsule_index": capsule_index,
                "configured_radius_m": float(spec.radius),
                "max_required_radius_m": 0.0,
                "max_radius_excess_m": float("-inf"),
                "worst_sample": -1,
                "vertex_count": int(mesh_data[mesh_link][1].shape[0]),
            }

        # A mesh-faithful, link-frame capsule proposal is used only for audit.
        # It avoids conflating a URDF collision origin with the joint-origin line.
        mesh_capsules: dict[str, dict[str, Any]] = {}
        for mesh_link, (_, vertices, shape) in mesh_data.items():
            start, end, radius = capsule_from_mesh_vertices(vertices, shape[5], shape[6])
            mesh_capsules[mesh_link] = {
                "start_link_frame": start.tolist(),
                "end_link_frame": end.tolist(),
                "radius_m": radius,
            }

        for sample in range(samples):
            if sample == 0:
                joint_positions = defaults
            else:
                joint_positions = defaults + rng.uniform(-0.45, 0.45, size=len(joint_ids))
            for joint_id, position in zip(joint_ids, joint_positions, strict=True):
                p.resetJointState(env.robot_id, joint_id, float(position), targetVelocity=0.0, physicsClientId=env.physics_client_id)
            p.stepSimulation(physicsClientId=env.physics_client_id)
            capsules = env._capsules()
            for mesh_link, (link_id, vertices, shape) in mesh_data.items():
                link_state = p.getLinkState(
                    env.robot_id,
                    link_id,
                    computeForwardKinematics=True,
                    physicsClientId=env.physics_client_id,
                )
                # Collision-shape frames are relative to the link inertial frame
                # (state[0]/state[1]); capsule endpoints use the link frame.
                world_vertices = transform_points(vertices, shape[5], shape[6], link_state[0], link_state[1])
                row = rows[mesh_link]
                capsule_state = capsules[row["capsule_index"]]
                required_radius = float(
                    np.max(point_segment_distances(world_vertices, capsule_state.start, capsule_state.end))
                )
                excess = required_radius - float(row["configured_radius_m"])
                if excess > row["max_radius_excess_m"]:
                    row["max_required_radius_m"] = required_radius
                    row["max_radius_excess_m"] = excess
                    row["worst_sample"] = sample

        for row in rows.values():
            row["coverage_ok_at_vertices"] = bool(row["max_radius_excess_m"] <= 1.0e-6)
            del row["capsule_index"]
        return {
            "config": str(config_path),
            "seed": int(seed),
            "samples": int(samples),
            "note": "Vertex-only coverage audit; passing does not prove full triangle-surface containment.",
            "links": rows,
            "mesh_fit_capsules": mesh_capsules,
            "all_vertices_covered": all(row["coverage_ok_at_vertices"] for row in rows.values()),
        }
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit collision-mesh vertices against configured capsules.")
    parser.add_argument("--config", default="configs/experiments/p3_diagnostics/b4_osqp_strict_margin30_heldout.yaml")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    result = audit_collision_coverage(args.config, args.samples, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_vertices_covered": result["all_vertices_covered"], "output": str(output)}))


if __name__ == "__main__":
    main()
