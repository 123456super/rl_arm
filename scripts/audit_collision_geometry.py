from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pybullet as p

from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config


def _vector(values: Any) -> list[float]:
    return [float(value) for value in values]


def _json_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _link_name_map(robot_id: int, physics_client_id: int) -> dict[int, str]:
    names = {-1: "base"}
    for link_id in range(p.getNumJoints(robot_id, physicsClientId=physics_client_id)):
        names[link_id] = p.getJointInfo(robot_id, link_id, physicsClientId=physics_client_id)[12].decode(
            "utf-8"
        )
    return names


def collect_geometry_audit(config_path: str | Path, seed: int) -> dict[str, Any]:
    config = load_config(config_path)
    method = str(config.get("eval", {}).get("method", "link_fixed"))
    env = UR5DynamicObstacleEnv(config, method=method)
    try:
        _, reset_info = env.reset(seed=seed)
        assert env.robot_id is not None
        link_names = _link_name_map(env.robot_id, env.physics_client_id)
        collision_shapes: list[dict[str, Any]] = []
        for link_id, link_name in sorted(link_names.items()):
            shapes = p.getCollisionShapeData(
                env.robot_id,
                link_id,
                physicsClientId=env.physics_client_id,
            )
            for shape in shapes or []:
                aabb_min, aabb_max = p.getAABB(
                    env.robot_id,
                    link_id,
                    physicsClientId=env.physics_client_id,
                )
                collision_shapes.append(
                    {
                        "link_id": link_id,
                        "link_name": link_name,
                        "geometry_type": int(shape[2]),
                        "dimensions": _vector(shape[3]),
                        "filename": str(shape[4]),
                        "local_frame_position": _vector(shape[5]),
                        "local_frame_orientation": _vector(shape[6]),
                        "world_aabb_min": _vector(aabb_min),
                        "world_aabb_max": _vector(aabb_max),
                    }
                )

        capsules = []
        for spec, state in zip(env.capsule_model.specs, env._capsules(), strict=True):
            parent_id = env.capsule_model.link_name_to_id[spec.parent_link_name]
            child_id = env.capsule_model.link_name_to_id[spec.child_link_name]
            capsules.append(
                {
                    "name": spec.name,
                    "parent_link_name": spec.parent_link_name,
                    "child_link_name": spec.child_link_name,
                    "parent_link_id": parent_id,
                    "child_link_id": child_id,
                    "radius_m": float(state.radius),
                    "start_world": _vector(state.start),
                    "end_world": _vector(state.end),
                }
            )
        return {
            "config": str(config_path),
            "seed": int(seed),
            "robot_urdf": str(env.robot_urdf),
            "reset_info": _json_safe(reset_info),
            "capsules": capsules,
            "collision_shapes": collision_shapes,
        }
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report URDF collision local transforms alongside configured capsule endpoints."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = collect_geometry_audit(args.config, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"saved collision geometry audit: {output}")


if __name__ == "__main__":
    main()
