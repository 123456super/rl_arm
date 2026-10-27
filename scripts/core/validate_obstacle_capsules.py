#!/usr/bin/env python3
"""Check original URDF collision-mesh coverage, including both distal wrist links."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.utils.config import load_config

GROUP_LINKS = {"upper_arm": ["upper_arm_link"], "forearm": ["forearm_link"],
               "wrist_1": ["wrist_1_link"], "wrist_2": ["wrist_2_link", "wrist_3_link"]}


def stl_vertices(path):
    content = path.read_bytes()
    triangles = struct.unpack_from("<I", content, 80)[0] if len(content) >= 84 else -1
    if len(content) == 84 + 50 * triangles:
        dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
        return np.frombuffer(content, dtype=dtype, count=triangles, offset=84)["vertices"].reshape(-1, 3).astype(float)
    return np.array([[float(v) for v in line.split()[1:]] for line in content.decode().splitlines()
                     if line.strip().startswith("vertex ")])


def collision_vertices(urdf):
    vertices = {}
    tree = ET.parse(urdf)
    for link in tree.findall("link"):
        rows = []
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                raise ValueError(f"coverage validator requires mesh collision geometry: {link.attrib['name']}")
            points = stl_vertices(urdf.parent / mesh.attrib["filename"])
            points *= np.fromstring(mesh.attrib.get("scale", "1 1 1"), sep=" ")
            origin = collision.find("origin")
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rotation = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(rpy))).reshape(3, 3)
            rows.append(points @ rotation.T + xyz)
        if rows:
            vertices[link.attrib["name"]] = np.concatenate(rows)
    return vertices


def validate_coverage(config, *, samples=8, wrist_samples=33, tolerance=1e-5, seed=73):
    config = copy.deepcopy(config)
    config["thesis"]["obstacles"]["enabled"] = False
    meshes = collision_vertices(Path(config["robot"]["urdf"]))
    env = ThesisHomotopyEnv(config)
    rng = np.random.default_rng(seed)
    result = {group: dict(max_axis_distance_m=0., radius_m=0., vertices_checked=0) for group in GROUP_LINKS}
    try:
        env.reset(seed=seed)
        lower, upper = env.robot.joint_lower_limits, env.robot.joint_upper_limits
        for _ in range(samples):
            q = rng.uniform(lower, upper)
            for angle in np.linspace(lower[-1], upper[-1], wrist_samples):
                q[-1] = angle
                env._set_joint_state(q)
                for capsule in env.robot.capsules():
                    if capsule.name not in GROUP_LINKS:
                        continue
                    row = result[capsule.name]
                    row["radius_m"] = capsule.radius
                    axis = capsule.end - capsule.start
                    for name in GROUP_LINKS[capsule.name]:
                        link = env.robot.capsule_model.link_name_to_id[name]
                        state = p.getLinkState(env.robot_id, link, computeForwardKinematics=True, physicsClientId=env.client_id)
                        rotation = np.array(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
                        points = meshes[name] @ rotation.T + state[4]
                        fractions = np.clip((points - capsule.start) @ axis / max(axis @ axis, 1e-30), 0, 1)
                        distance = np.linalg.norm(points - capsule.start - fractions[:, None] * axis, axis=1)
                        row["max_axis_distance_m"] = max(row["max_axis_distance_m"], float(distance.max()))
                        row["vertices_checked"] += len(points)
        for row in result.values():
            row["coverage_passed"] = row["max_axis_distance_m"] <= row["radius_m"] + tolerance
            row["minimum_radius_m"] = float(np.ceil((row["max_axis_distance_m"] + tolerance) * 1000) / 1000)
    finally:
        env.close()
    return dict(passed=all(row["coverage_passed"] for row in result.values()), groups=result,
                samples=samples, wrist_samples=wrist_samples, tolerance_m=tolerance, seed=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base/thesis_common.yaml")
    parser.add_argument("--output", default="outputs/capsule_coverage.json")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--wrist-samples", type=int, default=33)
    args = parser.parse_args()
    report = validate_coverage(load_config(args.config), samples=args.samples, wrist_samples=args.wrist_samples)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
