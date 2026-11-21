#!/usr/bin/env python3
"""Audit UR5 wrist capsule radii against the bundled collision meshes.

This is deliberately an offline geometry audit: it never mutates the robot
configuration. A single capsule cannot exactly reproduce the stepped wrist
meshes, so silently shrinking radii to remove false-positive proximity would
also create false-negative collision regions. The report states the minimum
radius that encloses every STL vertex for the configured capsule centreline.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _binary_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"invalid STL: {path}")
    triangles = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * triangles:
        raise ValueError(f"only binary STL is supported: {path}")
    vertices = np.empty((triangles * 3, 3), dtype=np.float64)
    for index in range(triangles):
        values = struct.unpack_from("<12fH", data, 84 + 50 * index)
        vertices[index * 3:(index + 1) * 3] = np.asarray(
            values[3:12], dtype=np.float64
        ).reshape(3, 3)
    return vertices


def _rotation_rpy(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _capsule_audit(
    vertices: np.ndarray,
    *,
    origin: tuple[float, float, float],
    rpy: tuple[float, float, float],
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    configured_radius: float,
) -> dict[str, float | int | bool]:
    transformed = (
        vertices @ _rotation_rpy(rpy).T
        + np.asarray(origin, dtype=np.float64)
    )
    first = np.asarray(start, dtype=np.float64)
    second = np.asarray(end, dtype=np.float64)
    axis = second - first
    length_squared = float(np.dot(axis, axis))
    if length_squared <= 1.0e-16:
        closest = np.repeat(first[None, :], len(transformed), axis=0)
        outside_fraction = 0.0
    else:
        raw_fraction = ((transformed - first) @ axis) / length_squared
        fraction = np.clip(raw_fraction, 0.0, 1.0)
        closest = first + fraction[:, None] * axis
        outside_fraction = float(np.mean(
            (raw_fraction < 0.0) | (raw_fraction > 1.0)
        ))
    distances = np.linalg.norm(transformed - closest, axis=1)
    required = float(np.max(distances))
    return {
        "vertex_count": int(len(transformed)),
        "configured_radius_m": float(configured_radius),
        "minimum_enclosing_radius_m": required,
        "configured_encloses_all_vertices": bool(
            configured_radius + 1.0e-9 >= required
        ),
        "radius_error_m": float(configured_radius - required),
        "distance_p99_m": float(np.quantile(distances, 0.99)),
        "vertices_beyond_segment_fraction": outside_fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    mesh_root = ROOT / "assets/robots/universal_robots/ur_models/meshes/ur5/collision"
    specifications = {
        "wrist_1": {
            "mesh": "wrist1.stl", "origin": (0.0, 0.0, -0.093),
            "rpy": (np.pi / 2.0, 0.0, 0.0),
            "start": (0.0, 0.0, 0.0), "end": (0.0, 0.0, 0.0),
            "radius": 0.067,
        },
        "wrist_2": {
            "mesh": "wrist2.stl", "origin": (0.0, 0.0, -0.095),
            "rpy": (0.0, 0.0, 0.0),
            "start": (0.0, 0.0, 0.0), "end": (0.0, 0.0, 0.0),
            "radius": 0.067,
        },
        "wrist_3": {
            "mesh": "wrist3.stl", "origin": (0.0, 0.0, -0.0818),
            "rpy": (np.pi / 2.0, 0.0, 0.0),
            "start": (0.0, 0.0, 0.0), "end": (0.0, 0.0, 0.0),
            "radius": 0.055,
        },
    }
    report = {}
    for name, spec in specifications.items():
        vertices = _binary_stl_vertices(mesh_root / str(spec["mesh"]))
        report[name] = _capsule_audit(
            vertices,
            origin=spec["origin"], rpy=spec["rpy"],
            start=spec["start"], end=spec["end"],
            configured_radius=float(spec["radius"]),
        )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
