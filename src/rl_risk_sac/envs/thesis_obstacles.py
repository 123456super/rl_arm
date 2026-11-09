"""Shared oriented-box geometry for static and dynamic obstacle modules."""

from __future__ import annotations

import numpy as np

from rl_risk_sac.tasks.reference_reaching import quaternion_to_matrix


def _horizontal_obb_surface_distance(
    position_a: np.ndarray,
    half_extents_a: np.ndarray,
    quaternion_a: np.ndarray,
    position_b: np.ndarray,
    half_extents_b: np.ndarray,
    quaternion_b: np.ndarray,
) -> float:
    """Exact distance between two OBBs whose rotations are horizontal yaw only."""

    def rectangle_corners(
        position: np.ndarray, half_extents: np.ndarray, quaternion: np.ndarray
    ) -> np.ndarray:
        rotation = quaternion_to_matrix(np.asarray(quaternion, dtype=np.float64))
        axes = rotation[:2, :2]
        signs = np.asarray(
            [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
            dtype=np.float64,
        )
        local = signs * np.asarray(half_extents[:2], dtype=np.float64)
        return np.asarray(position[:2], dtype=np.float64) + local @ axes.T

    def rectangles_overlap(a: np.ndarray, b: np.ndarray) -> bool:
        for polygon in (a, b):
            for index in range(2):
                edge = polygon[(index + 1) % 4] - polygon[index]
                axis = np.asarray([-edge[1], edge[0]], dtype=np.float64)
                norm = float(np.linalg.norm(axis))
                if norm <= 1.0e-12:
                    continue
                axis /= norm
                projection_a = a @ axis
                projection_b = b @ axis
                if projection_a.max() < projection_b.min() or projection_b.max() < projection_a.min():
                    return False
        return True

    def point_segment_distance(
        point: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> float:
        segment = end - start
        denominator = float(segment @ segment)
        if denominator <= 1.0e-15:
            return float(np.linalg.norm(point - start))
        fraction = float(np.clip((point - start) @ segment / denominator, 0.0, 1.0))
        return float(np.linalg.norm(point - (start + fraction * segment)))

    position_a = np.asarray(position_a, dtype=np.float64)
    position_b = np.asarray(position_b, dtype=np.float64)
    half_extents_a = np.asarray(half_extents_a, dtype=np.float64)
    half_extents_b = np.asarray(half_extents_b, dtype=np.float64)
    corners_a = rectangle_corners(position_a, half_extents_a, quaternion_a)
    corners_b = rectangle_corners(position_b, half_extents_b, quaternion_b)
    if rectangles_overlap(corners_a, corners_b):
        horizontal_distance = 0.0
    else:
        horizontal_distance = min(
            min(
                point_segment_distance(
                    point, polygon[(edge + 0) % 4], polygon[(edge + 1) % 4]
                )
                for point in points
                for edge in range(4)
            )
            for points, polygon in ((corners_a, corners_b), (corners_b, corners_a))
        )
    vertical_distance = max(
        0.0,
        abs(float(position_a[2] - position_b[2]))
        - float(half_extents_a[2] + half_extents_b[2]),
    )
    return float(np.hypot(horizontal_distance, vertical_distance))

def _swept_obb_envelope(
    positions: np.ndarray,
    half_extents: np.ndarray,
    quaternion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one conservative OBB containing all translated OBB poses.

    The obstacle orientation is fixed over the prediction horizon.  Expressing
    every predicted centre in that local OBB frame lets one box contain the
    complete piecewise-linear translation sweep without exposing prediction
    points as separate policy-set elements.
    """
    centres = np.asarray(positions, dtype=np.float64)
    extents = np.asarray(half_extents, dtype=np.float64)
    if centres.ndim != 2 or centres.shape[1] != 3 or len(centres) < 1:
        raise ValueError("swept OBB positions must have shape (N, 3), N >= 1")
    if extents.shape != (3,) or np.any(extents <= 0.0):
        raise ValueError("swept OBB half extents must be a positive 3-vector")
    if not np.isfinite(centres).all() or not np.isfinite(extents).all():
        raise ValueError("swept OBB inputs must be finite")
    rotation = quaternion_to_matrix(np.asarray(quaternion, dtype=np.float64))
    origin = centres[0]
    local_displacements = (centres - origin) @ rotation
    local_min = local_displacements.min(axis=0)
    local_max = local_displacements.max(axis=0)
    local_midpoint = 0.5 * (local_min + local_max)
    envelope_center = origin + rotation @ local_midpoint
    envelope_half_extents = extents + 0.5 * (local_max - local_min)
    return (
        envelope_center.astype(np.float32),
        envelope_half_extents.astype(np.float32),
    )

