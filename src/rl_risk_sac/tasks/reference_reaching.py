from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_risk_sac.robots.ur5_capsules import CapsuleState
from rl_risk_sac.utils.risk import closest_point_on_segment


REFERENCE_PPO_OBSERVATION_DIM = 25
SELF_DISTANCE_UPPER = 0.25


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a PyBullet ``[x, y, z, w]`` quaternion to a rotation matrix."""
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1.0e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def so3_log(rotation: np.ndarray) -> np.ndarray:
    """Principal SO(3) logarithm as a three-vector."""
    matrix = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew = np.asarray(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]],
        dtype=np.float64,
    )
    if angle < 1.0e-7:
        return (0.5 * skew).astype(np.float32)
    if np.pi - angle < 1.0e-5:
        diagonal = np.maximum((np.diag(matrix) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        largest = int(np.argmax(axis))
        if axis[largest] <= 1.0e-8:
            axis = np.asarray([1.0, 0.0, 0.0])
        else:
            if largest == 0:
                axis[1] = np.copysign(axis[1], matrix[0, 1] + matrix[1, 0])
                axis[2] = np.copysign(axis[2], matrix[0, 2] + matrix[2, 0])
            elif largest == 1:
                axis[0] = np.copysign(axis[0], matrix[0, 1] + matrix[1, 0])
                axis[2] = np.copysign(axis[2], matrix[1, 2] + matrix[2, 1])
            else:
                axis[0] = np.copysign(axis[0], matrix[0, 2] + matrix[2, 0])
                axis[1] = np.copysign(axis[1], matrix[1, 2] + matrix[2, 1])
            axis /= np.linalg.norm(axis)
        return (angle * axis).astype(np.float32)
    return (angle / (2.0 * np.sin(angle)) * skew).astype(np.float32)


def pose_error(
    goal_position: np.ndarray,
    goal_quaternion: np.ndarray,
    ee_position: np.ndarray,
    ee_quaternion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(goal_position, dtype=np.float32) - np.asarray(ee_position, dtype=np.float32)
    goal_rotation = quaternion_to_matrix(goal_quaternion)
    ee_rotation = quaternion_to_matrix(ee_quaternion)
    orientation = so3_log(goal_rotation @ ee_rotation.T)
    return position.astype(np.float32), orientation.astype(np.float32)


def reference_point_errors(
    goal_position: np.ndarray,
    goal_quaternion: np.ndarray,
    ee_position: np.ndarray,
    ee_quaternion: np.ndarray,
    *,
    offset_m: float = 0.001,
) -> np.ndarray:
    """Return the three 1 mm reference-point errors used by FTP4RM."""
    if offset_m <= 0.0:
        raise ValueError("reference point offset must be positive")
    goal_position = np.asarray(goal_position, dtype=np.float64)
    ee_position = np.asarray(ee_position, dtype=np.float64)
    if goal_position.shape != (3,) or ee_position.shape != (3,):
        raise ValueError("reference point positions must have shape (3,)")
    if np.asarray(goal_quaternion).shape != (4,) or np.asarray(ee_quaternion).shape != (4,):
        raise ValueError("reference point quaternions must have shape (4,)")
    offsets = float(offset_m) * np.eye(3, dtype=np.float64)
    return (
        goal_position[None, :] + offsets - ee_position[None, :] - offsets
    ).astype(np.float32)


def build_reference_observation(
    *,
    q: np.ndarray,
    ee_position: np.ndarray,
    ee_quaternion: np.ndarray,
    goal_position: np.ndarray,
    goal_quaternion: np.ndarray,
    orientation_error: np.ndarray,
    reference_point_offset_m: float = 0.001,
) -> np.ndarray:
    """Build the sole supported 25-dimensional FTP4RM observation."""
    point_errors = reference_point_errors(
        goal_position,
        goal_quaternion,
        ee_position,
        ee_quaternion,
        offset_m=reference_point_offset_m,
    )
    scalar_errors = np.asarray(
        [
            float(np.sum(np.linalg.norm(point_errors, axis=1))),
            float(np.linalg.norm(orientation_error)),
        ],
        dtype=np.float32,
    )
    # S0 has no obstacle. Three zero slots are the explicit absence sentinel.
    obstacle_slots = np.zeros(3, dtype=np.float32)
    observation = np.concatenate(
        [
            np.asarray(q, dtype=np.float32),
            np.asarray(ee_position, dtype=np.float32),
            np.asarray(ee_quaternion, dtype=np.float32),
            np.asarray(goal_position, dtype=np.float32),
            np.asarray(goal_quaternion, dtype=np.float32),
            scalar_errors,
            obstacle_slots,
        ]
    ).astype(np.float32)
    if observation.shape != (REFERENCE_PPO_OBSERVATION_DIM,):
        raise RuntimeError(
            f"reference PPO observation must have shape ({REFERENCE_PPO_OBSERVATION_DIM},), "
            f"got {observation.shape}"
        )
    if not np.isfinite(observation).all():
        raise ValueError("non-finite reference PPO observation")
    return observation


def reference_ppo_reward(
    *,
    reference_error: float,
    collision: bool,
    obstacle_clearances: tuple[float, ...] = (),
    obstacle_threshold: float = 0.08,
    collision_penalty: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """Exact dense reward used by the FTP4RM PPO main experiment."""
    errors = max(float(reference_error), 0.0)
    threshold = float(obstacle_threshold)
    if threshold <= 0.0:
        raise ValueError("reference obstacle threshold must be positive")
    proximity = sum(
        max(0.0, 1.0 - float(distance) / threshold)
        for distance in obstacle_clearances
    )
    dense_reward = -0.001 * errors**2 - float(np.log(errors**2 + 1.0e-4)) - 0.1 * proximity
    collision_cost = float(collision_penalty) * float(collision)
    reward = dense_reward - collision_cost
    return float(reward), {
        "r_goal": float(dense_reward),
        "c_proximity": float(proximity),
        "reference_error": float(errors),
        "reference_obstacle_penalty": float(0.1 * proximity),
        "reference_collision_penalty": collision_cost,
        "hard_penalty": 0.0,
        "safety_penalty": 0.0,
        "terminal_guard_penalty": 0.0,
        "timeout_guard": 0.0,
        "state_cost": 0.0,
        "position_state_cost": 0.0,
        "orientation_state_cost": 0.0,
        "velocity_cost": 0.0,
    }


@dataclass(frozen=True)
class ThesisGeometry:
    relative_vectors: np.ndarray
    distances: np.ndarray
    ttc: np.ndarray
    approach_velocities: np.ndarray
    risk_per_link: np.ndarray
    risk_max: float
    distance_min: float


@dataclass(frozen=True)
class SelfCollisionGeometry:
    distances: np.ndarray
    ttc: np.ndarray
    approach_velocities: np.ndarray
    risk_per_link: np.ndarray
    risk_max: float
    distance_min: float
    ttc_min: float
    approach_max: float
    closest_pair: tuple[int, int]


def compute_self_collision_geometry(
    pair_distances: np.ndarray,
    pair_approaches: np.ndarray,
    pair_links: list[tuple[int, int]],
    *,
    link_count: int = 6,
    d_safe: float = 0.005,
    query_distance: float = SELF_DISTANCE_UPPER,
    ttc_max: float = 3.0,
    approach_velocity_scale: float = 0.7,
    ttc_tau: float = 1.0,
    eps_v: float = 1.0e-4,
) -> SelfCollisionGeometry:
    distances_in = np.asarray(pair_distances, dtype=np.float32)
    approaches_in = np.asarray(pair_approaches, dtype=np.float32)
    if distances_in.shape != approaches_in.shape or distances_in.ndim != 1:
        raise ValueError("self pair distances and approaches must be equal one-dimensional arrays")
    if len(pair_links) != len(distances_in):
        raise ValueError("self pair link mapping must match the pair arrays")
    if d_safe <= 0.0 or query_distance <= d_safe:
        raise ValueError("self distances must satisfy 0 < d_safe < query_distance")
    if ttc_max <= 0.0 or approach_velocity_scale <= 0.0 or ttc_tau <= 0.0:
        raise ValueError("self risk scales must be positive")

    per_link_distance = np.full(link_count, query_distance, dtype=np.float32)
    per_link_ttc = np.full(link_count, ttc_max, dtype=np.float32)
    per_link_approach = np.zeros(link_count, dtype=np.float32)
    per_link_risk = np.zeros(link_count, dtype=np.float32)
    pair_ttc = np.full(len(distances_in), ttc_max, dtype=np.float32)
    pair_risk = np.zeros(len(distances_in), dtype=np.float32)
    for index, (distance_value, approach_value, links) in enumerate(
        zip(distances_in, approaches_in, pair_links)
    ):
        distance = float(min(distance_value, query_distance))
        approach = float(max(approach_value, 0.0))
        if distance <= d_safe:
            ttc = 0.0
        elif approach > eps_v:
            ttc = min((distance - d_safe) / approach, ttc_max)
        else:
            ttc = ttc_max
        clearance_risk = float(np.clip((d_safe - distance) / d_safe, 0.0, 1.0))
        predictive_risk = float(
            np.clip(approach / approach_velocity_scale, 0.0, 1.0)
            * np.exp(-ttc / ttc_tau)
        )
        risk = max(clearance_risk, predictive_risk)
        pair_ttc[index] = ttc
        pair_risk[index] = risk
        for link in links:
            if link < 0:
                continue
            if link >= link_count:
                raise ValueError(f"self pair link index {link} is outside [0, {link_count})")
            per_link_distance[link] = min(per_link_distance[link], distance)
            per_link_ttc[link] = min(per_link_ttc[link], ttc)
            per_link_approach[link] = max(per_link_approach[link], approach)
            per_link_risk[link] = max(per_link_risk[link], risk)
    if len(distances_in):
        closest_index = int(np.argmin(distances_in))
        closest_pair = tuple(int(value) for value in pair_links[closest_index])
        distance_min = float(distances_in[closest_index])
        risk_max = float(np.max(pair_risk))
        ttc_min = float(np.min(pair_ttc))
        approach_max = float(np.max(approaches_in))
    else:
        closest_pair = (-1, -1)
        distance_min = float(query_distance)
        risk_max = 0.0
        ttc_min = float(ttc_max)
        approach_max = 0.0
    return SelfCollisionGeometry(
        distances=per_link_distance,
        ttc=per_link_ttc,
        approach_velocities=per_link_approach,
        risk_per_link=per_link_risk,
        risk_max=risk_max,
        distance_min=distance_min,
        ttc_min=ttc_min,
        approach_max=approach_max,
        closest_pair=closest_pair,
    )


def compute_thesis_geometry(
    capsules: list[CapsuleState],
    obstacle_position: np.ndarray,
    obstacle_velocity: np.ndarray,
    closest_point_velocities: np.ndarray,
    *,
    obstacle_radius: float = 0.075,
    d_safe: float = 0.05,
    ttc_max: float = 3.0,
    eps_norm: float = 1.0e-8,
    eps_v: float = 1.0e-4,
) -> ThesisGeometry:
    relative, distances, ttc_values, approaches, risks = [], [], [], [], []
    obstacle_position = np.asarray(obstacle_position, dtype=np.float32)
    obstacle_velocity = np.asarray(obstacle_velocity, dtype=np.float32)
    for index, capsule in enumerate(capsules):
        closest, _ = closest_point_on_segment(obstacle_position, capsule.start, capsule.end)
        vector = obstacle_position - closest
        norm = float(np.linalg.norm(vector))
        distance = norm - float(capsule.radius) - obstacle_radius
        if norm <= eps_norm:
            approach, ttc = 0.0, 0.0
        else:
            normal = vector / norm
            distance_rate = float(np.dot(normal, obstacle_velocity - closest_point_velocities[index]))
            approach = max(-distance_rate, 0.0)
            if distance <= d_safe:
                ttc = 0.0
            elif approach > eps_v:
                ttc = (distance - d_safe) / approach
            else:
                ttc = ttc_max
        ttc_clipped = float(np.clip(ttc, 0.0, ttc_max))
        risk_distance = float(np.clip(np.exp(-(distance - d_safe) / 0.12), 0.0, 1.0))
        risk_velocity = float(np.clip(approach / 0.7, 0.0, 1.0))
        risk_ttc = float(np.exp(-ttc_clipped))
        risk = float(np.clip(0.5 * risk_distance + 0.2 * risk_velocity + 0.3 * risk_ttc, 0.0, 1.0))
        relative.append(vector)
        distances.append(distance)
        ttc_values.append(ttc_clipped)
        approaches.append(approach)
        risks.append(risk)
    return ThesisGeometry(
        relative_vectors=np.asarray(relative, dtype=np.float32),
        distances=np.asarray(distances, dtype=np.float32),
        ttc=np.asarray(ttc_values, dtype=np.float32),
        approach_velocities=np.asarray(approaches, dtype=np.float32),
        risk_per_link=np.asarray(risks, dtype=np.float32),
        risk_max=float(np.max(risks)),
        distance_min=float(np.min(distances)),
    )
