from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_risk_sac.robots.ur5_capsules import CapsuleState
from rl_risk_sac.utils.risk import closest_point_on_segment


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a PyBullet ``[x,y,z,w]`` quaternion to a rotation matrix."""
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
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
    """Principal SO(3) logarithm as a three-vector, stable near zero and pi."""
    matrix = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew = np.asarray(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]],
        dtype=np.float64,
    )
    if angle < 1e-7:
        return (0.5 * skew).astype(np.float32)
    if np.pi - angle < 1e-5:
        diagonal = np.maximum((np.diag(matrix) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        largest = int(np.argmax(axis))
        if axis[largest] <= 1e-8:
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


@dataclass(frozen=True)
class ThesisGeometry:
    relative_vectors: np.ndarray
    distances: np.ndarray
    ttc: np.ndarray
    approach_velocities: np.ndarray
    risk_per_link: np.ndarray
    risk_max: float
    distance_min: float


def compute_thesis_geometry(
    capsules: list[CapsuleState],
    obstacle_position: np.ndarray,
    obstacle_velocity: np.ndarray,
    closest_point_velocities: np.ndarray,
    *,
    obstacle_radius: float = 0.075,
    d_safe: float = 0.12,
    ttc_max: float = 3.0,
    eps_norm: float = 1e-8,
    eps_v: float = 1e-4,
) -> ThesisGeometry:
    relative = []
    distances = []
    ttc_values = []
    approaches = []
    risks = []
    obstacle_position = np.asarray(obstacle_position, dtype=np.float32)
    obstacle_velocity = np.asarray(obstacle_velocity, dtype=np.float32)
    for index, capsule in enumerate(capsules):
        closest, _ = closest_point_on_segment(obstacle_position, capsule.start, capsule.end)
        vector = obstacle_position - closest
        norm = float(np.linalg.norm(vector))
        distance = norm - float(capsule.radius) - obstacle_radius
        if norm <= eps_norm:
            approach = 0.0
            ttc = 0.0
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
        risk_ttc = float(np.exp(-ttc_clipped / 1.0))
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


def build_thesis_observation(
    *,
    q: np.ndarray,
    qdot: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    joint_velocity_limits: np.ndarray,
    position_error: np.ndarray,
    orientation_error: np.ndarray,
    obstacle_present: bool,
    obstacle_position: np.ndarray,
    obstacle_velocity: np.ndarray,
    geometry: ThesisGeometry | None,
) -> np.ndarray:
    q_normalized = np.clip(2.0 * (q - joint_lower) / (joint_upper - joint_lower) - 1.0, -1.0, 1.0)
    qdot_normalized = np.clip(qdot / joint_velocity_limits, -1.0, 1.0)
    if obstacle_present:
        if geometry is None:
            raise ValueError("geometry is required when an obstacle is present")
        low = np.asarray([0.22, -0.62, 0.18], dtype=np.float32)
        high = np.asarray([0.72, 0.62, 0.62], dtype=np.float32)
        obstacle_normalized = np.clip(2.0 * (obstacle_position - low) / (high - low) - 1.0, -1.0, 1.0)
        relative = np.clip(geometry.relative_vectors, -1.0, 1.0).reshape(-1)
        distance = 2.0 * (np.clip(geometry.distances, -0.20, 0.80) + 0.20) - 1.0
        ttc = np.clip(geometry.ttc, 0.0, 3.0) / 3.0
        obstacle_velocity_normalized = obstacle_velocity / 0.1
        present = 1.0
    else:
        relative = np.zeros(18, dtype=np.float32)
        obstacle_normalized = np.zeros(3, dtype=np.float32)
        obstacle_velocity_normalized = np.zeros(3, dtype=np.float32)
        distance = np.ones(6, dtype=np.float32)
        ttc = np.ones(6, dtype=np.float32)
        present = 0.0
    observation = np.concatenate(
        [
            q_normalized,
            qdot_normalized,
            np.clip(position_error, -1.0, 1.0),
            orientation_error / np.pi,
            relative,
            obstacle_normalized,
            obstacle_velocity_normalized,
            distance,
            ttc,
            np.asarray([present], dtype=np.float32),
        ]
    ).astype(np.float32)
    if observation.shape != (55,):
        raise RuntimeError(f"thesis observation must have shape (55,), got {observation.shape}")
    return observation


def homotopy_reward(
    *,
    rho_position: float,
    next_rho_position: float,
    rho_orientation: float,
    next_rho_orientation: float,
    smooth_velocity: float,
    task_reached: bool,
    hard_failure: bool,
    obstacle_collision: bool,
    terminal_obstacle_collision: bool = False,
    risk_max: float,
    distance_min: float,
    xi: float,
    d_safe: float = 0.12,
    gamma: float = 0.99,
    horizon: int = 240,
) -> tuple[float, dict[str, float]]:
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be in (0, 1)")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    state_cost = 2.0 * next_rho_position**2 + 0.5 * next_rho_orientation**2
    r_goal = (
        -state_cost
        +18.0 * (rho_position - next_rho_position)
        +4.0 * (rho_orientation - next_rho_orientation)
        -0.04 * smooth_velocity
        +20.0 * float(task_reached)
    )
    proximity = float(risk_max) + 3.0 * float(distance_min < d_safe)
    obstacle_only = float(obstacle_collision and not hard_failure)
    terminal_failure = float(hard_failure or terminal_obstacle_collision)
    discounted_horizon = (1.0 - gamma**horizon) / (1.0 - gamma)
    terminal_guard = terminal_failure * discounted_horizon * state_cost
    hard_penalty = 34.0 * float(hard_failure)
    safety_penalty = float(xi) * (4.0 * proximity + 34.0 * obstacle_only)
    reward = r_goal - hard_penalty - safety_penalty - terminal_guard
    return float(reward), {
        "r_goal": float(r_goal),
        "c_proximity": float(proximity),
        "hard_failure": float(hard_failure),
        "obstacle_collision": obstacle_only,
        "state_cost": float(state_cost),
        "hard_penalty": float(hard_penalty),
        "safety_penalty": float(safety_penalty),
        "terminal_guard_penalty": float(terminal_guard),
    }
