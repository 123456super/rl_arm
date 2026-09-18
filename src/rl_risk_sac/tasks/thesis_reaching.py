from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_risk_sac.robots.ur5_capsules import CapsuleState
from rl_risk_sac.utils.risk import closest_point_on_segment


THESIS_OBSERVATION_DIM = 122
WORLD_POSITION_SCALE = 1.0
APPROACH_VELOCITY_SCALE = 1.0
SELF_DISTANCE_LOWER = -0.02
SELF_DISTANCE_UPPER = 0.25


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


def rotation_6d(rotation: np.ndarray) -> np.ndarray:
    """Continuous SO(3) embedding formed by the first two matrix columns."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {matrix.shape}")
    return matrix[:, :2].reshape(6, order="F").astype(np.float32)


def quaternion_rotation_6d(quaternion: np.ndarray) -> np.ndarray:
    return rotation_6d(quaternion_to_matrix(quaternion))


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


def canonicalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Normalize a PyBullet quaternion and select one antipodal representation."""
    value = np.asarray(quaternion, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("zero quaternion")
    value = value / norm
    flip = value[3] < 0.0
    if abs(float(value[3])) <= 1e-7:
        first_nonzero = next((item for item in value[:3] if abs(float(item)) > 1e-7), 0.0)
        flip = first_nonzero < 0.0
    if flip:
        value = -value
    return value.astype(np.float32)


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
    eps_v: float = 1e-4,
) -> SelfCollisionGeometry:
    """Aggregate exact collision-mesh pair clearances into six link features.

    A pair endpoint of ``-1`` denotes the fixed base, which contributes to the
    global metrics and to the moving endpoint but has no observation slot.
    """
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
    joint_velocity_scale: np.ndarray,
    ee_position: np.ndarray,
    ee_quaternion: np.ndarray,
    ee_linear_velocity: np.ndarray,
    ee_angular_velocity: np.ndarray,
    ee_linear_velocity_scale: float,
    ee_angular_velocity_scale: float,
    goal_position: np.ndarray,
    goal_quaternion: np.ndarray,
    position_error: np.ndarray,
    orientation_error: np.ndarray,
    orientation_scale: float,
    remaining_time_fraction: float,
    obstacle_present: bool,
    obstacle_position: np.ndarray,
    obstacle_velocity: np.ndarray,
    geometry: ThesisGeometry | None,
    self_geometry: SelfCollisionGeometry | None = None,
) -> np.ndarray:
    q_normalized = np.clip(2.0 * (q - joint_lower) / (joint_upper - joint_lower) - 1.0, -1.0, 1.0)
    qdot_normalized = np.clip(qdot / joint_velocity_scale, -1.0, 1.0)
    ee_rotation = quaternion_to_matrix(ee_quaternion)
    goal_rotation = quaternion_to_matrix(goal_quaternion)
    relative_rotation = goal_rotation @ ee_rotation.T
    position_error_norm = float(np.linalg.norm(position_error))
    orientation_error_norm = float(np.linalg.norm(orientation_error))
    if obstacle_present:
        if geometry is None:
            raise ValueError("geometry is required when an obstacle is present")
        low = np.asarray([0.22, -0.62, 0.18], dtype=np.float32)
        high = np.asarray([0.72, 0.62, 0.62], dtype=np.float32)
        obstacle_normalized = np.clip(2.0 * (obstacle_position - low) / (high - low) - 1.0, -1.0, 1.0)
        relative = np.clip(geometry.relative_vectors, -1.0, 1.0).reshape(-1)
        distance = 2.0 * (np.clip(geometry.distances, -0.20, 0.80) + 0.20) - 1.0
        ttc = np.clip(geometry.ttc, 0.0, 3.0) / 3.0
        approach = np.clip(
            geometry.approach_velocities / APPROACH_VELOCITY_SCALE, 0.0, 1.0
        )
        risk = np.clip(geometry.risk_per_link, 0.0, 1.0)
        obstacle_velocity_normalized = obstacle_velocity / 0.1
        present = 1.0
    else:
        relative = np.zeros(18, dtype=np.float32)
        obstacle_normalized = np.zeros(3, dtype=np.float32)
        obstacle_velocity_normalized = np.zeros(3, dtype=np.float32)
        distance = np.ones(6, dtype=np.float32)
        ttc = np.ones(6, dtype=np.float32)
        approach = np.zeros(6, dtype=np.float32)
        risk = np.zeros(6, dtype=np.float32)
        present = 0.0
    if self_geometry is None:
        self_distance = np.ones(6, dtype=np.float32)
        self_ttc = np.ones(6, dtype=np.float32)
        self_approach = np.zeros(6, dtype=np.float32)
        self_risk = np.zeros(6, dtype=np.float32)
    else:
        span = SELF_DISTANCE_UPPER - SELF_DISTANCE_LOWER
        self_distance = (
            2.0
            * (np.clip(self_geometry.distances, SELF_DISTANCE_LOWER, SELF_DISTANCE_UPPER)
               - SELF_DISTANCE_LOWER)
            / span
            - 1.0
        )
        self_ttc = np.clip(self_geometry.ttc, 0.0, 3.0) / 3.0
        self_approach = np.clip(
            self_geometry.approach_velocities / APPROACH_VELOCITY_SCALE, 0.0, 1.0
        )
        self_risk = np.clip(self_geometry.risk_per_link, 0.0, 1.0)
    observation = np.concatenate(
        [
            q_normalized,
            qdot_normalized,
            np.clip(np.asarray(ee_position) / WORLD_POSITION_SCALE, -1.0, 1.0),
            rotation_6d(ee_rotation),
            np.clip(np.asarray(goal_position) / WORLD_POSITION_SCALE, -1.0, 1.0),
            rotation_6d(goal_rotation),
            np.clip(position_error, -1.0, 1.0),
            rotation_6d(relative_rotation),
            np.asarray([np.clip(position_error_norm / 1.0, 0.0, 1.0)], dtype=np.float32),
            np.asarray([np.clip(orientation_error_norm / np.pi, 0.0, 1.0)], dtype=np.float32),
            np.clip(np.asarray(ee_linear_velocity) / ee_linear_velocity_scale, -1.0, 1.0),
            np.clip(np.asarray(ee_angular_velocity) / ee_angular_velocity_scale, -1.0, 1.0),
            np.asarray([2.0 * np.clip(orientation_scale, 0.0, 1.0) - 1.0], dtype=np.float32),
            np.asarray(
                [2.0 * np.clip(remaining_time_fraction, 0.0, 1.0) - 1.0],
                dtype=np.float32,
            ),
            relative,
            obstacle_normalized,
            obstacle_velocity_normalized,
            distance,
            ttc,
            approach,
            risk,
            np.asarray([present], dtype=np.float32),
            self_distance,
            self_ttc,
            self_approach,
            self_risk,
        ]
    ).astype(np.float32)
    if observation.shape != (THESIS_OBSERVATION_DIM,):
        raise RuntimeError(
            f"thesis observation must have shape ({THESIS_OBSERVATION_DIM},), got {observation.shape}"
        )
    return observation


def homotopy_reward(
    *,
    rho_position: float,
    next_rho_position: float,
    rho_orientation: float,
    next_rho_orientation: float,
    smooth_velocity: float,
    velocity_magnitude: float,
    orientation_scale: float,
    task_reached: bool,
    hard_failure: bool,
    obstacle_collision: bool,
    terminal_obstacle_collision: bool = False,
    risk_max: float,
    distance_min: float,
    xi: float,
    self_risk_max: float = 0.0,
    self_distance_min: float = SELF_DISTANCE_UPPER,
    lambda_self: float = 0.0,
    d_safe: float = 0.12,
    d_self_safe: float = 0.005,
    gamma: float = 0.99,
    horizon: int = 240,
    position_sigma: float = 0.20,
    orientation_sigma: float = 1.00,
    micro_power: float = 4.0,
    orientation_priority_weight: float = 1.0,
    potential_scale: float = 20.0,
    success_bonus: float = 20.0,
    velocity_cost_weight: float = 0.04,
    smooth_cost_weight: float = 0.01,
    hard_failure_penalty: float = 10.0,
    safety_risk_weight: float = 2.0,
    safety_clearance_weight: float = 8.0,
    self_risk_weight: float = 2.0,
    self_clearance_weight: float = 8.0,
    external_safety_scale: float = 1.0,
    self_safety_scale: float = 1.0,
) -> tuple[float, dict[str, float]]:
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be in (0, 1)")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    eta = float(np.clip(orientation_scale, 0.0, 1.0))
    if position_sigma <= 0.0 or orientation_sigma <= 0.0:
        raise ValueError("pose quality sigmas must be positive")
    if micro_power <= 1.0:
        raise ValueError("micro_power must be greater than one")
    if orientation_priority_weight < 0.0 or potential_scale <= 0.0:
        raise ValueError("reward scales must be non-negative and potential_scale positive")
    if min(
        safety_risk_weight,
        safety_clearance_weight,
        self_risk_weight,
        self_clearance_weight,
        external_safety_scale,
        self_safety_scale,
    ) < 0.0:
        raise ValueError("safety reward weights must be non-negative")
    external_weight_sum = safety_risk_weight + safety_clearance_weight
    self_weight_sum = self_risk_weight + self_clearance_weight
    if external_weight_sum <= 0.0 or self_weight_sum <= 0.0:
        raise ValueError("each safety reward group must have a positive weight sum")

    def enhanced_quality(error: float, sigma: float) -> float:
        quality = float(np.exp(-max(float(error), 0.0) / sigma))
        return 0.5 * (quality + quality**micro_power)

    position_quality = enhanced_quality(rho_position, position_sigma)
    next_position_quality = enhanced_quality(next_rho_position, position_sigma)
    orientation_quality = enhanced_quality(rho_orientation, orientation_sigma)
    next_orientation_quality = enhanced_quality(next_rho_orientation, orientation_sigma)
    pose_potential = position_quality * (
        1.0 + orientation_priority_weight * eta * orientation_quality
    )
    next_pose_potential = next_position_quality * (
        1.0 + orientation_priority_weight * eta * next_orientation_quality
    )
    pose_potential_progress = float(gamma * next_pose_potential - pose_potential)
    position_progress = float(next_position_quality - position_quality)
    orientation_progress = float(next_orientation_quality - orientation_quality)
    orientation_reward_scale = float(
        orientation_priority_weight * eta * next_position_quality
    )
    velocity_cost = velocity_cost_weight * float(np.clip(velocity_magnitude, 0.0, 1.0))
    smooth_cost = smooth_cost_weight * float(np.clip(smooth_velocity, 0.0, 1.0))
    state_cost = 0.0
    r_goal = (
        potential_scale * pose_potential_progress
        -velocity_cost
        -smooth_cost
        +success_bonus * float(task_reached)
    )
    if d_safe <= 0.0 or d_self_safe <= 0.0:
        raise ValueError("safety distances must be positive")
    if not 0.0 <= float(lambda_self) <= 1.0:
        raise ValueError("lambda_self must be in [0, 1]")
    clearance_violation = float(np.clip((d_safe - distance_min) / d_safe, 0.0, 1.0))
    bounded_risk = float(np.clip(risk_max, 0.0, 1.0))
    self_clearance_violation = float(
        np.clip((d_self_safe - self_distance_min) / d_self_safe, 0.0, 1.0)
    )
    bounded_self_risk = float(np.clip(self_risk_max, 0.0, 1.0))
    external_proximity = bounded_risk + clearance_violation
    self_proximity = bounded_self_risk + self_clearance_violation
    proximity = external_proximity + self_proximity
    obstacle_only = float(obstacle_collision and not hard_failure)
    terminal_guard = hard_failure_penalty * float(
        terminal_obstacle_collision and not hard_failure
    )
    hard_penalty = hard_failure_penalty * float(hard_failure)
    # Each dense constraint group is normalized to [0, scale].  This keeps a
    # continuously active constraint from overwhelming the sparse task bonus.
    external_safety_cost = (
        safety_risk_weight * bounded_risk
        + safety_clearance_weight * clearance_violation
    ) / external_weight_sum
    self_safety_cost = (
        self_risk_weight * bounded_self_risk
        + self_clearance_weight * self_clearance_violation
    ) / self_weight_sum
    external_safety_penalty = (
        float(xi) * external_safety_scale * external_safety_cost
    )
    self_safety_penalty = (
        float(lambda_self) * self_safety_scale * self_safety_cost
    )
    safety_penalty = external_safety_penalty + self_safety_penalty
    reward = r_goal - hard_penalty - safety_penalty - terminal_guard
    return float(reward), {
        "r_goal": float(r_goal),
        "c_proximity": float(proximity),
        "hard_failure": float(hard_failure),
        "obstacle_collision": obstacle_only,
        "state_cost": float(state_cost),
        "position_state_cost": 0.0,
        "orientation_state_cost": 0.0,
        "velocity_cost": velocity_cost,
        "orientation_scale": eta,
        "orientation_reward_scale": float(orientation_reward_scale),
        "orientation_reward_gate": float(next_position_quality),
        "position_progress": position_progress,
        "orientation_progress": orientation_progress,
        "position_quality": float(next_position_quality),
        "orientation_quality": float(next_orientation_quality),
        "pose_potential": float(pose_potential),
        "next_pose_potential": float(next_pose_potential),
        "pose_potential_progress": pose_potential_progress,
        "smooth_cost": smooth_cost,
        "clearance_violation": clearance_violation,
        "self_clearance_violation": self_clearance_violation,
        "self_risk_max": bounded_self_risk,
        "lambda_self": float(lambda_self),
        "hard_penalty": float(hard_penalty),
        "external_safety_penalty": float(external_safety_penalty),
        "self_safety_penalty": float(self_safety_penalty),
        "external_safety_cost": float(external_safety_cost),
        "self_safety_cost": float(self_safety_cost),
        "safety_penalty": float(safety_penalty),
        "terminal_guard_penalty": float(terminal_guard),
    }
