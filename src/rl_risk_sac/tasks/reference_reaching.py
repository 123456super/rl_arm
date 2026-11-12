from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_risk_sac.robots.ur5_capsules import CapsuleState


REFERENCE_LEGACY_BASE_OBSERVATION_DIM = 28
REFERENCE_BASE_OBSERVATION_DIM = 34
OBSTACLE_FEATURE_DIM = 9
DEFAULT_MAX_OBSTACLES = 8
OBSTACLE_CLEARANCE_SCALE_M = 0.20

# Evaluation/deployment handoff defaults.  The Residual keeps full authority
# outside the near-goal region or whenever any arm capsule remains close to an
# obstacle.  It relinquishes control to the frozen Base only when both the
# target is near and the whole arm is clear.
RESIDUAL_HANDOFF_GOAL_BASE_M = 0.025
RESIDUAL_HANDOFF_GOAL_FULL_M = 0.035
RESIDUAL_HANDOFF_CLEARANCE_FULL_M = 0.080
RESIDUAL_HANDOFF_CLEARANCE_BASE_M = 0.100


def reference_ppo_observation_dim(
    max_obstacles: int = DEFAULT_MAX_OBSTACLES,
) -> int:
    """Flat transport size for base state, padded obstacle set, and mask."""
    if max_obstacles < 1:
        raise ValueError("max_obstacles must be positive")
    return REFERENCE_BASE_OBSERVATION_DIM + max_obstacles * (
        OBSTACLE_FEATURE_DIM + 1
    )


REFERENCE_PPO_OBSERVATION_DIM = reference_ppo_observation_dim()
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
    position_error: np.ndarray,
    orientation_error: np.ndarray,
    normalized_joint_velocity: np.ndarray | None = None,
    reference_point_offset_m: float = 0.001,
    obstacle_features: np.ndarray | None = None,
    obstacle_mask: np.ndarray | None = None,
    max_obstacles: int = DEFAULT_MAX_OBSTACLES,
) -> np.ndarray:
    """Build base state plus a padded, masked obstacle-set transport tensor.

    Each active obstacle row is ``[six capsule clearances, nearest direction]``.
    The direction is the unit vector from the closest robot-capsule centreline
    point towards the closest point on that obstacle's OBB.  OBB pose and size
    remain in the geometry layer and are not exposed to the policy.  Padding is
    only a batching representation; the PPO network consumes rows through a
    permutation-invariant set encoder and uses the mask for presence.
    """
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
    normalized_joint_velocity = (
        np.zeros(6, dtype=np.float32)
        if normalized_joint_velocity is None
        else np.asarray(normalized_joint_velocity, dtype=np.float32)
    )
    if normalized_joint_velocity.shape != (6,):
        raise ValueError("normalized_joint_velocity must have shape (6,)")
    base_observation = np.concatenate(
        [
            np.asarray(q, dtype=np.float32),
            np.asarray(ee_position, dtype=np.float32),
            np.asarray(ee_quaternion, dtype=np.float32),
            np.asarray(goal_position, dtype=np.float32),
            np.asarray(goal_quaternion, dtype=np.float32),
            scalar_errors,
            np.asarray(position_error, dtype=np.float32),
            np.asarray(orientation_error, dtype=np.float32),
            normalized_joint_velocity,
        ]
    ).astype(np.float32)
    if base_observation.shape != (REFERENCE_BASE_OBSERVATION_DIM,):
        raise RuntimeError(
            f"reference PPO base observation must have shape "
            f"({REFERENCE_BASE_OBSERVATION_DIM},), got {base_observation.shape}"
        )
    features = np.zeros(
        (max_obstacles, OBSTACLE_FEATURE_DIM), dtype=np.float32
    )
    mask = np.zeros(max_obstacles, dtype=np.float32)
    if obstacle_features is not None:
        supplied = np.asarray(obstacle_features, dtype=np.float32)
        if supplied.ndim != 2 or supplied.shape[1] != OBSTACLE_FEATURE_DIM:
            raise ValueError(
                f"obstacle_features must have shape (N, {OBSTACLE_FEATURE_DIM})"
            )
        if supplied.shape[0] > max_obstacles:
            raise ValueError(
                f"received {supplied.shape[0]} obstacles, maximum is {max_obstacles}"
            )
        features[: supplied.shape[0]] = supplied
        if obstacle_mask is None:
            mask[: supplied.shape[0]] = 1.0
        else:
            supplied_mask = np.asarray(obstacle_mask, dtype=np.float32)
            if supplied_mask.shape != (supplied.shape[0],):
                raise ValueError("obstacle_mask must have shape (N,)")
            if not np.logical_or(supplied_mask == 0.0, supplied_mask == 1.0).all():
                raise ValueError("obstacle_mask values must be 0 or 1")
            mask[: supplied.shape[0]] = supplied_mask
    elif obstacle_mask is not None:
        raise ValueError("obstacle_mask requires obstacle_features")
    observation = np.concatenate(
        [base_observation, features.reshape(-1), mask]
    ).astype(np.float32)
    expected_dim = reference_ppo_observation_dim(max_obstacles)
    if observation.shape != (expected_dim,):
        raise RuntimeError(
            f"reference PPO observation must have shape ({expected_dim},), "
            f"got {observation.shape}"
        )
    if not np.isfinite(observation).all():
        raise ValueError("non-finite reference PPO observation")
    return observation


def reference_ppo_reward(
    *,
    reference_error: float,
    obstacle_collision: bool = False,
    self_collision: bool = False,
    environment_collision: bool = False,
    obstacle_clearances: tuple[float, ...] = (),
    obstacle_threshold: float = 0.08,
    obstacle_clearance_weight: float = 1.0,
    obstacle_clearance_power: float = 1.0,
    obstacle_collision_penalty: float = 20.0,
    self_collision_penalty: float = 10.0,
    environment_collision_penalty: float = 10.0,
) -> tuple[float, dict[str, float]]:
    """Reference reaching reward plus literature-based avoidance penalties.

    ``obstacle_clearances`` contains one minimum clearance for each ordered
    robot collision capsule.  Signed penetrations are clamped to zero in the
    proximity term because physical contact is penalized separately.
    """
    errors = max(float(reference_error), 0.0)
    threshold = float(obstacle_threshold)
    if threshold <= 0.0:
        raise ValueError("reference obstacle threshold must be positive")
    penalties = (
        obstacle_clearance_weight,
        obstacle_collision_penalty,
        self_collision_penalty,
        environment_collision_penalty,
    )
    if not all(np.isfinite(value) and value >= 0.0 for value in penalties):
        raise ValueError("reward penalty weights must be finite and non-negative")
    if not np.isfinite(obstacle_clearance_power) or obstacle_clearance_power <= 0.0:
        raise ValueError("obstacle clearance power must be finite and positive")
    # The archived 8M S1 policy used the original linear normalized clearance
    # deficit.  Keep the exponent configurable for explicit ablations, while
    # retaining 1.0 as the reference-PPO default.
    proximity = sum(
        max(0.0, 1.0 - max(0.0, float(distance)) / threshold)
        ** obstacle_clearance_power
        for distance in obstacle_clearances
    )
    goal_reward = -0.001 * errors**2 - float(np.log(errors**2 + 1.0e-4))
    clearance_cost = float(obstacle_clearance_weight) * proximity
    obstacle_collision_cost = float(obstacle_collision_penalty) * float(obstacle_collision)
    self_collision_cost = float(self_collision_penalty) * float(self_collision)
    environment_collision_cost = float(environment_collision_penalty) * float(environment_collision)
    collision_cost = (
        obstacle_collision_cost + self_collision_cost + environment_collision_cost
    )
    reward = goal_reward - clearance_cost - collision_cost
    return float(reward), {
        "r_goal": float(goal_reward),
        "c_proximity": float(proximity),
        "reference_error": float(errors),
        "reference_obstacle_penalty": float(clearance_cost),
        "reference_obstacle_collision_penalty": obstacle_collision_cost,
        "reference_self_collision_penalty": self_collision_cost,
        "reference_environment_collision_penalty": environment_collision_cost,
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


def residual_action_penalty(
    residual_action: np.ndarray,
    goal_distance: float,
    obstacle_clearance: float,
    *,
    penalty_weight: float = 50.0,
    goal_start_m: float = 0.03,
    goal_full_m: float = 0.02,
    clearance_start_m: float = 0.08,
    clearance_full_m: float = 0.12,
) -> tuple[float, float]:
    """Penalize residual action only near a safely reachable goal.

    The goal weight rises smoothly from zero at ``goal_start_m`` to one at
    ``goal_full_m``.  The clearance weight independently rises from zero to
    one between ``clearance_start_m`` and ``clearance_full_m``.  Their product
    preserves full residual authority while the robot is still detouring or
    while any link remains inside the obstacle-avoidance working distance.
    """
    action = np.asarray(residual_action, dtype=np.float64)
    values = (
        float(goal_distance), float(obstacle_clearance), float(penalty_weight),
        float(goal_start_m), float(goal_full_m),
        float(clearance_start_m), float(clearance_full_m),
    )
    if action.ndim != 1 or not np.isfinite(action).all():
        raise ValueError("residual_action must be a finite vector")
    if not all(np.isfinite(value) for value in values):
        raise ValueError("residual penalty parameters must be finite")
    if penalty_weight < 0.0:
        raise ValueError("residual penalty weight must be non-negative")
    if goal_full_m < 0.0 or goal_start_m <= goal_full_m:
        raise ValueError(
            "goal residual penalty distances must satisfy 0 <= full < start"
        )
    if clearance_start_m < 0.0 or clearance_full_m <= clearance_start_m:
        raise ValueError(
            "clearance residual penalty distances must satisfy 0 <= start < full"
        )

    goal_phase = float(np.clip(
        (goal_start_m - goal_distance) / (goal_start_m - goal_full_m),
        0.0,
        1.0,
    ))
    clearance_phase = float(np.clip(
        (obstacle_clearance - clearance_start_m)
        / (clearance_full_m - clearance_start_m),
        0.0,
        1.0,
    ))
    goal_weight = goal_phase * goal_phase * (3.0 - 2.0 * goal_phase)
    clearance_weight = (
        clearance_phase * clearance_phase * (3.0 - 2.0 * clearance_phase)
    )
    combined_weight = float(goal_weight * clearance_weight)
    penalty = float(penalty_weight * combined_weight * np.dot(action, action))
    return penalty, combined_weight


def residual_handoff_authority(
    goal_distance: float,
    obstacle_clearance: float,
    *,
    goal_base_m: float = RESIDUAL_HANDOFF_GOAL_BASE_M,
    goal_full_m: float = RESIDUAL_HANDOFF_GOAL_FULL_M,
    clearance_full_m: float = RESIDUAL_HANDOFF_CLEARANCE_FULL_M,
    clearance_base_m: float = RESIDUAL_HANDOFF_CLEARANCE_BASE_M,
) -> float:
    """Return the Residual action authority in ``[0, 1]``.

    Authority is zero only in the near-goal, obstacle-clear region.  Either a
    distant goal or a nearby obstacle restores Residual authority.  Smoothstep
    transitions avoid a discontinuous action command at either boundary.
    """
    values = (
        float(goal_distance), float(obstacle_clearance), float(goal_base_m),
        float(goal_full_m), float(clearance_full_m), float(clearance_base_m),
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("residual handoff inputs and thresholds must be finite")
    if goal_base_m < 0.0 or goal_full_m <= goal_base_m:
        raise ValueError("goal handoff distances must satisfy 0 <= base < full")
    if clearance_full_m < 0.0 or clearance_base_m <= clearance_full_m:
        raise ValueError(
            "clearance handoff distances must satisfy 0 <= full < base"
        )

    goal_phase = float(np.clip(
        (goal_distance - goal_base_m) / (goal_full_m - goal_base_m),
        0.0,
        1.0,
    ))
    clearance_phase = float(np.clip(
        (obstacle_clearance - clearance_full_m)
        / (clearance_base_m - clearance_full_m),
        0.0,
        1.0,
    ))
    goal_authority = goal_phase * goal_phase * (3.0 - 2.0 * goal_phase)
    risk_authority = 1.0 - (
        clearance_phase * clearance_phase * (3.0 - 2.0 * clearance_phase)
    )
    return float(max(goal_authority, risk_authority))


def compose_base_residual_action(
    base_action: np.ndarray,
    residual_action: np.ndarray,
    goal_distance: float | np.ndarray,
    *,
    residual_cutoff_position_m: float | None = RESIDUAL_HANDOFF_GOAL_BASE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose the final inference action with the 25 mm Base handoff.

    This is the authoritative action-composition rule for evaluation,
    visualization, and robot deployment.  Training deliberately continues to
    optimize the original full ``Base + Residual`` policy.  Passing ``None``
    is reserved for the full-Residual ablation.
    """
    base = np.asarray(base_action)
    residual = np.asarray(residual_action)
    distance = np.asarray(goal_distance, dtype=np.float64)
    if base.ndim < 1 or base.shape != residual.shape:
        raise ValueError("base_action and residual_action must have equal shapes")
    if not np.isfinite(base).all() or not np.isfinite(residual).all():
        raise ValueError("base and residual actions must be finite")
    try:
        distance = np.broadcast_to(distance, base.shape[:-1])
    except ValueError as error:
        raise ValueError(
            "goal_distance must broadcast to the action batch shape"
        ) from error
    if not np.isfinite(distance).all() or np.any(distance < 0.0):
        raise ValueError("goal_distance must be finite and non-negative")
    if residual_cutoff_position_m is None:
        authority = np.ones(base.shape[:-1], dtype=np.float32)
    else:
        cutoff = float(residual_cutoff_position_m)
        if not np.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError("residual cutoff must be finite and positive")
        authority = (distance > cutoff).astype(np.float32)
    composed = base + authority[..., None] * residual
    return composed.astype(base.dtype, copy=False), authority


@dataclass(frozen=True)
class ThesisGeometry:
    relative_vectors: np.ndarray
    distances: np.ndarray
    nearest_direction: np.ndarray
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


def closest_points_segment_obb(
    start: np.ndarray,
    end: np.ndarray,
    box_center: np.ndarray,
    box_half_extents: np.ndarray,
    box_rotation: np.ndarray,
    *,
    eps: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return closest segment/OBB points, direction, distance, and segment fraction.

    The segment is transformed into the OBB frame.  Squared distance to an
    axis-aligned box is a convex piecewise quadratic in the segment fraction;
    its breakpoints occur when a segment coordinate crosses a box face.  We
    minimize each interval exactly instead of approximating the link by samples.
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    center = np.asarray(box_center, dtype=np.float64)
    half = np.asarray(box_half_extents, dtype=np.float64)
    rotation = np.asarray(box_rotation, dtype=np.float64)
    if start.shape != (3,) or end.shape != (3,) or center.shape != (3,):
        raise ValueError("segment and OBB points must have shape (3,)")
    if half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0.0):
        raise ValueError("OBB half extents must contain three positive finite values")
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("OBB rotation must have shape (3, 3) and be finite")

    local_start = rotation.T @ (start - center)
    local_end = rotation.T @ (end - center)
    delta = local_end - local_start
    breakpoints = [0.0, 1.0]
    for axis in range(3):
        if abs(float(delta[axis])) <= eps:
            continue
        for face in (-half[axis], half[axis]):
            fraction = float((face - local_start[axis]) / delta[axis])
            if 0.0 < fraction < 1.0:
                breakpoints.append(fraction)
    breakpoints = sorted(set(breakpoints))

    def evaluate(fraction: float) -> tuple[float, np.ndarray, np.ndarray]:
        segment_point = local_start + fraction * delta
        box_point = np.clip(segment_point, -half, half)
        squared = float(np.dot(segment_point - box_point, segment_point - box_point))
        return squared, segment_point, box_point

    candidates = set(breakpoints)
    for lower, upper in zip(breakpoints[:-1], breakpoints[1:], strict=True):
        midpoint = 0.5 * (lower + upper)
        point = local_start + midpoint * delta
        active = np.logical_or(point < -half, point > half)
        if not np.any(active):
            candidates.add(midpoint)
            continue
        target = np.where(point < -half, -half, half)
        denom = float(np.dot(delta[active], delta[active]))
        if denom > eps:
            numerator = float(
                np.dot(delta[active], local_start[active] - target[active])
            )
            candidates.add(float(np.clip(-numerator / denom, lower, upper)))

    best_fraction = 0.0
    best_squared = float("inf")
    best_segment_local = local_start.copy()
    best_box_local = np.clip(local_start, -half, half)
    for fraction in sorted(candidates):
        squared, segment_local, box_local = evaluate(float(fraction))
        if squared < best_squared:
            best_squared = squared
            best_fraction = float(fraction)
            best_segment_local = segment_local
            best_box_local = box_local

    centerline_distance = float(np.sqrt(max(best_squared, 0.0)))
    if centerline_distance > eps:
        normal_local = (best_box_local - best_segment_local) / centerline_distance
    else:
        # The capsule centreline intersects the OBB.  The nearest-face normal
        # remains well defined for learning even though the closest points
        # coincide and the geometric separation is zero.
        margins = half - np.abs(best_segment_local)
        axis = int(np.argmin(margins))
        normal_local = np.zeros(3, dtype=np.float64)
        normal_local[axis] = 1.0 if best_segment_local[axis] >= 0.0 else -1.0
        best_box_local = best_segment_local.copy()
        best_box_local[axis] = normal_local[axis] * half[axis]

    segment_world = center + rotation @ best_segment_local
    box_world = center + rotation @ best_box_local
    normal_world = rotation @ normal_local
    normal_world /= max(float(np.linalg.norm(normal_world)), eps)
    return (
        segment_world.astype(np.float32),
        box_world.astype(np.float32),
        normal_world.astype(np.float32),
        centerline_distance,
        best_fraction,
    )


def compute_thesis_geometry(
    capsules: list[CapsuleState],
    obstacle_position: np.ndarray,
    obstacle_velocity: np.ndarray,
    closest_point_velocities: np.ndarray,
    *,
    obstacle_half_extents: np.ndarray,
    obstacle_rotation: np.ndarray,
    d_safe: float = 0.05,
    ttc_max: float = 3.0,
    eps_v: float = 1.0e-4,
) -> ThesisGeometry:
    relative, directions, distances = [], [], []
    ttc_values, approaches, risks = [], [], []
    obstacle_position = np.asarray(obstacle_position, dtype=np.float32)
    obstacle_velocity = np.asarray(obstacle_velocity, dtype=np.float32)
    for index, capsule in enumerate(capsules):
        closest, obstacle_closest, normal, centerline_distance, _ = (
            closest_points_segment_obb(
                capsule.start,
                capsule.end,
                obstacle_position,
                obstacle_half_extents,
                obstacle_rotation,
            )
        )
        vector = obstacle_closest - closest
        distance = centerline_distance - float(capsule.radius)
        distance_rate = float(
            np.dot(normal, obstacle_velocity - closest_point_velocities[index])
        )
        approach = max(-distance_rate, 0.0)
        if distance <= d_safe:
            ttc = 0.0
        elif approach > eps_v:
            ttc = (distance - d_safe) / approach
        else:
            ttc = ttc_max
        ttc_clipped = float(np.clip(ttc, 0.0, ttc_max))
        risk_distance = float(
            np.clip(np.exp(-(distance - d_safe) / 0.12), 0.0, 1.0)
        )
        risk_velocity = float(np.clip(approach / 0.7, 0.0, 1.0))
        risk_ttc = float(np.exp(-ttc_clipped))
        risk = float(
            np.clip(
                0.5 * risk_distance + 0.2 * risk_velocity + 0.3 * risk_ttc,
                0.0,
                1.0,
            )
        )
        relative.append(vector)
        directions.append(normal)
        distances.append(distance)
        ttc_values.append(ttc_clipped)
        approaches.append(approach)
        risks.append(risk)
    distances_array = np.asarray(distances, dtype=np.float32)
    directions_array = np.asarray(directions, dtype=np.float32)
    closest_link = int(np.argmin(distances_array))
    return ThesisGeometry(
        relative_vectors=np.asarray(relative, dtype=np.float32),
        distances=distances_array,
        nearest_direction=directions_array[closest_link].copy(),
        ttc=np.asarray(ttc_values, dtype=np.float32),
        approach_velocities=np.asarray(approaches, dtype=np.float32),
        risk_per_link=np.asarray(risks, dtype=np.float32),
        risk_max=float(np.max(risks)),
        distance_min=float(distances_array[closest_link]),
    )
