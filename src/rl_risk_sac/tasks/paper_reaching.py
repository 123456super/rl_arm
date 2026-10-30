"""Reference observation/reward and sampling; no old curriculum or handoff terms."""
from __future__ import annotations

import numpy as np

PAPER_OBSERVATION_DIM = 25
PAPER_POSITION_THRESHOLDS_MM = (1, 5, 10, 20, 50, 100)


def normalized_quaternion(value):
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) <= 1e-12:
        raise ValueError("expected a finite nonzero quaternion")
    q = q / np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def pose_errors(position, quaternion, target_position, target_quaternion):
    distance = float(np.linalg.norm(np.asarray(position) - target_position))
    cosine = np.clip(abs(np.dot(normalized_quaternion(quaternion),
                                normalized_quaternion(target_quaternion))), 0., 1.)
    return distance, float(2 * np.arccos(cosine))


def reference_error_components(position_error, orientation_error):
    # The public implementation uses each rotation about its own axis. The
    # selected column is invariant, so its tri-point distance sum is 3*dpos.
    return np.asarray([3 * position_error, orientation_error], dtype=np.float64)


def combined_pose_error(components, reward_config):
    return float(reward_config["position_error_weight"] * components[0]
                 + reward_config["orientation_error_weight"] * components[1])


def paper_observation(q, position, quaternion, target_position, target_quaternion, distances):
    if np.asarray(q).shape != (6,) or np.asarray(distances).shape != (3,):
        raise ValueError("reference observation requires six joints/three obstacle distances")
    errors = reference_error_components(*pose_errors(position, quaternion, target_position, target_quaternion))
    obs = np.concatenate((q, position, normalized_quaternion(quaternion), target_position,
                          normalized_quaternion(target_quaternion), errors, distances)).astype(np.float32)
    if obs.shape != (PAPER_OBSERVATION_DIM,) or not np.isfinite(obs).all():
        raise ValueError("invalid 25D reference observation")
    return obs


def point_segment_distances(centers, starts, ends):
    """Pairwise point/finite-segment distances, including intentional zero-length links."""
    centers = np.asarray(centers, dtype=np.float64)
    starts, ends = np.asarray(starts, dtype=np.float64), np.asarray(ends, dtype=np.float64)
    vectors = ends - starts
    projection = np.sum((centers[:, None, :] - starts) * vectors, axis=-1)
    projection /= np.maximum(np.sum(vectors * vectors, axis=-1), 1e-20)
    nearest = starts + np.clip(projection, 0., 1.)[..., None] * vectors
    return np.linalg.norm(centers[:, None, :] - nearest, axis=-1)


def centerline_distances(centers, starts, ends):
    """One distance per obstacle, minimized over the arm's selected links."""
    return point_segment_distances(centers, starts, ends).min(axis=1)


def paper_reward(error_components, obstacle_distances, collision, config):
    error = combined_pose_error(error_components, config)
    distances = np.asarray(obstacle_distances, dtype=np.float64)
    if not np.isfinite(error) or not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("invalid reward geometry")
    tracking = -config["error_quadratic_weight"] * error**2 - np.log(error**2 + config["error_log_offset"])
    avoidance = -config["obstacle_weight"] * np.maximum(0., 1. - distances / config["obstacle_distance"]).sum()
    contact = -config["collision_penalty"] * bool(collision)
    return float(tracking + avoidance + contact), {
        "combined_error": error, "tracking_reward": float(tracking),
        "obstacle_reward": float(avoidance), "collision_reward": float(contact),
    }


def sample_goal_position(rng, config, base_position):
    base = np.asarray(base_position, dtype=np.float64)
    for _ in range(1024):
        direction = rng.normal(size=3)
        direction /= max(np.linalg.norm(direction), 1e-12)
        point = direction * config["workspace_radius"] * rng.random()**(1 / 3)
        if (point[2] >= config["min_z"]
                and np.linalg.norm(point[:2]) >= config["base_exclusion_radius"]
                and (not config["front_half"] or point[1] >= 0)):
            return base + point
    raise RuntimeError("goal workspace rejection budget exhausted")


def sample_obstacle_center(rng, target, base_position, config):
    """Uniform volume in a quarter spherical shell; orientation is an explicit choice."""
    inner, outer = config["shell_inner_radius"], config["shell_outer_radius"]
    horizontal = np.asarray(base_position, float) - target
    horizontal[2] = 0
    for _ in range(1024):
        direction = rng.normal(size=3)
        direction /= max(np.linalg.norm(direction), 1e-12)
        if direction[2] <= 0 and np.dot(direction, horizontal) >= 0:
            radius = (inner**3 + rng.random() * (outer**3 - inner**3))**(1 / 3)
            return np.asarray(target) + direction * radius
    raise RuntimeError("obstacle shell rejection budget exhausted")
