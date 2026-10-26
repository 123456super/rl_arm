"""Optional S1 potential shaping; no neural features or physical control changes."""
from __future__ import annotations

import numpy as np


COUPLING_KEYS = (
    "goal_obstacle_coupling_beta", "goal_obstacle_position_scale",
    "goal_obstacle_orientation_scale", "goal_obstacle_d_safe", "goal_obstacle_d_cut",
)


def coupling_parameters(config):
    """Absent/disabled settings produce no new keys in legacy reward contracts."""
    thesis = config["thesis"]
    settings = thesis.get("reward", {}).get("goal_obstacle_coupling", {})
    if not settings.get("enabled", False) or thesis.get("stage") != "s1":
        return {}
    if not thesis["obstacles"]["enabled"] or thesis["obstacles"]["dynamic"]:
        raise ValueError("goal-obstacle coupling requires static-obstacle S1")
    risk = thesis["obstacles"]["risk"]
    values = dict(zip(COUPLING_KEYS, (
        float(settings.get("beta", 1.)), float(settings.get("position_scale_m", .30)),
        float(settings.get("orientation_scale_rad", 1.)),
        float(risk["d_safe"]), float(risk["d_cut"]),
    )))
    validate_parameters(values)
    return values


def validate_parameters(parameters):
    beta, pos, ori, safe, cutoff = (float(parameters[k]) for k in COUPLING_KEYS)
    if (not np.isfinite([beta, pos, ori, safe, cutoff]).all()
            or beta < 0 or pos <= 0 or ori <= 0 or not 0 <= safe < cutoff):
        raise ValueError("invalid goal-obstacle coupling scales, beta or distance band")


def obstacle_distances(sets):
    """Reconstruct both endpoints from stored raw sets, including old replay."""
    distances, present = [], []
    for features in sets:
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[1] != 37 or not np.isfinite(features).all():
            raise ValueError("coupling requires preserved finite raw 37D obstacle sets")
        present.append(bool(len(features)))
        distances.append(float(features[:, :4].min()) if len(features) else .8)
    return np.asarray(distances, np.float64), np.asarray(present, bool)


def coupling_potential(position, orientation, distance, present, parameters):
    validate_parameters(parameters)
    position, orientation, distance, present = np.broadcast_arrays(
        np.asarray(position, np.float64), np.asarray(orientation, np.float64),
        np.asarray(distance, np.float64), np.asarray(present, bool),
    )
    if (not np.isfinite(position).all() or not np.isfinite(orientation).all()
            or not np.isfinite(distance).all() or (position < 0).any() or (orientation < 0).any()):
        raise ValueError("coupling requires finite distances and nonnegative pose errors")
    x = np.clip((distance - parameters[COUPLING_KEYS[3]]) /
                (parameters[COUPLING_KEYS[4]] - parameters[COUPLING_KEYS[3]]), 0., 1.)
    safety = x * x * (3. - 2. * x)
    pose = 1. / (1. + position / parameters[COUPLING_KEYS[1]]
                 + orientation / parameters[COUPLING_KEYS[2]])
    return parameters[COUPLING_KEYS[0]] * present * pose * safety


def coupling_shaping(position, next_position, orientation, next_orientation,
                     distance, next_distance, present, next_present, done, parameters, gamma):
    before = coupling_potential(position, orientation, distance, present, parameters)
    after = coupling_potential(next_position, next_orientation, next_distance, next_present, parameters)
    # All trainer dones (including horizon truncation) are absorbing. Do NOT
    # zero the whole reward on the last step: it must be -Phi(current).
    after = np.where(np.asarray(done, bool), 0., after)
    return gamma * after - before, before, after
