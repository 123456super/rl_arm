"""Analytic support maps, GJK distance and EPA penetration for convex sweeps.

No sampled hull replaces the original sphere, box or cylinder. EPA refines a
support polytope until its penetration bound converges; it fails explicitly
if convergence is not obtained, rather than silently returning a distance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product

import numpy as np
from scipy.spatial import ConvexHull

from rl_risk_sac.tasks.obstacle_prediction import line_segment

# Six axes plus six face diagonals; fixed in the robot base frame.
SUPPORT_DIRECTIONS = np.asarray([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1),
                                 (1,1,0),(-1,-1,0),(1,0,1),(-1,0,-1),(0,1,1),(0,-1,-1)], float)
SUPPORT_DIRECTIONS /= np.linalg.norm(SUPPORT_DIRECTIONS, axis=1)[:, None]
ARM_GROUPS = ("upper_arm", "forearm", "wrist_1", "wrist_2")


@dataclass
class ConvexSweep:
    shape: str
    size: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    emin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    emax: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        if self.shape not in {"sphere", "box", "cylinder"}:
            raise ValueError("unsupported geometry: nonconvex objects require explicit convex decomposition")
        for key in ("size", "position", "rotation", "emin", "emax"):
            setattr(self, key, np.asarray(getattr(self, key), dtype=np.float64))

    @property
    def center(self):
        return self.position + (self.emin + self.emax) / 2

    def support(self, direction):
        d = self.rotation.T @ np.asarray(direction, dtype=float)
        norm = np.linalg.norm(d)
        if norm < 1e-15:
            d, norm = np.array([1., 0., 0.]), 1.
        if self.shape == "sphere":
            local = self.size[0] * d / norm
        elif self.shape == "box":
            local = np.where(d >= 0, self.size, -self.size)
        else:
            radial = np.linalg.norm(d[:2])
            xy = self.size[0] * d[:2] / radial if radial > 1e-15 else np.zeros(2)
            local = np.r_[xy, self.size[1] / 2 * (1 if d[2] >= 0 else -1)]
        endpoint = self.emax if direction @ self.emax > direction @ self.emin else self.emin
        return self.position + self.rotation @ local + endpoint


@dataclass
class ConvexQuery:
    distance: float
    robot_point: np.ndarray
    obstacle_point: np.ndarray
    normal: np.ndarray


def _closest_simplex(vertices):
    """Minimum-norm convex combination, enumerating simplex faces (<=4)."""
    points = np.asarray([v[0] for v in vertices])
    best = None
    for count in range(1, min(4, len(vertices)) + 1):
        for ids in combinations(range(len(vertices)), count):
            subset = points[list(ids)]
            gram = subset @ subset.T
            matrix = np.block([[gram, np.ones((count, 1))], [np.ones((1, count)), np.zeros((1, 1))]])
            weights = np.linalg.lstsq(matrix, np.r_[np.zeros(count), 1.], rcond=None)[0][:count]
            if weights.min() < -1e-9:
                continue
            weights = np.maximum(weights, 0)
            weights /= weights.sum()
            point = weights @ subset
            value = float(point @ point)
            if best is None or value < best[0]:
                best = value, point, ids, weights
    _, point, ids, weights = best
    keep = weights > 1e-10
    return point, [vertices[i] for i, valid in zip(ids, keep) if valid], weights[keep] / weights[keep].sum()


def _witness(vertices, weights):
    return (sum(w * v[1] for w, v in zip(weights, vertices)),
            sum(w * v[2] for w, v in zip(weights, vertices)))


def signed_capsule_distance(sweep, capsule, tolerance=2e-6, max_iterations=128):
    """Return signed surface distance, witness points and obstacle->arm normal."""
    if sweep.shape == "sphere" and np.linalg.norm(sweep.emax - sweep.emin) < 1e-12:
        center = sweep.center
        axis = capsule.end - capsule.start
        fraction = np.clip((center - capsule.start) @ axis / max(axis @ axis, 1e-30), 0., 1.)
        closest = capsule.start + fraction * axis
        delta = closest - center
        length = np.linalg.norm(delta)
        if length > 1e-12:
            normal = delta / length
        else:
            normal = np.cross(axis, [1., 0., 0.])
            if np.linalg.norm(normal) < 1e-12:
                normal = np.array([0., 1., 0.])
            normal /= np.linalg.norm(normal)
        return ConvexQuery(float(length - capsule.radius - sweep.size[0]),
                           closest - capsule.radius * normal, center + sweep.size[0] * normal, normal)
    def support(direction):
        direction = np.asarray(direction, dtype=float)
        unit = direction / max(np.linalg.norm(direction), 1e-15)
        endpoint = capsule.end if direction @ capsule.end > direction @ capsule.start else capsule.start
        robot = endpoint + capsule.radius * unit
        obstacle = sweep.support(-direction)
        return robot - obstacle, robot, obstacle

    direction = (capsule.start + capsule.end) / 2 - sweep.center
    if np.linalg.norm(direction) < 1e-10:
        direction = np.array([1., 0., 0.])
    vertices = [support(-direction)]
    point, vertices, weights = _closest_simplex(vertices)
    for _ in range(max_iterations):
        norm = np.linalg.norm(point)
        if norm <= tolerance:
            break
        new = support(-point)
        gap = point @ (point - new[0])
        if gap <= tolerance * norm:
            robot, obstacle = _witness(vertices, weights)
            return ConvexQuery(float(norm), robot, obstacle, point / norm)
        if any(np.linalg.norm(new[0] - v[0]) < 1e-12 for v in vertices):
            robot, obstacle = _witness(vertices, weights)
            return ConvexQuery(float(norm), robot, obstacle, point / norm)
        point, vertices, weights = _closest_simplex(vertices + [new])
    else:
        raise RuntimeError("GJK distance did not converge")

    # Build an enclosing initial EPA polytope with deterministic support rays.
    candidates = vertices + [support(np.asarray(d, float)) for d in product((-1, 0, 1), repeat=3) if any(d)]
    vertices = []
    for v in candidates:
        if not any(np.linalg.norm(v[0] - old[0]) < 1e-10 for old in vertices):
            vertices.append(v)
    for _ in range(max_iterations * 2):
        hull = ConvexHull(np.asarray([v[0] for v in vertices]))
        depths = -hull.equations[:, 3]
        face_index = int(np.argmin(depths))
        normal = hull.equations[face_index, :3]
        depth = float(depths[face_index])
        new = support(normal)
        gap = float(normal @ new[0] - depth)
        if gap <= tolerance or any(np.linalg.norm(new[0] - v[0]) < 1e-10 for v in vertices):
            face = [vertices[i] for i in hull.simplices[face_index]]
            _, active, face_weights = _closest_simplex(face)
            robot, obstacle = _witness(active, face_weights)
            return ConvexQuery(-max(0., depth), robot, obstacle, -normal)
        vertices.append(new)
    raise RuntimeError("EPA penetration did not converge")


def sweep_features(records, predictions, capsules, epsilon_line, epsilon_pos):
    if not records:
        return np.empty((0, 37), np.float32), []
    groups = {c.name: c for c in capsules}
    features, queries = [], []
    for record in records:
        if record["id"] not in predictions:
            continue
        emin, emax, _ = line_segment(predictions[record["id"]] - record["position"], epsilon_line, epsilon_pos)
        sweep = ConvexSweep(record["shape"], record["size"], record["position"], record["rotation"], emin, emax)
        results = [signed_capsule_distance(sweep, groups[name]) for name in ARM_GROUPS]
        support_ranges = [d @ (sweep.support(d) - sweep.center) for d in SUPPORT_DIRECTIONS]
        features.append(np.r_[[q.distance for q in results], np.concatenate([q.normal for q in results]),
                              sweep.center, emin, emax, support_ranges])
        queries.append(results)
    return np.asarray(features, np.float32).reshape(-1, 37), queries


def swept_risk(features, *, d_safe, d_cut, tau, phi_max):
    if not (tau > 0 and phi_max > 0 and d_cut >= d_safe >= 0):
        raise ValueError("invalid external swept risk calibration")
    if not len(features):
        return 0.0
    nearest = np.min(np.asarray(features)[:, :4], axis=0)
    risk = np.where(nearest <= d_cut, np.minimum(1., np.logaddexp(0., (d_safe - nearest) / tau) / phi_max), 0.)
    return float(np.max(risk))
