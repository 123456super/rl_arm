"""Vision-free point-mass kinodynamic search followed by checked B-spline smoothing.

This is a small simulation implementation of the paper's planning architecture,
not the authors' unpublished manipulator/Fast-Planner integration. It checks the
end-effector envelope; link avoidance remains the learned controller's task.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import product

import numpy as np

from rl_risk_sac.tasks.paper_reaching import point_segment_distances


@dataclass
class TaskSpacePlan:
    positions: np.ndarray
    times: np.ndarray
    expanded: int
    smoothing: str

    def waypoints(self, spacing):
        distances = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        arc = np.concatenate(([0.], np.cumsum(distances)))
        if arc[-1] <= 1e-12:
            return self.positions[[0, -1]].copy()
        query = np.linspace(0, arc[-1], max(2, int(np.ceil(arc[-1] / spacing)) + 1))
        return np.stack([np.interp(query, arc, self.positions[:, j]) for j in range(3)], axis=1)


class SimulationTaskSpacePlanner:
    def __init__(self, config, obstacle_centers, obstacle_radius, *, workspace_radius=.85, base=(0, 0, 0)):
        self.cfg = config
        self.centers = np.asarray(obstacle_centers, dtype=np.float64).reshape(-1, 3)
        self.radius = float(obstacle_radius + config["end_effector_radius"] + config["margin"])
        self.base = np.asarray(base, float)
        self.workspace_radius = workspace_radius

    def free(self, points):
        points = np.asarray(points, float).reshape(-1, 3)
        if (not np.isfinite(points).all() or np.any(points[:, 2] < self.cfg["end_effector_radius"])
                or np.any(np.linalg.norm(points - self.base, axis=1) > self.workspace_radius + .05)):
            return False
        if not len(self.centers):
            return True
        if np.any(np.linalg.norm(points[:, None] - self.centers, axis=-1) <= self.radius):
            return False
        if len(points) > 1:
            # Sphere/segment clearance prevents stepping through small obstacles.
            return bool(np.all(point_segment_distances(self.centers, points[:-1], points[1:]) > self.radius))
        return True

    def _key(self, position, velocity):
        return tuple(np.rint(np.concatenate((position / self.cfg["resolution"],
                                             velocity / self.cfg["velocity_resolution"]))).astype(int))

    def _shot(self, position, velocity, target):
        distance = np.linalg.norm(target - position)
        duration = max(self.cfg["primitive_dt"], 2 * distance / self.cfg["max_speed"])
        for scale in (1., 2., 4.):
            total = duration * scale
            a2 = 3 * (target - position) / total**2 - 2 * velocity / total
            a3 = -2 * (target - position) / total**3 + velocity / total**2
            times = np.linspace(0, total, max(8, int(np.ceil(distance / .005)) + 1))
            points = position + times[:, None] * velocity + times[:, None]**2 * a2 + times[:, None]**3 * a3
            speeds = velocity + 2 * times[:, None] * a2 + 3 * times[:, None]**2 * a3
            acceleration = 2 * a2 + 6 * times[:, None] * a3
            if (np.max(np.linalg.norm(speeds, axis=1)) <= self.cfg["max_speed"] * 1.001
                    and np.max(np.linalg.norm(acceleration, axis=1)) <= self.cfg["max_acceleration"] * 1.001
                    and self.free(points)):
                return points, times
        return None

    def plan(self, start, target):
        start, target = np.asarray(start, float), np.asarray(target, float)
        if not self.free([start]) or not self.free([target]):
            raise RuntimeError("planner endpoint is outside the collision-free end-effector envelope")
        direct = self._shot(start, np.zeros(3), target)
        if direct is not None:
            return self._smooth(*direct, expanded=0)
        dt = self.cfg["primitive_dt"]
        controls = np.asarray(list(product((-1., 0., 1.), repeat=3))) * self.cfg["max_acceleration"] / np.sqrt(3)
        nodes = [(start, np.zeros(3), -1, None)]
        best = {self._key(start, np.zeros(3)): 0.}
        heap = [(np.linalg.norm(target - start) / self.cfg["max_speed"], 0., 0)]
        expanded = 0
        while heap and expanded < self.cfg["max_expansions"]:
            _, cost, index = heapq.heappop(heap)
            position, velocity, _, _ = nodes[index]
            if cost > best.get(self._key(position, velocity), float("inf")) + 1e-9:
                continue
            expanded += 1
            shot = self._shot(position, velocity, target) if np.linalg.norm(target - position) < .4 else None
            if shot is not None:
                segments = []
                cursor = index
                while nodes[cursor][2] >= 0:
                    segments.append(nodes[cursor][3])
                    cursor = nodes[cursor][2]
                positions, times, clock = [start], [0.], 0.
                for segment in reversed(segments):
                    positions.extend(segment[1:])
                    times.extend(clock + np.linspace(0, dt, len(segment))[1:])
                    clock += dt
                positions.extend(shot[0][1:])
                times.extend(clock + shot[1][1:])
                return self._smooth(np.asarray(positions), np.asarray(times), expanded)
            for acceleration in controls:
                next_velocity = velocity + acceleration * dt
                if np.linalg.norm(next_velocity) > self.cfg["max_speed"] + 1e-9:
                    continue
                times = np.linspace(0, dt, 13)
                segment = position + times[:, None] * velocity + .5 * times[:, None]**2 * acceleration
                following = segment[-1]
                key = self._key(following, next_velocity)
                next_cost = cost + dt * (1 + .05 * np.dot(acceleration, acceleration))
                if next_cost >= best.get(key, float("inf")) or not self.free(segment):
                    continue
                best[key] = next_cost
                nodes.append((following, next_velocity, index, segment))
                priority = next_cost + np.linalg.norm(target - following) / self.cfg["max_speed"]
                heapq.heappush(heap, (priority, next_cost, len(nodes) - 1))
        raise RuntimeError(f"kinodynamic search found no path within {expanded} expansions")

    def _smooth(self, positions, times, expanded):
        from scipy.interpolate import make_interp_spline

        positions, times = np.asarray(positions), np.asarray(times)
        if len(positions) < 4 or times[-1] <= 1e-9:
            return TaskSpacePlan(positions, times, expanded, "primitive")
        indices = np.unique(np.concatenate((np.arange(0, len(times), max(1, len(times) // 20)), [len(times) - 1])))
        spline = make_interp_spline(times[indices], positions[indices], k=min(3, len(indices) - 1))
        dense_times = np.linspace(0, times[-1], max(200, len(times) * 2))
        smoothed = spline(dense_times)
        if not self.free(smoothed):
            return TaskSpacePlan(positions, times, expanded, "primitive_collision_checked_fallback")
        vmax = np.max(np.linalg.norm(spline.derivative(1)(dense_times), axis=1))
        amax = np.max(np.linalg.norm(spline.derivative(2)(dense_times), axis=1))
        stretch = max(1., vmax / self.cfg["max_speed"], np.sqrt(amax / self.cfg["max_acceleration"]))
        return TaskSpacePlan(smoothed, dense_times * stretch, expanded, "bspline_retimed")
