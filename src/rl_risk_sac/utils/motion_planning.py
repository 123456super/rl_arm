from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


StateValidator = Callable[[np.ndarray], bool]
EdgeValidator = Callable[[np.ndarray, np.ndarray], bool]


@dataclass(frozen=True)
class RRTConnectConfig:
    step_size_rad: float = 0.25
    edge_resolution_rad: float = 0.06
    max_iterations: int = 1200
    goal_sample_probability: float = 0.10

    def __post_init__(self) -> None:
        if not np.isfinite(self.step_size_rad) or self.step_size_rad <= 0.0:
            raise ValueError("step_size_rad must be finite and positive")
        if not np.isfinite(self.edge_resolution_rad) or self.edge_resolution_rad <= 0.0:
            raise ValueError("edge_resolution_rad must be finite and positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not np.isfinite(self.goal_sample_probability) or not 0.0 <= self.goal_sample_probability <= 1.0:
            raise ValueError("goal_sample_probability must be in [0, 1]")


@dataclass(frozen=True)
class PlanResult:
    success: bool
    path: tuple[np.ndarray, ...]
    reason: str
    iterations: int
    sampled_states: int
    direct_path: bool = False


@dataclass
class _Tree:
    nodes: list[np.ndarray]
    parents: list[int]
    rooted_at_start: bool


class RRTConnectPlanner:
    """Dependency-free bidirectional RRT-Connect in bounded joint space.

    Robot-specific validity checks are injected by the environment.  Keeping
    the search independent of PyBullet makes the algorithm deterministic and
    unit-testable while the environment remains the single source of truth for
    joint, workspace and collision validity.
    """

    def __init__(
        self,
        lower_bounds_rad: np.ndarray,
        upper_bounds_rad: np.ndarray,
        state_is_valid: StateValidator,
        *,
        config: RRTConnectConfig | None = None,
        edge_is_valid: EdgeValidator | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.lower = np.asarray(lower_bounds_rad, dtype=np.float64)
        self.upper = np.asarray(upper_bounds_rad, dtype=np.float64)
        if self.lower.ndim != 1 or self.upper.shape != self.lower.shape:
            raise ValueError("joint bounds must be same-shaped vectors")
        if not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all():
            raise ValueError("joint bounds must be finite")
        if np.any(self.lower >= self.upper):
            raise ValueError("every lower joint bound must be below its upper bound")
        self.state_is_valid = state_is_valid
        self.config = config or RRTConnectConfig()
        self.edge_is_valid = edge_is_valid or self._interpolated_edge_is_valid
        self.rng = rng or np.random.default_rng()

    def plan(self, start_rad: np.ndarray, goal_rad: np.ndarray) -> PlanResult:
        start = self._state(start_rad)
        goal = self._state(goal_rad)
        if not self.state_is_valid(start):
            return PlanResult(False, (), "invalid_start", 0, 0)
        if not self.state_is_valid(goal):
            return PlanResult(False, (), "invalid_goal", 0, 0)
        if self.edge_is_valid(start, goal):
            return PlanResult(True, (start.copy(), goal.copy()), "direct_path", 0, 0, True)

        tree_a = _Tree([start.copy()], [-1], True)
        tree_b = _Tree([goal.copy()], [-1], False)
        sampled_states = 0
        for iteration in range(1, self.config.max_iterations + 1):
            bias_target = tree_b.nodes[0]
            sample = (
                bias_target.copy()
                if self.rng.random() < self.config.goal_sample_probability
                else self.rng.uniform(self.lower, self.upper)
            )
            sampled_states += 1
            new_a = self._extend(tree_a, sample)
            if new_a is not None:
                new_b, reached = self._connect(tree_b, tree_a.nodes[new_a])
                if reached and new_b is not None:
                    path = self._joined_path(tree_a, new_a, tree_b, new_b)
                    return PlanResult(
                        True,
                        tuple(point.copy() for point in self._shortcut(path)),
                        "rrt_connect",
                        iteration,
                        sampled_states,
                    )
            tree_a, tree_b = tree_b, tree_a
        return PlanResult(False, (), "max_iterations", self.config.max_iterations, sampled_states)

    def _state(self, value: np.ndarray) -> np.ndarray:
        state = np.asarray(value, dtype=np.float64)
        if state.shape != self.lower.shape:
            raise ValueError(f"joint state must have shape {self.lower.shape}")
        if not np.isfinite(state).all():
            raise ValueError("joint state must be finite")
        return state

    def _extend(self, tree: _Tree, target: np.ndarray) -> int | None:
        distances = [float(np.linalg.norm(node - target)) for node in tree.nodes]
        parent = int(np.argmin(distances))
        nearest = tree.nodes[parent]
        delta = target - nearest
        distance = float(np.linalg.norm(delta))
        if distance <= 1.0e-12:
            return parent
        candidate = nearest + min(self.config.step_size_rad, distance) * delta / distance
        candidate = np.clip(candidate, self.lower, self.upper)
        if not self.state_is_valid(candidate) or not self.edge_is_valid(nearest, candidate):
            return None
        tree.nodes.append(candidate)
        tree.parents.append(parent)
        return len(tree.nodes) - 1

    def _connect(self, tree: _Tree, target: np.ndarray) -> tuple[int | None, bool]:
        last: int | None = None
        while True:
            new_index = self._extend(tree, target)
            if new_index is None:
                return last, False
            last = new_index
            if np.linalg.norm(tree.nodes[new_index] - target) <= 1.0e-9:
                return new_index, True
            # _extend can return an existing node when target equals it.
            if tree.parents[new_index] < 0:
                return new_index, False

    @staticmethod
    def _branch(tree: _Tree, index: int) -> list[np.ndarray]:
        branch = []
        while index >= 0:
            branch.append(tree.nodes[index])
            index = tree.parents[index]
        branch.reverse()
        return branch

    def _joined_path(self, tree_a: _Tree, index_a: int, tree_b: _Tree, index_b: int) -> list[np.ndarray]:
        branch_a = self._branch(tree_a, index_a)
        branch_b = self._branch(tree_b, index_b)
        if tree_a.rooted_at_start:
            start_branch, goal_branch = branch_a, branch_b
        else:
            start_branch, goal_branch = branch_b, branch_a
        return start_branch + list(reversed(goal_branch[:-1]))

    def _shortcut(self, path: list[np.ndarray]) -> list[np.ndarray]:
        if len(path) <= 2:
            return path
        shortened = [path[0]]
        current = 0
        while current < len(path) - 1:
            next_index = len(path) - 1
            while next_index > current + 1 and not self.edge_is_valid(path[current], path[next_index]):
                next_index -= 1
            shortened.append(path[next_index])
            current = next_index
        return shortened

    def _interpolated_edge_is_valid(self, start: np.ndarray, end: np.ndarray) -> bool:
        max_delta = float(np.max(np.abs(end - start), initial=0.0))
        segments = max(1, int(np.ceil(max_delta / self.config.edge_resolution_rad)))
        for fraction in np.linspace(0.0, 1.0, segments + 1)[1:]:
            if not self.state_is_valid(start + fraction * (end - start)):
                return False
        return True
