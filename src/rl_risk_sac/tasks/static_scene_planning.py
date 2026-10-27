"""Bounded RRT-Connect witnesses for static training scenes.

Failure means no witness found within the budget, never a proof of no solution.
Edges are checked at a configured joint-space resolution, not continuously.
"""
from __future__ import annotations

import numpy as np


def joint_segment(start, goal, resolution):
    if resolution <= 0:
        raise ValueError("joint checking resolution must be positive")
    count = max(1, int(np.ceil(np.max(np.abs(np.asarray(goal) - start)) / resolution)))
    return np.linspace(start, goal, count + 1)


def path_band(distance, safe, cutoff):
    if not 0 < safe < cutoff:
        raise ValueError("path bands require 0 < safe < cutoff")
    return "far" if distance > cutoff else "near" if distance > safe else "blocking"


def rrt_connect(start, goal, lower, upper, valid, rng, *,
                resolution=.02, extension=.15, iterations=400, max_checks=4000,
                statistics=None):
    """Return a discretely checked path or None, with a strict check budget."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    if extension <= 0 or iterations < 1 or max_checks < 2:
        raise ValueError("invalid RRT budget")
    checks = 0
    if statistics is not None:
        statistics.update(checks=0, method="rrt_connect")

    def edge(a, b):
        nonlocal checks
        for q in joint_segment(a, b, resolution):
            if checks >= max_checks:
                return False
            checks += 1
            if statistics is not None:
                statistics["checks"] = checks
            if not valid(q):
                return False
        return True

    if not edge(start, start) or not edge(goal, goal):
        return None
    if edge(start, goal):
        if statistics is not None:
            statistics["method"] = "direct"
        return [start, goal]
    trees = [([start], [-1]), ([goal], [-1])]

    def extend(tree, target):
        nodes, parents = tree
        index = int(np.argmin([np.linalg.norm(q - target) for q in nodes]))
        delta = target - nodes[index]
        distance = float(np.max(np.abs(delta)))
        q = target.copy() if distance <= extension else nodes[index] + delta * extension / distance
        if not edge(nodes[index], q):
            return None, False
        nodes.append(q)
        parents.append(index)
        return len(nodes) - 1, distance <= extension

    def trace(tree, index):
        route = []
        while index >= 0:
            route.append(tree[0][index])
            index = tree[1][index]
        return route[::-1]

    for i in range(iterations):
        if checks >= max_checks:
            break
        side = i % 2
        a, b = trees[side], trees[1 - side]
        target = b[0][0] if rng.random() < .2 else rng.uniform(lower, upper)
        ia, _ = extend(a, target)
        if ia is None:
            continue
        while checks < max_checks:
            ib, reached = extend(b, a[0][ia])
            if ib is None:
                break
            if reached:
                left, right = trace(a, ia), trace(b, ib)
                return left + right[-2::-1] if side == 0 else right + left[-2::-1]
    return None


def rrt_goal_region(start, lower, upper, valid, goal_test, rng, *,
                    resolution=.02, extension=.15, iterations=400,
                    max_checks=4000, sample_target=None, statistics=None):
    """Bounded single-tree RRT for a goal region with no known target pose.

    Only nodes connected by checked edges are tested as goals. A miss means
    unconfirmed. ``sample_target`` may add task-specific sampling biases.
    """
    if resolution <= 0 or extension <= 0 or iterations < 1 or max_checks < 1:
        raise ValueError("invalid goal-region RRT budget")
    start = np.asarray(start, float)
    nodes, parents, checks = [start], [-1], 0
    if statistics is not None:
        statistics["checks"] = 0

    def check(q):
        nonlocal checks
        if checks >= max_checks:
            return False
        checks += 1
        if statistics is not None:
            statistics["checks"] = checks
        return valid(q)

    if not check(start):
        return None
    if goal_test(start):
        return [start, start.copy()]
    for _ in range(iterations):
        if checks >= max_checks:
            break
        target = rng.uniform(lower, upper) if sample_target is None else np.asarray(sample_target(), float)
        index = int(np.argmin([np.linalg.norm(q - target) for q in nodes]))
        delta = target - nodes[index]
        distance = float(np.max(np.abs(delta)))
        q = target.copy() if distance <= extension else nodes[index] + delta * extension / distance
        if not all(check(point) for point in joint_segment(nodes[index], q, resolution)[1:]):
            continue
        nodes.append(q)
        parents.append(index)
        if goal_test(q):
            route, cursor = [], len(nodes) - 1
            while cursor >= 0:
                route.append(nodes[cursor])
                cursor = parents[cursor]
            return route[::-1]
    return None
