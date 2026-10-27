import numpy as np
import pytest

from rl_risk_sac.tasks.static_scene_planning import joint_segment, path_band, rrt_connect, rrt_goal_region
from rl_risk_sac.utils.config import load_config
from rl_risk_sac.envs.thesis_homotopy_env import (
    ResetSamplingError, StaticSceneSamplingError, ThesisHomotopyEnv,
)
from scripts.core.train_thesis_homotopy import apply_stage_profile


def test_rrt_finds_detour_and_respects_budget():
    valid = lambda q: not (abs(q[0]) < .12 and q[1] < .4)
    start, goal = np.array([-.8, 0.]), np.array([.8, 0.])
    path = rrt_connect(start, goal, [-1, -1], [1, 1], valid,
                       np.random.default_rng(5), iterations=1000, max_checks=10000)
    assert path is not None and len(path) > 2
    np.testing.assert_allclose(path[0], start)
    np.testing.assert_allclose(path[-1], goal)
    assert all(valid(q) for a, b in zip(path, path[1:]) for q in joint_segment(a, b, .02))
    calls = []
    def disconnected(q):
        calls.append(q)
        return abs(q[0]) > .12
    assert rrt_connect(start, goal, [-1, -1], [1, 1], disconnected,
                       np.random.default_rng(5), max_checks=100) is None
    assert len(calls) <= 100


def test_path_band_boundaries():
    assert path_band(.10001, .025, .10) == "far"
    assert path_band(.10, .025, .10) == "near"
    assert path_band(.02501, .025, .10) == "near"
    assert path_band(.025, .025, .10) == "blocking"
    assert path_band(-.02, .025, .10) == "blocking"


def test_goal_region_rrt_finds_connected_exit_and_respects_budget():
    valid = lambda q: not (abs(q[0]) < .12 and q[1] < .4)
    goal = lambda q: q[0] > .6
    stats = {}
    path = rrt_goal_region([-.8, 0.], [-1, -1], [1, 1], valid, goal,
                           np.random.default_rng(5), iterations=1000, max_checks=10000,
                           statistics=stats)
    assert path is not None and goal(path[-1])
    np.testing.assert_allclose(path[0], [-.8, 0.])
    assert all(valid(q) for a, b in zip(path, path[1:]) for q in joint_segment(a, b, .02))
    assert stats["checks"] <= 10000
    stats = {}
    assert rrt_goal_region([-.8, 0.], [-1, -1], [1, 1],
                           lambda q: abs(q[0]) > .12, goal, np.random.default_rng(5),
                           max_checks=50, statistics=stats) is None
    assert stats["checks"] <= 50


@pytest.mark.parametrize("band", ["none", "far", "near", "blocking"])
def test_real_static_scene_band_and_witness(band):
    c = load_config("configs/base/thesis_common.yaml")
    apply_stage_profile(c, "s1")
    names = ["none", "far", "near", "blocking"]
    c["thesis"]["obstacles"]["path_sampling"]["probabilities"] = [float(n == band) for n in names]
    e = ThesisHomotopyEnv(c)
    try:
        level = c["thesis"]["joint_pose_curriculum"]["levels"][0]
        e.configure_episode("none", xi=1., strict=True,
                            position_tolerance=level["position_tolerance_m"],
                            orientation_tolerance=level["orientation_tolerance_rad"],
                            **{k: level[k] for k in ("target_distance_min_m", "target_distance_max_m",
                                                    "target_orientation_min_rad", "target_orientation_max_rad")})
        _, info = e.reset(seed=110 + names.index(band))
        sampling = info["obstacle_sampling"]
        assert sampling["path_band"] == band
        assert info["obstacle_count"] == int(band != "none")
        if band != "none":
            assert sampling["path_verified"]
            d = sampling["sampling_diagnostics"]
            assert d["proposals"] == 1 + sum(d[key] for key in (
                "start_rejections", "goal_rejections", "band_rejections", "planning_rejections",
            ))
            assert path_band(sampling["direct_path_min_distance_m"], .025, .10) == band
            q, qdot = e.robot.joint_state_exact()
            route = sampling["witness_waypoints"]
            np.testing.assert_allclose(route[0], q)
            np.testing.assert_allclose(route[-1], e.goal_joint_positions)
            for a, b in zip(route, route[1:]):
                assert all(e._static_planning_valid(state, e.obstacles[0], .025)
                           for state in joint_segment(np.asarray(a), np.asarray(b), .02))
            e._set_joint_state(q, qdot)
        saved = e.episode_state_dict()
        e.restore_episode_state(saved)
        assert e._scene_sampling == sampling
    finally:
        e.close()


def _blocking_env():
    config = load_config("configs/base/thesis_common.yaml")
    apply_stage_profile(config, "s1")
    config["thesis"]["obstacles"]["path_sampling"]["probabilities"] = [0., 0., 0., 1.]
    env = ThesisHomotopyEnv(config)
    level = config["thesis"]["joint_pose_curriculum"]["levels"][0]
    env.configure_episode(
        "none", xi=1., strict=True,
        position_tolerance=level["position_tolerance_m"],
        orientation_tolerance=level["orientation_tolerance_rad"],
        **{k: level[k] for k in ("target_distance_min_m", "target_distance_max_m",
                                "target_orientation_min_rad", "target_orientation_max_rad")},
    )
    return env


def test_exhausted_batch_retries_same_band_and_cell_then_finds_real_witness(monkeypatch):
    env = _blocking_env()
    original = env._try_static_scene_batch
    calls = []

    def fail_first(band, *args, **kwargs):
        calls.append((band, env._selected_task_space_position_bin,
                      env._selected_task_space_orientation_bin, kwargs["resample_first"]))
        if len(calls) == 1:
            kwargs["diagnostics"]["proposals"] += 4096
            return False
        return original(band, *args, **kwargs)

    monkeypatch.setattr(env, "_try_static_scene_batch", fail_first)
    try:
        _, info = env.reset(seed=113)
        sampling = info["obstacle_sampling"]
        assert len(calls) >= 2
        assert len({c[:3] for c in calls}) == 1
        assert not calls[0][3] and all(c[3] for c in calls[1:])
        assert sampling["path_band"] == "blocking" and sampling["path_verified"]
        assert sampling["sampling_proposals"] > 4096
        assert sampling["sampling_diagnostics"]["batches"] >= 2
        for a, b in zip(sampling["witness_waypoints"], sampling["witness_waypoints"][1:]):
            assert all(env._static_planning_valid(q, env.obstacles[0], .025)
                       for q in joint_segment(np.asarray(a), np.asarray(b), .02))
    finally:
        env.close()


def test_all_sampling_batches_exhaust_with_diagnostics_not_category_fallback(monkeypatch):
    env = _blocking_env()
    calls = []

    def fail(band, *args, **kwargs):
        calls.append(band)
        kwargs["diagnostics"]["proposals"] += 4096
        return False

    monkeypatch.setattr(env, "_try_static_scene_batch", fail)
    try:
        with pytest.raises(StaticSceneSamplingError, match="does not prove infeasibility") as caught:
            env.reset(seed=113)
        assert calls == ["blocking"] * 8
        assert caught.value.diagnostics["sampling_proposals"] == 32768
        assert caught.value.diagnostics["sampling_diagnostics"]["batches"] == 8
        assert not caught.value.diagnostics["path_verified"]
        assert not env.obstacles
    finally:
        env.close()


def test_reset_goal_budget_miss_is_retryable_but_programming_error_is_not(monkeypatch):
    env = _blocking_env()
    calls = []

    def fail_reset():
        calls.append(1)
        raise ResetSamplingError("test budget exhaustion")

    monkeypatch.setattr(env, "_sample_valid_reset", fail_reset)
    diagnostics = dict(goal_attempts=0, reset_goal_failures=0)
    try:
        assert not env._try_static_scene_batch(
            "blocking", {}, .025, .10, .02, 256, 3,
            resample_first=True, diagnostics=diagnostics,
        )
        assert len(calls) == diagnostics["reset_goal_failures"] == 3
        def bug():
            raise RuntimeError("unrelated programming error")
        monkeypatch.setattr(env, "_sample_valid_reset", bug)
        with pytest.raises(RuntimeError, match="unrelated programming error"):
            env._try_static_scene_batch(
                "blocking", {}, .025, .10, .02, 256, 3,
                resample_first=True, diagnostics=diagnostics,
            )
    finally:
        env.close()
