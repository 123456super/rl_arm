from pathlib import Path

import numpy as np

from rl_risk_sac.envs.thesis_static_obstacles import (
    _obb_vertical_half_extent,
    _sample_physical_layout_plan,
)
from rl_risk_sac.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "configs/experiments/"
    "paper_s1_single_obb_smooth_10deg_delta3_horizon120_2m.yaml"
)


def test_single_obb_config_uses_physical_layouts_and_literature_size_range():
    config = load_config(CONFIG)
    thesis = config["thesis"]
    assert thesis["obstacle_half_extents_min_m"] == [0.05, 0.05, 0.05]
    assert thesis["obstacle_half_extents_max_m"] == [0.15, 0.15, 0.15]
    assert thesis["static_obstacle_count_probabilities"] == {1: 1.0}
    assert thesis["static_obstacle_layout_probabilities"] == {
        "legacy_blocking": 0.0,
        "workspace_random": 0.0,
        "grounded_path": 0.40,
        "grounded_random": 0.10,
        "suspended_path": 0.40,
        "suspended_random": 0.10,
    }


def test_single_obb_config_matches_horizontal_s0_base_contract():
    s1 = load_config(CONFIG)
    s0 = load_config(
        ROOT / "configs/experiments/"
        "paper_s0_cb31_10deg_horizon120_delta3_horizontal_tool_smooth5m.yaml"
    )
    for key in (
        "control_dt", "horizon", "joint_delta_limit_rad",
        "qdot_observation_scale", "fixed_orientation_goal",
        "joint_motion_controller",
    ):
        assert s1["thesis"][key] == s0["thesis"][key]


def test_obb_vertical_extent_supports_arbitrary_attitude():
    import pybullet as p

    half = np.asarray([0.05, 0.10, 0.15], dtype=np.float32)
    identity = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    assert _obb_vertical_half_extent(half, identity) == np.float32(0.15)
    rotated = np.asarray(p.getQuaternionFromEuler([0.0, np.pi / 2, 0.0]))
    assert np.isclose(_obb_vertical_half_extent(half, rotated), 0.05)


def test_count_curriculum_uses_per_obstacle_physical_mixture():
    config = load_config(
        ROOT / "configs/experiments/paper_s1_obb_count_1_2_curriculum2m.yaml"
    )
    thesis = config["thesis"]
    assert thesis["static_obstacle_count_probabilities"] == {1: 0.5, 2: 0.5}
    assert thesis["static_obstacle_layout_probabilities"] == {
        "legacy_blocking": 0.0,
        "workspace_random": 0.0,
        "grounded_path": 0.0,
        "grounded_random": 0.0,
        "suspended_path": 0.0,
        "suspended_random": 0.0,
        "physical_mixed": 1.0,
    }


def test_physical_plan_caps_path_objects_and_samples_support_independently():
    rng = np.random.default_rng(11001)
    plans = [
        _sample_physical_layout_plan(
            5,
            challenging_probability=0.8,
            grounded_probability=0.5,
            extra_path_probability=0.25,
            max_path_obstacles=2,
            rng=rng,
        )
        for _ in range(10000)
    ]
    assert all(len(plan) == 5 for plan in plans)
    path_counts = np.asarray([
        sum(label.endswith("_path") for label in plan) for plan in plans
    ])
    grounded = np.asarray([
        label.startswith("grounded_") for plan in plans for label in plan
    ])
    assert int(path_counts.max()) == 2
    assert np.isclose(np.mean(path_counts > 0), 0.8, atol=0.02)
    assert np.isclose(np.mean(grounded), 0.5, atol=0.02)
