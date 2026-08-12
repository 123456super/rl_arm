from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import json
import numpy as np
import pytest

from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config
from scripts.evaluate_terminal_servo import _mean, _servo_blend, _terminal_servo_action
from scripts.export_static_waypoint_teacher import export_teacher


class _FakeServoEnv:
    joint_count = 2
    action_scale = 1.0
    risk_config = SimpleNamespace(d_safe=0.12)
    tool_link_id = 0

    def _goal_error(self) -> np.ndarray:
        return np.asarray([0.08, 0.0, 0.0], dtype=np.float64)

    @staticmethod
    def _link_origin_jacobian(link_id: int) -> np.ndarray:
        assert link_id == 0
        return np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)


def test_terminal_servo_blend_and_clearance_gate() -> None:
    env = _FakeServoEnv()
    observation = np.zeros(16, dtype=np.float32)
    observation[4:7] = np.asarray([0.08, 0.0, 0.0], dtype=np.float32)
    action, meta = _terminal_servo_action(
        env,
        observation,
        np.zeros(2, dtype=np.float32),
        {"d_min": 0.20},
        trigger_error_m=0.10,
        full_error_m=0.055,
        clearance_margin_m=0.03,
        gain=3.0,
        damping=0.05,
        max_speed_radps=None,
    )

    assert np.isclose(_servo_blend(0.08, 0.10, 0.055), 0.44444444444444464)
    assert meta["servo_safe_to_use"] is True
    assert meta["servo_active"] is True
    assert meta["servo_blend"] > 0.0
    assert action[0] > 0.0

    action, meta = _terminal_servo_action(
        env,
        observation,
        np.zeros(2, dtype=np.float32),
        {"d_min": 0.10},
        trigger_error_m=0.10,
        full_error_m=0.055,
        clearance_margin_m=0.03,
        gain=3.0,
        damping=0.05,
        max_speed_radps=None,
    )

    assert meta["servo_safe_to_use"] is False
    assert meta["servo_active"] is False
    assert np.allclose(action, 0.0)


def test_static_waypoint_teacher_export_round_trip(tmp_path) -> None:
    feasibility = tmp_path / "feasibility.json"
    feasibility.write_text(
        json.dumps(
            {
                "episodes_detail": [
                    {
                        "seed": 9001,
                        "initial_joint_positions_rad": [0.0, 0.0],
                        "goal_m": [0.4, -0.2, 0.4],
                        "obstacle_position_m": [0.5, 0.3, 0.4],
                        "obstacle_velocity_mps": [0.0, 0.0, 0.0],
                        "obstacle_speed_mps": 0.0,
                        "dynamic_obstacle_path": {
                            "status": "candidate_path_found",
                            "reason": "candidate_path_found",
                            "candidate_paths": [
                                {
                                    "goal_q_rad": [1.0, 0.0],
                                    "duration_s": 0.10,
                                    "required_clearance_m": 0.16,
                                    "min_capsule_clearance_m": 0.21,
                                    "max_workspace_violation_m": 0.0,
                                }
                            ],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    summary = export_teacher(
        feasibility,
        output_json=tmp_path / "teacher_summary.json",
        output_jsonl=tmp_path / "teacher_waypoints.jsonl",
        dt_s=0.05,
    )

    lines = (tmp_path / "teacher_waypoints.jsonl").read_text(encoding="utf-8").splitlines()
    assert summary["candidate_episode_count"] == 1
    assert summary["waypoint_count"] == 3
    assert len(lines) == 3
    first = json.loads(lines[0])
    last = json.loads(lines[-1])
    assert first["q_rad"] == [0.0, 0.0]
    assert last["q_rad"] == [1.0, 0.0]
    assert last["time_s"] == pytest.approx(0.10)
    assert last["required_clearance_m"] == pytest.approx(0.16)
    assert last["min_capsule_clearance_m"] == pytest.approx(0.21)


def test_static_waypoint_teacher_prefers_selected_path_metadata(tmp_path) -> None:
    feasibility = tmp_path / "feasibility.json"
    feasibility.write_text(
        json.dumps(
            {
                "episodes_detail": [
                    {
                        "seed": 9001,
                        "initial_joint_positions_rad": [0.0, 0.0],
                        "dynamic_obstacle_path": {
                            "status": "candidate_path_found",
                            "reason": "candidate_path_found",
                            "required_clearance_m": 0.10,
                            "min_capsule_clearance_m": 0.11,
                            "candidate_paths": [
                                {
                                    "goal_q_rad": [1.0, 0.0],
                                    "duration_s": 0.05,
                                    "required_clearance_m": 0.16,
                                    "min_capsule_clearance_m": 0.21,
                                    "max_workspace_violation_m": 0.01,
                                }
                            ],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = export_teacher(
        feasibility,
        output_json=tmp_path / "teacher_summary.json",
        output_jsonl=tmp_path / "teacher_waypoints.jsonl",
        dt_s=0.05,
    )

    row = json.loads((tmp_path / "teacher_waypoints.jsonl").read_text(encoding="utf-8").splitlines()[0])
    detail = summary["episodes_detail"][0]
    assert row["required_clearance_m"] == pytest.approx(0.16)
    assert row["min_capsule_clearance_m"] == pytest.approx(0.21)
    assert row["max_workspace_violation_m"] == pytest.approx(0.01)
    assert detail["required_clearance_m"] == pytest.approx(0.16)


def test_terminal_servo_norm_means_do_not_require_trace_rows() -> None:
    assert np.isfinite(_mean([0.1, 0.3]))
    assert _mean([0.1, 0.3]) == pytest.approx(0.2)


def test_residual_control_zero_action_uses_base_command() -> None:
    config = load_config("configs/default.yaml")
    config["device"] = "cpu"
    config["env"]["residual_control"]["enabled"] = True
    config["env"]["residual_control"]["residual_scale"] = 0.25
    config["env"]["obstacle"]["enabled"] = False
    config["env"]["max_episode_steps"] = 2
    env = UR5DynamicObstacleEnv(config, method="link_fixed")
    try:
        env.reset(seed=123)
        _, _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert info["residual_control_enabled"] is True
        assert info["residual_control_mode"] in {"goal_clf", "terminal_clf"}
        assert np.linalg.norm(info["residual_control_base_qdot"]) > 0.0
        assert np.allclose(info["residual_qdot"], 0.0)
        np.testing.assert_allclose(info["qdot_policy"], info["residual_control_base_qdot"])
    finally:
        env.close()


def test_residual_control_holds_direct_goal_clf_without_clearance() -> None:
    config = load_config("configs/default.yaml")
    config["device"] = "cpu"
    config["env"]["residual_control"]["enabled"] = True
    env = UR5DynamicObstacleEnv(config, method="link_fixed")
    try:
        env.reset(seed=123)
        ee_position, _ = env._end_effector_state()
        env.goal = np.asarray(ee_position + np.asarray([0.30, 0.0, 0.0]), dtype=np.float32)
        env.obstacle_center = np.asarray(ee_position + np.asarray([0.0, 0.30, 0.0]), dtype=np.float32)
        current_risk = env._compute_risk()
        current_risk.d_min = env.risk_config.d_safe

        base_qdot, info = env._residual_base_command(current_risk)

        assert info["residual_control_mode"] == "hold_for_clearance"
        assert info["residual_control_reason"] == "insufficient_clearance_for_goal_clf"
        assert np.allclose(base_qdot, 0.0)
    finally:
        env.close()


def test_residual_control_waypoint_activates_for_blocked_direct_path() -> None:
    config = load_config("configs/default.yaml")
    config["device"] = "cpu"
    config["env"]["residual_control"]["enabled"] = True
    env = UR5DynamicObstacleEnv(config, method="link_fixed")
    try:
        env.reset(seed=123)
        ee_position, _ = env._end_effector_state()
        env.goal = np.asarray(ee_position + np.asarray([0.30, 0.0, 0.0]), dtype=np.float32)
        env.obstacle_center = np.asarray(ee_position + np.asarray([0.15, 0.0, 0.0]), dtype=np.float32)
        _, _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert info["residual_control_mode"] == "waypoint"
        assert info["residual_control_waypoint_active"] is True
        assert info["residual_control_reason"] == "direct_path_blocked"
    finally:
        env.close()
