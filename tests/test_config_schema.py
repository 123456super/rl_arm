from copy import deepcopy
from pathlib import Path

import pytest

from rl_risk_sac.utils.config import load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]


def test_resolved_config_contains_only_active_pipeline_sections():
    config = load_config(ROOT / "configs/experiments/paper_full.yaml")
    assert set(config) == {
        "seed", "device", "device_selection", "robot", "thesis", "ppo", "train"
    }


def test_restored_reference_ppo_training_contract():
    config = load_config(
        ROOT / "configs/experiments/paper_s1_mixed_1_2_3_4_5.yaml"
    )
    thesis = config["thesis"]
    reward = thesis["reward"]
    assert thesis["control_dt"] == pytest.approx(100.0 / 240.0)
    assert thesis["horizon"] == 100
    assert thesis["joint_delta_limit_rad"] == pytest.approx(0.25)
    assert not thesis["joint_motion_controller"]["enabled"]
    assert reward["reference_obstacle_threshold_m"] == pytest.approx(0.08)
    assert reward["reference_obstacle_clearance_weight"] == pytest.approx(1.0)
    assert reward["reference_obstacle_clearance_power"] == pytest.approx(1.0)
    assert reward["reference_obstacle_collision_penalty"] == pytest.approx(20.0)
    assert reward["reference_self_collision_penalty"] == pytest.approx(10.0)
    assert reward["reference_environment_collision_penalty"] == pytest.approx(10.0)
    assert reward["reference_residual_penalty_weight"] == pytest.approx(0.0)
    assert config["train"]["num_envs"] == 8
    assert config["train"]["rollout_length"] == 512
    assert config["ppo"]["action_std_init"] == pytest.approx(0.08)


def test_cb31_10deg_training_contract_is_self_consistent():
    config = load_config(
        ROOT / "configs/experiments/paper_s0_cb31_10deg_horizon150.yaml"
    )
    thesis = config["thesis"]
    motion = thesis["joint_motion_controller"]
    assert motion["enabled"]
    assert motion["joint_velocity_limits_rad_s"] == pytest.approx(
        [0.17453292519943295] * 6
    )
    assert motion["joint_acceleration_limits_rad_s2"] == pytest.approx(
        [0.3490658503988659] * 6
    )
    assert thesis["joint_delta_limit_rad"] == pytest.approx(
        0.17453292519943295 * thesis["control_dt"]
    )
    assert thesis["qdot_observation_scale"] == pytest.approx(
        0.17453292519943295
    )
    assert thesis["horizon"] == 150
    assert motion["real_control_frequency_hz"] == pytest.approx(125.0)
    assert thesis["control_dt"] * motion["real_control_frequency_hz"] == pytest.approx(50.0)
    assert config["train"]["num_envs"] == 16
    assert config["train"]["rollout_length"] == 256
    assert config["train"]["total_steps"] == 1000000
    assert config["ppo"]["actor_lr"] == pytest.approx(3.0e-5)
    assert config["ppo"]["critic_lr"] == pytest.approx(1.0e-4)
    assert thesis["reward"]["reference_self_collision_penalty"] == pytest.approx(10.0)


def test_s1_single_obstacle_inherits_10deg_s0_contract():
    config = load_config(
        ROOT / "configs/experiments/paper_s1_single_10deg_horizon150.yaml"
    )
    thesis = config["thesis"]
    assert thesis["stage"] == "s1"
    assert not thesis["simple_reaching_mode"]
    assert thesis["blocking_obstacle_mode"]
    assert thesis["static_obstacle_count_probabilities"] == {1: 1.0}
    assert thesis["static_obstacle_layout_probabilities"] == {
        "legacy_blocking": 0.5,
        "workspace_random": 0.5,
    }
    assert thesis["scene_sample_max_attempts"] == 64
    assert thesis["obstacle_surface_clearance_min_m"] == pytest.approx(0.10)
    assert thesis["scene_mixture"]["s1"] == {"static": 1.0}
    assert thesis["joint_motion_controller"]["enabled"]
    assert thesis["joint_motion_controller"]["joint_velocity_limits_rad_s"] == pytest.approx(
        [0.17453292519943295] * 6
    )
    assert thesis["joint_delta_limit_rad"] == pytest.approx(
        0.17453292519943295 * thesis["control_dt"]
    )
    assert thesis["horizon"] == 150
    assert thesis["reward"]["reference_residual_penalty_weight"] == 0.0


def test_obsolete_top_level_section_is_rejected():
    config = load_config(ROOT / "configs/experiments/paper_full.yaml")
    invalid = deepcopy(config)
    invalid["reward"] = {"legacy": True}
    with pytest.raises(KeyError, match="obsolete or unknown"):
        validate_config(invalid)


def test_obsolete_thesis_curriculum_and_projection_are_rejected():
    config = load_config(ROOT / "configs/experiments/paper_full.yaml")
    for field in ("protocol", "joint_pose_curriculum", "self_safety_projection"):
        invalid = deepcopy(config)
        invalid["thesis"][field] = {}
        with pytest.raises(KeyError, match="obsolete thesis"):
            validate_config(invalid)


def test_named_experiment_configs_declare_their_actual_stage():
    for path in (ROOT / "configs/experiments").glob("paper_s1_*.yaml"):
        config = load_config(path)
        assert config["thesis"]["stage"] == "s1"
        assert not config["thesis"]["simple_reaching_mode"]
    s2 = load_config(
        ROOT / "configs/experiments/paper_s2_lstm_sweep_single.yaml"
    )
    assert s2["thesis"]["stage"] == "s2"
    assert not s2["thesis"]["simple_reaching_mode"]


def test_s2_lstm_training_and_inference_share_one_config():
    config = load_config(
        ROOT / "configs/experiments/paper_s2_lstm_sweep_single.yaml"
    )
    prediction = config["thesis"]["dynamic_prediction"]
    assert prediction["mode"] == "lstm"
    assert Path(prediction["training"]["output_dir"]) == Path(
        prediction["checkpoint"]
    ).parent
    assert prediction["history_length"] == 10
    assert prediction["history_sample_hz"] == 30.0
