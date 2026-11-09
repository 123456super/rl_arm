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
