from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError

import pytest

from rl_risk_sac.utils.config import load_config, validate_config
from rl_risk_sac.utils.runtime_config import RuntimeConfig


def test_resolved_yaml_is_converted_to_typed_frozen_runtime_config() -> None:
    runtime = RuntimeConfig.from_mapping(load_config("configs/default.yaml"))

    assert runtime.env.control_dt == 0.05
    assert runtime.env.execution.safety_qp.solver.max_iterations == 8
    assert runtime.env.observation.predictive_risk.horizon == 1.0
    assert runtime.robot.capsules[0].name == "shoulder"
    assert runtime.device_selection.memory_weight == 0.7
    assert runtime.sac.replay_size == 300_000
    assert runtime.train.progress_interval == 100
    assert runtime.eval.episodes == 20
    assert runtime.smoke.replay_capacity == 64
    with pytest.raises(FrozenInstanceError):
        runtime.env.control_dt = 0.1  # type: ignore[misc]


def test_runtime_parameter_missing_from_yaml_fails_during_validation() -> None:
    config = copy.deepcopy(load_config("configs/default.yaml"))
    del config["env"]["execution"]["safety_qp"]["gain"]

    with pytest.raises(KeyError, match="env.execution.safety_qp.gain"):
        validate_config(config)


def test_unlisted_runtime_parameter_missing_from_yaml_fails_during_validation() -> None:
    config = copy.deepcopy(load_config("configs/default.yaml"))
    del config["sac"]["replay_size"]

    with pytest.raises(KeyError, match="replay_size"):
        validate_config(config)
