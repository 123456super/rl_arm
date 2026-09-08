from __future__ import annotations

import numpy as np

from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config


def test_fixed_method_runs_with_butterworth_quintic_rtb() -> None:
    config = load_config("configs/experiments/random_crossing_link_fixed_penalty1_rtb.yaml")
    config["device"] = "cpu"
    env = UR5DynamicObstacleEnv(config, method="link_fixed")
    try:
        env.reset(seed=17)
        action = np.ones(env.action_space.shape, dtype=np.float32)
        _, _, _, _, _, info = env.step(action)
        assert info["fixed_smoothing_mode"] == "butterworth_quintic"
        assert info["observation_schema"] == "link_risk_v1"
        assert info["policy_rate_limited"]
        assert np.max(np.abs(info["qdot_policy_limited"])) <= 0.1 + 1e-6
        assert np.isfinite(info["physics_rms_acceleration"])
        assert np.isfinite(info["physics_rms_jerk"])
        assert np.max(np.abs(info["qdot_cmd"])) <= config["env"]["action_scale"]
    finally:
        env.close()
