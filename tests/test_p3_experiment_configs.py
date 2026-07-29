from pathlib import Path

import numpy as np

from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config


P3_CONFIGS = {
    "b1_ee_current": ("ee_fixed", "current", False),
    "b2_link_current": ("link_fixed", "current", False),
    "b3_link_predictive": ("link_fixed", "predictive", False),
    "b4_predictive_nonrobust_filter": ("link_fixed", "predictive", True),
    "b5_robust_predictive_filter": ("link_fixed", "robust_predictive", True),
}


P3_100K_CONFIGS = {
    "b1_ee_current": ("ee_fixed", "current", False),
    "b2_link_current": ("link_fixed", "current", False),
    "b3_link_predictive": ("link_fixed", "predictive", False),
    "b4_predictive_nonrobust_filter": ("link_fixed", "predictive", True),
    "b5_robust_predictive_filter": ("link_fixed", "robust_predictive", True),
}


def test_p3_configs_have_distinct_outputs_and_expected_factor_settings() -> None:
    output_dirs = set()
    for name, (method, representation, filter_enabled) in P3_CONFIGS.items():
        config = load_config(Path("configs/experiments/p3") / f"{name}.yaml")
        assert config["train"]["method"] == method
        assert config["eval"]["method"] == method
        assert config["smoke"]["method"] == method
        assert config["risk"]["representation"] == representation
        assert config["env"]["safety_filter"]["enabled"] is filter_enabled
        assert config["train"]["total_steps"] == 10000
        output_dirs.add(config["train"]["output_dir"])
    assert len(output_dirs) == len(P3_CONFIGS)


def test_predictive_risk_representation_runs_without_a_safety_filter() -> None:
    config = load_config("configs/experiments/p3/b3_link_predictive.yaml")
    env = UR5DynamicObstacleEnv(config, method="link_fixed")
    try:
        observation, info = env.reset(seed=4101)
        assert np.isfinite(observation).all()
        assert info["policy_risk_representation"] == "predictive"
        assert np.isfinite(info["policy_risk_global"])
        _, _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert "safety_filter_status" not in info
        assert info["policy_risk_representation"] == "predictive"
    finally:
        env.close()


def test_p3_100k_configs_preserve_factors_and_use_an_independent_validation_seed() -> None:
    for name, (method, representation, filter_enabled) in P3_100K_CONFIGS.items():
        short_config = load_config(Path("configs/experiments/p3") / f"{name}.yaml")
        config = load_config(Path("configs/experiments/p3_100k") / f"{name}.yaml")
        assert config["train"]["total_steps"] == 100000
        assert config["train"]["output_dir"].startswith("outputs/p3_dev_100k/")
        assert config["train"]["method"] == method
        assert config["risk"]["representation"] == representation
        assert config["env"]["safety_filter"]["enabled"] is filter_enabled
        assert config["env"]["safety_filter"] == short_config["env"]["safety_filter"]
        assert config["checkpoint_selection"] == {"seed": 5201, "episodes": 20, "metric": "mean_reward"}
        assert config["checkpoint_selection"]["seed"] != config["seed"]
        assert config["checkpoint_selection"]["seed"] != config["eval"]["seed"]


def test_p3_projection_diagnostics_only_change_projection_iterations() -> None:
    for name, base_name in {
        "b4_projection320": "b4_predictive_nonrobust_filter",
        "b5_projection320": "b5_robust_predictive_filter",
    }.items():
        base = load_config(Path("configs/experiments/p3_100k") / f"{base_name}.yaml")
        diagnostic = load_config(Path("configs/experiments/p3_diagnostics") / f"{name}.yaml")
        assert diagnostic["env"]["safety_filter"]["max_projection_iterations"] == 320
        assert diagnostic["env"]["safety_filter"]["fallback_projection_iterations"] == 640
        assert diagnostic["env"]["safety_filter"]["projection_failure_tolerance"] == base["env"]["safety_filter"]["projection_failure_tolerance"]
        assert diagnostic["risk"] == base["risk"]
        assert diagnostic["sac"] == base["sac"]


def test_osqp_diagnostic_configs_enable_only_the_qp_backend() -> None:
    for name in ("b4_osqp", "b5_osqp"):
        config = load_config(Path("configs/experiments/p3_diagnostics") / f"{name}.yaml")
        assert config["env"]["safety_filter"]["use_qp_solver"] is True
        assert config["env"]["safety_filter"]["fallback_projection_iterations"] == 640
