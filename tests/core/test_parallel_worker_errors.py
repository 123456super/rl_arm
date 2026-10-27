import pytest

from rl_risk_sac.envs.parallel_thesis_env import ParallelEnvWorkerError, ParallelThesisEnvPool
from rl_risk_sac.utils.config import load_config


def test_remote_error_preserves_traceback_drains_other_replies_and_closes():
    config = load_config("configs/base/thesis_common.yaml")
    config["thesis"]["obstacles"]["enabled"] = False
    pool = ParallelThesisEnvPool(config, [101, 102])
    try:
        with pytest.raises(ParallelEnvWorkerError, match="thesis-env-0.*reset") as caught:
            pool.reset_many({
                0: {"scene": "invalid", "xi": 1., "strict": True},
                1: {"scene": "none", "xi": 1., "strict": True},
            })
        assert "configure_episode" in caught.value.remote_traceback
        assert "ValueError" in str(caught.value)
        assert all(process.is_alive() for process in pool.processes)
        # Must be a state response, not worker 1's undrained reset response.
        assert pool.states([1])[1]["step_count"] == 0
        # A half-reset worker must not be serialized into a valid checkpoint.
        with pytest.raises(ParallelEnvWorkerError):
            pool.states()
    finally:
        pool.close()
    assert all(not process.is_alive() for process in pool.processes)


def test_sampling_exhaustion_reports_structured_diagnostics_from_worker():
    config = load_config("configs/base/thesis_common.yaml")
    from scripts.core.train_thesis_homotopy import apply_stage_profile
    apply_stage_profile(config, "s1")
    obstacle = config["thesis"]["obstacles"]
    obstacle["max_scene_attempts"] = 1
    obstacle["risk"].update(d_safe=10., d_cut=20.)
    obstacle["path_sampling"].update(
        probabilities=[0., 0., 0., 1.], max_goal_attempts=1, max_sampling_batches=2,
    )
    with ParallelThesisEnvPool(config, [103]) as pool:
        with pytest.raises(ParallelEnvWorkerError, match="StaticSceneSamplingError") as caught:
            pool.reset_many({0: {"scene": "none", "xi": 1., "strict": True}})
        d = caught.value.diagnostics
        assert d["path_band"] == "blocking" and not d["path_verified"]
        assert d["sampling_diagnostics"]["batches"] == 2
        assert d["sampling_diagnostics"]["start_rejections"] == 2
