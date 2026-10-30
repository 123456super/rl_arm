from types import SimpleNamespace

import pytest

from rl_risk_sac.utils.training_logs import collect_sac_diagnostics


@pytest.mark.parametrize("initial_steps", [0, 1464913, 1516045])
@pytest.mark.parametrize("actor_interval", [2, 3])
def test_diagnostics_follow_actual_actor_schedule_after_import_or_resume(initial_steps, actor_interval):
    agent = SimpleNamespace(update_steps=initial_steps, actor_update_interval=actor_interval)
    collected = []
    for run_update in range(1, 101):
        if collect_sac_diagnostics(agent, run_update):
            collected.append(agent.update_steps + 1)
        agent.update_steps += 1
    assert len(collected) >= 3
    assert all(step % actor_interval == 0 for step in collected[1:])
    assert collected[0] == initial_steps + 1


def test_imported_odd_sac_counter_reports_fresh_actor_metrics():
    import torch
    from rl_risk_sac.algorithms.dual_sac import ReachingSACAgent
    from rl_risk_sac.algorithms.replay_buffer import Batch
    from rl_risk_sac.tasks.reaching_handoff import recovery_config
    from rl_risk_sac.utils.config import load_config
    config = recovery_config(load_config("configs/experiments/paper_local_avoidance_geometry.yaml"))
    config["device"] = "cpu"
    config["sac"].update(hidden_dims=[16, 16], actor_update_interval=2)
    agent = ReachingSACAgent(241, 6, config)
    agent.update_steps = 1464913  # The real imported checkpoint's odd offset.
    batch = Batch(torch.randn(8, 241), torch.zeros(8, 6), torch.randn(8, 1),
                  torch.zeros(8, 1), torch.randn(8, 241), torch.zeros(8, 1))
    reports = []
    for run_update in range(1, 25):
        collect = collect_sac_diagnostics(agent, run_update)
        metrics = agent.update(batch, reference_observations=batch, collect_diagnostics=collect)
        if collect:
            reports.append(metrics)
    assert len(reports) >= 3
    assert all(row["actor_updated"] == 1 for row in reports)
    assert len({row["actor_loss"] for row in reports}) > 1
