from pathlib import Path

from scripts.select_checkpoint import actor_checkpoints, selection_key


def test_actor_checkpoints_are_sorted_by_step(tmp_path: Path) -> None:
    for name in ("actor_step_100000.pt", "actor_step_5000.pt", "actor.pt", "agent_state_step_10000.pt"):
        (tmp_path / name).touch()

    checkpoints = actor_checkpoints(tmp_path)

    assert [(step, path.name) for step, path in checkpoints] == [
        (5000, "actor_step_5000.pt"),
        (100000, "actor_step_100000.pt"),
    ]


def test_selection_key_uses_the_configured_metric_then_safety_tiebreakers() -> None:
    better_reward = {
        "mean_reward": 10.0,
        "success_rate": 0.5,
        "collision_rate": 0.8,
        "safety_violation_rate": 0.8,
        "mean_final_position_error": 0.8,
        "step": 10000.0,
    }
    safer_tie = {
        "mean_reward": 10.0,
        "success_rate": 0.4,
        "collision_rate": 0.1,
        "safety_violation_rate": 0.2,
        "mean_final_position_error": 0.3,
        "step": 5000.0,
    }

    assert selection_key(better_reward, "mean_reward") > selection_key(safer_tie, "mean_reward")
    assert selection_key(better_reward, "success_rate") < selection_key(safer_tie, "success_rate")
