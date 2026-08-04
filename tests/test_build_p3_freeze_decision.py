from scripts.build_p3_freeze_decision import per_seed_means


def test_per_seed_means_averages_available_formal_metrics() -> None:
    row = {
        "success": "1",
        "collision_capsule_overlap": "0",
        "collision_pybullet_contact": "0",
        "safety_violation_rate": "0.2",
        "safety_filter_intervention_rate": "0.3",
        "safety_filter_safe_stop_rate": "0.4",
        "safety_filter_infeasible_rate": "0.4",
        "safety_filter_compute_budget_stop_rate": "0",
        "reward": "2",
        "final_position_error": "0.1",
        "mean_safety_filter_solve_time_s": "0.005",
        "max_safety_filter_solve_time_s": "0.01",
    }
    result = per_seed_means([row, {**row, "success": "0", "reward": "4"}])

    assert result["success"] == 0.5
    assert result["reward"] == 3.0
