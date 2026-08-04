import pytest

from scripts.audit_initial_feasibility import audit_initial_feasibility


def test_audit_initial_feasibility_records_finite_margins() -> None:
    result = audit_initial_feasibility(
        "configs/experiments/p3/b4_predictive_nonrobust_filter.yaml", seed=7101, episodes=3
    )

    assert result["episodes"] == 3
    assert len(result["episodes_detail"]) == 3
    assert result["initially_unsafe_count"] <= 3
    assert result["initial_h_min_m"]["min"] <= result["initial_h_min_m"]["max"]


def test_audit_initial_feasibility_requires_a_filter() -> None:
    with pytest.raises(ValueError, match="enabled safety filter"):
        audit_initial_feasibility("configs/experiments/p3/b3_link_predictive.yaml", seed=7101, episodes=1)
