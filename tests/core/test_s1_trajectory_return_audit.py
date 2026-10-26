"""Accounting and comparison checks; no simulator or policy updates."""
import pytest

from scripts.calibration.audit_s1_trajectory_returns import (
    NEGATIVE, POSITIVE, paired_comparisons, rescore_self_weight, summarize_trace,
)


def row(reward, *, progress=0., success=0., timeout=0.):
    result = dict.fromkeys((*POSITIVE, *NEGATIVE), 0.)
    result.update(reward=reward, keypoint_progress_reward=progress,
                  success_reward=success, timeout_penalty=timeout)
    return result


def test_return_uses_first_reward_at_gamma_zero_power_and_terminal_once():
    result = summarize_trace([row(2., progress=2.), row(48., success=50., timeout=2.)], .5)
    assert result["discounted_return"] == 26.
    assert result["undiscounted_return"] == 50.
    assert result["discounted_components"]["success_reward"] == 25.
    assert result["discounted_components"]["timeout_penalty"] == 1.
    assert result["decomposition_residual"] == 0.


def test_unaccounted_reward_is_rejected():
    with pytest.raises(ValueError, match="decomposition"):
        summarize_trace([row(1.)], .99)


def test_no_success_claim_or_comparison_for_failed_witness():
    assert paired_comparisons([dict(episode=1, mode="witness", cause="timeout")]) == []


def test_comparison_reports_inverted_order_without_hiding_low_clearance():
    common = dict(episode=1, seed=31002, band="blocking", sampled_mesh_margin_passed=False)
    rows = [dict(common, mode="witness", cause="success", discounted_return=1.),
            dict(common, mode="latest1250k", cause="timeout", discounted_return=2.)]
    result = paired_comparisons(rows)[0]
    assert not result["successful_route_preferred"]
    assert result["success_minus_failure"] == -1.
    assert not result["witness_sampled_mesh_margin_passed"]


def test_self_weight_relabel_is_linear_and_does_not_mutate_source():
    source = dict(lambda_self=1., discounted_return=2., undiscounted_return=3.,
                  discounted_components={"self_safety_penalty": .5},
                  trace=[{"self_safety_penalty": 1.}])
    result = rescore_self_weight([source], .2)[0]
    assert result["discounted_return"] == pytest.approx(2.4)
    assert result["undiscounted_return"] == pytest.approx(3.8)
    assert result["discounted_components"]["self_safety_penalty"] == pytest.approx(.1)
    assert source["discounted_components"]["self_safety_penalty"] == .5
    assert source["discounted_return"] == 2.


def test_comparisons_can_use_a_different_executed_reference():
    common = dict(episode=1, seed=31002, band="blocking", sampled_mesh_margin_passed=True)
    rows = [dict(common, mode="witness_gain5", cause="success", discounted_return=4.),
            dict(common, mode="latest1250k", cause="timeout", discounted_return=2.)]
    result = paired_comparisons(rows, "witness_gain5")
    assert len(result) == 1
    assert result[0]["success_mode"] == "witness_gain5"
    assert result[0]["success_minus_failure"] == 2.
