import numpy as np
import pytest
import torch

from scripts.evaluate_model_plasticity import (
    churn_metrics,
    effective_rank,
    normalized_activation_scores,
    policy_update_direction,
    q_value_churn,
    relative_parameter_update,
    twin_q_disagreement,
)


def test_normalized_activation_scores_and_dormant_units():
    features = np.asarray([[0.0, 1.0, 2.0], [0.0, -1.0, -2.0]])
    scores = normalized_activation_scores(features)
    assert scores.mean() == pytest.approx(1.0)
    assert scores[0] == 0.0
    assert scores[2] == pytest.approx(2.0)


def test_churn_metrics_are_split_by_orientation_group():
    previous = np.zeros((3, 2), dtype=np.float32)
    current = np.asarray([[.01, 0.0], [.2, 0.0], [.3, .4]], dtype=np.float32)
    groups = np.asarray(["o0_o4", "o5_o6", "o7_o9"])
    metrics = churn_metrics(previous, current, groups, (.1, .2))
    assert metrics["all"]["mean_l2"] == pytest.approx((.01 + .2 + .5) / 3)
    assert metrics["o0_o4"]["fractions"]["0.1"] == 0.0
    assert metrics["o7_o9"]["mean_l2"] == pytest.approx(.5)


def test_parameter_update_ratio_and_effective_rank():
    previous = {"weight": torch.tensor([3.0, 4.0])}
    current = {"weight": torch.tensor([3.0, 5.0])}
    assert relative_parameter_update(previous, current) == pytest.approx(.2)
    rank = effective_rank(np.eye(4, dtype=np.float32))
    assert rank["effective_rank"] == pytest.approx(4.0)
    assert rank["rank_99pct_energy"] == 4


def test_q_value_churn_and_twin_disagreement_are_grouped():
    groups = np.asarray(["o0_o4", "o5_o6", "o7_o9"])
    previous = {
        "q1": np.asarray([1.0, 2.0, 4.0]),
        "q2": np.asarray([1.0, 3.0, 2.0]),
        "min_q": np.asarray([1.0, 2.0, 2.0]),
    }
    current = {
        "q1": np.asarray([1.5, 1.0, 5.0]),
        "q2": np.asarray([1.0, 4.0, 4.0]),
        "min_q": np.asarray([1.0, 1.0, 4.0]),
    }
    churn = q_value_churn(previous, current, groups)
    assert churn["q1"]["all"]["mean"] == pytest.approx(5.0 / 6.0)
    assert churn["min_q"]["o7_o9"]["mean"] == pytest.approx(2.0)
    assert churn["min_q"]["o7_o9"][
        "relative_to_previous_mean_abs_q"
    ] == pytest.approx(1.0)

    disagreement = twin_q_disagreement(
        current["q1"], current["q2"], groups,
    )
    assert disagreement["all"]["mean"] == pytest.approx(1.5)
    assert disagreement["o5_o6"]["mean"] == pytest.approx(3.0)


def test_policy_update_direction_detects_reversal_by_group():
    older = np.zeros((3, 2), dtype=np.float32)
    previous = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    current = np.asarray([[2.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    groups = np.asarray(["o0_o4", "o5_o6", "o7_o9"])
    direction = policy_update_direction(older, previous, current, groups)
    assert direction["all"]["valid_samples"] == 2
    assert direction["all"]["mean_cosine"] == pytest.approx(0.0)
    assert direction["all"]["negative_fraction"] == pytest.approx(.5)
    assert direction["o0_o4"]["aligned_fraction"] == pytest.approx(1.0)
    assert direction["o5_o6"]["strong_reversal_fraction"] == pytest.approx(1.0)
    assert direction["o7_o9"]["valid_samples"] == 0
