from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from scripts.audit_safe_stop import audit_trace_dir
from scripts.audit_collision_coverage import point_segment_distances


def test_audit_safe_stop_reports_margin_drift(tmp_path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    path = trace_dir / "episode_0000.csv"
    fields = ["step", "safety_filter_status", "predictive_h_min_m", "collision"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {"step": 3, "safety_filter_status": "safe_stop_infeasible", "predictive_h_min_m": "0.02", "collision": "0"},
                {"step": 4, "safety_filter_status": "safe_stop_infeasible", "predictive_h_min_m": "0.01", "collision": "0"},
                {"step": 5, "safety_filter_status": "safe_stop_infeasible", "predictive_h_min_m": "0.00", "collision": "1"},
            ]
        )

    rows = audit_trace_dir(trace_dir, 0.05)

    assert len(rows) == 1
    assert rows[0]["infeasible_steps"] == 3
    assert rows[0]["h_drift_mps"] == pytest.approx(-0.2)
    assert rows[0]["drift_class"] == "dynamic_drift"
    assert rows[0]["collision"] is True


def test_point_segment_distances_handles_degenerate_capsule() -> None:
    points = point_segment_distances(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.zeros(3),
        np.zeros(3),
    )
    np.testing.assert_allclose(points, [0.0, 1.0])
