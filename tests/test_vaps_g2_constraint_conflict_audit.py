from __future__ import annotations

import numpy as np

from scripts.audit_vaps_g2_constraint_conflicts import (
    ConstraintTerm,
    find_minimal_category_conflicts,
    irreducible_constraint_core,
)


def test_minimal_category_conflict_and_irreducible_core() -> None:
    terms = (
        ConstraintTerm("predictive_link_0", "predictive_barrier", np.asarray([1.0]), 1.0),
        ConstraintTerm("workspace_x_lower", "workspace", np.asarray([1.0]), -2.0),
        ConstraintTerm("joint_0_velocity_upper", "joint_velocity", np.asarray([-1.0]), 0.0),
    )

    categories, probes = find_minimal_category_conflicts(terms, dimension=1, tolerance=1.0e-6)

    assert categories == ("joint_velocity", "predictive_barrier", "workspace")
    classifications = {tuple(probe["categories"]): probe["classification"] for probe in probes}
    assert classifications[("joint_velocity", "predictive_barrier")] == "primal_infeasible"
    assert classifications[("joint_velocity", "workspace")] == "strict_feasible"
    core = irreducible_constraint_core(
        (terms[0], terms[2]), dimension=1, tolerance=1.0e-6
    )
    assert [term.label for term in core] == ["predictive_link_0", "joint_0_velocity_upper"]
