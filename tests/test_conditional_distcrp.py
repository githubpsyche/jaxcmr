from typing import Any

import jax.numpy as jnp

from jaxcmr.analyses.conditional_distcrp import dist_crp as conditional_dist_crp
from jaxcmr.analyses.distcrp import dist_crp as unconditional_dist_crp
from jaxcmr.helpers import make_dataset


def test_matches_unconditional_when_all_transitions_tabulated():
    """Behavior: Conditional CRP equals unconditional when all included.

    Given:
      - Two 3-item trials with 2 transitions each, all-True should_tabulate.
      - Distance matrix: d(1,2)=1, d(1,3)=2, d(2,3)=1.
      - Single bin edge at 1.5 (near / far).
    When:
      - Both ``conditional_dist_crp`` and ``unconditional_dist_crp`` are called.
    Then:
      - Results are identical: [0.75, 0.5].
    Why this matters:
      - With no filtering, the conditional variant must reproduce the
        unconditional result exactly.
    """
    # Arrange / Given
    dm = jnp.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
    bin_edges = jnp.array([1.5])
    recalls = jnp.array([[1, 2, 3], [1, 3, 2]], dtype=jnp.int32)
    pres_ids = jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32)
    base = make_dataset(recalls)
    dataset_uncond: Any = {**base, "pres_itemids": pres_ids}
    dataset_cond: Any = {
        **base,
        "pres_itemids": pres_ids,
        "_should_tabulate": jnp.ones_like(recalls, dtype=bool),
    }

    # Act / When
    result_uncond = unconditional_dist_crp(dataset_uncond, dm, bin_edges)
    result_cond = conditional_dist_crp(dataset_cond, dm, bin_edges)

    # Assert / Then
    assert jnp.allclose(result_cond, result_uncond, equal_nan=True)
    assert jnp.allclose(result_cond, jnp.array([0.75, 0.5]))


def test_filtering_excludes_transitions_and_changes_crp():
    """Behavior: Skipping transitions changes CRP vs the unconditional.

    Given:
      - Trial 1: [1,2,3]. Transition 1->2 (dist=1, near) then 2->3 (dist=1, near).
      - Trial 2: [1,3,2]. Transition 1->3 (dist=2, far) then 3->2 (dist=1, near).
      - ``should_tabulate`` skips the 1st transition (index 1) in each trial,
        keeping only the 2nd transition (index 2).
    When:
      - ``conditional_dist_crp`` is called.
    Then:
      - Only 2->3 (near) and 3->2 (near) are counted: CRP near = 1.0.
      - No far transitions counted: CRP far = NaN.
    Why this matters:
      - Verifies that the should_tabulate mask genuinely excludes
        transitions from the CRP numerator and denominator while
        availability tracking still proceeds.
    """
    # Arrange / Given
    dm = jnp.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
    bin_edges = jnp.array([1.5])
    recalls = jnp.array([[1, 2, 3], [1, 3, 2]], dtype=jnp.int32)
    pres_ids = jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32)
    should_tab = jnp.array([[True, False, True], [True, False, True]], dtype=bool)
    dataset: Any = {
        **make_dataset(recalls),
        "pres_itemids": pres_ids,
        "_should_tabulate": should_tab,
    }

    # Act / When
    result = conditional_dist_crp(dataset, dm, bin_edges)

    # Assert / Then
    assert jnp.isclose(result[0], 1.0)  # near bin: 2 near transitions / 2 available
    assert jnp.isnan(result[1])          # far bin: no counted transitions


def test_returns_all_nan_when_no_transitions_tabulated():
    """Behavior: Return all NaN when every transition is excluded.

    Given:
      - A two-trial dataset with ``_should_tabulate`` set to all False.
    When:
      - ``conditional_dist_crp`` is called.
    Then:
      - Every element of the result is NaN (0 / 0 division).
    Why this matters:
      - When no transitions are counted, the denominator is zero for every
        bin, so NaN is the only valid output.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 3, 2, 0], [2, 1, 3, 0]], dtype=jnp.int32)
    positions = jnp.arange(4, dtype=float)
    dm = jnp.abs(positions[:, None] - positions[None, :]).astype(float)
    bin_edges = jnp.array([1.5])
    dataset: Any = {
        **make_dataset(recalls),
        "pres_itemids": jnp.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=jnp.int32),
        "_should_tabulate": jnp.zeros_like(recalls, dtype=bool),
    }

    # Act / When
    result = conditional_dist_crp(dataset, dm, bin_edges)

    # Assert / Then
    assert jnp.all(jnp.isnan(result))
