from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import jax.numpy as jnp
import pytest
from matplotlib.axes import Axes

from jaxcmr.analyses.conditional_distcrp import (
    dist_crp as conditional_dist_crp,
    plot_dist_crp as plot_conditional_dist_crp,
)
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


def test_plot_conditional_dist_crp_accepts_distance_matrix():
    """Behavior: conditional plotting accepts a precomputed distance matrix.

    Given:
      - A small dataset with a transition mask.
      - Explicit bin edges and centers.
    When:
      - ``plot_dist_crp`` is called with ``distance_matrix``.
    Then:
      - A Matplotlib Axes is returned.
    Why this matters:
      - Conditional Distance-CRP should support the same distance sources as
        the standard Distance-CRP plotter.
    """
    # Arrange / Given
    dm = jnp.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
    recalls = jnp.array([[1, 2, 3], [1, 3, 2]], dtype=jnp.int32)
    pres_ids = jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32)
    dataset: Any = {**make_dataset(recalls), "pres_itemids": pres_ids}
    trial_mask = jnp.array([True, True])
    should_tabulate = jnp.ones_like(recalls, dtype=bool)

    # Act / When
    axis = plot_conditional_dist_crp(
        dataset,
        trial_mask,
        should_tabulate,
        distance_matrix=dm,
        bin_edges=jnp.array([1.5]),
        bin_centers=jnp.array([1., 2.]),
    )

    # Assert / Then
    assert isinstance(axis, Axes)
    plt.close(axis.figure)


def test_plot_conditional_dist_crp_defaults_to_percentile_bins():
    """Behavior: conditional plotting defaults to percentile bins.

    Given:
      - A small dataset and all-True transition mask.
    When:
      - ``plot_dist_crp`` is called without specifying ``bin_edges``.
    Then:
      - A Matplotlib Axes is returned.
    Why this matters:
      - Conditional and standard Distance-CRP plotters should share the same
        default binning rule.
    """
    # Arrange / Given
    positions = jnp.arange(5, dtype=float)
    dm = jnp.abs(positions[:, None] - positions[None, :])
    recalls = jnp.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=jnp.int32)
    pres_ids = jnp.tile(jnp.arange(1, 6)[None, :], (2, 1))
    dataset: Any = {
        **make_dataset(recalls, subject=jnp.array([1, 2])),
        "pres_itemids": pres_ids,
    }
    trial_mask = jnp.array([True, True])
    should_tabulate = jnp.ones_like(recalls, dtype=bool)

    # Act / When
    axis = plot_conditional_dist_crp(
        dataset,
        trial_mask,
        should_tabulate,
        distance_matrix=dm,
    )

    # Assert / Then
    assert isinstance(axis, Axes)
    plt.close(axis.figure)


def test_plot_conditional_dist_crp_accepts_min_count_bin_string():
    """Behavior: conditional plotting accepts the min_count binning option.

    Given:
      - A small dataset and all-True transition mask.
    When:
      - ``plot_dist_crp`` is called with ``bin_edges="min_count"``.
    Then:
      - A Matplotlib Axes is returned.
    Why this matters:
      - The previous adaptive binning behavior remains selectable for the
        conditional plotter.
    """
    # Arrange / Given
    positions = jnp.arange(4, dtype=float)
    dm = jnp.abs(positions[:, None] - positions[None, :])
    recalls = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32)
    pres_ids = jnp.tile(jnp.arange(1, 5)[None, :], (2, 1))
    dataset: Any = {
        **make_dataset(recalls, subject=jnp.array([1, 2])),
        "pres_itemids": pres_ids,
    }
    trial_mask = jnp.array([True, True])
    should_tabulate = jnp.ones_like(recalls, dtype=bool)

    # Act / When
    axis = plot_conditional_dist_crp(
        dataset,
        trial_mask,
        should_tabulate,
        distance_matrix=dm,
        bin_edges="min_count",
        min_transitions_per_subject=1,
    )

    # Assert / Then
    assert isinstance(axis, Axes)
    plt.close(axis.figure)


def test_plot_conditional_dist_crp_requires_one_distance_input():
    """Behavior: conditional plotting requires exactly one distance source.

    Given:
      - A small dataset and transition mask.
    When:
      - Neither ``features`` nor ``distance_matrix`` is supplied.
      - Both ``features`` and ``distance_matrix`` are supplied.
    Then:
      - Both calls raise ValueError.
    Why this matters:
      - The conditional plotter should reject ambiguous distance inputs.
    """
    # Arrange / Given
    dm = jnp.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
    recalls = jnp.array([[1, 2, 3], [1, 3, 2]], dtype=jnp.int32)
    pres_ids = jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32)
    dataset: Any = {**make_dataset(recalls), "pres_itemids": pres_ids}
    trial_mask = jnp.array([True, True])
    should_tabulate = jnp.ones_like(recalls, dtype=bool)

    # Act / When / Assert / Then
    with pytest.raises(ValueError, match="Exactly one"):
        plot_conditional_dist_crp(
            dataset,
            trial_mask,
            should_tabulate,
            bin_edges=jnp.array([1.5]),
            bin_centers=jnp.array([1., 2.]),
        )

    with pytest.raises(ValueError, match="Exactly one"):
        plot_conditional_dist_crp(
            dataset,
            trial_mask,
            should_tabulate,
            features=jnp.eye(3),
            distance_matrix=dm,
            bin_edges=jnp.array([1.5]),
            bin_centers=jnp.array([1., 2.]),
        )
    plt.close("all")
