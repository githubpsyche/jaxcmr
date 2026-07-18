from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import jax.numpy as jnp
import pytest
from matplotlib.axes import Axes

from jaxcmr.analyses.distcrp import (
    compute_distance_bins_percentiles,
    dist_crp,
    plot_dist_crp,
)
from jaxcmr.helpers import make_dataset


def test_bin_edges_exact_percentile_values():
    """Behavior: Interior edges match upper-triangle percentiles.

    Given:
      - A 4x4 positional distance matrix with upper-triangle values
        [1, 1, 1, 2, 2, 3].
      - Percentile at [50].
    When:
      - ``compute_distance_bins_percentiles`` is called.
    Then:
      - Interior edge = median of [1,1,1,2,2,3] = 1.5.
      - Two bin centers straddle that edge.
    Why this matters:
      - Verifies that bin edges are derived from the exact percentile
        distribution of the upper triangle, not from all matrix entries.
    """
    # Arrange / Given
    positions = jnp.arange(4, dtype=float)
    distance_matrix = jnp.abs(positions[:, None] - positions[None, :])

    # Act / When
    edges, centers = compute_distance_bins_percentiles(
        distance_matrix, jnp.array([50.0])
    )

    # Assert / Then
    upper = distance_matrix[jnp.triu_indices(4, k=1)]
    expected_edge = jnp.percentile(upper, 50.0)
    assert edges.shape == (1,)
    assert jnp.isclose(edges[0], expected_edge)
    assert centers.shape == (2,)
    assert centers[0] < edges[0] < centers[1]


def test_bin_edges_exact_values_at_tertiles():
    """Behavior: Tertile edges produce exact values for known distribution.

    Given:
      - A 4x4 positional distance matrix.
      - Percentiles at [33, 66].
    When:
      - ``compute_distance_bins_percentiles`` is called.
    Then:
      - Edges = [1.0, 2.0] (33rd and 66th percentiles of [1,1,1,2,2,3]).
      - 3 bin centers correspond to low, mid, and high distance bins.
    Why this matters:
      - Validates multi-edge binning with exact percentile cutpoints.
    """
    # Arrange / Given
    positions = jnp.arange(4, dtype=float)
    distance_matrix = jnp.abs(positions[:, None] - positions[None, :])

    # Act / When
    edges, centers = compute_distance_bins_percentiles(
        distance_matrix, jnp.array([33.0, 66.0])
    )

    # Assert / Then
    assert edges.shape == (2,)
    assert jnp.isclose(edges[0], 1.0)
    assert jnp.isclose(edges[1], 2.0)
    assert centers.shape == (3,)


def test_bin_edges_percentiles_ignore_nan_values():
    """Behavior: Percentile binning ignores missing pairwise distances.

    Given:
      - A sparse distance matrix with NaN values for missing item pairs.
      - Percentile at [50].
    When:
      - ``compute_distance_bins_percentiles`` is called.
    Then:
      - The edge is the median of the finite upper-triangle values.
    Why this matters:
      - Precomputed distance matrices may be sparse when only some item pairs
        have rated distances.
    """
    # Arrange / Given
    distance_matrix = jnp.array([
        [0., 1., jnp.nan, 3.],
        [1., 0., 2., jnp.nan],
        [jnp.nan, 2., 0., 4.],
        [3., jnp.nan, 4., 0.],
    ])

    # Act / When
    edges, centers = compute_distance_bins_percentiles(
        distance_matrix, jnp.array([50.0])
    )

    # Assert / Then
    assert edges.shape == (1,)
    assert jnp.isclose(edges[0], 2.5)
    assert centers.shape == (2,)


def test_dist_crp_exact_values_with_two_bins():
    """Behavior: CRP per distance bin matches hand-traced counts.

    Given:
      - 3-item distance matrix: d(1,2)=1, d(1,3)=2, d(2,3)=1.
      - Single bin edge at 1.5 splits into near (<=1.5) and far (>1.5).
      - Trial 1 recalls [1,2]: transition 1->2, dist=1 (near bin).
      - Trial 2 recalls [1,3]: transition 1->3, dist=2 (far bin).
      - Each trial has 2 available items: one near, one far.
    When:
      - ``dist_crp`` is called.
    Then:
      - Near bin: actual 1, available 2 → CRP = 0.5.
      - Far bin:  actual 1, available 2 → CRP = 0.5.
    Why this matters:
      - Verifies exact distance-binned CRP from aggregated per-trial
        actual / available transition counts.
    """
    # Arrange / Given
    dm = jnp.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
    bin_edges = jnp.array([1.5])
    recalls = jnp.array([[1, 2, 0], [1, 3, 0]], dtype=jnp.int32)
    dataset: Any = {
        **make_dataset(recalls),
        "pres_itemids": jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32),
    }

    # Act / When
    result = dist_crp(dataset, dm, bin_edges)

    # Assert / Then
    expected = jnp.array([0.5, 0.5])
    assert jnp.allclose(result, expected)


def test_dist_crp_favors_near_bin_for_near_transitions():
    """Behavior: Exclusively near transitions yield CRP 1.0 in near bin.

    Given:
      - 3-item distance matrix: d(1,2)=1, d(1,3)=2, d(2,3)=1.
      - Both trials recall [1,2]: both transitions at dist=1 (near).
    When:
      - ``dist_crp`` is called.
    Then:
      - Near bin: actual 2, available 2 → CRP = 1.0.
      - Far bin:  actual 0, available 2 → CRP = 0.0.
    Why this matters:
      - Confirms that when all transitions target near items, the near
        bin CRP is 1.0 and the far bin CRP is 0.0.
    """
    # Arrange / Given
    dm = jnp.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
    bin_edges = jnp.array([1.5])
    recalls = jnp.array([[1, 2, 0], [1, 2, 0]], dtype=jnp.int32)
    dataset: Any = {
        **make_dataset(recalls),
        "pres_itemids": jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32),
    }

    # Act / When
    result = dist_crp(dataset, dm, bin_edges)

    # Assert / Then
    assert jnp.isclose(result[0], 1.0)  # near bin: 2/2
    assert jnp.isclose(result[1], 0.0)  # far bin: 0/2


def test_plot_dist_crp_accepts_distance_matrix():
    """Behavior: plotting accepts a precomputed distance matrix.

    Given:
      - A small dataset with item IDs already aligned to a distance matrix.
      - Explicit bin edges and centers.
    When:
      - ``plot_dist_crp`` is called with ``distance_matrix``.
    Then:
      - A Matplotlib Axes is returned.
    Why this matters:
      - Precomputed pairwise distances should not need to be converted into
        feature vectors before plotting.
    """
    # Arrange / Given
    dm = jnp.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
    recalls = jnp.array([[1, 2, 0], [1, 3, 0]], dtype=jnp.int32)
    dataset: Any = {
        **make_dataset(recalls),
        "pres_itemids": jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32),
    }
    trial_mask = jnp.array([True, True])

    # Act / When
    axis = plot_dist_crp(
        dataset,
        trial_mask,
        distance_matrix=dm,
        bin_edges=jnp.array([1.5]),
        bin_centers=jnp.array([1., 2.]),
    )

    # Assert / Then
    assert isinstance(axis, Axes)
    plt.close(axis.figure)


def test_plot_dist_crp_defaults_to_percentile_bins():
    """Behavior: plotting defaults to percentile distance bins.

    Given:
      - A small dataset and distance matrix.
    When:
      - ``plot_dist_crp`` is called without specifying ``bin_edges``.
    Then:
      - A Matplotlib Axes is returned.
    Why this matters:
      - The default plotting behavior should use the comparable percentile
        binning rule.
    """
    # Arrange / Given
    positions = jnp.arange(6, dtype=float)
    dm = jnp.abs(positions[:, None] - positions[None, :])
    recalls = jnp.array([
        [1, 2, 3, 4, 5, 6],
        [6, 5, 4, 3, 2, 1],
        [1, 3, 5, 2, 4, 6],
    ], dtype=jnp.int32)
    dataset: Any = {
        **make_dataset(recalls, subject=jnp.array([1, 2, 3])),
        "pres_itemids": jnp.tile(jnp.arange(1, 7)[None, :], (3, 1)),
    }
    trial_mask = jnp.array([True, True, True])

    # Act / When
    axis = plot_dist_crp(dataset, trial_mask, distance_matrix=dm)

    # Assert / Then
    assert isinstance(axis, Axes)
    plt.close(axis.figure)


def test_plot_dist_crp_accepts_min_count_bin_string():
    """Behavior: plotting accepts the min_count binning option.

    Given:
      - A small dataset and distance matrix.
    When:
      - ``plot_dist_crp`` is called with ``bin_edges="min_count"``.
    Then:
      - A Matplotlib Axes is returned.
    Why this matters:
      - The previous adaptive binning behavior remains selectable.
    """
    # Arrange / Given
    positions = jnp.arange(4, dtype=float)
    dm = jnp.abs(positions[:, None] - positions[None, :])
    recalls = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32)
    dataset: Any = {
        **make_dataset(recalls, subject=jnp.array([1, 2])),
        "pres_itemids": jnp.tile(jnp.arange(1, 5)[None, :], (2, 1)),
    }
    trial_mask = jnp.array([True, True])

    # Act / When
    axis = plot_dist_crp(
        dataset,
        trial_mask,
        distance_matrix=dm,
        bin_edges="min_count",
        min_transitions_per_subject=1,
    )

    # Assert / Then
    assert isinstance(axis, Axes)
    plt.close(axis.figure)


def test_plot_dist_crp_rejects_unknown_bin_string():
    """Behavior: plotting rejects unknown named binning rules.

    Given:
      - A small dataset and distance matrix.
    When:
      - ``plot_dist_crp`` is called with an unknown string for ``bin_edges``.
    Then:
      - A ValueError is raised.
    Why this matters:
      - One-word binning rules should be explicit and bounded.
    """
    # Arrange / Given
    dm = jnp.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
    recalls = jnp.array([[1, 2, 0], [1, 3, 0]], dtype=jnp.int32)
    dataset: Any = {
        **make_dataset(recalls),
        "pres_itemids": jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32),
    }
    trial_mask = jnp.array([True, True])

    # Act / When / Assert / Then
    with pytest.raises(ValueError, match="bin_edges"):
        plot_dist_crp(
            dataset,
            trial_mask,
            distance_matrix=dm,
            bin_edges="unknown",
        )
    plt.close("all")


def test_plot_dist_crp_requires_one_distance_input():
    """Behavior: plotting requires exactly one distance input source.

    Given:
      - A small dataset.
    When:
      - Neither ``features`` nor ``distance_matrix`` is supplied.
      - Both ``features`` and ``distance_matrix`` are supplied.
    Then:
      - Both calls raise ValueError.
    Why this matters:
      - The plotting wrapper should not silently choose between incompatible
        distance sources.
    """
    # Arrange / Given
    dm = jnp.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
    recalls = jnp.array([[1, 2, 0], [1, 3, 0]], dtype=jnp.int32)
    dataset: Any = {
        **make_dataset(recalls),
        "pres_itemids": jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32),
    }
    trial_mask = jnp.array([True, True])

    # Act / When / Assert / Then
    with pytest.raises(ValueError, match="Exactly one"):
        plot_dist_crp(
            dataset,
            trial_mask,
            bin_edges=jnp.array([1.5]),
            bin_centers=jnp.array([1., 2.]),
        )

    with pytest.raises(ValueError, match="Exactly one"):
        plot_dist_crp(
            dataset,
            trial_mask,
            features=jnp.eye(3),
            distance_matrix=dm,
            bin_edges=jnp.array([1.5]),
            bin_centers=jnp.array([1., 2.]),
        )
    plt.close("all")
