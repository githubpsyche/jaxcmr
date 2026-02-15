from typing import Any

import jax.numpy as jnp

from jaxcmr.analyses.distcrp import (
    compute_distance_bins_percentiles,
    dist_crp,
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
