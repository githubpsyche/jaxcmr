from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from jax import numpy as jnp
from matplotlib.axes import Axes

from jaxcmr.analyses.cat_recall_rate import (
    cat_recall_rate,
    plot_cat_recall_rate,
    subject_cat_recall_rate,
)
from jaxcmr.analyses.cat_recall_rate import (
    test_cat_recall_rate_pair as run_test_cat_recall_rate_pair,
)
from jaxcmr.helpers import make_dataset


def _point_x_positions(axis: Axes) -> np.ndarray:
    positions = []
    for line in axis.lines:
        if line.get_marker() == "o":
            positions.extend(np.asarray(line.get_xdata(), dtype=float))
    return np.asarray(positions)


def test_cat_recall_rate_exact_values():
    """Behavior: scalar category recall rate matches hand-calculated values.

    Given:
      - Two trials with categories [1, 2, 1, 2].
      - Trial 1 recalls positions 1 and 2.
      - Trial 2 recalls position 3.
    When:
      - ``cat_recall_rate`` is called for category 1 and category 2.
    Then:
      - Category 1 has 2 recalls / 4 studied positions = 0.5.
      - Category 2 has 1 recall / 4 studied positions = 0.25.
    Why this matters:
      - Verifies the scalar aggregation, not only per-position counting.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 2, 0, 0], [3, 0, 0, 0]])
    dataset: Any = {
        **make_dataset(recalls),
        "condition": jnp.array([[1, 2, 1, 2], [1, 2, 1, 2]]),
    }

    # Act / When
    category_1 = cat_recall_rate(dataset, "condition", 1)
    category_2 = cat_recall_rate(dataset, "condition", 2)

    # Assert / Then
    assert float(category_1) == pytest.approx(0.5)
    assert float(category_2) == pytest.approx(0.25)


def test_cat_recall_rate_counts_recalled_position_once():
    """Behavior: repeated recall events do not inflate the rate.

    Given:
      - One category-1 item is recalled twice in the same trial.
    When:
      - ``cat_recall_rate`` is called.
    Then:
      - The recalled study position contributes one hit, not two.
    Why this matters:
      - Recall proportion is a studied-position statistic, not an event count.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 1, 0, 0]])
    dataset: Any = {
        **make_dataset(recalls),
        "condition": jnp.array([[1, 1, 1, 1]]),
    }

    # Act / When
    result = cat_recall_rate(dataset, "condition", 1)

    # Assert / Then
    assert float(result) == pytest.approx(0.25)


def test_cat_recall_rate_nan_when_category_absent():
    """Behavior: absent categories return NaN.

    Given:
      - A dataset with only category-1 items.
    When:
      - ``cat_recall_rate`` is called for category 2.
    Then:
      - Result is NaN because the denominator is zero.
    Why this matters:
      - Pure-list cases need missing category cells rather than fake zeros.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 2, 0, 0]])
    dataset: Any = {
        **make_dataset(recalls),
        "condition": jnp.array([[1, 1, 1, 1]]),
    }

    # Act / When
    result = cat_recall_rate(dataset, "condition", 2)

    # Assert / Then
    assert jnp.isnan(result)


def test_cat_recall_rate_valid_field_excludes_padding():
    """Behavior: valid_field excludes padded study positions.

    Given:
      - Four category-1 positions, but only the first two are valid.
      - Position 1 is recalled.
    When:
      - ``cat_recall_rate`` is called with ``valid_field='pres_itemnos'``.
    Then:
      - Rate is 1 / 2 = 0.5 instead of 1 / 4 = 0.25.
    Why this matters:
      - Variable-length and padded HDF5 inputs need generic validity handling.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 0, 0, 0]])
    dataset: Any = {
        **make_dataset(recalls),
        "condition": jnp.array([[1, 1, 1, 1]]),
        "pres_itemnos": jnp.array([[1, 2, 0, 0]]),
    }

    # Act / When
    result = cat_recall_rate(dataset, "condition", 1, valid_field="pres_itemnos")

    # Assert / Then
    assert float(result) == pytest.approx(0.5)


def test_subject_cat_recall_rate_shape_and_values():
    """Behavior: subject-level rates return subjects by categories.

    Given:
      - Two subjects, each with two trials.
      - Category-1 and category-2 recall rates differ by subject.
    When:
      - ``subject_cat_recall_rate`` summarizes both categories.
    Then:
      - The result has shape (2, 2) with hand-calculated subject means.
    Why this matters:
      - Plotting and statistics depend on stable subject-level matrices.
    """
    # Arrange / Given
    recalls = jnp.array([
        [1, 2, 0, 0],
        [1, 0, 0, 0],
        [3, 0, 0, 0],
        [4, 0, 0, 0],
    ])
    dataset: Any = {
        **make_dataset(recalls, subject=jnp.array([1, 1, 2, 2])),
        "condition": jnp.array([
            [1, 2, 1, 2],
            [1, 2, 1, 2],
            [1, 2, 1, 2],
            [1, 2, 1, 2],
        ]),
    }
    trial_mask = jnp.array([True, True, True, True])

    # Act / When
    result = subject_cat_recall_rate(dataset, trial_mask, "condition", [1, 2])

    # Assert / Then
    expected = np.array([
        [0.5, 0.25],
        [0.25, 0.25],
    ])
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, expected)


def test_cat_recall_rate_pair_test_values():
    """Behavior: paired helper returns valid sample size and mean difference.

    Given:
      - Paired subject-level rates with one missing pair.
    When:
      - ``test_cat_recall_rate_pair`` is called.
    Then:
      - The missing pair is excluded and the mean difference is hand-calculated.
    Why this matters:
      - Notebook contrasts need clear paired handling with NaN cells.
    """
    # Arrange / Given
    left = np.array([0.8, 0.7, np.nan, 0.9])
    right = np.array([0.6, 0.5, 0.4, 0.8])

    # Act / When
    result = run_test_cat_recall_rate_pair(left, right, "left", "right")

    # Assert / Then
    assert result.n == 3
    assert result.mean_diff == pytest.approx((0.2 + 0.2 + 0.1) / 3)
    assert np.isfinite(result.t_stat)
    assert np.isfinite(result.t_pval)


def test_plot_cat_recall_rate_returns_axes():
    """Behavior: plot function returns a Matplotlib Axes.

    Given:
      - A small two-subject dataset with two category values.
    When:
      - ``plot_cat_recall_rate`` is called.
    Then:
      - An Axes is returned and points are drawn.
    Why this matters:
      - Confirms the plotting wrapper composes subject-level values.
    """
    # Arrange / Given
    recalls = jnp.array([
        [1, 2, 0, 0],
        [3, 4, 0, 0],
    ])
    dataset: Any = {
        **make_dataset(recalls, subject=jnp.array([1, 2])),
        "condition": jnp.array([[1, 2, 1, 2], [1, 2, 1, 2]]),
    }
    trial_mask = jnp.array([True, True])

    # Act / When
    axis = plot_cat_recall_rate(
        dataset,
        trial_mask,
        "condition",
        [1, 2],
        labels=["All"],
        category_labels=["A", "B"],
        contrast_name="Category",
    )

    # Assert / Then
    assert isinstance(axis, Axes)
    assert len(_point_x_positions(axis)) == 2
    plt.close(axis.figure)


def test_plot_cat_recall_rate_centers_present_category_points():
    """Behavior: absent category cells do not leave empty point slots.

    Given:
      - One mixed trial mask with category-1 and category-2 items.
      - One pure trial mask with only category-1 items.
    When:
      - ``plot_cat_recall_rate`` plots both masks with both requested categories.
    Then:
      - The pure group has one centered point, not one offset point plus a gap.
    Why this matters:
      - Pure-list categories are undefined missing cells, not zero-height bars.
    """
    # Arrange / Given
    recalls = jnp.array([
        [1, 2, 0, 0],
        [1, 2, 0, 0],
    ])
    dataset: Any = {
        **make_dataset(recalls, subject=jnp.array([1, 2])),
        "condition": jnp.array([
            [1, 2, 1, 2],
            [1, 1, 1, 1],
        ]),
    }
    mixed_mask = jnp.array([True, False])
    pure_mask = jnp.array([False, True])

    # Act / When
    axis = plot_cat_recall_rate(
        [dataset, dataset],
        [mixed_mask, pure_mask],
        "condition",
        [1, 2],
        labels=["Mixed", "Pure"],
        category_labels=["A", "B"],
    )

    # Assert / Then
    point_positions = _point_x_positions(axis)
    assert len(point_positions) == 3
    np.testing.assert_allclose(point_positions, [-0.175, 0.175, 1.0])
    plt.close(axis.figure)
