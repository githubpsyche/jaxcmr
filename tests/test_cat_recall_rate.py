from jaxcmr.analyses.statistics import compare_paired
from typing import Any
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from jax import numpy as jnp
from matplotlib.axes import Axes
from jaxcmr.analyses.cat_recall_rate import plot_cat_recall_rate, subject_cat_recall_rate

from jaxcmr.dataset import make_dataset


def _point_x_positions(axis: Axes) -> np.ndarray:
    positions = []
    for line in axis.lines:
        if line.get_marker() == "o":
            positions.extend(np.asarray(line.get_xdata(), dtype=float))
    return np.asarray(positions)


def test_subject_cat_recall_rate_shape_and_values():
    """subject-level rates return subjects by categories."""
    recalls = jnp.array([[1, 2, 0, 0], [1, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]])
    dataset: Any = {
        **make_dataset(recalls, subject=jnp.array([1, 1, 2, 2])),
        "condition": jnp.array([[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2]]),
    }
    trial_mask = jnp.array([True, True, True, True])
    result = subject_cat_recall_rate(dataset, trial_mask, "condition", [1, 2])
    expected = np.array([[0.5, 0.25], [0.25, 0.25]])
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, expected)


def test_cat_recall_rate_pair_test_values():
    """paired helper returns valid sample size and mean difference."""
    left = np.array([0.8, 0.7, np.nan, 0.9])
    right = np.array([0.6, 0.5, 0.4, 0.8])
    result = compare_paired(left, right, labels=["left vs right"]).iloc[0]
    assert result.n == 3
    assert result.mean_difference == pytest.approx((0.2 + 0.2 + 0.1) / 3)
    # Paired differences .2,.2,.1 have mean 1/6 and SEM 1/30: t=5.
    # With df=2, the two-sided p-value is 1 - t/sqrt(t**2 + 2).
    assert result.t_stat == pytest.approx(5)
    assert result.t_pvalue == pytest.approx(1 - 5 / np.sqrt(27))


def test_plot_cat_recall_rate_centers_present_category_points():
    """absent category cells do not leave empty point slots."""
    recalls = jnp.array([[1, 2, 0, 0], [1, 2, 0, 0]])
    dataset: Any = {
        **make_dataset(recalls, subject=jnp.array([1, 2])),
        "condition": jnp.array([[1, 2, 1, 2], [1, 1, 1, 1]]),
    }
    mixed_mask = jnp.array([True, False])
    pure_mask = jnp.array([False, True])
    axis = plot_cat_recall_rate(
        [dataset, dataset],
        [mixed_mask, pure_mask],
        "condition",
        [1, 2],
        labels=["Mixed", "Pure"],
        category_labels=["A", "B"],
    )
    point_positions = _point_x_positions(axis)
    assert len(point_positions) == 3
    np.testing.assert_allclose(point_positions, [-0.175, 0.175, 1.0])
    plt.close(axis.figure)
