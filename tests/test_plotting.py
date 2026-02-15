import matplotlib

matplotlib.use("Agg", force=True)

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxcmr.helpers import make_dataset
from jaxcmr.plotting import prepare_plot_inputs, segment_by_nan


def test_segment_by_nan_single_segment_when_no_nans():
    """Behavior: ``segment_by_nan`` returns one segment for NaN-free input.

    Given:
      - A vector with no NaN values.
    When:
      - ``segment_by_nan`` is called.
    Then:
      - Returns a single segment spanning the full vector.
    Why this matters:
      - Continuous curves should be drawn as one line.
    """
    # Arrange / Given
    vec = jnp.array([1.0, 2.0, 3.0])

    # Act / When
    segments = segment_by_nan(vec)

    # Assert / Then
    assert segments == [(0, 3)]


def test_segment_by_nan_splits_at_nan():
    """Behavior: ``segment_by_nan`` splits at NaN boundaries.

    Given:
      - A vector with a NaN in the middle.
    When:
      - ``segment_by_nan`` is called.
    Then:
      - Returns two segments, one before and one after the NaN.
    Why this matters:
      - Plot lines must break at undefined data points.
    """
    # Arrange / Given
    vec = jnp.array([1.0, jnp.nan, 3.0])

    # Act / When
    segments = segment_by_nan(vec)

    # Assert / Then
    assert len(segments) == 2
    assert segments[0] == (0, 1)
    assert segments[1] == (2, 3)


def test_prepare_plot_inputs_wraps_single_dataset():
    """Behavior: ``prepare_plot_inputs`` normalizes a single dataset to a list.

    Given:
      - A single dataset and mask (not wrapped in lists).
    When:
      - ``prepare_plot_inputs`` is called.
    Then:
      - Datasets and masks are returned as single-element lists.
    Why this matters:
      - Analysis plotters accept both single and multi-dataset inputs.
    """
    # Arrange / Given
    ds = make_dataset(recalls=jnp.array([[1, 2, 0]]))
    mask = jnp.array([True])

    # Act / When
    axis, datasets, masks, colors = prepare_plot_inputs(ds, mask, None, None)

    # Assert / Then
    assert isinstance(datasets, list)
    assert len(datasets) == 1
    assert isinstance(masks, list)
    assert len(masks) == 1
    assert isinstance(colors, list)
    assert len(colors) > 0
    plt.close(axis.figure)  # type: ignore[arg-type]


