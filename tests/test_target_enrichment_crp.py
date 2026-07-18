from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from jax import jit
import jax.numpy as jnp
from matplotlib.axes import Axes

from jaxcmr.analyses.target_enrichment_crp import (
    plot_target_enrichment_crp,
    tabulate_trial,
    target_enrichment_crp,
)
from jaxcmr.helpers import make_dataset


def test_tabulate_trial_counts_target_and_total_lags():
    """Behavior: exact target and total counts are tracked separately."""
    trial = jnp.array([2, 3, 0], dtype=jnp.int32)
    presentation = jnp.array([1, 2, 3], dtype=jnp.int32)
    conditions = jnp.array([0, 0, -1], dtype=jnp.int32)
    should_tabulate = jnp.ones_like(trial, dtype=bool)
    lag_range = presentation.size - 1

    actual_target, actual_total, avail_target, avail_total = tabulate_trial(
        trial,
        presentation,
        conditions,
        should_tabulate,
        target_value=-1,
        target_values=jnp.array([-1, 0], dtype=jnp.int32),
        size=3,
    )

    expected_actual_target = np.zeros(5, dtype=int)
    expected_actual_total = np.zeros(5, dtype=int)
    expected_avail_target = np.zeros(5, dtype=int)
    expected_avail_total = np.zeros(5, dtype=int)
    expected_actual_target[lag_range + 1] = 1
    expected_actual_total[lag_range + 1] = 1
    expected_avail_target[lag_range + 1] = 1
    expected_avail_total[lag_range - 1] = 1
    expected_avail_total[lag_range + 1] = 1

    np.testing.assert_array_equal(np.asarray(actual_target), expected_actual_target)
    np.testing.assert_array_equal(np.asarray(actual_total), expected_actual_total)
    np.testing.assert_array_equal(np.asarray(avail_target), expected_avail_target)
    np.testing.assert_array_equal(np.asarray(avail_total), expected_avail_total)


def test_target_enrichment_crp_exact_lag_enrichment():
    """Behavior: enrichment is observed target fraction minus availability."""
    recalls = jnp.array([[2, 3, 0], [2, 1, 0]], dtype=jnp.int32)
    pres = jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32)
    dataset: Any = {
        **make_dataset(recalls, pres),
        "condition": jnp.array([[0, 0, -1], [0, 0, 0]], dtype=jnp.int32),
    }
    lag_range = pres.shape[1] - 1

    result = target_enrichment_crp(
        dataset,
        target_field="condition",
        target_value=-1,
        target_values=[-1, 0],
        source_field="condition",
        source_values=[0],
        size=3,
    )

    assert result.shape == (1, 5)
    assert float(result[0, lag_range + 1]) == pytest.approx(0.5)
    assert float(result[0, lag_range - 1]) == pytest.approx(0.0)
    assert jnp.isnan(result[0, lag_range]).item()


def test_target_enrichment_crp_respects_should_tabulate():
    """Behavior: existing transition masks exclude events from enrichment."""
    recalls = jnp.array([[2, 3, 0], [2, 1, 0]], dtype=jnp.int32)
    pres = jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32)
    dataset: Any = {
        **make_dataset(recalls, pres),
        "_should_tabulate": jnp.zeros_like(recalls, dtype=bool),
        "condition": jnp.array([[0, 0, -1], [0, 0, 0]], dtype=jnp.int32),
    }

    result = target_enrichment_crp(
        dataset,
        target_field="condition",
        target_value=-1,
        target_values=[-1, 0],
        source_field="condition",
        source_values=[0],
        size=3,
    )

    assert jnp.all(jnp.isnan(result)).item()


def test_target_enrichment_crp_rejects_incomplete_source_arguments():
    """Behavior: source field and values must be supplied together."""
    dataset: Any = {
        **make_dataset(
            jnp.array([[1, 2, 0]], dtype=jnp.int32),
            jnp.array([[1, 2, 3]], dtype=jnp.int32),
        ),
        "condition": jnp.array([[0, -1, 0]], dtype=jnp.int32),
    }

    with pytest.raises(ValueError):
        target_enrichment_crp(
            dataset,
            target_field="condition",
            target_value=-1,
            target_values=[-1, 0],
            source_field="condition",
            size=3,
        )
    with pytest.raises(ValueError):
        target_enrichment_crp(
            dataset,
            target_field="condition",
            target_value=-1,
            target_values=[-1, 0],
            source_values=[0],
            size=3,
        )


def test_plot_target_enrichment_crp():
    """Behavior: plotting supports repeated datasets and trial masks."""
    recalls = jnp.array([[2, 3, 0], [2, 1, 0]], dtype=jnp.int32)
    pres = jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32)
    dataset: Any = {
        **make_dataset(recalls, pres, subject=jnp.array([1, 2])),
        "condition": jnp.array([[0, 0, -1], [0, 0, 0]], dtype=jnp.int32),
    }
    mask = jnp.array([True, True])

    axis = plot_target_enrichment_crp(
        [dataset, dataset],
        [mask, mask],
        target_field="condition",
        target_value=-1,
        target_values=[-1, 0],
        source_field="condition",
        source_values=[0],
        max_lag=2,
        labels=["A", "B"],
        contrast_name="Dataset",
    )

    assert isinstance(axis, Axes)
    plt.close(axis.figure)


def test_target_enrichment_crp_jit_compatible():
    """Behavior: target enrichment can be JIT compiled."""
    recalls = jnp.array([[2, 3, 0], [2, 1, 0]], dtype=jnp.int32)
    pres = jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32)
    dataset: Any = {
        **make_dataset(recalls, pres),
        "condition": jnp.array([[0, 0, -1], [0, 0, 0]], dtype=jnp.int32),
    }
    target_values = jnp.array([-1, 0], dtype=jnp.int32)
    source_values = jnp.array([0], dtype=jnp.int32)

    result_nojit = target_enrichment_crp(
        dataset,
        target_field="condition",
        target_value=-1,
        target_values=target_values,
        source_field="condition",
        source_values=source_values,
        size=3,
    )
    result_jit = jit(
        target_enrichment_crp,
        static_argnames=("target_field", "target_value", "source_field", "size"),
    )(
        dataset,
        target_field="condition",
        target_value=-1,
        target_values=target_values,
        source_field="condition",
        source_values=source_values,
        size=3,
    )

    np.testing.assert_allclose(
        np.asarray(result_nojit), np.asarray(result_jit), equal_nan=True
    )
