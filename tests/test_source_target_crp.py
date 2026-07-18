from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from jax import jit
import jax.numpy as jnp
from matplotlib.axes import Axes

from jaxcmr.analyses.source_target_crp import (
    plot_source_target_crp,
    source_target_crp,
    subject_source_target_crp,
    tabulate_trial,
)
from jaxcmr.helpers import make_dataset


def test_tabulate_trial_counts_item_denominators():
    """Behavior: denominator counts available target items, not bins."""
    trial = jnp.array([1, 3, 0], dtype=jnp.int32)
    presentation = jnp.array([1, 2, 3], dtype=jnp.int32)
    conditions = jnp.array([-1, 1, 1], dtype=jnp.int32)
    should_tabulate = jnp.ones_like(trial, dtype=bool)
    source_values = jnp.array([-1], dtype=jnp.int32)
    target_values = jnp.array([1], dtype=jnp.int32)

    actual, possible = tabulate_trial(
        trial,
        presentation,
        conditions,
        conditions,
        should_tabulate,
        source_values,
        target_values,
        size=3,
    )

    np.testing.assert_array_equal(np.asarray(actual), np.array([[1]]))
    np.testing.assert_array_equal(np.asarray(possible), np.array([[2]]))


def test_source_target_crp_exact_values_with_neutral_target():
    """Behavior: neutral targets update state but are not plotted target hits."""
    recalls = jnp.array([[1, 3, 2, 4, 0]], dtype=jnp.int32)
    pres = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    dataset: Any = {
        **make_dataset(recalls, pres),
        "valence": jnp.array([[-1, 0, 1, 1]], dtype=jnp.int32),
    }

    result = source_target_crp(
        dataset,
        source_field="valence",
        source_values=[-1, 1],
        target_field="valence",
        target_values=[-1, 1],
        size=3,
    )

    assert jnp.isnan(result[0, 0]).item()
    assert float(result[0, 1]) == pytest.approx(0.5)
    assert jnp.isnan(result[1, 0]).item()
    assert float(result[1, 1]) == pytest.approx(0.0)


def test_source_target_crp_should_tabulate_updates_state():
    """Behavior: excluded transitions still update previous item and availability."""
    recalls = jnp.array([[1, 2, 3, 0]], dtype=jnp.int32)
    pres = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    should_tabulate = jnp.array([[True, False, True, True]], dtype=bool)
    dataset: Any = {
        **make_dataset(recalls, pres),
        "_should_tabulate": should_tabulate,
        "valence": jnp.array([[-1, 0, 1]], dtype=jnp.int32),
    }

    result = source_target_crp(
        dataset,
        source_field="valence",
        source_values=[-1, 0],
        target_field="valence",
        target_values=[1],
        size=3,
    )

    assert jnp.isnan(result[0, 0]).item()
    assert float(result[1, 0]) == pytest.approx(1.0)


def test_source_target_crp_subject_aggregation():
    """Behavior: subject-level helper returns one source-target table per subject."""
    recalls = jnp.array([[1, 2, 0], [1, 3, 0]], dtype=jnp.int32)
    pres = jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32)
    dataset: Any = {
        **make_dataset(recalls, pres, subject=jnp.array([1, 2])),
        "valence": jnp.array([[-1, 1, 1], [-1, 1, 1]], dtype=jnp.int32),
    }
    trial_mask = jnp.array([True, True])

    result = subject_source_target_crp(
        dataset,
        trial_mask,
        source_field="valence",
        source_values=[-1],
        target_field="valence",
        target_values=[1],
        size=3,
    )

    assert result.shape == (2, 1, 1)
    np.testing.assert_allclose(result[:, 0, 0], np.array([0.5, 0.5]))


def test_plot_source_target_crp():
    """Behavior: plotting supports repeated datasets and trial masks."""
    recalls = jnp.array([[1, 2, 0], [1, 3, 0]], dtype=jnp.int32)
    pres = jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32)
    dataset: Any = {
        **make_dataset(recalls, pres, subject=jnp.array([1, 2])),
        "valence": jnp.array([[-1, 1, 1], [-1, 1, 1]], dtype=jnp.int32),
    }
    mask = jnp.array([True, True])

    axis = plot_source_target_crp(
        [dataset, dataset],
        [mask, mask],
        source_field="valence",
        source_values=[-1],
        target_field="valence",
        target_values=[1],
        source_labels=["Negative"],
        labels=["A", "B"],
        contrast_name="Dataset",
        size=3,
    )

    assert isinstance(axis, Axes)
    plt.close(axis.figure)


def test_source_target_crp_jit_compatible():
    """Behavior: source-target CRP can be JIT compiled."""
    recalls = jnp.array([[1, 2, 0], [1, 3, 0]], dtype=jnp.int32)
    pres = jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32)
    dataset: Any = {
        **make_dataset(recalls, pres),
        "valence": jnp.array([[-1, 1, 1], [-1, 1, 1]], dtype=jnp.int32),
    }
    source_values = jnp.array([-1], dtype=jnp.int32)
    target_values = jnp.array([1], dtype=jnp.int32)

    result_nojit = source_target_crp(
        dataset,
        source_field="valence",
        source_values=source_values,
        target_field="valence",
        target_values=target_values,
        size=3,
    )
    result_jit = jit(
        source_target_crp,
        static_argnames=("source_field", "target_field", "size"),
    )(
        dataset,
        source_field="valence",
        source_values=source_values,
        target_field="valence",
        target_values=target_values,
        size=3,
    )

    np.testing.assert_allclose(np.asarray(result_nojit), np.asarray(result_jit))
