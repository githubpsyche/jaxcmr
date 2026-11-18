"""Regression checks for model-state diagnostic plots."""

import matplotlib
import pytest
from support import CMR_PARAMS

matplotlib.use("Agg", force=True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from matplotlib.patches import Rectangle

from jaxcmr.components.context import TemporalContext, init
from jaxcmr.components.instance_memory import init_mcf
from jaxcmr.experimental.state_analysis import (
    instance_memory_heatmap,
    plot_context_and_outcomes,
)
from jaxcmr.models.cmr import CMR
from jaxcmr.models.positional_cmr import CMR as PositionalCMR

_PARAMETERS = CMR_PARAMS | {
    "stop_probability_scale": 0.05,
    "stop_probability_growth": 0.2,
    "mfc_sensitivity": 1.0,
}


def test_plot_after_transition_handles_expanded_context_without_showing(monkeypatch):
    """Plot a JAX-updated model with extra context units and caller labels."""
    model = CMR(4, _PARAMETERS, context_create_fn=TemporalContext.init_expanded).experience(
        jnp.int32(1)
    )
    labels = [f"feature {index}" for index in range(9)]
    monkeypatch.setattr(
        plt, "show", lambda: (_ for _ in ()).throw(AssertionError("unexpected show"))
    )
    figure, supplied_axes = plt.subplots(1, 2, figsize=(12, 3))
    positions = [axis.get_position().bounds for axis in supplied_axes]
    returned_figure, (context_axis, outcome_axis) = plot_context_and_outcomes(
        model, context_labels=labels, axes=tuple(supplied_axes)
    )
    try:
        assert returned_figure is figure
        assert context_axis is supplied_axes[0] and outcome_axis is supplied_axes[1]
        np.testing.assert_allclose([axis.get_position().bounds for axis in supplied_axes], positions)
        np.testing.assert_allclose([bar.get_height() for bar in context_axis.patches], model.context.state)
        assert len(context_axis.patches) == 9
        assert [tick.get_text() for tick in context_axis.get_xticklabels()] == labels
        assert len(outcome_axis.patches) == 5
    finally:
        plt.close(figure)


def test_repeated_items_do_not_get_occurrence_based_outcome_colors():
    """Plot item outcomes without treating occurrence slots as item status."""
    model = PositionalCMR(4, _PARAMETERS)
    for item in (1, 2, 1, 3):
        model = model.experience(jnp.int32(item))
    model = model.start_retrieving().retrieve(jnp.int32(1))
    figure, (_, outcome_axis) = plot_context_and_outcomes(model)
    try:
        bars = outcome_axis.patches
        assert len({to_hex(bar.get_facecolor()) for bar in bars[1:]}) == 1
        heights = [bar.get_height() for bar in bars if isinstance(bar, Rectangle)]
        assert len(heights) == len(bars)
        assert np.allclose(heights, np.asarray(model.outcome_probabilities()))
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    "features,include_preexperimental,size,columns",
    [
        ("both", True, 0, slice(None)),
        ("inputs", True, 0, slice(None, 4)),
        ("outputs", True, 0, slice(4, None)),
        ("outputs", False, 3, slice(4, None)),
    ],
)
def test_instance_memory_heatmap_respects_input_width_and_trace_selection(
    features, include_preexperimental, size, columns
):
    memory = init_mcf(
        3, {"item_support": 1.0, "shared_support": 0.1, "mcf_trace_sensitivity": 1.0}, init(3)
    )
    original = np.array(memory.state, copy=True)
    expected = original[size:, columns]
    figure, axis = instance_memory_heatmap(
        memory,
        pre_experimental_size=size,
        features=features,
        include_preexperimental=include_preexperimental,
    )
    try:
        np.testing.assert_allclose(np.asarray(axis.collections[0].get_array()), expected)
        np.testing.assert_array_equal(memory.state, original)
        np.testing.assert_allclose(figure.get_size_inches(), (10, 6))
        feature_labels = [f"Input {index}" for index in range(4)] + [
            f"Output {index}" for index in range(3)
        ]
        assert [tick.get_text() for tick in axis.get_xticklabels()] == feature_labels[columns]
        assert [tick.get_text() for tick in axis.get_yticklabels()] == [
            str(index) for index in range(size, original.shape[0])
        ]
        assert axis.get_ylabel() == "Trace Index"
    finally:
        plt.close(figure)
