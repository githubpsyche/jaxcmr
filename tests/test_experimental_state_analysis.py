"""Regression checks for model-state diagnostic plots."""

import matplotlib

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
    plot_model_state,
)
from jaxcmr.models.cmr import CMR
from jaxcmr.models.positional_cmr import CMR as PositionalCMR

_PARAMETERS = {
    "encoding_drift_rate": 0.5,
    "start_drift_rate": 0.4,
    "recall_drift_rate": 0.6,
    "shared_support": 1.0,
    "item_support": 1.5,
    "learning_rate": 0.3,
    "primacy_scale": 1.0,
    "primacy_decay": 0.1,
    "stop_probability_scale": 0.05,
    "stop_probability_growth": 0.2,
    "choice_sensitivity": 2.0,
    "mfc_sensitivity": 1.0,
    "learn_after_context_update": False,
    "allow_repeated_recalls": False,
}


def test_plot_after_transition_handles_expanded_context_without_showing(monkeypatch):
    """Plot a JAX-updated model with extra context units and caller labels."""
    model = CMR(
        4, _PARAMETERS, context_create_fn=TemporalContext.init_expanded
    ).experience(jnp.int32(1))
    labels = [f"feature {index}" for index in range(9)]
    monkeypatch.setattr(
        plt, "show", lambda: (_ for _ in ()).throw(AssertionError("unexpected show"))
    )

    figure, (context_axis, outcome_axis) = plot_model_state(
        model, context_labels=labels
    )
    try:
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

    figure, (_, outcome_axis) = plot_model_state(model)
    try:
        bars = outcome_axis.patches
        assert len({to_hex(bar.get_facecolor()) for bar in bars[1:]}) == 1
        heights = [bar.get_height() for bar in bars if isinstance(bar, Rectangle)]
        assert len(heights) == len(bars)
        assert np.allclose(heights, np.asarray(model.outcome_probabilities()))
    finally:
        plt.close(figure)


def test_instance_memory_heatmap_respects_input_width_and_trace_selection():
    """Context-to-item traces have unequal input and output widths."""
    memory = init_mcf(
        3,
        {"item_support": 1.0, "shared_support": 0.1, "mcf_trace_sensitivity": 1.0},
        init(3),
    )
    state = np.asarray(memory.state)
    views = (
        (True, False, True, 0, state[:, :4]),
        (False, True, True, 0, state[:, 4:]),
        (False, True, False, 3, state[3:, 4:]),
    )

    for (
        include_inputs,
        include_outputs,
        include_preexperimental,
        size,
        expected,
    ) in views:
        figure, axis = instance_memory_heatmap(
            memory,
            pre_experimental_size=size,
            include_inputs=include_inputs,
            include_outputs=include_outputs,
            include_preexperimental=include_preexperimental,
        )
        try:
            plotted = np.asarray(axis.collections[0].get_array())
            assert plotted.shape == expected.shape
            np.testing.assert_allclose(plotted, expected)
            np.testing.assert_allclose(figure.get_size_inches(), (10, 6))
        finally:
            plt.close(figure)
