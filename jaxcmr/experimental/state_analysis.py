"""Inspect memory traces and the current context and retrieval probabilities.

``instance_memory_heatmap`` selects input/output features and traces from an
instance memory, then draws them with ``jaxcmr.plotting.matrix_heatmap``.
``plot_context_and_outcomes`` shows a model's current context activations and
probabilities for stopping or recalling each item. Both accept existing axes
for inclusion in larger diagnostic figures.

These views read the supplied state without advancing the model. They remain
experimental; callers choose the study or retrieval event to inspect. Memory
heatmaps retain the stored trace indices, including unused zero-filled rows.
"""

from collections.abc import Sequence
from typing import Literal, Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure, SubFigure

from jaxcmr.components.instance_memory import InstanceMemory
from jaxcmr.plotting import matrix_heatmap
from jaxcmr.typing import Context, MemorySearch

__all__ = ["instance_memory_heatmap", "plot_context_and_outcomes"]


def instance_memory_heatmap(
    memory: InstanceMemory,
    pre_experimental_size: int = 0,
    include_preexperimental: bool = True,
    title: str = "Memory Traces",
    *,
    features: Literal["inputs", "outputs", "both"] = "both",
    figsize: tuple[float, float] = (10, 6),
    axis: Axes | None = None,
    colorbar: bool = True,
    norm: Normalize | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    """Draw selected memory traces with their original trace and feature identities.

    Parameters
    ----------
    memory : InstanceMemory
        Memory whose rows concatenate input and output feature values. All
        selected storage rows are shown, including unused zero-filled rows.
    pre_experimental_size : int, optional
        Number of initial rows to omit when include_preexperimental is false.
    include_preexperimental : bool, optional
        Include the initial traces as well as experimental traces.
    title : str, optional
        Base title, prefixed to describe selected features and traces.
    features : {"inputs", "outputs", "both"}, optional
        Which part of each stored trace to display. Input and output feature
        indices each start at zero and retain those labels in every view.
    figsize : tuple of float, optional
        Figure size in inches when axis is omitted.
    axis : Axes, optional
        Existing axes for the heatmap, in a figure or subfigure.
    colorbar : bool, optional
        Include a colorbar.
    norm : matplotlib.colors.Normalize, optional
        Color normalization passed to matrix_heatmap. Use fixed limits shared
        across views to make their colors comparable.

    Returns
    -------
    tuple of Figure or SubFigure and Axes
        The containing figure or subfigure and heatmap axes. Neither is shown,
        saved or closed.

    Raises
    ------
    ValueError
        The chosen traces and features form an empty view.
    """
    state = np.asarray(memory.state)
    input_size = int(memory.input_size)
    input_labels = [f"Input {index}" for index in range(input_size)]
    output_labels = [f"Output {index}" for index in range(int(memory.output_size))]

    if features == "both":
        plotted_memory = state
        column_labels = input_labels + output_labels
    elif features == "inputs":
        plotted_memory = state[:, :input_size]
        column_labels = input_labels
        title = f"Input {title}"
    else:
        plotted_memory = state[:, input_size:]
        column_labels = output_labels
        title = f"Output {title}"

    first_trace = 0 if include_preexperimental else pre_experimental_size
    plotted_memory = plotted_memory[first_trace:]
    if not include_preexperimental:
        title = f"Experimental {title}"

    if 0 in plotted_memory.shape:
        raise ValueError("The selected memory view is empty")

    return matrix_heatmap(
        plotted_memory,
        figsize=figsize,
        axis=axis,
        title=title,
        xlabel="Memory Features",
        ylabel="Trace Index",
        row_labels=[str(index) for index in range(first_trace, state.shape[0])],
        column_labels=column_labels,
        colorbar=colorbar,
        norm=norm,
    )


class _ContextualMemorySearch(MemorySearch, Protocol):
    """Memory-search model that exposes a context representation."""

    context: Context
    item_count: int


def plot_context_and_outcomes(
    model: _ContextualMemorySearch,
    title: str | None = None,
    *,
    context_labels: Sequence[str] | None = None,
    axes: tuple[Axes, Axes] | None = None,
    figsize: tuple[float, float] = (10, 4),
) -> tuple[Figure | SubFigure, tuple[Axes, Axes]]:
    """Draw current context activations and retrieval-outcome probabilities.

    Parameters
    ----------
    model : MemorySearch with context
        Model at the event to inspect. This view reads context.state and calls
        outcome_probabilities without advancing study or retrieval.
    title : str, optional
        Title for the containing figure, including when axes are supplied.
    context_labels : sequence of str, optional
        One label per context feature; defaults to zero-based indices. Supply
        meaningful labels when known, such as the start-of-list unit.
    axes : tuple of Axes and Axes, optional
        Context and outcome axes in the same figure or subfigure. Omit to create two panels.
        Supplied axes retain their layout; bars are added to existing contents.
    figsize : tuple of float, optional
        Figure size in inches when axes are omitted.

    Returns
    -------
    tuple of Figure or SubFigure and tuple of Axes
        Containing figure or subfigure and the context and outcome axes. The function
        does not show, save or close the figure.

    Notes
    -----
    Outcome zero denotes stopping; outcomes 1 through item_count denote item
    recalls. Item bars share one color. The view does not infer item availability
    from context units or occurrence slots. Association matrices and other model
    components are outside this two-panel view.
    """
    context_values = np.asarray(model.context.state)
    labels = (
        [str(index) for index in range(context_values.size)]
        if context_labels is None
        else list(context_labels)
    )
    item_count = int(model.item_count)
    probabilities = np.asarray(model.outcome_probabilities())

    if axes is None:
        figure, (context_axis, probability_axis) = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw={"height_ratios": [1, 1.2]}
        )
    else:
        context_axis, probability_axis = axes
        figure = context_axis.figure
    if title:
        figure.suptitle(title, fontsize=12, fontweight="bold")

    context_positions = np.arange(context_values.size)
    context_axis.bar(context_positions, context_values, color="#3498db")
    context_axis.set_xticks(context_positions, labels)
    context_axis.set_ylabel("Activation")
    context_axis.set_ylim(
        min(0.0, float(np.min(context_values)) * 1.1),
        max(1.0, float(np.max(context_values)) * 1.1),
    )

    outcome_positions = np.arange(item_count + 1)
    outcome_labels = ["Stop", *[str(index + 1) for index in range(item_count)]]
    outcome_colors = ["#95a5a6", *["#3498db"] * item_count]
    probability_axis.bar(outcome_positions, probabilities, color=outcome_colors)
    probability_axis.set_xticks(outcome_positions, outcome_labels)
    probability_axis.set_ylabel("P(outcome)")
    probability_axis.set_xlabel("Outcome")
    probability_axis.set_ylim(0, max(0.1, float(np.max(probabilities)) * 1.15))

    if axes is None:
        cast(Figure, figure).tight_layout()
    return figure, (context_axis, probability_axis)
