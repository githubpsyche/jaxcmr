"""Experimental visualizations of memory-search models and components."""

from collections.abc import Sequence
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from jaxcmr.components.instance_memory import InstanceMemory
from jaxcmr.state_analysis import matrix_heatmap
from jaxcmr.typing import Context, MemorySearch

__all__ = ["instance_memory_heatmap", "plot_model_state"]


def instance_memory_heatmap(
    memory: InstanceMemory,
    pre_experimental_size: int = 0,
    include_inputs: bool = True,
    include_outputs: bool = True,
    include_preexperimental: bool = True,
    title: str = "Memory Traces",
    *,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[Figure, Axes]:
    """Plot selected input/output columns and traces from an instance memory.

    Args:
        memory: Memory whose state and input/output widths define the view.
        pre_experimental_size: Number of initial rows to omit when
            ``include_preexperimental`` is false.
        include_inputs: Show the input features.
        include_outputs: Show the output features.
        include_preexperimental: Show initial pre-experimental traces.
        title: Figure title.
        figsize: Figure size in inches, independent of memory dimensions.

    Returns:
        The figure and heatmap axis.
    """
    if not include_inputs and not include_outputs:
        raise ValueError("At least one of inputs or outputs must be included")

    state = np.asarray(memory.state)
    if state.ndim != 2:
        raise ValueError("memory.state must be a two-dimensional matrix")
    input_size = int(memory.input_size)
    output_size = int(memory.output_size)
    if state.shape[1] != input_size + output_size:
        raise ValueError("memory.state width must equal input_size + output_size")

    if include_inputs and include_outputs:
        plotted_memory = state
    elif include_inputs:
        plotted_memory = state[:, :input_size]
        title = f"Input {title}"
    else:
        plotted_memory = state[:, input_size:]
        title = f"Output {title}"

    if not include_preexperimental:
        if not 0 <= pre_experimental_size <= state.shape[0]:
            raise ValueError("pre_experimental_size must be within the trace count")
        plotted_memory = plotted_memory[pre_experimental_size:]
        title = f"Experimental {title}"

    if 0 in plotted_memory.shape:
        raise ValueError("The selected memory view is empty")

    return matrix_heatmap(plotted_memory, figsize=figsize, title=title)


class _ContextualMemorySearch(MemorySearch, Protocol):
    """Memory-search model that exposes a context representation."""

    context: Context


def plot_model_state(
    model: _ContextualMemorySearch,
    title: str | None = None,
    *,
    context_labels: Sequence[str] | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot context state and retrieval outcome probabilities.

    Context labels default to feature indices. Supply ``context_labels`` when
    the units have known meanings, such as a start-of-list unit. Probability
    bars describe outcomes without inferring item availability from the model's
    internal representation.

    Args:
        model: A memory-search model with a context state.
        title: Optional figure title.
        context_labels: Optional label for each context feature.

    Returns:
        The figure and its context and outcome axes.
    """
    context_values = np.asarray(model.context.state)
    if context_values.ndim != 1 or context_values.size == 0:
        raise ValueError("context.state must be a nonempty one-dimensional vector")

    labels = (
        [str(index) for index in range(context_values.size)]
        if context_labels is None
        else list(context_labels)
    )
    if len(labels) != context_values.size:
        raise ValueError("context_labels must match the number of context features")

    item_count = int(model.item_count)
    probabilities = np.asarray(model.outcome_probabilities())
    if probabilities.shape != (item_count + 1,):
        raise ValueError(
            "outcome probabilities must contain stop plus one value per item"
        )

    figure, (context_axis, probability_axis) = plt.subplots(
        2, 1, figsize=(10, 4), gridspec_kw={"height_ratios": [1, 1.2]}
    )
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

    figure.tight_layout()
    return figure, (context_axis, probability_axis)
