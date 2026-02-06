from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from jaxcmr.typing import MemorySearch

__all__ = [
    "matrix_heatmap",
    "instance_memory_heatmap",
    "plot_model_state",
]


def matrix_heatmap(
    matrix,
    figsize=(15, 15),
    axis=None,
    label_font_size=32,
    annot_font_size=14,
    print_threshold=0.005,
    title="",
):
    """Plots an array of model states as a value-annotated heatmap with an arbitrary title. Omits annotations for cells
    where values are effectively zero.

    Args:
        matrix: an array of model states; columns representing unique feature indices and rows identifying unique update indices
        title: a title for the generated plot,
        label_font_size: font size for the axis labels
        annot_font_size: font size for the annotations within each cell
        axis: an existing matplotlib axis (optional)

    Returns:
        (fig, axis): the figure and axis objects for the generated heatmap
    """

    if matrix.ndim == 1:
        matrix = np.expand_dims(matrix, axis=0)

    if axis is None:
        fig, axis = plt.subplots(figsize=figsize)
    else:
        fig = axis.figure

    annot = np.array(
        [
            [
                "" if -print_threshold < val < print_threshold else f"{val:.2f}"
                for val in row
            ]
            for row in matrix
        ]
    )

    sns.heatmap(
        matrix,
        annot=annot,
        fmt="",
        annot_kws={"size": annot_font_size},
        linewidths=0.5,
        ax=axis,
        cbar=True,
    )

    axis.set_xlabel("Feature Index", fontsize=label_font_size)
    axis.set_ylabel("Update Index", fontsize=label_font_size)
    axis.set_title(title, fontsize=label_font_size)

    return fig, axis


def instance_memory_heatmap(
    memory_state,
    pre_experimental_size=0,
    include_inputs=True,
    include_outputs=True,
    include_preexperimental=True,
    title="Memory Traces",
):
    assert include_inputs or include_outputs, (
        "At least one of inputs or outputs must be included"
    )
    memory_shape = np.shape(memory_state)
    fig_size = list(reversed(memory_shape))
    plotted_memory = memory_state.copy()

    if not include_inputs:
        fig_size[0] /= 2  # type: ignore
        plotted_memory = plotted_memory[:, int(memory_shape[1] / 2) :]
        title = f"Output {title}"

    if not include_outputs:
        fig_size[0] /= 2  # type: ignore
        plotted_memory = plotted_memory[:, : int(memory_shape[1] / 2)]
        title = f"Input {title}"

    if not include_preexperimental:
        fig_size[1] -= pre_experimental_size
        plotted_memory = plotted_memory[int(pre_experimental_size) :]
        title = f"Experimental {title}"

    return matrix_heatmap(plotted_memory, fig_size, title=title)


def plot_model_state(model: MemorySearch, title: Optional[str] = None) -> None:
    """Plot context state and outcome probabilities for a memory search model.

    Parameters
    ----------
    model : MemorySearch
        A memory search model instance.
    title : str, optional
        Title for the figure.

    """
    fig, (ax_ctx, ax_probs) = plt.subplots(
        2, 1, figsize=(10, 4), gridspec_kw={"height_ratios": [1, 1.2]}
    )
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")

    # Context state
    ctx = np.array(model.context.state)
    ctx_labels = ["Start"] + [str(i + 1) for i in range(model.item_count)]
    ctx_colors = ["#2ecc71"] + ["#3498db"] * model.item_count
    ax_ctx.bar(ctx_labels, ctx, color=ctx_colors)
    ax_ctx.set_ylabel("Activation")
    ax_ctx.set_ylim(0, max(1.0, float(np.max(ctx)) * 1.1))

    # Outcome probabilities
    probs = np.array(model.outcome_probabilities())
    prob_labels = ["Stop"] + [str(i + 1) for i in range(model.item_count)]
    prob_colors = ["#95a5a6"] + [
        "#3498db" if model.recallable[i] else
        "#e74c3c" if i < int(model.study_index) else "#bdc3c7"
        for i in range(model.item_count)
    ]
    ax_probs.bar(prob_labels, probs, color=prob_colors)
    ax_probs.set_ylabel("P(outcome)")
    ax_probs.set_xlabel("Outcome")
    max_prob = float(np.max(probs))
    ax_probs.set_ylim(0, max(0.1, max_prob * 1.15))

    plt.tight_layout()
    plt.show()
