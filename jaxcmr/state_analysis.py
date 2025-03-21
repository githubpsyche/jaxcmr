import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ["matrix_heatmap", "instance_memory_heatmap"]


def matrix_heatmap(
    matrix,
    figsize=(15, 15),
    axis=None,
    label_font_size=32,
    annot_font_size=14,
    print_threshold=.005,
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
            ["" if -print_threshold < val < print_threshold else f"{val:.2f}" for val in row]
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
        cbar=True
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
    assert (
        include_inputs or include_outputs
    ), "At least one of inputs or outputs must be included"
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