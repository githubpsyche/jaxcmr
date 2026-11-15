"""Model state visualization utilities.

Provides heatmap plotting for association matrices and instance
memory stores, useful for inspecting internal model state during
debugging and development.

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from jaxcmr.typing import Array, Real

__all__ = [
    "matrix_heatmap",
]


def matrix_heatmap(
    matrix: Real[Array | np.ndarray, " values"] | Real[Array | np.ndarray, " rows columns"],
    figsize: tuple[float, float] = (15, 15),
    axis: Axes | None = None,
    label_font_size: float = 32,
    annot_font_size: float = 14,
    print_threshold: float = 0.005,
    title: str = "",
    xlabel: str = "Feature Index",
    ylabel: str = "Update Index",
) -> tuple[Figure, Axes]:
    """Plots an array of model states as a value-annotated heatmap with an arbitrary title. Omits annotations for cells
    where values are effectively zero. Uses the magma palette; nonfinite cells are blank.

    Args:
        matrix: an array of model states; columns representing unique feature indices and rows identifying unique update indices
        title: a title for the generated plot,
        label_font_size: font size for the axis labels
        annot_font_size: font size for the annotations within each cell
        axis: an existing matplotlib axis (optional)
        xlabel: label for the horizontal axis
        ylabel: label for the vertical axis

    Returns:
        (fig, axis): the figure and axis objects for the generated heatmap
    """

    matrix = np.asarray(matrix)
    if matrix.ndim == 1:
        matrix = np.expand_dims(matrix, axis=0)

    if axis is None:
        figure, axis = plt.subplots(figsize=figsize)
    else:
        figure = axis.figure

    mesh = axis.pcolormesh(
        np.ma.masked_invalid(matrix),
        cmap="magma",
        linewidths=0.5,
        edgecolors="white",
        shading="flat",
    )
    row_count, column_count = matrix.shape
    axis.set_xlim(0, column_count)
    axis.set_ylim(row_count, 0)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.colorbar(mesh, ax=axis)

    # Label cell centres, thinning indices to fit the available axis space.
    for dimension, count in ((axis.xaxis, column_count), (axis.yaxis, row_count)):
        tick_capacity = max(1, dimension.get_tick_space())
        tick_step = max(1, int(np.ceil(count / tick_capacity)))
        indices = np.arange(0, count, tick_step)
        dimension.set_ticks(indices + 0.5, labels=[str(index) for index in indices])

    for (row, column), value in np.ndenumerate(matrix):
        if not np.isfinite(value) or -print_threshold < value < print_threshold:
            continue
        colour = mesh.cmap(mesh.norm(value))
        brightness = np.dot(colour[:3], (0.299, 0.587, 0.114))
        axis.text(
            column + 0.5,
            row + 0.5,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=annot_font_size,
            color="black" if brightness > 0.5 else "white",
        )

    axis.set_xlabel(xlabel, fontsize=label_font_size)
    axis.set_ylabel(ylabel, fontsize=label_font_size)
    axis.set_title(title, fontsize=label_font_size)

    return figure, axis
