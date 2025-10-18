from typing import Mapping, Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from jaxcmr.math import cosine_similarity_matrix, lb
from jaxcmr.typing import (
    Array,
    Float,
    Float_,
    Integer,
    MemorySearch,
    MemorySearchCreateFn,
    RecallDataset,
)

__all__ = [
    "matrix_heatmap",
    "instance_memory_heatmap",
    "MemorySearchModelFactory",
    "build_trial_connections",
]


def build_trial_connections(
    present_lists: np.ndarray,
    features: Optional[Float[Array, " word_pool_items features_count"]],
) -> Float[Array, " trials study_events study_events"]:
    """Returns per-trial connection matrices aligned to study lists.

    Args:
      present_lists: Study lists indexed by trial, containing 1-indexed item ids.
      features: Wordpool-wide feature matrix or ``None`` to disable semantics.
    """

    # If no connections are provided, return zero matrices
    if features is None:
        list_length = present_lists.shape[1]
        blank = jnp.zeros((list_length, list_length))
        return jnp.stack([blank] * present_lists.shape[0])

    connections = cosine_similarity_matrix(features)

    # Clip to non-negative values and zero the diagonal
    clipped = jnp.clip(connections, a_min=lb, a_max=None)
    zeroed = clipped * (1.0 - jnp.eye(clipped.shape[0]))

    # Extract trial-specific submatrices
    trial_blocks: list[jnp.ndarray] = []
    for trial_idx in range(present_lists.shape[0]):
        present = present_lists[trial_idx]
        valid = present > 0
        zero_based = jnp.array(jnp.where(valid, present - 1, 0), dtype=jnp.int32)
        block = zeroed[zero_based[:, None], zero_based[None, :]]
        keep_mask = jnp.logical_and(valid[:, None], valid[None, :])
        trial_blocks.append(jnp.where(keep_mask, block, 0.0).astype(jnp.float32))

    return jnp.stack(trial_blocks)


class MemorySearchModelFactory:
    def __init__(
        self,
        dataset: RecallDataset,
        features: Optional[Float[Array, " word_pool_items features_count"]],
        model_create_fn: MemorySearchCreateFn,
    ):
        self.model_create_fn = model_create_fn
        self.max_list_length = np.max(dataset["listLength"]).item()
        self.present_lists = np.array(dataset["pres_itemids"])
        self.trial_connections = build_trial_connections(self.present_lists, features)

    def create_model(self, parameters: Mapping[str, Float_]) -> MemorySearch:
        return self.model_create_fn(
            self.max_list_length, parameters, self.trial_connections[0]
        )

    def create_trial_model(
        self,
        trial_index: Integer[Array, ""],
        parameters: Mapping[str, Float_],
    ) -> MemorySearch:
        return self.model_create_fn(
            self.max_list_length,
            parameters,
            self.trial_connections[trial_index],
        )


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
