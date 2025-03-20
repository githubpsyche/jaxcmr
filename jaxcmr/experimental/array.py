from typing import Any, Callable

import numpy as np
from jax import numpy as jnp, vmap, lax

from jaxcmr.typing import Array, Float, Real, Integer
from numba import types
from numba.typed import Dict

# from numba import njit

__all__ = [
    "repmat",
    "all_rows_identical",
    "np_segment_array_by_index",
    "segment_by_index",
    "sub_connectivity",
    "subset_connectivity_matrix",
    "np_connectivity_by_index",
    "cos_sim",
    "compute_similarity_matrix",
    "segment_by_nan",
    "njit_apply_along_axis",
    "filter_repeated_recalls",
]

def to_numba_typed_dict(py_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Converts a Python dictionary of 2D int64 arrays to a Numba-typed dictionary.

    Parameters
    ----------
        py_dict (dict[str, np.ndarray]): A Python dictionary containing 2D int64 arrays.

    Returns
    -------
        dict[str, np.ndarray]: A Numba-typed dictionary containing 2D int32 arrays.
    """
    # Define the key and value types for the Numba typed dict
    key_type = types.unicode_type
    value_type = types.Array(types.int32, 2, "C")  # 2D int64 array

    # Initialize an empty Numba typed dict
    numba_dict = Dict.empty(key_type=key_type, value_type=value_type)

    # Populate the Numba typed dict
    for key, value in py_dict.items():
        numba_dict[key] = np.ascontiguousarray(value.astype(np.int32))

    return numba_dict

def filter_trial(trial):
    seen = set()
    filtered_trial = [
        item for item in trial if item > 0 and (item not in seen and not seen.add(item))
    ]
    # Pad with zeros to maintain the original trial length
    return filtered_trial + [0] * (len(trial) - len(filtered_trial))


def filter_repeated_recalls(recalls: jnp.ndarray):
    """Filters out repeated recalls within each trial, keeping only the first occurrence and padding with 0s.

    Args:
        recalls: A trial by recall position array of recalled items. 1-indexed; 0 for no recall.

    Returns:
        A trial by recall position array with repeated recalls removed and padded with 0s.
    """
    # Convert to list of lists for easy manipulation
    recalls_list = recalls.tolist()

    return [filter_trial(trial) for trial in recalls_list]


def repmat(matrix: Real[Array, " ..."], m: int, n: int) -> Real[Array, " x y"]:
    """Return matrix replicated and tiled m x n times.

    Args:
        matrix: 2-D array to be replicated and tiled.
        m: Number of rows to replicate.
        n: Number of columns to replicate.
    """

    # Handling 1D array
    if matrix.ndim == 1:
        row_count = 1
        col_count = matrix.shape[0]
        result = jnp.zeros((m, col_count * n), dtype=matrix.dtype)
        for col_rep in range(n):
            result[:, col_rep * col_count : (col_rep + 1) * col_count] = matrix

    # Handling 2D array
    else:
        row_count, col_count = matrix.shape
        result = jnp.zeros((row_count * m, col_count * n), dtype=matrix.dtype)
        for row_rep in range(m):  # sourcery skip: use-itertools-product
            for col_rep in range(n):
                result[
                    row_rep * row_count : (row_rep + 1) * row_count,
                    col_rep * col_count : (col_rep + 1) * col_count,
                ] = matrix

    return result


def all_rows_identical(arr: Real[Array, " x y"]) -> bool:
    """Return whether all rows in the 2D array are identical."""
    return jnp.all(arr == arr[0])  # type: ignore


def np_segment_array_by_index(
    data: Real[Array, " ..."], index_vector: Real[Array, " ..."]
) -> tuple[Real[Array, " ..."], Real[Array, " ..."]]:
    """Returns array and unique indices by segmenting the input data array based on index vectors.

    Applies padding to handle variable sizes within groups.

    Args:
        data: The multidimensional array to segment.
        index_vector: Array of indices used to determine how to group elements of the data array.
    """
    # Find unique indices and their positions
    unique_indices, inverse_index, counts = jnp.unique(
        index_vector, return_inverse=True, return_counts=True
    )
    # Determine the maximum count of any index
    max_count = jnp.max(counts)

    # Initialize the output array with zeros (for padding)
    output_shape = (len(unique_indices), max_count, data.shape[1])
    output_array = jnp.zeros(output_shape, dtype=data.dtype)

    # Temporary array to track insertion positions for each unique index
    positions = jnp.zeros_like(unique_indices)

    # Iterate over each element in the original data array
    for i in range(data.shape[0]):
        idx = inverse_index[i]  # Find the group index for the current row
        pos = positions[idx]  # Find the current insert position for this group
        # Check if we still have space to add new items in the group
        if pos < max_count:
            # Insert the data row into the correct position in the output array
            output_array[idx, pos] = data[i]
            # Update the position tracker
            positions[idx] = pos + 1

    return output_array, unique_indices


def segment_by_index(vector, index_vector) -> tuple[list[np.ndarray], np.ndarray]:
    unique_indices, first_indices = np.unique(index_vector, return_index=True)
    unique_indices = unique_indices[np.argsort(first_indices)]
    return [vector[index_vector == idx] for idx in unique_indices], unique_indices


def sub_connectivity(
    trial_item_ids: Integer[Array, " item_ids"], connectivity: Float[Array, " N N"]
) -> jnp.ndarray:
    """Returns a connectivity matrix subset and zero-padded based on non-zero, 1-indexed trial_item_ids.

    Args:
        trial_item_ids: Array of item IDs with 0 indicating padding.
        connectivity: Full NxN connectivity matrix.
    """

    def connection_at(i, j):
        return lax.cond(
            (i * j) == 0,
            lambda: 0.,
            lambda: connectivity[i - 1, j - 1],
        )

    item_count = trial_item_ids.shape[0]
    output_matrix = jnp.zeros((item_count, item_count))
    return lax.fori_loop(
        0,
        item_count,
        lambda i, matrix: lax.fori_loop(
            0,
            item_count,
            lambda j, matrix: matrix.at[i, j].set(
                connection_at(trial_item_ids[i], trial_item_ids[j])
            ),
            matrix,
        ),
        output_matrix,
    )


def subset_connectivity_matrix(
    trialwise_item_ids: Float[Array, " trial list_length"],
    connectivity: Float[Array, " N N"],
) -> Float[Array, " trial list_length list_length"]:
    """Return a subset of the connectivity matrix based on the specified 1-indexed item_ids.

    The output matrix is zero-padded to match the dimensions of item_ids.

    Args:
        trialwise_item_ids: Trial by pres matrix of item IDs, with 0 for padding.
        connectivity: Full NxN connectivity matrix.
    """
    return vmap(sub_connectivity, in_axes=(0, None))(trialwise_item_ids, connectivity)


def np_connectivity_by_index(connectivity, item_ids, index_vector):
    return jnp.array(
        [
            subset_connectivity_matrix(item_ids, connectivity)
            for item_ids in np_segment_array_by_index(item_ids, index_vector)[0]
        ]
    )

def cos_sim(x: Float[Array, "N D"], y: Float[Array, "M D"], epsilon: float = 1e-8) -> Float[Array, "N M"]:
    """
    Compute cosine similarity between two sets of vectors.

    Args:
        x: Array of shape (N, D) containing N D-dimensional vectors.
        y: Array of shape (M, D) containing M D-dimensional vectors.
        epsilon: Small value to ensure numerical stability.

    Returns:
        Array of shape (N, M) containing pairwise cosine similarities.
    """
    # Compute norms
    x_norm = jnp.linalg.norm(x, axis=1, keepdims=True)
    y_norm = jnp.linalg.norm(y, axis=1, keepdims=True)

    # Normalize vectors, handling potential zero norms
    x_normalized = jnp.where(x_norm > epsilon, x / x_norm, 0)
    y_normalized = jnp.where(y_norm > epsilon, y / y_norm, 0)

    # Compute cosine similarity
    similarities = jnp.dot(x_normalized, y_normalized.T)  # type: ignore

    # Handle potential numerical instabilities
    similarities = jnp.nan_to_num(similarities, nan=0.0)
    similarities = jnp.clip(similarities, -1, 1)

    return similarities

def compute_similarity_matrix(embeddings: Float[Array, " words features"]) -> Real[Array, " N N"]:
    """
    Returns an N x N similarity matrix from an N x Z embeddings array.
    Diagonal elements are zero and all other connections are at least 0.

    Args:
        embeddings: An N x Z matrix where each row represents an embedding.
    """
    cosine_scores = cos_sim(embeddings, embeddings)

    # Ensure all values are non-negative
    cosine_scores = jnp.maximum(cosine_scores, 0)

    # Set diagonal elements to zero
    cosine_scores = cosine_scores.at[jnp.diag_indices_from(cosine_scores)].set(0)

    return cosine_scores


def segment_by_nan(vector: jnp.ndarray) -> list[tuple[int, int]]:
    "Returns list of tuples segmenting the vector by its NaN values."
    segments = []
    start_idx = 0
    for i in range(len(vector)):
        if jnp.isnan(vector[i]):
            segments.append((start_idx, i))
            start_idx = i + 1
    if start_idx < len(vector):
        segments.append((start_idx, len(vector)))
    return segments


def njit_apply_along_axis(
    func: Callable, array: jnp.ndarray, *args: Any
) -> list[jnp.ndarray]:
    """Returns a list of results by applying a jit-compiled function along the first axis of a numpy array.

    Args:
        func: Jit-compiled function to be applied. It takes an array slice and additional arguments.
        array: Numpy array to be processed.
        *args: Additional arguments to pass to the function.
    """
    return [func(array[i], *args) for i in range(len(array))]
