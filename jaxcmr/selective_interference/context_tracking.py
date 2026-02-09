"""Context trajectory tracking during interference encoding.

Records the temporal context vector at each step of an interference
encoding phase, producing a trajectory that can be analyzed for
contextual drift and overlap with target memories.

"""

from jax import lax
from jax import numpy as jnp

from jaxcmr.math import normalize_magnitude
from jaxcmr.typing import Array, Float, Integer, MemorySearch


__all__ = ["track_context_trajectory"]


def track_context_trajectory(
    model: MemorySearch,
    items: Integer[Array, " n_items"],
    reference: Float[Array, " context_size"],
) -> Float[Array, " n_items"]:
    """Record context similarity to a reference after each encoding step.

    Parameters
    ----------
    model : MemorySearch
        Model state at the start of the encoding sequence.
    items : Integer[Array, " n_items"]
        One-indexed items to encode sequentially.
    reference : Float[Array, " context_size"]
        Reference context vector (e.g. end-of-film state).

    Returns
    -------
    Float[Array, " n_items"]
        Cosine similarity between context and reference after each step.

    """
    ref_normed = normalize_magnitude(reference)

    def step(
        m: MemorySearch, item: Integer[Array, ""]
    ) -> tuple[MemorySearch, Float[Array, ""]]:
        m = m.experience(item)
        sim = jnp.dot(normalize_magnitude(m.context.state), ref_normed)
        return m, sim

    _, similarities = lax.scan(step, model, items)
    return similarities
