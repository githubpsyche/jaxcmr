"""Linear memory sums learned associations in a matrix. Studying the same
input twice adds both output patterns to the same associations. Probing
with a cue multiplies the cue by that matrix.

CMR uses a separate memory for each direction. `init_mfc` lets an item
cue its context. `init_mcf` lets context cue the items.
"""



from jax import numpy as jnp

from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.typing import Array, Context, Float, Float_, Int_, Parameters

__all__ = [
    "LinearMemory",
    "init_mcf",
    "init_mfc",
]


class LinearMemory(Pytree):
    """An association matrix with one row per input feature and one column per output feature.

    Parameters
    ----------
    state : floating array, shape (input_size, output_size)
        Initial association weights.
    """

    def __init__(
        self,
        state: Float[Array, " input_size output_size"],
    ) -> None:
        """Initialize memory with an association matrix.

        Args:
            state: Association weights. Shape (input_size, output_size).
        """
        self.state = state
        self.input_size = self.state.shape[0]
        self.output_size = self.state.shape[1]

    def associate(
        self,
        in_pattern: Float[Array, " input_size"],
        out_pattern: Float[Array, " output_size"],
        learning_rate: Float_,
    ) -> "LinearMemory":
        """Add the weighted outer product of the input and output patterns.

        Parameters
        ----------
        in_pattern : floating array, shape (input_size,)
            Input pattern.
        out_pattern : floating array, shape (output_size,)
            Output pattern to associate with the input.
        learning_rate : float
            Weight given to this association.

        Returns
        -------
        LinearMemory
            Updated memory. The original matrix retains its weights.
        """
        return self.replace(
            state=self.state + (learning_rate * jnp.outer(in_pattern, out_pattern))
        )

    def probe(
        self,
        in_pattern: Float[Array, " input_size"],
    ) -> Float[Array, " output_size"]:
        """Retrieve the weighted sum of the rows selected by an input cue.

        Parameters
        ----------
        in_pattern : floating array, shape (input_size,)
            Input pattern.

        Returns
        -------
        floating array, shape (output_size,)
            The product of the input row vector and the association matrix.
        """
        return jnp.dot(in_pattern, self.state)

    def zero_out(
        self,
        index: Int_,
    ) -> "LinearMemory":
        """Clear all outgoing associations of one input feature.

        Parameters
        ----------
        index : int
            Zero-based row index in the association matrix.

        Returns
        -------
        LinearMemory
            Updated memory with the selected row set to zero.
        """
        return self.replace(state=self.state.at[index].set(0))

def init_mfc(
    list_length: int,
    parameters: Parameters,
    context: Context,
) -> "LinearMemory":
    """Create item-to-context memory with one initial context association per item.

    Parameters
    ----------
    list_length : int
        Number of item slots in the trial.
    parameters : Parameters
        The initial association to the corresponding item unit is `1 - learning_rate`.
    context : Context
        Context whose size sets the number of context units.

    Returns
    -------
    LinearMemory
        Probing with an item cue returns a pattern of length `context.size`.
    """
    context_feature_count = context.size
    learning_rate = parameters["learning_rate"]
    item_feature_count = list_length
    return LinearMemory(
        jnp.eye(item_feature_count, context_feature_count, 1) * (1 - learning_rate),
    )

def init_mcf(
    list_length: int,
    parameters: Parameters,
    context: Context,
) -> "LinearMemory":
    """Create context-to-item memory with item-specific and shared initial support.

    Parameters
    ----------
    list_length : int
        Number of item slots in the trial.
    parameters : Parameters
        An in-list context unit gives its matching item `item_support` and every
        other item `shared_support`.
    context : Context
        Context whose size sets the number of context units.

    Returns
    -------
    LinearMemory
        Probing with a context cue returns one value per item slot. Start-of-list
        and out-of-list units initially give every item zero support.
    """
    item_support = parameters["item_support"]
    shared_support = parameters["shared_support"]

    context_rows = jnp.arange(context.size)[:, None]
    in_list = (context_rows > 0) & (context_rows <= list_length)
    baseline = jnp.where(in_list, shared_support, 0.0)
    # Context row 1 corresponds to item column 0.
    matching_items = jnp.eye(context.size, list_length, k=-1)
    return LinearMemory(baseline + matching_items * (item_support - shared_support))
