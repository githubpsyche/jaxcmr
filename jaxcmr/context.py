from jax import numpy as jnp
from simple_pytree import Pytree

from jaxcmr.helpers import lb
from jaxcmr.typing import Array, Float, Float_


def linalg_norm(
    vector: Float[Array, " features"],
) -> Float[Array, " features"]:
    """Return the input vector normalized to unit length."""
    return vector / jnp.sqrt(jnp.sum(vector**2) + lb)


class TemporalContext(Pytree):
    """Temporal context representation for memory search models.

    Attributes:
        state: the current state of the context.
        initial_state: the initial state of the context.
        next_outlist_unit: the index of the next out-of-list context unit.
        outlist_contexts: the out-of-list context representations.
    """

    def __init__(self, item_count: int, size: int):
        """Create a new temporal context model.

        Args:
            item_count: the number of items in the context model.
        """
        self.size = size
        self.zeros = jnp.zeros(size)
        self.state = self.zeros.at[0].set(1)
        self.initial_state = self.zeros.at[0].set(1)
        self.next_outlist_unit = item_count + 1

    def integrate(
        self,
        context_input: Float[Array, " context_feature_units"],
        drift_rate: Float_,
    ) -> "TemporalContext":
        """Returns context after integrating input representation, preserving unit length.

        Args:
            context_input: the input representation to be integrated into the contextual state.
            drift_rate: The drift rate parameter.
        """
        context_input = linalg_norm(context_input)
        rho = jnp.sqrt(
            1 + jnp.square(drift_rate) * (jnp.square(self.state * context_input) - 1)
        ) - (drift_rate * (self.state * context_input))
        return self.replace(
            state=linalg_norm((rho * self.state) + (drift_rate * context_input))
        )

    @classmethod
    def init(cls, item_count: int) -> "TemporalContext":
        """Initialize a new context model.

        Args:
            item_count: the number of items in the context model.
        """
        return cls(item_count, item_count + 1)
