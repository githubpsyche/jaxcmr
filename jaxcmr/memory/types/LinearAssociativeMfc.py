"""
Linear Associative $M^{FC}$

Initialization functions for a feature-to-context linear associative memory as specified for CMR.
Item representations and a space of one-hot context states are associated according to the Hebbian learning rule based on a provided learning rate.

When initialized with an item count, items are represented as one-hot vectors.
Otherwise, items can be explicitly provided as a matrix of item vectors.
"""

# %% Imports
from jaxcmr.memory.types.OneWayMemory import LinearAssociativeMemory
from jaxcmr.helpers import Float, Array, ScalarInteger, ScalarFloat
from plum import dispatch
from jax import lax, jit, numpy as jnp
from functools import partial

# %% Exports

__all__ = [
    "LinearAssociativeMfc",
]

# %% Base Type for Linear Associative Mfc

class LinearAssociativeMfc(LinearAssociativeMemory, mutable=True):
    @dispatch
    def __init__(self, state: Float[Array, "item_features context_features"]):
        self.state = state

    @classmethod
    @dispatch
    def create(
        cls,
        item_count: ScalarInteger,
        learning_rate: ScalarFloat,
    ):
        return cls(basic_init_mfc(item_count, learning_rate))

    @classmethod
    @dispatch
    def create(
        cls,
        items: Float[Array, "item_count item_features"],
        learning_rate: ScalarFloat,
    ):
        return cls(generalized_init_mfc(items, learning_rate))
    
    @property
    def input_features(self):
        return self.state.shape[0]

    @property
    def output_features(self):
        return self.state.shape[1]

# %% Type-Specific Helper Functions

@partial(jit, static_argnums=(0,))
def basic_init_mfc(
    item_count: ScalarInteger, learning_rate: ScalarFloat
) -> Float[Array, "item_features context_features"]:
    """A linear associative feature-to-context memory assuming one-hot item representations"""
    memory = jnp.eye(item_count, item_count + 2)
    memory = jnp.hstack([jnp.zeros((item_count, 1)), memory[:, :-1]])
    return memory * (1 - learning_rate)


@jit
def generalized_init_mfc(
    items: Float[Array, "item_count item_features"],
    learning_rate: ScalarFloat,
) -> Float[Array, "item_features context_features"]:
    """A linear associative feature-to-context memory from arbitrary item representations"""
    item_count = items.shape[0]
    item_feature_count = items.shape[1]
    context_feature_count = item_count + 2
    contexts = jnp.eye(item_count, context_feature_count, 1)
    memory = jnp.zeros((item_feature_count, context_feature_count))
    return lax.fori_loop(
        0,
        item_count,
        lambda i, m: hebbian_associate(
            m, 1 - learning_rate, items[i], contexts[i]
        ),
        memory,
    )

@jit
@dispatch
def hebbian_associate(
    memory_state: Float[Array, "input_features output_features"],
    learning_rate: ScalarFloat,
    input_feature_pattern: Float[Array, "input_features"],
    output_feature_pattern: Float[Array, "output_features"],
) -> Float[Array, "input_features output_features"]:
    """Associate input and output feature patterns in a M x N linear associative memory state"""
    return memory_state + (
        learning_rate * jnp.outer(input_feature_pattern, output_feature_pattern)
    )