from jax import lax
from jax import numpy as jnp
from simple_pytree import Pytree

from jaxcmr.typing import Array, Float, Float_


def power_scale(value: Float_, scale: Float_) -> Float:
    """Returns value scaled by the exponent factor using logsumexp trick."""
    log_activation = jnp.log(value)
    return lax.cond(
        jnp.logical_and(jnp.any(value != 0), scale != 1),
        lambda _: jnp.exp(scale * (log_activation - jnp.max(log_activation))),
        lambda _: value,
        None,
    )


class LinearMemory(Pytree):
    """Linear associative memory model for CMR.

    Attributes:
        input_size: the size of the input representation.
        output_size: the size of the output representation.
        state: the current state of the memory.
    """

    def __init__(
        self,
        state: Float[Array, " input_size output_size"],
        activation_scale: Float_,
    ):
        self.state = state
        self.activation_scale = activation_scale
        self.input_size = self.state.shape[0]
        self.output_size = self.state.shape[1]

    def associate(
        self,
        in_pattern: Float[Array, " input_size"],
        out_pattern: Float[Array, " output_size"],
        learning_rate: Float_,
    ) -> "LinearMemory":
        """Return the updated memory after associating input and output patterns.

        Args:
            in_pattern: a feature pattern for an input.
            out_pattern: a feature pattern for an output.
            learning_rate: the learning rate parameter.
        """
        return self.replace(
            state=self.state + (learning_rate * jnp.outer(in_pattern, out_pattern))
        )

    def probe(
        self,
        in_pattern: Float[Array, " input_size"],
    ) -> Float[Array, " output_size"]:
        """Return the output pattern associated with the input pattern in memory.

        Args:
            memory: the current memory state.
            input_pattern: the input feature pattern.
        """
        return power_scale(jnp.dot(in_pattern, self.state), self.activation_scale)

    @classmethod
    def init_mfc(
        cls,
        item_count: int,
        context_feature_count: int,
        learning_rate: Float_,
        activation_scale: Float_,
    ) -> "LinearMemory":
        """Return a new linear associative item-to-context memory model.

        Initially, all items are associated with a unique context unit by `1-learning_rate`.
        To allow out-of-list contexts, set context_feature_count to `list_length + list_length + 1`.
        Otherwise use `list_length+1`.

        Args:
            item_count: the number of items in the memory model.
            context_feature_count: the number of unique units in context.
            learning_rate: the learning rate parameter.
            activation_scale: the activation scaling factor.
        """
        item_feature_count = item_count
        return cls(
            jnp.eye(item_feature_count, context_feature_count, 1) * (1 - learning_rate),
            activation_scale,
        )

    @classmethod
    def init_mcf(
        cls,
        item_count: int,
        context_feature_count: int,
        item_support: Float_,
        shared_support: Float_,
        activation_scale: Float_,
    ) -> "LinearMemory":
        """Return a new linear associative context-to-item memory model.

        Initially, in-list context units are associated with all items according to shared_support.
        They are also associated with a unique item according to item_support.
        Start-of-list and out-of-list context units receive no initial associations.

        To allow out-of-list contexts, set context_feature_count to `list_length + list_length + 1`.
        Otherwise use `list_length+1`.

        Args:
            item_count: the number of items in the memory model.
            context_feature_count: the number of unique units in context.
            item_support: initial association between an item and its own context feature
            shared_support: initial association between an item and all other contextual features
            activation_scale: the activation scaling factor.
        """
        base_memory = jnp.full((context_feature_count - 1, item_count), shared_support)
        base_memory = lax.fori_loop(
            0, item_count, lambda i, m: m.at[i, i].set(item_support), base_memory
        )
        start_list = jnp.zeros((1, item_count))
        return cls(jnp.vstack((start_list, base_memory)), activation_scale)
