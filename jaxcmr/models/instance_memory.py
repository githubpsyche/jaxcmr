from jax import numpy as jnp
from simple_pytree import Pytree

from jaxcmr.math import power_scale
from jaxcmr.typing import Array, Float, Float_, Int_


class InstanceMemory(Pytree):
    """Instance-based memory model for CMR.

    Attributes:
        state: the current state of the memory.
        _probe: pre-allocated probe array for memory retrieval.
        study_index: index for the next trace to be stored.
        feature_activation_scale: the scaling factor for activated output features.
        trace_activation_scale: the scaling factor for activated traces.
        output_size: the size of the output representation.
        input_size: the size of the input representation.
    """

    def __init__(
        self,
        state: Float[Array, " trace_count input_size+output_size"],
        probe: Float[Array, " input_size+output_size"],
        study_index: Int_,
        feature_activation_scale: Float_,
        trace_activation_scale: Float_,
        input_size: int,
        output_size: int,
    ):
        self.state = state
        self.input_size = input_size
        self.output_size = output_size
        self.study_index = study_index
        self.feature_activation_scale = feature_activation_scale
        self.trace_activation_scale = trace_activation_scale
        self._probe = probe

    def associate(
        self,
        in_pattern: Float[Array, " input_size"],
        out_pattern: Float[Array, " output_size"],
        learning_rate: Float_,
    ) -> "InstanceMemory":
        """Return the updated memory after associating input and output patterns.

        Args:
            in_pattern: a feature pattern for an input.
            out_pattern: a feature pattern for an output.
            learning_rate: the learning rate parameter.
        """
        return self.replace(
            state=self.state.at[self.study_index].set(
                jnp.concatenate((in_pattern, out_pattern * learning_rate))
            ),
            study_index=self.study_index + 1,
        )

    def trace_activations(
        self, input_pattern: Float[Array, " input_size+output_size"]
    ) -> Float[Array, " trace_count"]:
        """Return the activations of each trace based on the input pattern.

        Args:
            input_pattern: the input feature pattern.
            trace_activation_scale: the scaling factor for activated traces.
        """
        activation = jnp.dot(self.state, input_pattern)
        return power_scale(activation, self.trace_activation_scale)

    def probe(
        self,
        in_pattern: Float[Array, " input_size"],
    ) -> Float[Array, " output_size"]:
        """Return the output pattern associated with the input pattern in memory.

        Args:
            input_pattern: the input feature pattern.
        """
        t = self.trace_activations(self._probe.at[: in_pattern.size].set(in_pattern))
        a = jnp.dot(t, self.state)[in_pattern.size :]
        return power_scale(a, self.feature_activation_scale)
    

    def probe_without_scale(
        self,
        in_pattern: Float[Array, " input_size"],
    ) -> Float[Array, " output_size"]:
        """Return the output pattern associated with the input pattern in memory.

        Args:
            input_pattern: the input feature pattern.
        """
        t = self.trace_activations(self._probe.at[: in_pattern.size].set(in_pattern))
        return jnp.dot(t, self.state)[in_pattern.size :]
    
    def zero_out(
        self,
        index: Int_,
    ) -> "InstanceMemory":
        """Return the updated memory after zeroing out associations for a given input index.

        Args:
            index: the index to zero out.
        """
        ...

    @classmethod
    def init_mfc(
        cls,
        list_length: int,
        context_feature_count: int,
        size: int,
        learning_rate: Float_,
        feature_activation_scale: Float_,
        trace_activation_scale: Float_,
    ) -> "InstanceMemory":
        """Return a new instance-based item-to-context memory model.

        Initially, all items are associated with a unique context unit by `1-learning_rate`.
        We pre-allocate for `list_length` study events.

        To allow multiplex traces, set size to `list_length + list_length + list_length`.
        To allow out-of-list contexts, set context_feature_count to `list_length + list_length + 1`.

        Args:
            list_length: the max number of study events that could occur.
            context_feature_count: the number of unique units in context.
            size: maximum number of additional traces to allow to be stored in memory.
            learning_rate: the learning rate parameter.
            feature_activation_scale: the activation scaling factor for output features.
            trace_activation_scale: the activation scaling factor for traces.
        """
        item_feature_count = list_length
        items = jnp.eye(list_length)
        contexts = jnp.eye(list_length, context_feature_count, 1) * (1 - learning_rate)
        presentations = (
            jnp.zeros((list_length + size, item_feature_count + context_feature_count))
            .at[:list_length, :item_feature_count]
            .set(items)
        )

        return cls(
            state=presentations.at[:list_length, item_feature_count:].set(contexts),
            probe=jnp.zeros(item_feature_count + context_feature_count),
            study_index=list_length,
            input_size=item_feature_count,
            output_size=context_feature_count,
            feature_activation_scale=feature_activation_scale,
            trace_activation_scale=trace_activation_scale,
        )

    @classmethod
    def init_mcf(
        cls,
        list_length: int,
        context_feature_count: int,
        size: int,
        item_support: Float_,
        shared_support: Float_,
        feature_activation_scale: Float_,
        trace_activation_scale: Float_,
    ) -> "InstanceMemory":
        """Return a new instance-based context-to-item memory model.

        Initially, in-list context units are associated with all items according to shared_support.
        They are also associated with a unique item according to item_support.
        Start-of-list and out-of-list context units receive no initial associations.
        We pre-allocate for `list_length` study events.

        To allow multiplex traces, set size to `list_length + list_length + list_length`.
        To allow out-of-list contexts, set context_feature_count to `list_length + list_length + 1`.
        Otherwise, use `list_length` and `list_length + 1` respectively.

        Args:
            list_length: the max number of study events that could occur.
            context_feature_count: the number of unique units in context.
            size: maximum number of additional traces to allow to be stored in memory.
            item_support: initial association between an item and its own context feature
            shared_support: initial association between an item and all other contextual features
            feature_activation_scale: the activation scaling factor for output features.
            trace_activation_scale: the activation scaling factor for traces.
        """
        item_feature_count = list_length
        contexts = jnp.eye(list_length, context_feature_count, 1)
        items = (
            jnp.eye(list_length) * (item_support - shared_support)
        ) + shared_support
        presentations = (
            jnp.zeros((size + list_length, context_feature_count + item_feature_count))
            .at[:list_length, :context_feature_count]
            .set(contexts)
        )
        return cls(
            state=presentations.at[:list_length, context_feature_count:].set(items),
            probe=jnp.zeros(item_feature_count + context_feature_count),
            study_index=list_length,
            input_size=context_feature_count,
            output_size=item_feature_count,
            feature_activation_scale=feature_activation_scale,
            trace_activation_scale=trace_activation_scale,
        )
