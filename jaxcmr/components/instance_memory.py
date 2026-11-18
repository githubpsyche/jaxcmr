"""Instance-based memory operations for CMR.

Implements ``InstanceMemory``, which stores item-context
associations as separate memory traces (instances) rather than a
summed outer-product matrix, supporting instance-based retrieval
models.

"""

from typing import Self

from jax import numpy as jnp

from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.math import relative_power_scale
from jaxcmr.typing import Array, Context, Float, Float_, Int_, Parameters

__all__ = [
    "InstanceMemory",
    "init_mcf",
    "init_mfc",
]


class InstanceMemory(Pytree):
    """Associations stored as separate input/output traces.

    Each row of ``state`` concatenates an input pattern and its learned output.
    ``associate`` writes a new row, multiplying only the output by the learning
    rate. It returns a new memory and leaves the previous memory unchanged.

    For an input cue, each trace receives its input-pattern dot product as
    support. ``probe`` weights stored outputs by ``(support / max_support) **
    trace_activation_scale``; zero-support traces contribute nothing. All-zero
    support produces a zero output. This normalization applies at sensitivity
    one too. ``probe_without_scale`` instead uses raw dot-product weights.

    Callers supply finite, nonnegative supports and a positive sensitivity.
    Unused rows must be zero, and ``study_index`` must leave a free row for
    each association. The constructors below reserve one row per study event.

    Attributes:
        state: Rows of concatenated input and learned output patterns.
        _probe: Cue buffer; its output portion must be zero for input-only probes.
        study_index: Row to fill with the next association.
        trace_activation_scale: Positive power applied to relative trace support.
        output_size: Number of output features.
        input_size: Number of input features.
    """

    def __init__(
        self,
        state: Float[Array, " trace_count input_size+output_size"],
        probe: Float[Array, " input_size+output_size"],
        study_index: Int_,
        trace_activation_scale: Float_,
        input_size: int,
        output_size: int,
    ) -> None:
        self.state = state
        self.input_size = input_size
        self.output_size = output_size
        self.study_index = study_index
        self.trace_activation_scale = trace_activation_scale
        self._probe = probe

    def associate(
        self,
        in_pattern: Float[Array, " input_size"],
        out_pattern: Float[Array, " output_size"],
        learning_rate: Float_,
    ) -> Self:
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
            input_pattern: Full input/output cue. Use zeros in its output
                portion to probe only the stored inputs.

        The largest positive support has weight one at every sensitivity;
        unsupported traces, including unused rows, have weight zero.
        """
        activation = jnp.dot(self.state, input_pattern)
        return relative_power_scale(activation, self.trace_activation_scale)

    def probe(
        self,
        in_pattern: Float[Array, " input_size"],
    ) -> Float[Array, " output_size"]:
        """Return the output pattern associated with the input pattern in memory.

        Args:
            in_pattern: Input cue, with one value per input feature.
        """
        cue = self._probe.at[: in_pattern.size].set(in_pattern)
        trace_weights = self.trace_activations(cue)
        return jnp.dot(trace_weights, self.state)[in_pattern.size :]

    def probe_without_scale(
        self,
        in_pattern: Float[Array, " input_size"],
    ) -> Float[Array, " output_size"]:
        """Sum stored outputs weighted by raw input-cue dot products.

        This bypasses both normalization and the sensitivity power. It equals
        probing the summed outer-product association matrix represented by
        the stored traces.

        Args:
            in_pattern: Input cue, with one value per input feature.
        """
        cue = self._probe.at[: in_pattern.size].set(in_pattern)
        trace_weights = jnp.dot(self.state, cue)
        return jnp.dot(trace_weights, self.state)[in_pattern.size :]


def init_mfc(
    list_length: int,
    parameters: Parameters,
    context: Context,
) -> InstanceMemory:
    """Return a new instance-based item-to-context memory model.

    Initially, all items are associated with a unique context unit by
    `1-learning_rate`. We pre-allocate for `list_length` study events.
    `mfc_trace_sensitivity` sets the trace activation power.

    Args:
        list_length: the max number of study events that could occur.
        parameters: mapping of parameter names to values.
        context: context instance supplying feature dimension.
    """
    context_feature_count = context.size
    learning_rate = parameters["learning_rate"]
    trace_activation_scale = parameters.get("mfc_trace_sensitivity", 1.0)

    item_feature_count = list_length
    items = jnp.eye(list_length)
    contexts = jnp.eye(list_length, context_feature_count, 1) * (1 - learning_rate)
    presentations = (
        jnp.zeros((2 * list_length, item_feature_count + context_feature_count))
        .at[:list_length, :item_feature_count]
        .set(items)
    )

    return InstanceMemory(
        state=presentations.at[:list_length, item_feature_count:].set(contexts),
        probe=jnp.zeros(item_feature_count + context_feature_count),
        study_index=list_length,
        input_size=item_feature_count,
        output_size=context_feature_count,
        trace_activation_scale=trace_activation_scale,
    )


def init_mcf(
    list_length: int,
    parameters: Parameters,
    context: Context,
) -> InstanceMemory:
    """Return a new instance-based context-to-item memory model.

    Initially, in-list context units are associated with all items according to shared_support.
    They are also associated with a unique item according to item_support.
    Start-of-list and out-of-list context units receive no initial associations.
    We pre-allocate for `list_length` study events.

    Args:
        list_length: the maximum number of study events that could occur.
        parameters: mapping of parameter names to values.
        context: context instance supplying feature dimension.
    """
    context_feature_count = context.size
    item_support = parameters["item_support"]
    shared_support = parameters["shared_support"]
    trace_activation_scale = parameters["mcf_trace_sensitivity"]

    item_feature_count = list_length
    contexts = jnp.eye(list_length, context_feature_count, 1)
    items = (jnp.eye(list_length) * (item_support - shared_support)) + shared_support
    presentations = (
        jnp.zeros((2 * list_length, context_feature_count + item_feature_count))
        .at[:list_length, :context_feature_count]
        .set(contexts)
    )
    return InstanceMemory(
        state=presentations.at[:list_length, context_feature_count:].set(items),
        probe=jnp.zeros(item_feature_count + context_feature_count),
        study_index=list_length,
        input_size=context_feature_count,
        output_size=item_feature_count,
        trace_activation_scale=trace_activation_scale,
    )
