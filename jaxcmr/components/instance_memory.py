"""Instance memory stores each association as a separate trace. A retrieval
cue gives each trace a weight based on its match to the stored input.
The response is the sum of the stored outputs multiplied by those weights.

`probe` scales the strongest positive match to one and raises each
relative match to the trace sensitivity. `probe_without_scale` uses the raw
matches, giving the same result as a linear memory built from those traces.
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
    """A table of input patterns and their learned outputs.

    Parameters
    ----------
    state : floating array, shape (trace_count, input_size + output_size)
        Each row contains the input followed by its learning-rate-weighted output.
        Unused rows must be zero.
    probe : floating array, shape (input_size + output_size,)
        Cue buffer. Its output portion must be zero for an input-only probe.
    study_index : int
        Next row to fill. Each association needs one free row.
    trace_activation_scale : float
        Positive exponent for relative trace supports. Cue supports must be finite
        and nonnegative.
    input_size : int
        Number of input features at the start of each row.
    output_size : int
        Number of output features following the input.
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
        """Store one input pattern and its weighted output in the next free trace.

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
        InstanceMemory
            Updated memory with the next free row advanced by one.
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
        """Calculate the weight of each trace for a full cue.

        Parameters
        ----------
        input_pattern : floating array, shape (input_size + output_size,)
            Cue for a dot product with each stored row. Set its output features to
            zero to match only the stored inputs.

        Returns
        -------
        floating array, shape (trace_count,)
            Positive supports receive `(support / max_support) **
            trace_activation_scale`. Zero supports receive zero, including when
            every support is zero.
        """
        activation = jnp.dot(self.state, input_pattern)
        return relative_power_scale(activation, self.trace_activation_scale)

    def probe(
        self,
        in_pattern: Float[Array, " input_size"],
    ) -> Float[Array, " output_size"]:
        """Retrieve outputs using relative trace weights.

        Parameters
        ----------
        in_pattern : floating array, shape (input_size,)
            Input pattern.

        Returns
        -------
        floating array, shape (output_size,)
            Sum of stored outputs weighted by `trace_activations` for the input cue.
        """
        cue = self._probe.at[: in_pattern.size].set(in_pattern)
        trace_weights = self.trace_activations(cue)
        return jnp.dot(trace_weights, self.state)[in_pattern.size :]

    def probe_without_scale(
        self,
        in_pattern: Float[Array, " input_size"],
    ) -> Float[Array, " output_size"]:
        """Retrieve outputs using raw cue-to-input dot products.

        Parameters
        ----------
        in_pattern : floating array, shape (input_size,)
            Input pattern.

        Returns
        -------
        floating array, shape (output_size,)
            Sum of stored outputs weighted by their raw matches. This equals a
            probe of the summed outer-product matrix for the same traces.
        """
        cue = self._probe.at[: in_pattern.size].set(in_pattern)
        trace_weights = jnp.dot(self.state, cue)
        return jnp.dot(trace_weights, self.state)[in_pattern.size :]


def init_mfc(
    list_length: int,
    parameters: Parameters,
    context: Context,
) -> InstanceMemory:
    """Create item-to-context traces with one initial trace per item.

    Parameters
    ----------
    list_length : int
        Number of item slots in the trial.
    parameters : Parameters
        The matching context unit has weight `1 - learning_rate`.
        `mfc_trace_sensitivity` defaults to one.
    context : Context
        Context whose size sets the number of context units.

    Returns
    -------
    InstanceMemory
        Memory with room for `list_length` further associations. Each probe
        returns a pattern of length `context.size`.
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
    """Create context-to-item traces with one initial trace per in-list context unit.

    Parameters
    ----------
    list_length : int
        Number of item slots in the trial.
    parameters : Parameters
        The matching item has weight `item_support` and the other items have
        `shared_support`. `mcf_trace_sensitivity` sets the exponent for this
        direction.
    context : Context
        Context whose size sets the number of context units.

    Returns
    -------
    InstanceMemory
        Memory with room for `list_length` further associations. Start-of-list and
        fresh out-of-list cues initially give every item zero support.
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
