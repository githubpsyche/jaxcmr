"""Temporal context is the cue a CMR model uses to retrieve items. At the start
of a list, only the start-of-list unit is active. Studying or recalling an
item can move context toward the pattern associated with that item.

Use `context.init` when the model supplies only start-of-list and item cues.
`TemporalContext.init_expanded` also reserves one fresh unit per item.
Outlist CMR uses a fresh unit when it studies an item again. Source-switch
models use one when a change of source triggers drift. Each call to
`integrate_with_outlist` consumes one of these units.
"""

__all__ = ["OutlistContext", "OutlistContextCreateFn", "TemporalContext", "integrate_normalized"]

import base64
import io
from typing import Protocol

from jax import numpy as jnp

from jaxcmr._vendor.simple_pytree import Pytree

from ..math import normalize_magnitude
from ..typing import Array, Context, ContextCreateFn, Float, Float_


class OutlistContext(Context, Protocol):
    """Context that can mix in a fresh out-of-list cue during study.

    Outlist CMR and the source-switch models call `integrate_with_outlist`
    in addition to the ordinary `Context.integrate` operation.
    """

    def integrate_with_outlist(
        self,
        inlist_input: Float[Array, " context_feature_units"],
        ratio: Float_,
        drift_rate: Float_,
    ) -> "OutlistContext":
        """Integrate in-list input with weight ratio and fresh input with 1-ratio.

        The context must have enough out-of-list capacity for this update.
        """
        ...


class OutlistContextCreateFn(ContextCreateFn, Protocol):
    """Create an `OutlistContext` from an item count."""

    def __call__(self, item_count: int) -> OutlistContext:
        """Return context with fresh units available for out-of-list updates."""
        ...


def integrate_normalized(
    state: Float[Array, " units"],
    context_input: Float[Array, " units"],
    drift_rate: Float_,
) -> Float[Array, " units"]:
    """Update a context vector from a unit-length or zero input.

    Parameters
    ----------
    state : floating array, shape (units,)
        Current context.
    context_input : floating array, shape (units,)
        Unit-length input pattern or an all-zero vector.
    drift_rate : float
        Weight given to the input, between zero and one.

    Returns
    -------
    updated_state : floating array, shape (units,)
        With unit-length state and input, the result has unit length.
        A zero input scales the state by `sqrt(1 - drift_rate**2)`.
    """
    overlap = jnp.dot(state, context_input)
    rho = jnp.sqrt(1 + drift_rate**2 * (overlap**2 - 1)) - drift_rate * overlap
    return rho * state + drift_rate * context_input


class TemporalContext(Pytree):
    """Temporal context for a single trial.

    Unit zero represents the start of the list. Units 1 through `item_count`
    represent the items. Any remaining units are available for out-of-list
    input, in the order in which the model uses them. An update returns a
    new context. Its `initial_state` still holds the start-of-list pattern.

    Parameters
    ----------
    item_count : int
        Number of item units.
    size : int
        Total number of units, including the start and any out-of-list units.
    """

    def __init__(self, item_count: int, size: int) -> None:
        """Set the start-of-list unit to one and all other units to zero."""
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
        """Move the current state toward an input pattern.

        Parameters
        ----------
        context_input : floating array, shape (context_feature_units,)
            Input in context coordinates. A nonzero pattern is scaled to unit
            length before integration.
        drift_rate : float
            Weight given to the input, between zero and one.

        Returns
        -------
        TemporalContext
            Updated context with the same next out-of-list unit. A zero input
            scales the current state by `sqrt(1 - drift_rate**2)`.
        """
        context_input = normalize_magnitude(context_input)
        return self.replace(
            state=integrate_normalized(self.state, context_input, drift_rate)
        )

    def _repr_markdown_(self) -> str:
        """Returns a markdown ``img`` tag of the current state."""
        import matplotlib.pyplot as plt

        from ..plotting import matrix_heatmap

        fig, ax = plt.subplots(figsize=(6, 0.6))
        matrix_heatmap(self.state, axis=ax, colorbar=False, xlabel="", ylabel="")
        ax.set_yticks([])

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)

        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f'<img src="data:image/png;base64,{encoded}" />'

    @property
    def outlist_input(self) -> Float[Array, " context_feature_units"]:
        """One-hot cue for the next unused out-of-list unit.

        Reading the cue leaves it available for the next update. The context
        must have an out-of-list unit remaining.
        """
        return self.zeros.at[self.next_outlist_unit].set(1)

    def integrate_with_outlist(
        self,
        inlist_input: Float[Array, " context_feature_units"],
        ratio: Float_,
        drift_rate: Float_,
    ) -> "TemporalContext":
        """Mix an item cue with a fresh out-of-list cue and update context.

        Parameters
        ----------
        inlist_input : floating array, shape (context_feature_units,)
            In-list input in context coordinates. A nonzero input is scaled
            to unit length before mixing.
        ratio : float
            Weight on the in-list cue. The fresh unit receives weight
            `1 - ratio`. The mixture is normalized before integration.
        drift_rate : float
            Weight given to the mixed cue, between zero and one.

        Returns
        -------
        TemporalContext
            Updated context with the next out-of-list index advanced by one.
            Each call requires one unused unit, even when `ratio` is one or
            `drift_rate` is zero.
        """
        context_input = normalize_magnitude(
            (normalize_magnitude(inlist_input) * ratio)
            + (self.outlist_input * (1 - ratio))
        )
        # Square Python scalar rates at JAX precision, as for array rates.
        return self.replace(
            state=integrate_normalized(self.state, context_input, jnp.asarray(drift_rate)),
            next_outlist_unit=self.next_outlist_unit + 1,
        )

    @classmethod
    def init_expanded(cls, item_count: int) -> "TemporalContext":
        """Create context with one fresh out-of-list unit per item.

        Parameters
        ----------
        item_count : int
            Number of item units and number of available out-of-list updates.

        Returns
        -------
        TemporalContext
            Context with `1 + 2 * item_count` units. The first fresh unit has
            index `item_count + 1`.
        """
        return cls(item_count, item_count + item_count + 1)


def init(item_count: int) -> "TemporalContext":
    """Create context with one start-of-list unit and one unit per item.

    Parameters
    ----------
    item_count : int
        Number of item units.

    Returns
    -------
    TemporalContext
        Context with `1 + item_count` units. Use this constructor for models
        that call only `integrate`.
    """
    return TemporalContext(item_count, item_count + 1)
