"""Termination policies for memory search models."""

from __future__ import annotations

from jax import lax
from jax import numpy as jnp

from jaxcmr.math import lb
from jaxcmr.typing import Array, Float, Float_, MemorySearch, TerminationPolicy

__all__ = [
    "NoStopTermination",
    "PositionalTermination",
    "SupportRatioTermination",
    "RetrievalDependentTermination",
]


class NoStopTermination(TerminationPolicy):
    """Termination probability is always zero as long as recalls remain."""

    def __init__(self) -> None:
        pass

    def stop_probability(self, model: MemorySearch) -> Float[Array, ""]:
        total_recallable = jnp.sum(model.recallable)
        return lax.cond(
            jnp.logical_or(total_recallable == 0, ~model.is_active),
            lambda: jnp.array(1.0),
            lambda: jnp.array(0.0),
        )


class PositionalTermination(TerminationPolicy):
    """Termination probability is an exponential function of recall position."""

    def __init__(
        self,
        scale: Float_,
        growth: Float_,
        recall_capacity: int,
    ):
        self._stop_probability = scale * jnp.exp(jnp.arange(recall_capacity) * growth)

    def stop_probability(self, model: MemorySearch) -> Float[Array, ""]:
        total_recallable = jnp.sum(model.recallable)
        return lax.cond(
            jnp.logical_or(total_recallable == 0, ~model.is_active),
            lambda: jnp.array(1.0),
            lambda: jnp.minimum(
                1.0 - (lb * total_recallable),
                self._stop_probability[model.recall_total],
            ),
        )


class RetrievalDependentTermination(TerminationPolicy):
    """Termination probability is the probability of retrieving a STOP item."""

    def __init__(self) -> None:
        pass

    def stop_probability(self, model: MemorySearch) -> Float[Array, ""]:
        total_recallable = jnp.sum(model.recallable)
        return lax.cond(
            jnp.logical_or(total_recallable == 0, ~model.is_active),
            lambda: jnp.array(1.0),
            lambda: jnp.minimum(
                1.0 - (lb * total_recallable),
                model.outcome_probability(model.item_count),
            ),
        )


class SupportRatioTermination(TerminationPolicy):
    """Termination probability is based on ratio of recalled to not recalled support."""

    def __init__(self, scale: Float_, growth: Float_):
        self.scale = scale
        self.growth = growth

    def stop_probability(self, model: MemorySearch) -> Float[Array, ""]:
        total_recallable = jnp.sum(model.recallable)

        def compute_stop_probability() -> Float[Array, ""]:
            """Returns stop probability derived from support ratios."""
            support = model.mcf.probe(model.context.state)
            recalled_mask = jnp.logical_and(model.studied, ~model.recallable)
            not_recalled_mask = jnp.logical_and(model.studied, model.recallable)
            recalled_support = jnp.sum(support * recalled_mask)
            not_recalled_support = jnp.sum(support * not_recalled_mask)
            support_ratio = lax.cond(
                recalled_support > 0.0,
                lambda: not_recalled_support / recalled_support,
                lambda: jnp.inf,
            )
            stop_probability = self.scale + jnp.exp(-self.growth * support_ratio)
            return jnp.minimum(
                stop_probability,
                1.0 - (lb * total_recallable),
            )

        return lax.cond(
            jnp.logical_or(total_recallable == 0, ~model.is_active),
            lambda: jnp.array(1.0),
            compute_stop_probability,
        )
