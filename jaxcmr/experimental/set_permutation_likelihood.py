"""Estimate the likelihood of recalling specified items in any order.

For observed items A and B, ``RecallSetLikelihoodLoss`` scores the event
“A, B, STOP or B, A, STOP.” ``RecallPrefixLikelihoodLoss`` scores the event
“the first two responses are A and B in either order.” Later responses
are unrestricted in the prefix calculation. Repeated items retain their
observed multiplicities.

Both losses sample orders uniformly and multiply the mean sequence
probability by the number of distinct orders. This estimates the sum
over orders without enumerating them all. The probability estimate is
unbiased, though a finite sample can exceed one. Taking its logarithm
introduces sampling bias into the fitted score.
"""

from math import lgamma

import jax.numpy as jnp
import numpy as np
from jax import random, vmap
from jax.scipy.special import logsumexp

from jaxcmr.loss.sequence_likelihood import (
    MemorySearchLikelihoodLoss,
    sequence_probabilities,
)
from jaxcmr.simulation import simulate_study
from jaxcmr.typing import (
    Array,
    Float,
    Integer,
    MemorySearchModelFactory,
    Parameters,
    PRNGKeyArray,
    RecallDataset,
)

__all__ = ["RecallPrefixLikelihoodLoss", "RecallSetLikelihoodLoss"]


def _sample_permutations(
    key: PRNGKeyArray, items: Integer[Array, " recall_events"], count: int
) -> Integer[Array, " permutations recall_events"]:
    permutations = vmap(lambda draw: random.permutation(draw, items))(
        random.split(key, count)
    )
    order = jnp.argsort(permutations == 0, axis=-1, stable=True)
    return jnp.take_along_axis(permutations, order, axis=-1)


class RecallSetLikelihoodLoss(MemorySearchLikelihoodLoss):
    """Estimate the negative log probability of complete unordered recalls.

    Parameters
    ----------
    model_factory_cls : type[MemorySearchModelFactory]
        Factory for a trial-specific memory-search model.
    dataset : RecallDataset
        Position-coded recalls. Repeated responses are retained with their
        multiplicities; unused slots are zero. The supplied model must support
        any repeated responses being scored.
    features : Array or None
        Vocabulary features passed to the factory.
    permutation_count : int, optional
        Uniform draws per trial, default 50, fixed across parameter candidates.
    seed : int, optional
        Seed for the sampled orders, default zero.

    Notes
    -----
    A STOP event is appended even when the observation array has no padding.
    An empty complete recall scores immediate STOP. For k items with counts
    m_i, the multiplier is k! / product(m_i!). Calculations use log space.
    """

    _include_stop = True

    def __init__(
        self,
        model_factory_cls: type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Float[Array, " word_pool_items features_count"] | None,
        *,
        permutation_count: int = 50,
        seed: int = 0,
    ) -> None:
        super().__init__(model_factory_cls, dataset, features)
        # An extra slot makes complete-sequence scoring independent of storage padding.
        items = jnp.pad(self.trials, ((0, 0), (0, 1)))
        keys = random.split(random.PRNGKey(seed), items.shape[0])
        self.trial_permutations = vmap(_sample_permutations, in_axes=(0, 0, None))(
            keys, items, permutation_count
        )
        totals = jnp.count_nonzero(items, axis=1)
        self.trial_masks = (
            jnp.arange(items.shape[1])[None, :]
            < (totals + int(self._include_stop))[:, None]
        )
        self.log_order_counts = jnp.asarray(
            [
                lgamma(np.count_nonzero(row) + 1)
                - sum(
                    lgamma(int(n) + 1)
                    for n in np.unique(row[row != 0], return_counts=True)[1]
                )
                for row in np.asarray(items)
            ]
        )

    def predict_trial(
        self, trial_index: Integer[Array, ""], parameters: Parameters
    ) -> Float[Array, ""]:
        """Return the log of the sampled unordered-event probability."""
        model = self.create_model(trial_index, parameters)
        model = simulate_study(
            model, self.present_lists[trial_index]
        ).start_retrieving()
        probabilities = vmap(sequence_probabilities, in_axes=(None, 0))(
            model, self.trial_permutations[trial_index]
        )
        logs = jnp.log(
            jnp.where(self.trial_masks[trial_index], probabilities, 1.0)
        ).sum(axis=-1)
        return self.log_order_counts[trial_index] + logsumexp(logs) - jnp.log(logs.size)

    def present_and_predict_trials_loss(
        self, trial_indices: Integer[Array, " trials"], parameters: Parameters
    ) -> Float[Array, ""]:
        """Sum negative log estimates for the selected trials."""
        return -vmap(self.predict_trial, in_axes=(0, None))(
            trial_indices, parameters
        ).sum()


class RecallPrefixLikelihoodLoss(RecallSetLikelihoodLoss):
    """Estimate unordered-prefix probability, without scoring subsequent STOP.

    Constructor options are identical to RecallSetLikelihoodLoss. The event is
    that the first k responses contain the observed multiset; subsequent recall
    is unrestricted. An empty prefix has probability one. Item probabilities
    still include the supplied model's stopping rule.
    """

    _include_stop = False
