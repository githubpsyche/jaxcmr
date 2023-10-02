import jax.numpy as jnp
from jax import vmap, jit, lax
from simple_pytree import Pytree, dataclass
from jaxcmr.helpers import Bool, Array, ScalarInteger, ScalarBool, replace, Callable
from plum import dispatch


def crp(recalls, list_length):
    lag_range = list_length - 1
    total_actual_lags = jnp.zeros(lag_range * 2 + 1)
    total_possible_lags = jnp.zeros(lag_range * 2 + 1)
    terminus = jnp.sum(recalls != 0, axis=1)

    def update_lags(i, totals):
        actual_lags, possible_lags = totals
        possible_items = jnp.arange(list_length) + 1
        previous_item = 0
        term = terminus[i]

        for recall_index in range(term):
            if recall_index > 0:
                current_lag = recalls[i, recall_index] - previous_item + lag_range
                actual_lags = jax.ops.index_add(actual_lags, jax.ops.index[current_lag], 1)

                possible_lags_range = possible_items - previous_item + lag_range
                possible_lags = jax.ops.index_add(possible_lags, jax.ops.index[possible_lags_range], 1)

            previous_item = recalls[i, recall_index]
            possible_items = possible_items[possible_items != previous_item]

        return actual_lags, possible_lags

    total_actual_lags, total_possible_lags = jax.lax.fori_loop(0, len(recalls), update_lags, (total_actual_lags, total_possible_lags))

    total_possible_lags = jax.ops.index_add(total_possible_lags, jax.ops.index[total_actual_lags == 0], 1)

    return total_actual_lags / total_possible_lags

import jax



# %% Data Class


@dataclass
class CRP(Pytree, mutable=True):
    def __init__(self, item_count):
        self.recall_mask = jnp.ones(self.item_count, jnp.bool_)
        self.recall_total = 0


# %% Single Event Tabulation (and Simulation)


def tabulate_and_simulate_retrieval(
    state: CRP, choice: ScalarInteger, tabulate_fn: Callable
) -> CRP:
    """Tabulate and simulate a specified retrieval outcome given a model state"""
    return retrieve(state, choice), tabulate_fn(state, choice)


# %% Incrementing Analysis State


def _retrieve_item(state: CRP, choice: ScalarInteger) -> CRP:
    """Retrieve item with index choice-1"""

    return replace(
        state,
        recall_mask=state.recall_mask.at[choice - 1].set(True),
        recall_total=state.recall_total + 1,
    )


def stop_recall(state: CRP, _: ScalarInteger = 0) -> CRP:
    """The model shifts to inactive mode and will not retrieve any more items"""
    return replace(state, is_active=False)


def retrieve(state: CRP, choice: ScalarInteger) -> CRP:
    """Perform retrieval event, either item recall (choice > 0) or termination (choice = 0)"""
    return lax.cond(choice > 0, _retrieve_item, stop_recall, state, choice)


# %% Tabulating Lag Transition Rates


def possible_transitions(state: CRP, list_length) -> Bool[Array, "possible_lags"]:
    """Return a boolean array indicating which lag transitions are possible"""

    lag_range = list_length - 1
    possible_lags = jnp.zeros(lag_range * 2 + 1)
    CRP()
