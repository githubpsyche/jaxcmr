"""Simulation utilities for memory-search models.

Provides helpers to run study and free-recall simulations per trial and
per subject, and to generate HDF5-shaped datasets from model outputs.
"""

from typing import Mapping, Optional, Sequence, Type

import jax.numpy as jnp
from jax import lax, random, vmap
from jax.tree_util import tree_map

from jaxcmr.typing import (
    Array,
    Bool,
    Float,
    Float_,
    Int_,
    Integer,
    MemorySearch,
    MemorySearchModelFactory,
    PRNGKeyArray,
    RecallDataset,
)


def item_to_study_positions(
    item: Int_,
    presentation: Integer[Array, " list_length"],
    size: int,
) -> Integer[Array, " size"]:
    """Returns one-indexed study positions for an item in a sequence.

    Args:
      item: Item index.
      presentation: Presentation sequence for a single trial.
      size: Number of nonzero matches to return.
    """
    return lax.cond(
        item == 0,
        lambda: jnp.zeros(size, dtype=int),
        lambda: jnp.nonzero(presentation == item, size=size, fill_value=-1)[0] + 1,
    )


def _single_free_recall(
    model: MemorySearch, rng: PRNGKeyArray
) -> tuple[MemorySearch, Integer[Array, ""]]:
    """Returns model and choice after a free-recall event.

    Args:
      model: Retrieval-ready memory search model.
      rng: Random key.
    """
    p_all = model.outcome_probabilities()
    choice = random.choice(rng, p_all.shape[0], p=p_all)
    return model.retrieve(choice), choice


def _maybe_free_recall(
    model: MemorySearch, rng: PRNGKeyArray
) -> tuple[MemorySearch, Integer[Array, ""]]:
    """Returns model and choice for a single step if active; otherwise no-op."""
    return lax.cond(
        model.is_active,
        _single_free_recall,
        lambda m, _: (m, 0),
        model,
        rng,
    )


def simulate_free_recall(
    model: MemorySearch, list_length: int, rng: PRNGKeyArray
) -> tuple[MemorySearch, Integer[Array, " recall_events"]]:
    """Returns model and choices from repeated free-recall steps.

    Args:
      model: Retrieval-ready memory search model.
      list_length: Upper bound on recall-event steps to simulate.
      rng: Random key.
    """
    return lax.scan(_maybe_free_recall, model, random.split(rng, list_length))


def simulate_study_and_free_recall(
    model: MemorySearch, present: Integer[Array, " study_events"], rng: PRNGKeyArray
) -> tuple[MemorySearch, Integer[Array, " recall_events"]]:
    """Simulates study then free recall for a single trial.

    Args:
      model: Memory search model in study mode.
      present: One-indexed study sequence for the trial.
      rng: Random key.
    """
    model = lax.fori_loop(0, present.size, lambda i, m: m.experience(present[i]), model)
    model = model.start_retrieving()
    return simulate_free_recall(model, present.size, rng)


class MemorySearchSimulator:
    """Stateless trial-level simulator usable with vmap over trials."""

    def __init__(
        self,
        model_factory: Type[MemorySearchModelFactory],
        dataset: RecallDataset,
        connections: Optional[Integer[Array, " word_pool_items word_pool_items"]],
    ) -> None:
        """Initializes trial-conditioned model creation from dataset and connections."""
        factory = model_factory(dataset, connections)
        self.create_model = factory.create_trial_model
        self.present_lists = jnp.array(dataset["pres_itemnos"])
        self.empty = jnp.zeros(dataset["recalls"].shape[-1], jnp.int32)

    def simulate_trial(
        self,
        trial_index: Integer[Array, ""],
        subject_index: Integer[Array, " subject_count"],
        parameters: Mapping[str, Float_],
        rng: PRNGKeyArray,
    ) -> Integer[Array, " recall_events"]:
        """Returns recalled item indices for a single trial.

        Uses per-subject parameters to create the trial model and simulates
        study plus free recall for that trial.
        """
        model = self.create_model(
            trial_index, tree_map(lambda p: p[subject_index], parameters)
        )
        return simulate_study_and_free_recall(
            model, self.present_lists[trial_index], rng
        )[1]


def preallocate_for_h5_dataset(
    data: RecallDataset, trial_mask: Bool[Array, " trial_count"], experiment_count: int
) -> RecallDataset:
    """Returns a dataset dict replicated by experiment_count for selected trials.

    Arrays are replicated for each key in the input. Optional keys linked to
    recall behavior may not generalize across replication.

    Args:
      data: Input dataset.
      trial_mask: Mask selecting trials to include.
      experiment_count: Replication count per trial.
    """
    return tree_map(
        lambda x: jnp.repeat(x[trial_mask], experiment_count, axis=0),
        data,
    )


def simulate_h5_from_h5(
    model_factory: Type[MemorySearchModelFactory],
    dataset: RecallDataset,
    connections: Optional[Integer[Array, " word_pool_items word_pool_items"]],
    parameters: dict[str, Float[Array, " subject_count"]],
    trial_mask: Bool[Array, " trial_count"],
    experiment_count: int,
    rng: PRNGKeyArray,
    size: int = 3,
) -> RecallDataset:
    """Returns a simulated dataset using per-subject parameters from an H5-like dataset.

    Args:
      model_factory: Factory class for memory search models.
      dataset: Original dataset containing trial data.
      connections: Optional connectivity among word-pool items.
      parameters: Simulation parameters per subject.
      trial_mask: Mask selecting trials to simulate.
      experiment_count: Number of replications per selected trial.
      rng: Random key.
      size: Max number of study positions to keep when reindexing recalls.
    """

    sim_h5 = preallocate_for_h5_dataset(dataset, trial_mask, experiment_count)
    simulator = MemorySearchSimulator(model_factory, sim_h5, connections)

    # Flat trial + subject index vectors (static shapes)
    total_trials = sim_h5["subject"].size
    trial_indices = jnp.arange(total_trials, dtype=jnp.int32)
    subject_indices = sim_h5["subject"].flatten()
    rngs = random.split(rng, total_trials)

    # One jit-compiled vmap over trials
    recalls = vmap(simulator.simulate_trial, in_axes=(0, 0, None, 0))(
        trial_indices, subject_indices, parameters, rngs
    )

    # Reindex study positions
    sim_h5["recalls"] = vmap(
        vmap(item_to_study_positions, in_axes=(0, None, None)), in_axes=(0, 0, None)
    )(recalls, sim_h5["pres_itemnos"], size)[:, :, 0]
    return sim_h5


def parameter_shifted_simulate_h5_from_h5(
    model_factory: Type[MemorySearchModelFactory],
    dataset: RecallDataset,
    connections: Optional[Integer[Array, " word_pool_items word_pool_items"]],
    parameters: dict[str, Float[Array, " subject_count"]],
    trial_mask: Bool[Array, " trial_count"],
    experiment_count: int,
    varied_parameter: str,
    parameter_values: Sequence[float],
    rng: PRNGKeyArray,
    size: int = 3,
) -> Sequence[RecallDataset]:
    """Returns multiple simulated datasets by sweeping a single parameter.

    Args:
      model_factory: Factory class for memory search models.
      dataset: Original dataset containing trial data.
      connections: Optional connectivity among word-pool items.
      parameters: Simulation parameters per subject.
      trial_mask: Mask selecting trials to simulate.
      experiment_count: Number of replications per selected trial.
      varied_parameter: Name of the parameter to sweep.
      parameter_values: Values to assign to the swept parameter.
      rng: Random key.
      size: Max number of study positions to keep when reindexing recalls.
    """

    results: list[RecallDataset] = []
    for value in parameter_values:
        rng_key, this_rng = random.split(rng)
        swept_params = {
            **parameters,
            varied_parameter: jnp.full_like(parameters[varied_parameter], value),
        }
        sim_data = simulate_h5_from_h5(
            model_factory,
            dataset,
            connections,
            swept_params,
            trial_mask,
            experiment_count,
            this_rng,
            size,
        )
        results.append(sim_data)

    return results
