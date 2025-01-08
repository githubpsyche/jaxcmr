from typing import Optional, Type, Mapping

import numpy as np
from jax import lax, random, vmap
from jax import numpy as jnp
from jax.tree_util import tree_map

from jaxcmr.typing import (
    Array,
    Bool,
    Float,
    Float_,
    Int_,
    Integer,
    MemorySearchModelFactory,
    MemorySearch,
    PRNGKeyArray,
)


def segment_by_index(vector, index_vector) -> tuple[list[np.ndarray], np.ndarray]:
    """Segments a vector by index vector."""
    unique_indices, first_indices = np.unique(index_vector, return_index=True)
    unique_indices = unique_indices[np.argsort(first_indices)]
    return [vector[index_vector == idx] for idx in unique_indices], unique_indices


def item_to_study_positions(
    item: Int_,
    presentation: Integer[Array, " list_length"],
    size: int,
):
    """Return the one-indexed study positions of an item in a 1D presentation sequence.

    Args:
        item: the item index.
        presentation: the 1D presentation sequence.
        size: number of non-zero entries to return.
    """
    return lax.cond(
        item == 0,
        lambda: jnp.zeros(size, dtype=int),
        lambda: jnp.nonzero(presentation == item, size=size, fill_value=-1)[0] + 1,
    )

# %%

def single_free_recall(
    model: MemorySearch, rng: PRNGKeyArray
) -> tuple[MemorySearch, Integer[Array, ""]]:
    """Return model state and choice after performing a free recall event.

    Args:
        model: the current memory search model, after starting retrieval.
        rng: key for random number generation.
    """
    p_all = model.outcome_probabilities()
    choice = random.choice(rng, p_all.shape[0], p=p_all)
    return model.retrieve(choice), choice


def maybe_free_recall(model, rng):
    """Return model state and choice after performing a free recall event if the model is active.

    Args:
        model: the current memory search model, after starting retrieval.
        rng: key for random number generation.
    """
    return lax.cond(
        model.is_active,
        single_free_recall,
        lambda m, _: (m, 0),
        model,
        rng,
    )


def simulate_free_recall(
    model: MemorySearch, list_length: int, rng: PRNGKeyArray
) -> tuple[MemorySearch, Integer[Array, ""] | PRNGKeyArray]:
    """Return model state and choices after performing free recall events until termination.

    Args:
        model: the current memory search model, after starting retrieval.
        list_length: the length of the study and recall sequences.
        rng: key for random number generation.
    """
    return lax.scan(maybe_free_recall, model, random.split(rng, list_length))


def simulate_study_and_free_recall(
    model: MemorySearch, present: Integer[Array, " study_events"], rng: PRNGKeyArray
) -> tuple[MemorySearch, Integer[Array, ""]]:
    """Return model state and choices after simulating a trial.

    Args:
        model: the current memory search model.
        present: the indices of the items to present (1-indexed).
        rng: key for random number generation.
    """
    model = lax.fori_loop(0, present.size, lambda i, m: m.experience(present[i]), model)
    model = model.start_retrieving()
    return simulate_free_recall(model, present.size, rng)


class MemorySearchSimulator:
    def __init__(
        self,
        model_factory: Type[MemorySearchModelFactory],
        dataset: dict[str, Integer[Array, " trials ?"]],
        connections: Optional[Integer[Array, " word_pool_items word_pool_items"]],
    ) -> None:
        """Initialize the factory with the specified trials and trial data."""
        self.factory = model_factory(dataset, connections)
        self.create_model = self.factory.create_model
        self.present_lists = jnp.array(dataset["pres_itemnos"])

    def simulate_trial(
        self,
        trial_index: Integer[Array, ""],
        parameters: Mapping[str, Float_],
        rng: Integer[Array, " rng"],
    ) -> tuple[MemorySearch, Integer[Array, ""]]:
        present = self.present_lists[trial_index]
        model = self.create_model(trial_index, parameters)
        return simulate_study_and_free_recall(model, present, rng)

    def present_and_simulate_trials(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
        rng: Integer[Array, " rng"],
    ) -> Integer[Array, " trials recall_events"]:
        return vmap(self.simulate_trial, in_axes=(0, None, 0))(
            trial_indices, parameters, random.split(rng, trial_indices.size)
        )[1]

#%%

def preallocate_for_h5_dataset(
    data: dict, trial_mask: Bool[Array, " trial_count"], experiment_count: int
) -> dict:
    """Pre-allocates dictionary of numpy arrays based on trial mask and experiment count.

    Arrays are allocated for each key in the input data.
    For 'recalls', the array is initialized with zeros.

    Args:
        data: Dictionary containing dataset arrays.
        trial_mask: Boolean array to select trials.
        experiment_count: Number of times to replicate each array.

    Returns:
        Dictionary with same keys as `data`; each is an array replicated by `experiment_count`.
    """
    """Pre-allocate a dictionary of arrays for a given trial mask."""
    return tree_map(
        lambda x: jnp.repeat(x[trial_mask], experiment_count, axis=0),
        data,
    )


def simulate_h5_from_h5(
    model_factory: Type[MemorySearchModelFactory],
    dataset: dict[str, Integer[Array, " trials ?"]],
    connections: Optional[Integer[Array, " word_pool_items word_pool_items"]],
    parameters: dict[str, Float[Array, " subject_count"]],
    trial_mask: Bool[Array, " trial_count"],
    experiment_count: int,
    rng: PRNGKeyArray,
    size=3,
) -> dict[str, Integer[Array, " trials ?"]]:
    """Simulates dataset from existing dataset using a memory search model parameterized by subject."""

    sim_h5 = preallocate_for_h5_dataset(dataset, trial_mask, experiment_count)
    simulator = MemorySearchSimulator(model_factory, sim_h5, connections)
    subjects, _ = jnp.unique(dataset["subject"][trial_mask], return_counts=True)
    trial_indices, _ = segment_by_index(
        jnp.arange(sim_h5["recalls"].shape[0], dtype=int), sim_h5["subject"].flatten()
    )

    # Handle parameter sampling if the number of subjects in parameters doesn't match
    if len(parameters["subject"]) != len(trial_indices):
        rng, rng_iter = random.split(rng)
        shuffled_indices = random.choice(
            rng_iter,
            len(parameters["subject"]),
            shape=(len(trial_indices),),
            replace=True,
        )
        sim_parameters = {key: parameters[key][shuffled_indices] for key in parameters}
    else:
        sim_parameters = parameters

    reordering = jnp.concatenate(trial_indices)
    for key in sim_h5:
        sim_h5[key] = sim_h5[key][reordering]

    # Run simulations for each subject in a single call
    rngs = random.split(rng, len(subjects))
    recalls = [
        simulator.present_and_simulate_trials(
            jnp.array(trials),
            {key: sim_parameters[key][subject] for key in sim_parameters},
            rng_key,
        )
        for subject, trials, rng_key in zip(subjects, trial_indices, rngs)
    ]

    sim_h5["recalls"] = jnp.concatenate(recalls)

    # Reindex item positions
    reindex_fn = vmap(
        vmap(item_to_study_positions, in_axes=(0, None, None)), in_axes=(0, 0, None)
    )
    sim_h5["recalls"] = reindex_fn(sim_h5["recalls"], sim_h5["pres_itemnos"], size)
    sim_h5["recalls"] = sim_h5["recalls"][:, :, 0]
    return sim_h5
