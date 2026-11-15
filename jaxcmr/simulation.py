"""Simulation utilities for memory-search models.

Provides functions to run forward simulations of memory-search models,
converting model parameters and study lists into simulated recall
sequences. Supports single-trial, multi-trial (vmapped), and
dataset-level simulation with optional study-phase replay.

"""

import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import lax, random, vmap

from jaxcmr.dataset import _replace_recalls, _select_trials
from jaxcmr.helpers.io import load_data
from jaxcmr.repetition import item_to_study_positions
from jaxcmr.typing import (
    Array,
    Bool,
    Float,
    Int_,
    Integer,
    MemorySearch,
    MemorySearchModelFactory,
    PRNGKeyArray,
    Real,
    RecallDataset,
    TrialSimulator,
)

__all__ = [
    "MemorySearchSimulator",
    "item_to_study_positions",
    "load_matching_simulation",
    "parameter_shifted_simulate_h5_from_h5",
    "preallocate_for_h5_dataset",
    "simulate_free_recall",
    "simulate_free_recall_count",
    "simulate_h5_from_h5",
    "simulate_study_and_free_recall",
    "simulate_study_first_recall_and_forced_stop",
    "simulate_study_first_recall_and_free_recall",
    "simulate_study_free_recall_and_forced_stop",
]


def load_matching_simulation(
    response_path: str | Path, expected_settings: Mapping[str, Any]
) -> RecallDataset | None:
    """Load saved responses when their recorded settings match the request.

    Args:
        response_path: HDF5 response file. Its settings are stored alongside it
            with the same basename and a ``.json`` suffix.
        expected_settings: Requested JSON-compatible settings. Every supplied
            field must be present and equal; extra saved fields are allowed.
            Older records can supply ``jaxcmr_version`` and ``jax_enable_x64``
            inside ``implementation`` when absent at the top level.

    Returns:
        The dataset loaded by ``load_data``, or ``None`` if either file is absent
        or the requested settings differ. Missing fields differ from explicit
        ``None`` values. Neither file nor the requested settings are modified.

    JSON parsing and dataset-reading errors propagate when loading is attempted.
    The caller decides whether reuse is allowed after a refit or an explicit
    regeneration request. Matching settings do not detect edits to input files
    or model code at the same location.
    """
    response_path = Path(response_path)
    metadata_path = response_path.with_suffix(".json")
    if not response_path.exists() or not metadata_path.exists():
        return None
    saved_settings = json.loads(metadata_path.read_text())
    if not _simulation_metadata_matches(saved_settings, expected_settings):
        return None
    return load_data(response_path)


def _simulation_metadata_matches(
    saved: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    """Compare requested fields, accepting the older version-metadata layout."""
    saved_metadata = dict(saved)
    implementation = saved_metadata.get("implementation", {})
    for key in ("jaxcmr_version", "jax_enable_x64"):
        if key not in saved_metadata and key in implementation:
            saved_metadata[key] = implementation[key]
    return all(
        key in saved_metadata and saved_metadata[key] == value
        for key, value in expected.items()
    )


def _reindex_recalls(
    recalls: Integer[Array, " trial_count recall_events"],
    presentations: Integer[Array, " trial_count list_length"],
    size: int,
) -> Integer[Array, " trial_count recall_events"]:
    """Reindex recall indices as study positions.

    Parameters
    ----------
    recalls : Integer[Array, " trial_count recall_events"]
        Simulated recall indices per trial.
    presentations : Integer[Array, " trial_count list_length"]
        Study sequences aligned with each trial.
    size : int
        Number of positions to retain.

    Returns
    -------
    Integer[Array, " trial_count recall_events"]
        Recalls expressed as study positions.

    """
    return vmap(
        vmap(item_to_study_positions, in_axes=(0, None, None)), in_axes=(0, 0, None)
    )(recalls, presentations, size)[:, :, 0]


def _single_free_recall(
    model: MemorySearch, rng: PRNGKeyArray
) -> tuple[MemorySearch, Integer[Array, ""]]:
    """Execute one free-recall event.

    Parameters
    ----------
    model : MemorySearch
        Retrieval-ready memory search model.
    rng : PRNGKeyArray
        Random key.

    Returns
    -------
    tuple[MemorySearch, Integer[Array, ""]]
        Updated model and chosen item index.

    """
    p_all = model.outcome_probabilities()
    choice = random.choice(rng, p_all.shape[0], p=p_all)
    return model.retrieve(choice), choice


def _maybe_free_recall(
    model: MemorySearch, rng: PRNGKeyArray
) -> tuple[MemorySearch, Integer[Array, ""]]:
    """Perform one recall step if active; no-op otherwise."""
    return lax.cond(
        model.is_active,
        _single_free_recall,
        lambda m, _: (m, 0),
        model,
        rng,
    )


def _maybe_free_recall_count(
    model: MemorySearch, rng: PRNGKeyArray, recall_count: Int_
) -> tuple[MemorySearch, Integer[Array, ""]]:
    """Perform one recall step if under count limit; no-op otherwise."""
    return lax.cond(
        model.recall_total < recall_count,
        _single_free_recall,
        lambda m, _: (m, 0),
        model,
        rng,
    )


def simulate_free_recall(
    model: MemorySearch, list_length: int, rng: PRNGKeyArray
) -> tuple[MemorySearch, Integer[Array, " recall_events"]]:
    """Simulate repeated free-recall steps from a model.

    Parameters
    ----------
    model : MemorySearch
        Retrieval-ready memory search model.
    list_length : int
        Upper bound on recall steps to simulate.
    rng : PRNGKeyArray
        Random key.

    Returns
    -------
    tuple[MemorySearch, Integer[Array, " recall_events"]]
        Updated model and recalled item indices.

    """
    return lax.scan(_maybe_free_recall, model, random.split(rng, list_length))


def simulate_free_recall_count(
    model: MemorySearch,
    recall_count: Int_,
    list_length: int,
    rng: PRNGKeyArray,
) -> tuple[MemorySearch, Integer[Array, " recall_events"]]:
    """Simulate free recall truncated to a fixed event count.

    Parameters
    ----------
    model : MemorySearch
        Retrieval-ready memory search model.
    recall_count : Int_
        Maximum retrieval events to simulate.
    list_length : int
        Total recall slots in the output.
    rng : PRNGKeyArray
        Random key.

    Returns
    -------
    tuple[MemorySearch, Integer[Array, " recall_events"]]
        Updated model and recalled item indices.

    """

    return lax.scan(
        lambda m, rng: _maybe_free_recall_count(m, rng, recall_count),
        model,
        random.split(rng, list_length),
    )


def simulate_study_and_free_recall(
    model: MemorySearch,
    present: Integer[Array, " study_events"],
    trial: Integer[Array, " recalls"],
    rng: PRNGKeyArray,
) -> tuple[MemorySearch, Integer[Array, " recall_events"]]:
    """Simulate study then free recall for a single trial.

    Parameters
    ----------
    model : MemorySearch
        Memory search model in study mode.
    present : Integer[Array, " study_events"]
        One-indexed study sequence for the trial.
    trial : Integer[Array, " recalls"]
        Observed recall sequence for the trial.
    rng : PRNGKeyArray
        Random key.

    Returns
    -------
    tuple[MemorySearch, Integer[Array, " recall_events"]]
        Updated model and recalled item indices.

    """
    model = lax.fori_loop(0, present.size, lambda i, m: m.experience(present[i]), model)
    model = model.start_retrieving()
    return simulate_free_recall(model, present.size, rng)


def simulate_study_first_recall_and_free_recall(
    model: MemorySearch,
    present: Integer[Array, " study_events"],
    trial: Integer[Array, " recalls"],
    rng: PRNGKeyArray,
) -> tuple[MemorySearch, Integer[Array, " recall_events"]]:
    """Simulate study, forced first recall, then free recall.

    Parameters
    ----------
    model : MemorySearch
        Memory search model in study mode.
    present : Integer[Array, " study_events"]
        One-indexed study sequence for the trial.
    trial : Integer[Array, " recalls"]
        Observed recall sequence; first element is forced.
    rng : PRNGKeyArray
        Random key.

    Returns
    -------
    tuple[MemorySearch, Integer[Array, " recall_events"]]
        Updated model and recalled item indices.

    """
    model = lax.fori_loop(0, present.size, lambda i, m: m.experience(present[i]), model)
    model = model.start_retrieving()
    model = model.retrieve(trial[0])
    model, recalls = simulate_free_recall(model, present.size - 1, rng)
    return model, jnp.concatenate((trial[:1], recalls))


def simulate_study_free_recall_and_forced_stop(
    model: MemorySearch,
    present: Integer[Array, " study_events"],
    trial: Integer[Array, " recalls"],
    rng: PRNGKeyArray,
) -> tuple[MemorySearch, Integer[Array, " recall_events"]]:
    """Simulate study and free recall with stop timing from data.

    Parameters
    ----------
    model : MemorySearch
        Memory search model in study mode.
    present : Integer[Array, " study_events"]
        One-indexed study sequence for the trial.
    trial : Integer[Array, " recalls"]
        Observed recall sequence; first zero marks stop.
    rng : PRNGKeyArray
        Random key.

    Returns
    -------
    tuple[MemorySearch, Integer[Array, " recall_events"]]
        Updated model and masked recall indices.

    """
    model = lax.fori_loop(0, present.size, lambda i, m: m.experience(present[i]), model)
    model = model.start_retrieving()
    model, recalls = simulate_free_recall(model, trial.size, rng)
    return model, recalls * (trial != 0)


def simulate_study_first_recall_and_forced_stop(
    model: MemorySearch,
    present: Integer[Array, " study_events"],
    trial: Integer[Array, " recalls"],
    rng: PRNGKeyArray,
) -> tuple[MemorySearch, Integer[Array, " recall_events"]]:
    """Simulate study and recall conditional on the first item and total count.

    Parameters
    ----------
    model : MemorySearch
        Memory search model in study mode. Use NoStopTermination to prevent
        early stopping, and disable repeated recalls for first-mention data.
    present : Integer[Array, " study_events"]
        One-indexed study sequence for the trial.
    trial : Integer[Array, " recalls"]
        Compact observed sequence with trailing zeros. Its first item is
        forced and its nonzero count sets the total number of recalls.
    rng : PRNGKeyArray
        Random key for the generated recalls after the first item.

    Returns
    -------
    tuple[MemorySearch, Integer[Array, " recall_events"]]
        Updated model and recalls with the same width as trial. The forced
        first recall updates retrieval state exactly once. Zero- and
        one-recall trials generate no additional choices.

    """
    model = lax.fori_loop(0, present.size, lambda i, m: m.experience(present[i]), model)
    model = model.start_retrieving()
    if trial.size == 0:
        return model, trial
    recall_count = jnp.count_nonzero(trial)
    model = lax.cond(
        recall_count > 0, lambda m: m.retrieve(trial[0]), lambda m: m, model
    )
    # The count primitive compares model.recall_total with an absolute target.
    # The forced first recall is already included in that total.
    model, recalls = simulate_free_recall_count(model, recall_count, trial.size - 1, rng)
    return model, jnp.concatenate((trial[:1], recalls))


class MemorySearchSimulator:
    """Stateless trial-level simulator usable with vmap."""

    def __init__(
        self,
        model_factory_cls: type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Float[Array, " word_pool_items features_count"] | None,
        simulate_trial_fn: TrialSimulator = simulate_study_and_free_recall,
    ) -> None:
        """Initialize simulator from a dataset and model factory.

        Parameters
        ----------
        model_factory_cls : type[MemorySearchModelFactory]
            Factory class for memory search models.
        dataset : RecallDataset
            Dataset with presentation and recall sequences.
        features : Optional[Float[Array, " ..."]]
            Feature matrix for the word pool, or None.
        simulate_trial_fn : TrialSimulator
            Trial simulation function.

        """
        factory = model_factory_cls(dataset, features)
        self.create_model = factory.create_trial_model
        self.present_lists = jnp.array(dataset["pres_itemnos"])
        self._simulate_trial_fn = simulate_trial_fn

        # Reindex the recalled items so they match the "present_lists" indexing
        trials = np.array(dataset["recalls"])
        for trial_index in range(trials.shape[0]):
            present = self.present_lists[trial_index]
            recall = trials[trial_index]
            reindexed = np.array(
                [(present[item - 1] if item else 0) for item in recall]
            )
            trials[trial_index] = reindexed

        self.trials = jnp.array(trials)

    def simulate_trial(
        self,
        trial_index: Integer[Array, ""],
        subject_index: Integer[Array, ""],
        parameters: Mapping[str, Real[Array, " participants"] | Bool[Array, " participants"]],
        rng: PRNGKeyArray,
    ) -> Integer[Array, " recall_events"]:
        """Simulate a single trial using per-subject parameters.

        Parameters
        ----------
        trial_index : Integer[Array, ""]
            Index of the trial to simulate.
        subject_index : Integer[Array, ""]
            Index into per-subject parameter arrays.
        parameters : Mapping[str, Array]
            Numeric or Boolean columns with one value per participant.
            An optional ``subject`` identity
            column is not passed to the model constructor.
        rng : PRNGKeyArray
            Random key.

        Returns
        -------
        Integer[Array, " recall_events"]
            Recalled item indices for the trial.

        """
        # Participant identity selects the row; it is not a model parameter.
        model_parameters = {
            name: values[subject_index]
            for name, values in parameters.items() if name != "subject"
        }
        model = self.create_model(trial_index, model_parameters)
        return self._simulate_trial_fn(
            model, self.present_lists[trial_index], self.trials[trial_index], rng
        )[1]


def _parameter_indices_for_subjects(
    subject_ids: Integer[Array, " trial_count"],
    parameters: Mapping[str, Real[Array, " subject_count"] | Bool[Array, " subject_count"]],
) -> Integer[Array, " trial_count"]:
    """Map integer participant IDs to parameter rows under the simulation contract.

    With a subject column, IDs are unique and cover the selected participants.
    Without it, dataset IDs already are valid zero-based parameter-row indices.
    Participant identities are concrete host inputs; numerical parameter columns
    remain available for JAX transformations.
    """
    if "subject" not in parameters:
        return subject_ids
    parameter_subject_ids = np.asarray(parameters["subject"])
    row_by_subject = {subject: index for index, subject in enumerate(parameter_subject_ids)}
    return jnp.asarray(
        [row_by_subject[subject] for subject in np.asarray(subject_ids)], dtype=jnp.int32
    )


def preallocate_for_h5_dataset(
    data: RecallDataset, trial_mask: Bool[Array, " trial_count"], experiment_count: int
) -> RecallDataset:
    """Replicate selected trials for batch simulation.

    Parameters
    ----------
    data : RecallDataset
        Input dataset.
    trial_mask : Bool[Array, " trial_count"]
        Mask selecting trials to include.
    experiment_count : int
        Replication count per trial.

    Returns
    -------
    RecallDataset
        Dataset with selected trials replicated.

    """
    trial_indices = jnp.repeat(
        jnp.flatnonzero(trial_mask), experiment_count
    )
    return _select_trials(data, trial_indices)


def simulate_h5_from_h5(
    model_factory_cls: type[MemorySearchModelFactory],
    dataset: RecallDataset,
    features: Float[Array, " word_pool_items features_count"] | None,
    parameters: dict[str, Real[Array, " subject_count"] | Bool[Array, " subject_count"]],
    trial_mask: Bool[Array, " trial_count"],
    experiment_count: int,
    rng: PRNGKeyArray,
    size: int = 3,
    simulate_trial_fn: TrialSimulator = simulate_study_and_free_recall,
    recall_dependent_fields: Collection[str] = (),
) -> RecallDataset:
    """Simulate a dataset using per-subject parameters.

    Dataset selection and participant identities are prepared on the host.
    Numerical parameter columns can be mapped with JAX while those inputs
    remain fixed; they are not converted to NumPy.

    Parameters
    ----------
    model_factory_cls : type[MemorySearchModelFactory]
        Factory class for memory search models.
    dataset : RecallDataset
        Original dataset containing trial data.
    features : Optional[Float[Array, " ..."]]
        Feature matrix for the word pool, or None.
    parameters : dict[str, Array]
        Nonempty, aligned one-dimensional numeric or Boolean parameter columns.
        An optional ``subject`` column contains unique integer IDs covering all
        selected dataset participants. Floating-point IDs, Boolean IDs and pooled
        rows are excluded; assign pooled fits with ``parameters_for_subjects``.
        Without ``subject``, dataset IDs must be valid zero-based indices into
        every parameter column. These are caller preconditions, not runtime checks.
    trial_mask : Bool[Array, " trial_count"]
        Mask selecting trials to simulate.
    experiment_count : int
        Replications per selected trial.
    rng : PRNGKeyArray
        Random key.
    size : int
        Max study positions per item when reindexing.
    simulate_trial_fn : TrialSimulator
        Trial-level simulation function.
    recall_dependent_fields : Collection[str]
        Additional fields tied to the original recall events, omitted from output.
        The known fields ``rec_itemids``, ``irt``, ``cue_clips``, and
        ``_should_tabulate`` are always omitted when recalls are regenerated.

    Returns
    -------
    RecallDataset
        Simulated dataset with recall sequences.

    """

    simulation_data = preallocate_for_h5_dataset(dataset, trial_mask, experiment_count)
    simulator = MemorySearchSimulator(
        model_factory_cls, simulation_data, features, simulate_trial_fn
    )

    # Flat trial + subject index vectors (static shapes)
    total_trials = simulation_data["subject"].size
    trial_indices = jnp.arange(total_trials, dtype=jnp.int32)
    subject_ids = simulation_data["subject"].flatten()
    parameter_row_indices = _parameter_indices_for_subjects(subject_ids, parameters)
    trial_keys = random.split(rng, total_trials)

    # One jit-compiled vmap over trials
    recalls = vmap(simulator.simulate_trial, in_axes=(0, 0, None, 0))(
        trial_indices, parameter_row_indices, parameters, trial_keys
    )

    # Reindex study positions
    simulated_recalls = _reindex_recalls(recalls, simulation_data["pres_itemnos"], size)
    return _replace_recalls(simulation_data, simulated_recalls, recall_dependent_fields)


def parameter_shifted_simulate_h5_from_h5(
    model_factory_cls: type[MemorySearchModelFactory],
    dataset: RecallDataset,
    features: Float[Array, " word_pool_items features_count"] | None,
    parameters: dict[str, Real[Array, " subject_count"] | Bool[Array, " subject_count"]],
    trial_mask: Bool[Array, " trial_count"],
    experiment_count: int,
    varied_parameter: str,
    parameter_values: Sequence[float],
    rng: PRNGKeyArray,
    size: int = 3,
    simulate_trial_fn: TrialSimulator = simulate_study_and_free_recall,
    recall_dependent_fields: Collection[str] = (),
) -> Sequence[RecallDataset]:
    """Simulate multiple datasets by sweeping a single parameter.

    Parameters
    ----------
    model_factory_cls : type[MemorySearchModelFactory]
        Factory class for memory search models.
    dataset : RecallDataset
        Original dataset containing trial data.
    features : Optional[Float[Array, " ..."]]
        Feature matrix for the word pool, or None.
    parameters : dict[str, Array]
        Nonempty, aligned one-dimensional numeric or Boolean parameter columns.
        An optional ``subject`` column contains unique integer IDs covering all
        selected dataset participants. Floating-point IDs, Boolean IDs and pooled
        rows are excluded; assign pooled fits with ``parameters_for_subjects``.
        Without ``subject``, dataset IDs must be valid zero-based indices into
        every parameter column. These are caller preconditions, not runtime checks.
    trial_mask : Bool[Array, " trial_count"]
        Mask selecting trials to simulate.
    experiment_count : int
        Replications per selected trial.
    varied_parameter : str
        Name of the parameter to sweep.
    parameter_values : Sequence[float]
        Values to assign to the swept parameter.
    rng : PRNGKeyArray
        Random key.
    size : int
        Max study positions per item when reindexing.
    simulate_trial_fn : TrialSimulator
        Trial-level simulation function.
    recall_dependent_fields : Collection[str]
        Additional fields tied to the original recall events, omitted from output.
        The known fields ``rec_itemids``, ``irt``, ``cue_clips``, and
        ``_should_tabulate`` are always omitted when recalls are regenerated.

    Returns
    -------
    Sequence[RecallDataset]
        One simulated dataset per parameter value.

    """

    template = preallocate_for_h5_dataset(dataset, trial_mask, experiment_count)
    simulator = MemorySearchSimulator(
        model_factory_cls, template, features, simulate_trial_fn
    )
    total_trials = template["subject"].size
    trial_indices = jnp.arange(total_trials, dtype=jnp.int32)
    subject_ids = template["subject"].flatten()
    def run_for_parameters(
        parameter_columns: Mapping[str, Real[Array, " participants"] | Bool[Array, " participants"]],
        sweep_rng: PRNGKeyArray,
    ) -> RecallDataset:
        trial_keys = random.split(sweep_rng, total_trials)
        parameter_row_indices = _parameter_indices_for_subjects(subject_ids, parameter_columns)
        recalls = vmap(simulator.simulate_trial, in_axes=(0, 0, None, 0))(
            trial_indices, parameter_row_indices, parameter_columns, trial_keys
        )
        simulated_recalls = _reindex_recalls(recalls, template["pres_itemnos"], size)
        return _replace_recalls(template, simulated_recalls, recall_dependent_fields)

    results: list[RecallDataset] = []
    for value in parameter_values:
        rng, sweep_key = random.split(rng)
        swept_parameters = {
            **parameters,
            varied_parameter: jnp.full_like(parameters[varied_parameter], value),
        }
        results.append(run_for_parameters(swept_parameters, sweep_key))

    return results
