"""Construct and transform recall datasets while keeping trial fields aligned.

``make_dataset`` builds a dataset from recall sequences and study information.
``select_trials`` selects, reorders or repeats whole trial rows, carrying
participant IDs, study lists and custom fields along with the responses.
``replace_recalls`` supplies new responses and removes metadata tied to the
old recall events.

A recorded response identifies a one-based study position. Models use the
identity of the item presented there. For the study list A, B, A, a response
at position 3 becomes A under ``recall_positions_to_items``. Converting A
back with ``recall_items_to_positions`` gives position 1, its first occurrence.

Use ``select_conditions`` to inspect which trials and participants meet a
selection, and ``select_trials`` to take those rows. ``apply_by_subject``
groups selected trials by participant and calls an analysis on each group.
Condition inspection and participant grouping run on the host. Array-based
selection and recall replacement support JAX transformations with fixed
output shapes.
"""

from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Concatenate, NamedTuple, cast

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from jaxcmr.typing import Array, Bool, Integer, RecallDataset

__all__ = [
    "TrialSelection",
    "apply_by_subject",
    "find_max_list_length",
    "generate_recall_mask",
    "generate_trial_mask",
    "limit_to_first_subjects",
    "make_dataset",
    "make_subject_trial_masks",
    "recall_items_to_positions",
    "recall_positions_to_items",
    "replace_recalls",
    "select_conditions",
    "select_trials",
]

_RECALL_EVENT_FIELDS = frozenset(
    {"rec_itemids", "irt", "cue_clips", "_should_tabulate"}
)


def _prepare_study_lists(
    recalls: Integer[Array, " recall_trials recall_events"],
    presentation_items: ArrayLike | None,
    list_lengths: ArrayLike | None,
) -> tuple[
    Integer[Array, " #study_trials study_events"],
    Integer[Array, " #study_trials 1"],
]:
    """Normalize supplied study lists or generate padded sequential items."""
    if presentation_items is not None:
        presentations = jnp.atleast_2d(jnp.asarray(presentation_items, dtype=jnp.int32))
        if list_lengths is None:
            list_lengths = presentations.shape[1]
        length_column = jnp.asarray(list_lengths, dtype=jnp.int32).reshape(-1, 1)
        return presentations, length_column

    if list_lengths is None:
        list_lengths = max(recalls.shape[1], int(recalls.max()))
    length_column = jnp.asarray(list_lengths, dtype=jnp.int32).reshape(-1, 1)
    # Keep a supplied integer static so generation also works under JIT.
    study_width = (
        list_lengths
        if isinstance(list_lengths, (int, np.integer))
        else int(length_column.max())
    )
    study_positions = jnp.arange(1, study_width + 1, dtype=jnp.int32)[None, :]
    presentations = jnp.where(study_positions <= length_column, study_positions, 0)
    return presentations, length_column


def make_dataset(
    recalls: ArrayLike,
    pres_itemnos: ArrayLike | None = None,
    listLength: ArrayLike | None = None,
    subject: (
        int
        | np.integer
        | Sequence[int]
        | Sequence[Sequence[int]]
        | Integer[Array | np.ndarray, "..."]
    ) = 1,
    *,
    reps: int = 1,
) -> RecallDataset:
    """Construct the four core trial arrays, broadcasting singleton inputs.

    Parameters
    ----------
    recalls : array-like
        One- or two-dimensional study-position recall codes. A vector denotes
        one trial; zero marks an unused response slot.
    pres_itemnos : array-like or None, optional
        One- or two-dimensional within-list item codes. A vector denotes one
        trial. Omit to generate sequential codes padded to the maximum length.
    listLength : array-like or None, optional
        Scalar, vector or single-column array of study lengths. With supplied
        presentations, defaults to their storage width. Supply explicit lengths
        for padded lists. Without presentations or lengths, the default is
        max(response width, highest recall code); this cannot establish an
        experiment's actual study length.
    subject : integer or array-like, optional
        Integer participant IDs as a scalar, vector or single column. Defaults
        to 1. Floating-point and Boolean IDs are outside this interface.
    reps : int, optional
        Positive number of copies of the complete trial sequence. Two trials
        A and B with reps=2 produce A, B, A, B. Simulation's ``replicate_trials``
        instead places copies of each trial together.

    Returns
    -------
    RecallDataset
        New int32 arrays for recalls, pres_itemnos, listLength and subject.
        Singleton rows broadcast to the common trial count before repetition.
        Input arrays are unchanged.

    Notes
    -----
    Inputs must be integer-valued and representable as int32, with compatible
    trial counts. Lengths must be nonnegative and agree with presentation
    padding. These are caller responsibilities, not runtime value checks.

    With explicit presentations, construction supports JIT and vmap; reps must
    be static. Generating presentations also works with a static integer
    listLength. Inferring their width from array values requires concrete data
    and must happen before JIT.
    """
    recall_matrix = jnp.atleast_2d(jnp.asarray(recalls, dtype=jnp.int32))
    presentation_matrix, list_lengths = _prepare_study_lists(
        recall_matrix, pres_itemnos, listLength
    )
    data: RecallDataset = {
        "recalls": recall_matrix,
        "pres_itemnos": presentation_matrix,
        "listLength": list_lengths,
        "subject": jnp.asarray(subject, dtype=jnp.int32).reshape(-1, 1),
    }
    trial_count = max(values.shape[0] for values in data.values())
    for field, values in data.items():
        trial_rows = jnp.broadcast_to(values, (trial_count, values.shape[1]))
        data[field] = jnp.tile(trial_rows, (reps, 1))
    return data


def select_trials(
    data: RecallDataset, trial_indices: Integer[Array, " selected_trials"]
) -> RecallDataset:
    """Select, reorder or repeat complete trial rows without changing their fields.

    Parameters
    ----------
    data : RecallDataset
        Dataset whose fields share a leading trial dimension, including any
        custom trial, study-event and recall-event arrays.
    trial_indices : Integer[Array, "selected_trials"]
        Zero-based row indices in the desired output order. Repeated indices
        repeat whole trials. To use a Boolean mask, first obtain its indices
        with ``jnp.flatnonzero(mask)`` outside JIT.

    Returns
    -------
    RecallDataset
        New dataset with the same fields and selected rows. An empty index
        array produces zero trial rows, retaining the remaining dimensions.
        The input dataset is unchanged.

    Notes
    -----
    This operation supports JAX transformations when the index-array shape is
    fixed. Vocabulary-wide features and other non-trial metadata belong outside
    the dataset and are not selected here.
    """
    selected = data.copy()
    for field, values in data.items():
        selected[field] = values[trial_indices]
    return selected


def replace_recalls(
    data: RecallDataset,
    recalls: Integer[Array, "n_trials num_recalled"],
    additional_recall_fields: Collection[str] = (),
) -> RecallDataset:
    """Replace responses and discard metadata belonging to the old recall events.

    Parameters
    ----------
    data : RecallDataset
        Dataset supplying trial and study information for the new responses.
    recalls : Integer[Array, "n_trials num_recalled"]
        New study-position recall codes, with one row per original trial.
        Response width may differ from the old recalls. Item-code responses
        must be translated to study positions before calling this operation.
    additional_recall_fields : collection of str, optional
        Custom fields tied to the previous responses, to remove as well.
        These must not include required dataset fields or study metadata.

    Returns
    -------
    RecallDataset
        New mapping with the supplied recalls. The known recall-event fields
        rec_itemids, irt, cue_clips and _should_tabulate are removed when present,
        along with additional_recall_fields. Other fields are retained unchanged.
        No metadata is recalculated and the original mapping is unchanged.
    """
    replaced = data.copy()
    for field in _RECALL_EVENT_FIELDS | set(additional_recall_fields):
        replaced.pop(field, None)
    replaced["recalls"] = recalls
    return replaced


def generate_trial_mask(
    data: RecallDataset, trial_query: str | None
) -> Bool[Array, " trial_count"]:
    """Evaluate a query selecting whole trials.

    Parameters
    ----------
    data : RecallDataset
        Trial data available to the query as data.
    trial_query : str or None
        Trusted Python expression producing one Boolean per trial. The names
        data, np and jnp are available. None selects every trial.

    Returns
    -------
    Bool[Array, "trial_count"]
        One-dimensional JAX mask. Column-shaped query results are flattened.
        The dataset is unchanged.

    Examples
    --------
    >>> mask = generate_trial_mask(data, "data['subject'] == 3")
    """
    if trial_query is None:
        return jnp.ones(data["recalls"].shape[0], dtype=bool)
    return jnp.asarray(eval(trial_query), dtype=bool).flatten()


def generate_recall_mask(
    data: RecallDataset, recall_query: str | None
) -> Bool[Array, " trial_count recall_events"]:
    """Evaluate a query selecting individual recall events.

    Parameters
    ----------
    data : RecallDataset
        Trial data available to the query as data.
    recall_query : str or None
        Trusted Python expression producing a Boolean array aligned to recalls.
        The names data, np and jnp are available. None selects every slot,
        including padding; likelihood-specific stopping rules belong to the loss.

    Returns
    -------
    Bool[Array, "trial_count recall_events"]
        JAX mask with the same shape as recalls. The dataset is unchanged.
    """
    if recall_query is None:
        return jnp.ones_like(data["recalls"], dtype=bool)
    return jnp.asarray(eval(recall_query), dtype=bool)


class TrialSelection(NamedTuple):
    """A condition's mask and participant coverage, computed on the host.

    Attributes
    ----------
    mask : Bool[Array, "trials"]
        One-dimensional JAX mask aligned to the original dataset.
    trial_count : int
        Number of selected trial rows.
    subject_ids : Integer[np.ndarray, "subjects"]
        Sorted unique IDs of selected participants.
    participant_count : int
        Read-only property giving the number of selected participants.

    Notes
    -----
    The record does not contain or copy selected data. Pass its mask to an
    analysis, or use its nonzero indices with ``select_trials``.
    """

    mask: Bool[Array, " trials"]
    trial_count: int
    subject_ids: Integer[np.ndarray, " subjects"]

    @property
    def participant_count(self) -> int:
        """Number of distinct selected participants."""
        return len(self.subject_ids)


def select_conditions(
    data: RecallDataset,
    selections: Sequence[
        str | Bool[Array | np.ndarray, " trials"] | Sequence[bool] | None
    ],
    *,
    complete_lists_only: bool = False,
    require_nonempty: bool = True,
) -> list[TrialSelection]:
    """Build condition masks and report their trial and participant coverage.

    Parameters
    ----------
    data : RecallDataset
        Trial-indexed recall data, including subject identities.
    selections : sequence of str, Boolean vectors or None
        Ordered conditions. Strings use ``generate_trial_mask``; Boolean vectors
        have one entry per trial; None selects every trial. Supply an outer
        sequence even for a single condition: [query] or [mask].
    complete_lists_only : bool, optional
        Intersect each condition with rows whose pres_itemnos are positive in
        every stored position. This excludes any padded row, including a fully
        observed shorter list stored alongside longer lists.
    require_nonempty : bool, optional
        Reject conditions with no eligible trials. False permits a zero count
        and empty participant IDs.

    Returns
    -------
    list of TrialSelection
        One record per condition, preserving input order. No dataset fields are
        modified or sliced. This inspection runs outside JIT.

    Raises
    ------
    ValueError
        A condition is empty and require_nonempty is true.

    Examples
    --------
    >>> selection = select_conditions(data, [data["condition"].ravel() == 1])[0]
    >>> selected_data = select_trials(data, jnp.flatnonzero(selection.mask))
    """
    subject_ids = np.asarray(data["subject"]).reshape(-1)
    complete = None
    if complete_lists_only:
        presentations = jnp.asarray(data["pres_itemnos"])
        complete = jnp.all(presentations > 0, axis=1)
    selections_result = []
    for index, selection in enumerate(selections):
        mask = (
            generate_trial_mask(data, selection)
            if selection is None or isinstance(selection, str)
            else jnp.asarray(selection)
        )
        if complete is not None:
            mask = mask & complete
        trial_count = int(mask.sum())
        if require_nonempty and not trial_count:
            raise ValueError(
                f"Condition {index}: the selection contains no eligible lists."
            )
        selections_result.append(
            TrialSelection(mask, trial_count, np.unique(subject_ids[np.asarray(mask)]))
        )
    return selections_result


def make_subject_trial_masks(
    trial_mask: Bool[Array | np.ndarray, " trials"],
    subject_vector: Integer[Array | np.ndarray, " trials"],
) -> tuple[list[Bool[Array, " trials"]], Integer[np.ndarray, " subjects"]]:
    """Partition selected trials into nonempty participant groups.

    Parameters
    ----------
    trial_mask : Bool[Array | np.ndarray, "trials"]
        Selection aligned to the original trial rows.
    subject_vector : Integer[Array | np.ndarray, "trials"]
        Participant ID for each trial, as a one-dimensional vector.

    Returns
    -------
    tuple of list of JAX masks and NumPy participant IDs
        One mask per selected participant, ordered by increasing ID. Each mask
        still indexes the original dataset and includes only selected trials.
        An empty selection returns an empty mask list and ID array.

    Notes
    -----
    Participant discovery reads values on the host and runs outside JIT.
    """
    unique_subjects = np.unique(np.asarray(subject_vector)[np.asarray(trial_mask)])
    subject_vector = jnp.asarray(subject_vector)
    trial_mask = jnp.asarray(trial_mask)
    subject_masks = [
        (subject_vector == subject_id) & trial_mask for subject_id in unique_subjects
    ]
    return subject_masks, unique_subjects


def apply_by_subject[Result, **Parameters](
    data: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    func: Callable[Concatenate[RecallDataset, Parameters], Result],
    *args: Parameters.args,
    **kwargs: Parameters.kwargs,
) -> list[Result]:
    """Apply an operation to the selected trial rows of each participant.

    Parameters
    ----------
    data : RecallDataset
        Dataset with aligned trial fields, including any custom arrays.
    trial_mask : Bool[Array, "trial_count"]
        Trials to include before grouping by participant.
    func : callable
        Operation called as func(subject_data, *args, **kwargs). Each
        subject_data contains only that participant's selected trial rows.
    *args
        Additional positional arguments passed to func.
    **kwargs
        Additional keyword arguments passed to func.

    Returns
    -------
    list of Result
        One result per selected participant, ordered by increasing ID. No
        empty groups are passed to func; an empty selection returns [].

    Notes
    -----
    Grouping runs on the host. The supplied operation may itself be JIT-compiled;
    its selected dataset shapes can differ between participants. This function
    does not aggregate results or change the input dataset.
    """
    masks, _ = make_subject_trial_masks(trial_mask, data["subject"].flatten())
    results: list[Result] = []
    for subject_mask in masks:
        subject_dataset = select_trials(data, jnp.flatnonzero(subject_mask))
        results.append(func(subject_dataset, *args, **kwargs))
    return results


def limit_to_first_subjects(
    data: Mapping[str, Array],
    max_subjects: int,
) -> dict[str, Array]:
    """Retain trials from the lowest participant IDs without reordering rows.

    Parameters
    ----------
    data : Mapping[str, Array]
        Trial-indexed arrays with a subject column. This mapping need not
        contain the other core recall fields.
    max_subjects : int
        Nonnegative participant limit. Zero retains all participants, matching
        the convention in ``load_data``. A limit above the participant count
        also retains everyone.

    Returns
    -------
    dict[str, Array]
        New mapping with selected rows of every field, in their original trial
        order. Empty input remains empty. The supplied mapping is unchanged.
    """
    subject_ids = data["subject"].reshape(-1)
    unique_subjects = np.unique(np.asarray(subject_ids))
    retained_subjects = (
        unique_subjects[:max_subjects] if max_subjects else unique_subjects
    )
    include_mask = jnp.isin(subject_ids, retained_subjects)
    return {key: value[include_mask] for key, value in data.items()}


def find_max_list_length(
    datasets: Sequence[RecallDataset],
    trial_masks: Sequence[Bool[Array, " trial_count"]],
) -> int:
    """Find the greatest declared study length among selected trials.

    Parameters
    ----------
    datasets : sequence of RecallDataset
        Nonempty collection of datasets containing listLength columns.
    trial_masks : sequence of Bool[Array, "trial_count"]
        Corresponding nonempty trial selections, in dataset order.

    Returns
    -------
    int
        Maximum selected listLength value. This is the number of declared
        study events, not the width of a padded presentation array.

    Notes
    -----
    This inspection reads a scalar value on the host and runs outside JIT.
    """
    return max(
        jnp.max(data["listLength"][trial_mask]).item()
        for data, trial_mask in zip(datasets, trial_masks)
    )


def _require_pres_itemids(
    data: RecallDataset,
) -> Integer[Array, "n_trials num_presented"]:
    """Read global item IDs where callers require this optional dataset field.

    This typed accessor performs ordinary key lookup, not value validation.
    """
    return cast(Mapping[str, Array], data)["pres_itemids"]


def recall_positions_to_items(
    recalls: Integer[Array, " trials recall_events"],
    presentations: Integer[Array, " trials study_positions"],
) -> Integer[Array, " trials recall_events"]:
    """Convert recorded study positions to within-list item codes.

    Parameters
    ----------
    recalls : Integer[Array, " trials recall_events"]
        One-based study positions; zero denotes an unused response slot.
    presentations : Integer[Array, " trials study_positions"]
        Within-list item codes for the same trials.

    Returns
    -------
    Integer[Array, " trials recall_events"]
        Presented item codes, preserving zeros, row order and repetitions.
        Inputs remain unchanged; the operation supports JAX transformations.
    """
    items = jnp.take_along_axis(presentations, jnp.maximum(recalls - 1, 0), axis=1)
    return jnp.where(recalls > 0, items, 0)


def recall_items_to_positions(
    recalls: Integer[Array, " trials recall_events"],
    presentations: Integer[Array, " trials study_positions"],
) -> Integer[Array, " trials recall_events"]:
    """Convert recalled item codes to their first matching study positions.

    Parameters
    ----------
    recalls : Integer[Array, " trials recall_events"]
        Within-list item codes; zero denotes an unused response slot.
    presentations : Integer[Array, " trials study_positions"]
        Within-list item codes for the same trials, with zero padding.

    Returns
    -------
    Integer[Array, " trials recall_events"]
        One-based first study positions. Zero and unpresented items map to zero.
        Repeated presentations share their first position; analyses can recover
        all presentations from the dataset. Inputs remain unchanged.
    """
    matches = (recalls[:, :, None] == presentations[:, None, :]) & (
        recalls[:, :, None] > 0
    )
    return jnp.where(matches.any(axis=-1), jnp.argmax(matches, axis=-1) + 1, 0)
