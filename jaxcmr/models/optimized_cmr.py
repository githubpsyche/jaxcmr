"""Efficient regular CMR with event and model-owned sequence computations.

This model implements standard linear temporal CMR, learning before context
integration and suppressing repeated recall responses. NoStopTermination is
fixed. It retains every repeated study event. Event and bulk paths share item
input, primacy, context integration, initial support and choice functions.

Sequence preparation knows the complete study schedule, so it can discard
association updates that no later presentation needs. The event interface
retains every study context because future presentations are not known.
Bulk scoring is valid here because an observed item determines reinstatement
without consulting its retrieval probability. These are properties of this
model implementation, not assumptions made by a likelihood adapter.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from simple_pytree import Pytree

from jaxcmr.components import context
from jaxcmr.components.context import integrate_normalized
from jaxcmr.components.termination import NoStopTermination
from jaxcmr.math import (
    exponential_primacy_decay,
    lb,
    normalize_magnitude,
    relative_power_scale,
)
from jaxcmr.models.sequence import PreparedPrediction
from jaxcmr.typing import Array, Bool, Float, Float_, Int_, Integer, RecallDataset

__all__ = ["CMR", "ModelFactory"]

_FLAGS = {"learn_after_context_update", "allow_repeated_recalls"}


class _CMRParameters(NamedTuple):
    """Named scalar parameters shared by the event and sequence calculations.

    JAX treats a named tuple as a pytree, so names remain available inside
    compiled scans and vmaps. All values have the same floating-point dtype.
    The public constructor continues to accept the usual parameter mapping.
    """

    encoding_drift_rate: Float[Array, ""]
    start_drift_rate: Float[Array, ""]
    recall_drift_rate: Float[Array, ""]
    shared_support: Float[Array, ""]
    item_support: Float[Array, ""]
    learning_rate: Float[Array, ""]
    primacy_scale: Float[Array, ""]
    primacy_decay: Float[Array, ""]
    choice_sensitivity: Float[Array, ""]

    @property
    def dtype(self) -> np.dtype:
        """Return the common dtype used when allocating model state."""
        return self.encoding_drift_rate.dtype


def _parse_parameters(parameters: Mapping[str, Float_]) -> _CMRParameters:
    """Validate this implementation's scope and construct named parameters.

    Both configuration flags must be fixed False. A traced flag would permit
    a scientific configuration this implementation does not calculate.
    """
    required = set(_CMRParameters._fields) | _FLAGS
    missing = required - parameters.keys()
    extra = parameters.keys() - required
    if missing or extra:
        raise ValueError(
            f"optimized_cmr parameter mismatch: missing={sorted(missing)}, unknown={sorted(extra)}"
        )
    for flag in _FLAGS:
        if isinstance(parameters[flag], jax.core.Tracer) or bool(parameters[flag]):
            raise ValueError(f"optimized_cmr requires a fixed {flag}=False.")
    values = jnp.asarray([parameters[name] for name in _CMRParameters._fields])
    return _CMRParameters(*values)


def _item_input(
    parameters: _CMRParameters,
    item_index: Array | int,
    association: Array,
    item_count: int,
) -> Array:
    """Combine an item's pre-experimental and learned context inputs.

    ``item_index`` is zero-based; context unit zero is the start-of-list unit.
    ``association`` sums the contexts preceding this item's study occurrences,
    with shape (..., item_count + 1). It has not yet been scaled by learning.
    Leading dimensions permit the same rule to process many recalled items.
    """
    basis = jax.nn.one_hot(item_index + 1, item_count + 1, dtype=parameters.dtype)
    return (
        1 - parameters.learning_rate
    ) * basis + parameters.learning_rate * association


def _primacy(parameters: _CMRParameters, positions: Array) -> Array:
    """Return context-to-item learning weights at zero-based study positions."""
    return exponential_primacy_decay(
        positions, parameters.primacy_scale, parameters.primacy_decay
    )


def _retrieval_support(
    parameters: _CMRParameters, states: Array, learned: Array
) -> Array:
    """Add pre-experimental support to learned context-to-item support.

    ``states`` has shape (..., context_units); ``learned`` has shape
    (..., items). The start-of-list unit has no pre-experimental item links.
    Each item receives item_support from its own unit and shared_support
    from all other item units.
    """
    item = states[..., 1:]
    shared = item.sum(axis=-1, keepdims=True) - item
    return parameters.shared_support * shared + parameters.item_support * item + learned


def _choice_weights(
    parameters: _CMRParameters, support: Array, candidates: Array
) -> Array:
    """Return unnormalized Luce weights for one set of available items.

    Support and candidate masks are one-dimensional item vectors. The small
    positive floor preserves nonzero probability for available items with no
    support; unavailable items retain exactly zero weight.
    """
    return (
        relative_power_scale(support * candidates, parameters.choice_sensitivity) + lb
    ) * candidates


class CMR(Pytree):
    """Event-level CMR backed by saved study contexts instead of dense memories.

    Public item labels remain 1..list_length; zero means no study event or STOP.
    The model can use the existing simulation and ordinary likelihood drivers.
    It owns a TemporalContext object for ordinary state inspection, but no MFC
    or MCF matrix. Model-specific memory-inspection utilities need an explicit
    adapter rather than an inferred dense representation.
    """

    def __init__(self, list_length: int, parameters: Mapping[str, Float_]) -> None:
        """Initialize an empty temporal CMR model.

        Parameters
        ----------
        list_length : int
            Study-event capacity and maximum one-indexed item label.
        parameters : Mapping[str, Float_]
            The nine regular-CMR parameters. Both learn_after_context_update
            and allow_repeated_recalls must be fixed False.

        """
        self.parameters = _parse_parameters(parameters)
        self.item_count = list_length
        self.context = context.init(list_length)
        self.termination_policy = NoStopTermination(list_length, parameters)
        self.study_contexts = jnp.zeros(
            (list_length, list_length + 1), dtype=self.parameters.dtype
        )
        self.study_items = jnp.zeros(list_length, dtype=int)
        self.study_index = jnp.array(0, dtype=int)
        self.studied = jnp.zeros(list_length, dtype=bool)
        self.recallable = jnp.zeros(list_length, dtype=bool)
        self.recalls = jnp.zeros(list_length, dtype=int)
        self.recall_total = jnp.array(0, dtype=int)
        self.is_active = jnp.array(True)

    def _association(self, item_index: Int_) -> Array:
        """Sum the saved pre-update contexts associated with a zero-based item.

        This is the learned part of an MFC row before learning-rate scaling.
        Unused study slots are excluded even when their stored label is zero.
        """
        included = (self.study_items == item_index + 1) & (
            jnp.arange(self.study_items.size) < self.study_index
        )
        return jnp.sum(jnp.where(included[:, None], self.study_contexts, 0.0), axis=0)

    def experience_item(self, item_index: Int_) -> "CMR":
        """Encode an item and retain its pre-update context for later probes.

        Parameters
        ----------
        item_index : Int_
            Zero-indexed item identity; repeated presentations use the same index.

        Returns
        -------
        CMR
            Model after learning this occurrence and updating temporal context.

        """
        memory_input = _item_input(
            self.parameters,
            item_index,
            self._association(item_index),
            self.study_items.size,
        )
        new_state = integrate_normalized(
            self.context.state,
            normalize_magnitude(memory_input),
            self.parameters.encoding_drift_rate,
        )
        # Both association directions learn the context before this update.
        # Saving that vector preserves separate contributions from repetitions.
        return self.replace(
            context=self.context.replace(state=new_state),
            study_contexts=self.study_contexts.at[self.study_index].set(
                self.context.state
            ),
            study_items=self.study_items.at[self.study_index].set(item_index + 1),
            study_index=self.study_index + 1,
            studied=self.studied.at[item_index].set(True),
            recallable=self.recallable.at[item_index].set(True),
        )

    def experience(self, choice: Int_) -> "CMR":
        """Simulate a study event, leaving the model unchanged for padding.

        Parameters
        ----------
        choice : Int_
            One-indexed item identity. Zero denotes padding.

        Returns
        -------
        CMR
            Model after the specified study event.

        """
        return lax.cond(
            choice == 0, lambda: self, lambda: self.experience_item(choice - 1)
        )

    def start_retrieving(self) -> "CMR":
        """Reinstate the start-of-list input before the first recall.

        Returns
        -------
        CMR
            Model with context drifted toward its initial state.

        """
        state = integrate_normalized(
            self.context.state,
            self.context.initial_state,
            self.parameters.start_drift_rate,
        )
        return self.replace(context=self.context.replace(state=state))

    def retrieve_item(self, item_index: Int_) -> "CMR":
        """Reinstate an item's learned context and suppress further recall of it.

        Parameters
        ----------
        item_index : Int_
            Zero-indexed recalled item identity.

        Returns
        -------
        CMR
            Model with updated context, availability and recall history.

        """
        memory_input = _item_input(
            self.parameters,
            item_index,
            self._association(item_index),
            self.study_items.size,
        )
        state = integrate_normalized(
            self.context.state,
            normalize_magnitude(memory_input),
            self.parameters.recall_drift_rate,
        )
        return self.replace(
            context=self.context.replace(state=state),
            recallable=self.recallable.at[item_index].set(False),
            recalls=self.recalls.at[self.recall_total].set(item_index + 1),
            recall_total=self.recall_total + 1,
        )

    def retrieve(self, choice: Int_) -> "CMR":
        """Process an observed or simulated recall outcome.

        Parameters
        ----------
        choice : Int_
            One-indexed item identity, or zero to terminate recall.

        Returns
        -------
        CMR
            Model after reinstatement or termination.

        """
        return lax.cond(
            choice == 0,
            lambda: self.replace(is_active=False),
            lambda: self.retrieve_item(choice - 1),
        )

    def candidate_activations(self, candidates: Array) -> Array:
        """Calculate retrieval weights from context-to-item associations.

        Parameters
        ----------
        candidates : Bool[Array, "item_count"]
            Mask identifying items eligible for competition.

        Returns
        -------
        Float[Array, "item_count"]
            Unnormalized Luce weights, with zero for excluded items.

        """
        item_count = self.study_items.size
        valid = jnp.arange(item_count) < self.study_index
        overlaps = self.study_contexts @ self.context.state
        contributions = (
            overlaps * _primacy(self.parameters, jnp.arange(item_count)) * valid
        )
        item_codes = jax.nn.one_hot(
            self.study_items - 1, item_count, dtype=self.parameters.dtype
        )
        # Summing occurrence contributions by identity reconstructs an MCF
        # probe without materializing the dense association matrix.
        learned = contributions @ item_codes
        support = _retrieval_support(self.parameters, self.context.state, learned)
        return _choice_weights(self.parameters, support, candidates)

    def activations(self) -> Array:
        """Return unnormalized retrieval weights for currently recallable items."""
        return self.candidate_activations(self.recallable)

    def stop_probability(self) -> Array:
        """Return one after exhaustion or an explicit stop, and zero otherwise."""
        return self.termination_policy.stop_probability(self)

    def item_probability(self, item_index: Int_) -> Array:
        """Return an item's conditional probability given continued recall.

        Parameters
        ----------
        item_index : Int_
            Zero-indexed item identity.

        Returns
        -------
        Float[Array, ""]
            Normalized choice weight; zero if no items are available.

        """
        weights = self.activations()
        return weights[item_index] / jnp.where(weights.sum() != 0, weights.sum(), 1.0)

    def outcome_probability(self, choice: Int_) -> Array:
        """Return the probability of a recall or termination event.

        Parameters
        ----------
        choice : Int_
            One-indexed item identity, or zero for STOP.

        Returns
        -------
        Float[Array, ""]
            Probability under the current context and availability state.

        """
        stop = self.stop_probability()
        return lax.cond(
            choice == 0,
            lambda: stop,
            lambda: (1 - stop) * self.item_probability(choice - 1),
        )

    def outcome_probabilities(self) -> Array:
        """Return probabilities for STOP followed by all reserved item labels.

        Returns
        -------
        Float[Array, "item_count+1"]
            Probabilities indexed by the same outcome codes as retrieve().

        """
        stop = self.stop_probability()
        weights = self.activations()
        return jnp.concatenate(
            (
                jnp.asarray([stop]),
                (1 - stop)
                * weights
                / jnp.where(weights.sum() != 0, weights.sum(), 1.0),
            )
        )


def _validate_trial_arrays(
    present: np.ndarray, recalls: np.ndarray, lengths: np.ndarray, subjects: np.ndarray
) -> None:
    """Require integer study/recall matrices and one length and subject per trial."""
    if (
        present.ndim != 2
        or recalls.ndim != 2
        or not present.shape[0]
        or not present.shape[1]
    ):
        raise ValueError(
            "Expected nonempty trial-by-position presentations and trial-by-output recalls."
        )
    if recalls.shape[0] != present.shape[0] or lengths.shape != (present.shape[0],):
        raise ValueError("Presentation, recall and listLength trial counts must match.")
    if subjects.size != present.shape[0]:
        raise ValueError("subject must contain one entry per trial.")
    if any(
        not np.issubdtype(array.dtype, np.integer)
        for array in (present, recalls, lengths)
    ):
        raise ValueError(
            "Presentations, recalls and listLength must use integer arrays."
        )


def _validate_responses(present: np.ndarray, recalls: np.ndarray) -> None:
    """Require in-list position codes, trailing padding and no repeated items.

    Different position codes can identify the same repeated item. Validation
    therefore checks response identity after mapping positions to study labels.
    """
    if np.any((recalls < 0) | (recalls > present.shape[1])):
        raise ValueError(
            "Recalls must be in-list position codes followed by zero padding."
        )
    counts = np.count_nonzero(recalls, axis=1)
    if not np.array_equal(
        recalls != 0, np.arange(recalls.shape[1])[None, :] < counts[:, None]
    ):
        raise ValueError("Only trailing recall padding is supported.")
    for row, response, count in zip(present, recalls, counts, strict=True):
        if len(np.unique(row[response[:count] - 1])) != count:
            raise ValueError(
                "Repeated item responses require a different likelihood configuration."
            )


def _validated_data(dataset: RecallDataset) -> tuple[np.ndarray, np.ndarray]:
    """Snapshot host-side observations supported by this CMR implementation.

    Returns integer matrices with shapes (trials, study_positions) and
    (trials, recall_outputs). Presentations retain one-indexed item labels;
    recalls retain one-indexed study positions, with zero denoting STOP/padding.
    This validation runs during factory construction, outside JIT.
    """
    present = np.asarray(dataset["pres_itemnos"])
    recalls = np.asarray(dataset["recalls"])
    lengths = np.asarray(dataset["listLength"]).reshape(-1)
    _validate_trial_arrays(present, recalls, lengths, np.asarray(dataset["subject"]))
    if np.any(lengths != present.shape[1]):
        raise ValueError(
            "optimized_cmr requires complete, unpadded study lists of equal length."
        )
    if np.any((present < 1) | (present > present.shape[1])):
        raise ValueError("Presented labels must lie in 1..list_length.")
    _validate_responses(present, recalls)
    # Copy once: a caller editing the original NumPy data must not change what
    # an already compiled or prepared selection means.
    return present.astype(np.int32, copy=True), recalls.astype(np.int32, copy=True)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _Group:
    """Prepared trials sharing a compressed item space and array geometry.

    ``present`` contains distinct study patterns, rather than one row per trial;
    ``mapping`` selects each trial's pattern. ``rows`` preserves the original
    selection order for inspection; padded rows use -1. Item and study-position
    indices are zero-based throughout this internal representation.

    ``positions`` lists each item's study occurrences; ``exists`` excludes its
    padding. The corresponding recall_* arrays gather those occurrences for
    each observed response. ``slots`` and ``remember`` describe which learned
    item inputs must be retained for later study presentations. ``eligible``
    describes item availability immediately before each output, including
    outputs excluded from scoring. ``valid`` distinguishes real recalls from
    padding. ``recall_count`` permits a constant prediction for empty groups.

    Shape names: patterns = distinct study schedules, trials = selected lists,
    outputs = retained recall width, items = distinct identities, occurrences =
    maximum presentations of an item. Geometry scalars are static JIT metadata.
    """

    rows: Integer[Array, " trials"]
    item_count: int = field(metadata={"static": True})
    present: Integer[Array, "patterns study_positions"]
    mapping: Integer[Array, " trials"]
    recalled: Integer[Array, "trials outputs"]
    valid: Bool[Array, "trials outputs"]
    eligible: Bool[Array, "trials outputs items"]
    positions: Integer[Array, "patterns items occurrences"]
    exists: Bool[Array, "patterns items occurrences"]
    slots: Integer[Array, "patterns study_positions"]
    remember: Bool[Array, "patterns study_positions"]
    cache_size: int = field(metadata={"static": True})
    recall_positions: Integer[Array, "trials outputs occurrences"]
    recall_exists: Bool[Array, "trials outputs occurrences"]
    recall_count: int = field(metadata={"static": True})


def _occurrence_lookup(
    patterns: np.ndarray, item_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Index study positions by pattern and item, padding the occurrence axis.

    Returns position indices and a Boolean existence mask, both shaped
    (patterns, items, max_occurrences). Every repeated presentation is retained.
    """
    counts = np.asarray([np.bincount(row, minlength=item_count) for row in patterns])
    positions = np.zeros((len(patterns), item_count, int(counts.max())), dtype=np.int32)
    exists = np.zeros_like(positions, dtype=bool)
    for trial, row in enumerate(patterns):
        for item in range(item_count):
            occurrences = np.flatnonzero(row == item)
            positions[trial, item, : len(occurrences)] = occurrences
            exists[trial, item, : len(occurrences)] = True
    return positions, exists


def _encoding_cache(
    patterns: np.ndarray, occurrence_exists: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Plan the item associations that later study events will probe.

    Returns one cache slot and update mask per study event, plus cache size.
    Only repeated items need accumulated context. An item's last presentation
    needs no further cache update; its context is still saved for retrieval.
    """
    item_count = occurrence_exists.shape[1]
    repeated = occurrence_exists.sum(axis=-1) > 1
    max_repeated = int(repeated.sum(axis=1).max())
    # Mixed lists need an untouched zero slot for singleton inputs. They must
    # never probe a repeated item's accumulated context by sharing its slot.
    cache_size = max_repeated + int(0 < max_repeated < item_count)
    slots = np.full_like(patterns, max_repeated if max_repeated < item_count else 0)
    remember = np.zeros_like(patterns, dtype=bool)
    for trial, row in enumerate(patterns):
        for slot, item in enumerate(np.flatnonzero(repeated[trial])):
            occurrences = np.flatnonzero(row == item)
            slots[trial, occurrences] = slot
            remember[trial, occurrences[:-1]] = True
    return slots, remember, cache_size


class _RecallLayout(NamedTuple):
    """Host-side response identities, conditioning masks and scoring requests."""

    recalled: np.ndarray
    valid: np.ndarray
    eligible: np.ndarray
    scoring: np.ndarray
    count: int


def _recall_layout(
    present: np.ndarray, recalls: np.ndarray, mask: np.ndarray, item_count: int
) -> _RecallLayout:
    """Map position-coded responses to items and availability before each recall.

    Inputs are trial matrices; present contains compressed zero-based labels.
    Keep only the longest real recall sequence, or one inert output if empty.
    Scoring masks never remove real responses from the availability history.
    """
    counts = np.count_nonzero(recalls, axis=1)
    width = max(1, int(counts.max()))
    retained = min(width, recalls.shape[1])
    codes = np.zeros((len(present), width), dtype=np.int32)
    codes[:, :retained] = recalls[:, :retained]
    recalled = np.take_along_axis(present, np.maximum(codes - 1, 0), axis=1)
    valid = np.arange(width)[None, :] < counts[:, None]
    eligible = np.ones((len(present), width, item_count), dtype=bool)
    for trial, count in enumerate(counts):
        for output in range(count):
            eligible[trial, output, recalled[trial, :output]] = False
    scoring = np.zeros_like(valid)
    scoring[:, :retained] = mask[:, :retained]
    return _RecallLayout(recalled, valid, eligible, scoring & valid, int(counts.sum()))


def _prepare_group(
    present: np.ndarray,
    recalls: np.ndarray,
    mask: np.ndarray,
    rows: np.ndarray,
    item_count: int,
) -> tuple[_Group, Array]:
    """Prepare one common item space and return its aligned scoring mask.

    Standard temporal CMR is invariant to permutation of item labels. Compress
    unused item units and share identical study schedules; preserve all study
    occurrences, selected trials and response histories.
    """
    remapped = np.asarray(
        [np.unique(row, return_inverse=True)[1] for row in present], dtype=np.int32
    )
    patterns, mapping = np.unique(remapped, axis=0, return_inverse=True)
    positions, exists = _occurrence_lookup(patterns, item_count)
    slots, remember, cache_size = _encoding_cache(patterns, exists)
    layout = _recall_layout(remapped, recalls, mask, item_count)
    group = _Group(
        rows=jnp.asarray(rows, dtype=jnp.int32),
        item_count=item_count,
        present=jnp.asarray(patterns),
        mapping=jnp.asarray(mapping, dtype=jnp.int32),
        recalled=jnp.asarray(layout.recalled),
        valid=jnp.asarray(layout.valid),
        eligible=jnp.asarray(layout.eligible),
        positions=jnp.asarray(positions),
        exists=jnp.asarray(exists),
        slots=jnp.asarray(slots),
        remember=jnp.asarray(remember),
        cache_size=cache_size,
        recall_positions=jnp.asarray(positions[mapping[:, None], layout.recalled]),
        recall_exists=jnp.asarray(exists[mapping[:, None], layout.recalled]),
        recall_count=layout.count,
    )
    return group, jnp.asarray(layout.scoring)


def _stop_predictions(recalls: np.ndarray, item_counts: np.ndarray) -> Array:
    """Predict STOP without a context trajectory under NoStopTermination.

    The first zero has probability zero before exhaustion and one afterward.
    Later zeros follow an explicit stop, so they always have probability one.
    Nonzero response positions are unused; the caller supplies a STOP mask.
    """
    counts = np.count_nonzero(recalls, axis=1)
    first_stop = np.arange(recalls.shape[1])[None, :] == counts[:, None]
    return jnp.asarray(np.where(first_stop, (counts >= item_counts)[:, None], 1.0))


class _CMRSequence:
    """Host-side preparation of one trial selection into probability blocks.

    Numerical model calculations live in the functions below. Preparation only
    determines observation indexing, sparse storage and aligned scoring masks.
    """

    def __init__(
        self, present: np.ndarray, recalls: np.ndarray, mask: np.ndarray
    ) -> None:
        """Group selected trials by distinct-item count and prepare STOP requests."""
        self.trial_count = len(present)
        self.groups: list[_Group] = []
        self.masks: list[Array] = []
        item_counts = np.asarray([len(np.unique(row)) for row in present])
        for item_count in np.unique(item_counts):
            rows = np.flatnonzero(item_counts == item_count)
            group, scoring = _prepare_group(
                present[rows], recalls[rows], mask[rows], rows, int(item_count)
            )
            self.groups.append(group)
            self.masks.append(scoring)

        stop_mask = mask & (recalls == 0)
        self.stops = None
        if np.any(stop_mask):
            self.stops = _stop_predictions(recalls, item_counts)
            self.masks.append(jnp.asarray(stop_mask))

    def prepared(self) -> PreparedPrediction:
        """Package dynamic observations with their shared compiled evaluator."""
        return PreparedPrediction(
            (tuple(self.groups), self.stops), tuple(self.masks), _predict
        )


def _encode_study_pattern(
    parameters: _CMRParameters,
    present: Array,
    slots: Array,
    remember: Array,
    item_count: int,
    cache_size: int,
) -> tuple[Array, Array]:
    """Encode one study schedule using only associations probed later in study.

    present, slots and remember each have shape (study_positions,). Returns the
    initial recall context (items + 1,) and every pre-update study context
    (study_positions, items + 1). Repetitions retain separate study contexts.
    The cache is only temporary encoding storage, not the retrieval memory.
    """
    initial = jnp.zeros(item_count + 1, dtype=parameters.dtype).at[0].set(1)
    if cache_size == 0:

        def unique_step(state, item):
            memory_input = _item_input(
                parameters, item, jnp.zeros_like(state), item_count
            )
            updated = integrate_normalized(
                state, normalize_magnitude(memory_input), parameters.encoding_drift_rate
            )
            return updated, state

        end, study = lax.scan(unique_step, initial, present)
    else:

        def repeated_step(carry, event):
            state, history = carry
            item, slot, keep = event
            memory_input = _item_input(parameters, item, history[slot], item_count)
            updated = integrate_normalized(
                state, normalize_magnitude(memory_input), parameters.encoding_drift_rate
            )
            history = history.at[slot].add(jnp.where(keep, state, 0.0))
            return (updated, history), state

        cache = jnp.zeros((cache_size, item_count + 1), dtype=parameters.dtype)
        (end, _), study = lax.scan(
            repeated_step, (initial, cache), (present, slots, remember)
        )
    start = integrate_normalized(end, initial, parameters.start_drift_rate)
    return start, study


def _recall_contexts(
    parameters: _CMRParameters, start: Array, normalized_inputs: Array
) -> Array:
    """Return context immediately before each observed recall in one trial.

    normalized_inputs and the result have shape (outputs, context_units).
    Regular CMR's observed item determines reinstatement without retrieval
    scores. This permits constructing the trajectory before scoring choices.
    """

    def step(state, memory_input):
        return integrate_normalized(
            state, memory_input, parameters.recall_drift_rate
        ), state

    return lax.scan(step, start, normalized_inputs)[1]


def _reinstatement_inputs(
    parameters: _CMRParameters, study: Array, group: _Group
) -> Array:
    """Reconstruct normalized MFC inputs for all observed responses.

    study has shape (trials, study_positions, context_units). Summing every
    occurrence's saved context reconstructs the learned item association.
    Returns (trials, outputs, context_units), ready for sequential integration.
    """
    associations = jax.vmap(
        lambda states, positions, exists: (states[positions] * exists[..., None]).sum(
            axis=-2
        )
    )(study, group.recall_positions, group.recall_exists)
    memory_inputs = _item_input(
        parameters, group.recalled, associations, group.item_count
    )
    return jax.vmap(jax.vmap(normalize_magnitude))(memory_inputs)


def _learned_retrieval_support(
    parameters: _CMRParameters, states: Array, study: Array, group: _Group
) -> Array:
    """Reconstruct learned MCF support from context overlaps and primacy weights.

    states and study contain one vector per recall output and study position,
    respectively, in each trial. Sum all presentations of an item after weighting
    each by its own study position. Returns (trials, outputs, items).
    """
    overlaps = jnp.einsum("bkd,bld->bkl", states, study)
    occurrence_overlaps = (
        jax.vmap(lambda values, positions: values[:, positions])(
            overlaps, group.positions[group.mapping]
        )
        * group.exists[group.mapping, None, :, :]
    )
    primacy = _primacy(parameters, group.positions[group.mapping])
    return (occurrence_overlaps * primacy[:, None, :, :]).sum(axis=-1)


def _group_probabilities(parameters: _CMRParameters, group: _Group) -> Array:
    """Predict a group's observed recalls, returning (trials, outputs).

    Encode distinct study schedules once, then reinstate each trial's observed
    items and score their choices. Inert output slots receive probability one.
    """
    if not group.recall_count:
        return jnp.ones(group.valid.shape, dtype=parameters.dtype)
    start, study = jax.vmap(_encode_study_pattern, in_axes=(None, 0, 0, 0, None, None))(
        parameters,
        group.present,
        group.slots,
        group.remember,
        group.item_count,
        group.cache_size,
    )
    study_by_trial = study[group.mapping]
    inputs = _reinstatement_inputs(parameters, study_by_trial, group)
    states = jax.vmap(_recall_contexts, in_axes=(None, 0, 0))(
        parameters, start[group.mapping], inputs
    )
    learned = _learned_retrieval_support(parameters, states, study_by_trial, group)
    support = _retrieval_support(parameters, states, learned)
    activations = jax.vmap(
        jax.vmap(_choice_weights, in_axes=(None, 0, 0)), in_axes=(None, 0, 0)
    )(parameters, support, group.eligible)
    probabilities = jnp.take_along_axis(
        activations, group.recalled[..., None], axis=-1
    )[..., 0] / activations.sum(axis=-1)
    return jnp.where(group.valid, probabilities, 1.0)


def _group_geometry(group: _Group) -> tuple[int, int, int, int]:
    """Identify structural dimensions that common-layout padding leaves fixed."""
    return (
        group.item_count,
        group.cache_size,
        group.present.shape[1],
        group.positions.shape[-1],
    )


def _pad_array(
    array: Array, shape: tuple[int, ...], *, edge: bool = False, value: float = 0
) -> Array:
    """Pad a prepared host array to a larger shape, rejecting accidental truncation."""
    values = np.asarray(array)
    widths = [
        (0, target - actual) for actual, target in zip(values.shape, shape, strict=True)
    ]
    if any(width < 0 for _, width in widths):
        raise ValueError("Prepared layout would discard observations.")
    result = (
        np.pad(values, widths, mode="edge")
        if edge
        else np.pad(values, widths, constant_values=value)
    )
    return jnp.asarray(result)


class _GroupShape(NamedTuple):
    """Dimensions that may grow to share a compiled calculation across selections."""

    patterns: int
    trials: int
    outputs: int


def _pad_group(group: _Group, shape: _GroupShape) -> _Group:
    """Pad observations without adding scored events or changing real predictions.

    Duplicate edge study patterns keep dummy context calculations well-defined;
    false validity/scoring masks ensure those dummy events contribute nothing.
    """
    patterns, trials, outputs = shape

    n, _, length, occurrences = _group_geometry(group)
    return replace(
        group,
        rows=_pad_array(group.rows, (trials,), value=-1),
        present=_pad_array(group.present, (patterns, length), edge=True),
        mapping=_pad_array(group.mapping, (trials,)),
        recalled=_pad_array(group.recalled, (trials, outputs)),
        valid=_pad_array(group.valid, (trials, outputs), value=False),
        eligible=_pad_array(group.eligible, (trials, outputs, n), value=True),
        positions=_pad_array(group.positions, (patterns, n, occurrences), edge=True),
        exists=_pad_array(group.exists, (patterns, n, occurrences), edge=True),
        slots=_pad_array(group.slots, (patterns, length), edge=True),
        remember=_pad_array(group.remember, (patterns, length), edge=True),
        recall_positions=_pad_array(
            group.recall_positions, (trials, outputs, occurrences)
        ),
        recall_exists=_pad_array(
            group.recall_exists, (trials, outputs, occurrences), value=False
        ),
        # Keep only the empty/nonempty distinction static. Empty groups must
        # remain constant even at singular, unused context-drift boundaries.
        # Ordinary changes in recall count do not change the compilation key.
        recall_count=int(bool(group.recall_count)),
    )


def _common_group_shapes(sequences: list[_CMRSequence]) -> tuple[_GroupShape, ...]:
    """Find padding sizes for corresponding groups in compatible selections."""
    return tuple(
        _GroupShape(
            max(group.present.shape[0] for group in groups),
            max(group.valid.shape[0] for group in groups),
            max(group.valid.shape[1] for group in groups),
        )
        for groups in zip(*(sequence.groups for sequence in sequences), strict=True)
    )


def _pad_sequence(
    sequence: _CMRSequence, shapes: tuple[_GroupShape, ...], stop_trials: int
) -> PreparedPrediction:
    """Align one selection's probability blocks and masks with a shared layout."""
    groups = tuple(
        _pad_group(group, shape)
        for group, shape in zip(sequence.groups, shapes, strict=True)
    )
    masks = tuple(
        _pad_array(mask, (shape.trials, shape.outputs))
        for mask, shape in zip(sequence.masks[: len(groups)], shapes, strict=True)
    )
    stops = sequence.stops
    if stops is not None:
        stop_shape = (stop_trials, stops.shape[1])
        stops = _pad_array(stops, stop_shape, value=1.0)
        masks += (_pad_array(sequence.masks[-1], stop_shape),)
    return PreparedPrediction((groups, stops), masks, _predict)


def _prepare_compatible_selections(
    sequences: list[_CMRSequence],
) -> tuple[PreparedPrediction, ...]:
    """Share array layouts where model geometry permits, preserving input order.

    Study length, item space, cache geometry and occurrence capacity remain
    fixed within a family. Other axes can grow through masked padding. This
    enables compilation reuse without sharing observations between subjects.
    """
    families: dict[tuple, list[int]] = {}
    for index, sequence in enumerate(sequences):
        key = (
            tuple(_group_geometry(group) for group in sequence.groups),
            sequence.stops is not None,
        )
        families.setdefault(key, []).append(index)
    result: dict[int, PreparedPrediction] = {}
    for members in families.values():
        family = [sequences[index] for index in members]
        shapes = _common_group_shapes(family)
        stop_trials = max(sequence.trial_count for sequence in family)
        for index in members:
            result[index] = _pad_sequence(sequences[index], shapes, stop_trials)
    return tuple(result[index] for index in range(len(sequences)))


def _predict(
    data: tuple[tuple[_Group, ...], Array | None], parameters: Mapping[str, Float_]
) -> tuple[Array, ...]:
    """Evaluate prepared observation blocks with a complete parameter mapping."""
    groups, stops = data
    parsed = _parse_parameters(parameters)
    values = tuple(_group_probabilities(parsed, group) for group in groups)
    return values + (() if stops is None else (stops,))


class ModelFactory:
    """Construct efficient CMR events or prepare its observed-sequence prediction.

    The same factory determines both paths. There is no foreign model factory
    whose equations can be ignored. Unsupported features or parameters fail
    explicitly. Preparation accepts position-coded dataset recalls, snapshots
    observations, and preserves requested trials' order and multiplicity.
    """

    def __init__(self, dataset: RecallDataset, features: Array | None = None) -> None:
        """Validate and snapshot the study and recall arrays.

        Parameters
        ----------
        dataset : RecallDataset
            Equal-length, unpadded study lists. Repeated presentations are
            supported; repeated recalled identities are not. Recalls contain
            one-indexed study positions followed by zero padding.
        features : Array or None, optional
            Must be None: this implementation has no semantic feature pathway.

        """
        if features is not None:
            raise ValueError(
                "optimized_cmr has no semantic feature pathway; features must be None."
            )
        self._present, self._recalls = _validated_data(dataset)
        self.max_list_length = self._present.shape[1]

    def create_model(self, parameters: Mapping[str, Float_]) -> CMR:
        """Create an empty event model with this dataset's study capacity.

        Parameters
        ----------
        parameters : Mapping[str, Float_]
            Complete regular-CMR parameter mapping, including the two fixed flags.

        Returns
        -------
        CMR
            An unstudied model suitable for simulation or sequential likelihoods.

        """
        return CMR(self.max_list_length, parameters)

    def create_trial_model(
        self, trial_index: Int_, parameters: Mapping[str, Float_]
    ) -> CMR:
        """Create an empty model for the ordinary per-trial driver.

        Parameters
        ----------
        trial_index : Int_
            Accepted for factory compatibility. This model has no trial-specific
            initialization; study events subsequently supply all trial inputs.
        parameters : Mapping[str, Float_]
            Complete regular-CMR parameter mapping.

        Returns
        -------
        CMR
            An unstudied event model.

        """
        return self.create_model(parameters)

    def _sequence(self, trial_indices: Array, mask: Array | None) -> _CMRSequence:
        """Validate a concrete selection and prepare it outside compiled model code."""
        indices = np.asarray(trial_indices)
        if (
            indices.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any((indices < 0) | (indices >= len(self._present)))
        ):
            raise ValueError(
                "trial_indices must be a one-dimensional array of valid trial indices."
            )
        shape = (len(indices), self._recalls.shape[1])
        mask = np.ones(shape, dtype=bool) if mask is None else np.asarray(mask)
        if mask.shape != shape or mask.dtype != np.bool_:
            raise ValueError(
                "Prediction mask must be Boolean and match the selected recall array."
            )
        return _CMRSequence(self._present[indices], self._recalls[indices], mask)

    def prepare_predictions(
        self, trial_indices: Array, mask: Array | None = None
    ) -> PreparedPrediction:
        """Prepare probability blocks for requested outputs, including STOP.

        With no mask, predict all supplied outcomes. A false mask only excludes
        scoring that outcome; it never removes a real recall from conditioning.
        Grouped output blocks retain all requested events and their multiplicity.

        Parameters
        ----------
        trial_indices : Integer[Array, "trials"]
            Concrete, zero-indexed dataset rows. Repeated indices are allowed.
        mask : Bool[Array, "trials recall_outputs"] or None, optional
            Scoring requests aligned with the selected recall matrix. None
            requests every supplied output, including zero-coded STOP events.

        Returns
        -------
        PreparedPrediction
            Immutable probability plan with aligned blocks and scoring masks.
            Preparation runs outside JIT; probability evaluation supports JIT,
            vmap and differentiation with respect to model parameters.

        """
        return self._sequence(trial_indices, mask).prepared()

    def prepare_prediction_selections(
        self,
        trial_indices: Iterable[Array],
        masks: Iterable[Array],
    ) -> tuple[PreparedPrediction, ...]:
        """Prepare multiple selections with common layouts where possible.

        Parameters
        ----------
        trial_indices : Iterable[Array]
            One concrete trial-index vector per selection, in requested order.
        masks : Iterable[Array]
            One Boolean scoring matrix per selection, aligned with its recalls.

        Returns
        -------
        tuple[PreparedPrediction, ...]
            One prediction plan per selection. Compatible selections have
            matching array shapes to reuse compilation; masked padding does
            not contribute observations or change their conditioning history.

        """
        sequences = [
            self._sequence(indices, mask)
            for indices, mask in zip(trial_indices, masks, strict=True)
        ]
        return _prepare_compatible_selections(sequences)
