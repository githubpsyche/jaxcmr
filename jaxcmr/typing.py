"""Data formats and callable interfaces for recall models, simulation and fitting.

``RecallDataset`` describes the arrays recorded on each trial, and
``Parameters`` maps parameter names to scalar model values. Array annotations
specify the dtype and dimensions a caller supplies.

``Memory``, ``Context``, ``TerminationPolicy`` and ``ResponsePolicy`` describe components used to
build a model. ``MemorySearch`` describes how a model experiences study
items and produces recall responses. A ``MemorySearchModelFactory``
constructs that model for a particular trial. A ``TrialSimulator`` uses it
to carry out a study and recall procedure.

For fitting, a ``LossFactory`` constructs an objective for a dataset.
``LossFn`` scores candidate parameter sets, and a ``FittingAlgorithm``
returns estimates and objective values in a ``FitResult``. These protocols
specify the operations an implementation must supply. A model, loss or
fitter can satisfy its protocol without inheriting from it.
"""

from collections.abc import Iterable, Mapping
from typing import Any, NotRequired, Protocol, runtime_checkable

from jaxtyping import Array, ArrayLike, Bool, Float, Integer, PRNGKeyArray, Real, Shaped
from typing_extensions import TypedDict

type Float_ = Float[Array, ""] | float | int
"""Scalar floating-point JAX array or Python numeric value."""

type Int_ = Integer[Array, ""] | int
"""Scalar integer JAX array or Python integer."""

type Bool_ = Bool[Array, ""] | bool
"""Scalar Boolean JAX array or Python Boolean."""

type Parameters = Mapping[str, Float_ | Int_ | Bool_]
"""Model parameter names mapped to scalar numeric or Boolean values."""

__all__ = [
    "Array",
    "ArrayLike",
    "Bool",
    "Bool_",
    "Context",
    "ContextCreateFn",
    "FitResult",
    "FittingAlgorithm",
    "Float",
    "Float_",
    "Int_",
    "Integer",
    "LikelihoodMaskFn",
    "LossFactory",
    "LossFn",
    "Memory",
    "MemoryCreateFn",
    "MemorySearch",
    "MemorySearchModelFactory",
    "PRNGKeyArray",
    "Parameters",
    "Real",
    "RecallDataset",
    "ResponsePolicy",
    "ResponsePolicyCreateFn",
    "Shaped",
    "TerminationPolicy",
    "TerminationPolicyCreateFn",
    "TrialSimulator",
]


class RecallDataset(TypedDict, extra_items=Array):
    """Trial-indexed arrays for free or serial recall experiments.

    The required fields are ``subject``, ``listLength``, ``pres_itemnos``,
    and ``recalls``. Their annotated shapes and integer dtypes appear below;
    zeros pad item and recall sequences.

    ``pres_itemids`` identifies items across lists and is required by semantic
    models and analyses, but not by every recall dataset. Other named fields
    are optional. Additional fields may contain arrays of any dtype or shape,
    such as valence or LPP measurements. Callers must supply the optional or
    additional fields required by their chosen operation. Every field must
    have a leading trial dimension; non-trial metadata belongs outside this dict.
    """

    # REQUIRED FIELDS

    subject: Integer[Array, "n_trials 1"]
    """Subject ID for each trial (shape: [n_trials, 1])."""

    listLength: Integer[Array, "n_trials 1"]
    """Number of valid study events per trial, including repeated presentations
    (shape: [n_trials, 1])."""

    pres_itemnos: Integer[Array, "n_trials num_presented"]
    """Per-trial within-list item codes (shape: [n_trials, num_presented]).
    Positive item codes repeat when an item is presented again; 0 pads unused
    study slots. These codes are distinct from study positions and global IDs."""

    recalls: Integer[Array, "n_trials num_recalled"]
    """Study-position codes for recalled items (shape: [n_trials, num_recalled]).
    A positive code p identifies the item at pres_itemnos[trial, p - 1];
    0 marks an unused response slot. Model adapters translate positions to
    within-list item codes, and dataset simulation returns first-occurrence
    study positions when an item was presented repeatedly."""

    # OPTIONAL FIELD REQUIRED BY SEMANTIC ANALYSES
    pres_itemids: NotRequired[Integer[Array, "n_trials num_presented"]]
    """Per-trial cross-list item IDs (shape: [n_trials, num_presented]). 
    These IDs reference a global word pool and may repeat across trials."""

    # OPTIONAL FIELDS
    rec_itemids: NotRequired[Integer[Array, "n_trials num_recalled"]]
    """Cross-list item IDs for recalled items (shape: [n_trials, num_recalled])."""

    # OPTIONAL FIELDS, MISC
    irt: NotRequired[Integer[Array, "n_trials num_recalled"]]
    """Item response times for recalled items (shape: [n_trials, num_recalled])."""

    session: NotRequired[Integer[Array, "n_trials 1"]]
    """Session IDs for each trial (shape: [n_trials, 1])."""

    listtype: NotRequired[Integer[Array, "n_trials 1"]]
    """List type for each trial (shape: [n_trials, 1])."""

    list_type: NotRequired[Integer[Array, "n_trials 1"]]
    """List type for each trial (shape: [n_trials, 1])."""


@runtime_checkable
class Memory(Protocol):
    """Associate and retrieve feature patterns through a shared memory interface.

    Linear memory stores an input-by-output matrix; instance memory stores
    rows of concatenated input/output patterns. ``probe`` and ``associate``
    provide the same operations without requiring callers to interpret that
    storage layout.

    Attributes
    ----------
    state : Float[Array, "memory_rows memory_columns"]
        Stored associations, with layout determined by the implementation."""

    state: Float[Array, " memory_rows memory_columns"]

    @property
    def input_size(self) -> int:
        "The size of the input feature space."
        ...

    @property
    def output_size(self) -> int:
        "The size of the output feature space."
        ...

    def associate(
        self,
        in_pattern: Float[Array, " input_size"],
        out_pattern: Float[Array, " output_size"],
        learning_rate: Float_,
    ) -> "Memory":
        """Return memory after associating the supplied patterns.

        Parameters
        ----------
        in_pattern : Float[Array, "input_size"]
            Input feature pattern.
        out_pattern : Float[Array, "output_size"]
            Output feature pattern.
        learning_rate : Float_
            Strength of the association update.

        Returns
        -------
        Memory
            Updated memory; the original instance is unchanged."""
        ...

    def probe(
        self,
        in_pattern: Float[Array, " input_size"],
    ) -> Float[Array, " output_size"]:
        """Retrieve the output pattern associated with an input cue.

        Parameters
        ----------
        in_pattern : Float[Array, "input_size"]
            Input cue in the memory's feature space.

        Returns
        -------
        Float[Array, "output_size"]
            Retrieved output pattern, without updating memory."""
        ...


class MemoryCreateFn(Protocol):
    """Construct one associative memory for a model.

    The same interface is used separately for item-to-context (MFC) and
    context-to-item (MCF) memory. The chosen constructor determines the
    direction and storage representation."""

    def __call__(
        self,
        list_length: int,
        parameters: Parameters,
        context: "Context",
    ) -> Memory:
        """Create one memory using the experiment and context dimensions.

        Parameters
        ----------
        list_length : int
            Number of study slots available to the model.
        parameters : Parameters
            Model parameter values used to configure this memory.
        context : Context
            Context representation determining the context feature space.

        Returns
        -------
        Memory
            A newly initialized associative memory."""
        ...


@runtime_checkable
class Context(Protocol):
    """Integrate new input into a model's context representation.

    Attributes
    ----------
    state : Float[Array, "context_feature_units"]
        Current context vector.
    initial_state : Float[Array, "context_feature_units"]
        Initial context, also available as a retrieval-start cue.
    size : int
        Number of context feature units."""

    state: Float[Array, " context_feature_units"]
    initial_state: Float[Array, " context_feature_units"]
    size: int

    def integrate(
        self,
        context_input: Float[Array, " context_feature_units"],
        drift_rate: Float_,
    ) -> "Context":
        """Return context after integrating an input representation.

        Parameters
        ----------
        context_input : Float[Array, "context_feature_units"]
            Input in the same feature space as context.
        drift_rate : Float_
            Rate of context change toward the input.

        Returns
        -------
        Context
            Updated context; the original instance is unchanged."""
        ...


class ContextCreateFn(Protocol):
    """Construct a context representation for the model's item capacity."""

    def __call__(self, item_count: int) -> Context:
        """Initialize context before study begins.

        Parameters
        ----------
        item_count : int
            Number of item slots reserved by the model. The constructor determines
            the corresponding context dimensions and any additional units.

        Returns
        -------
        Context
            A newly initialized context representation."""
        ...


@runtime_checkable
class TerminationPolicy[ModelT](Protocol):
    """A stopping rule accepting the model or view described by ``ModelT``.

    Concrete policies declare their own structural input requirements. A policy
    need not accept every ``MemorySearch`` model, and its input need not be a
    complete memory-search model.
    """

    def stop_probability(self, model: ModelT) -> Float[Array, ""]:
        """Calculate the probability of stopping in the supplied state.

        Parameters
        ----------
        model : ModelT
            Model or view exposing the information required by this policy.

        Returns
        -------
        Float[Array, ""]
            Probability of a stop event, without advancing the model."""
        ...


class TerminationPolicyCreateFn[ModelT](Protocol):
    """Create a stopping rule compatible with the supplied model type.

    Constructors and factories name the concrete model type their policy must
    accept. Variants forwarding a policy to a base constructor retain that base
    model's contract. Models passing a view to their policy use the view's type.
    These annotations check structural compatibility without changing model
    behavior or performing runtime validation.
    """

    def __call__(
        self,
        list_length: int,
        parameters: Parameters,
    ) -> TerminationPolicy[ModelT]:
        """Construct a stopping rule compatible with the chosen model.

        Parameters
        ----------
        list_length : int
            Number of study slots available to the model.
        parameters : Parameters
            Model parameters used to configure the stopping rule.

        Returns
        -------
        TerminationPolicy[ModelT]
            Policy accepting the model or view described by ModelT."""
        ...


class ResponsePolicy[ModelT](Protocol):
    """Score one retrieval attempt without advancing the model.

    Entry zero is STOP. Remaining entries correspond to one-indexed item
    codes. Inactive and exhausted models place all probability on STOP.
    """

    def outcome_probabilities(self, model: ModelT) -> Float[Array, " outcomes"]: ...


class ResponsePolicyCreateFn[ModelT](Protocol):
    """Construct a response policy whose evidence requirements the model supplies."""

    def __call__(self, list_length: int, parameters: Parameters) -> ResponsePolicy[ModelT]: ...


@runtime_checkable
class MemorySearch(Protocol):
    """Study and retrieval operations used by simulation and likelihood evaluation.

    Item choices are one-indexed within-list codes. Zero is ignored during
    study and denotes stopping during retrieval. Update methods return the
    new model state; callers retain the original state until they replace it.
    Implementations used by the numerical routines must support JAX tracing
    and carry their changing state in a JAX pytree.

    Internal representations of studied items, context and retrieval support
    belong to the concrete model. Response policies and diagnostic views
    specify any additional information they require separately.

    Attributes
    ----------
    is_active : Bool[Array, ""]
        Whether retrieval can continue.
    recall_total : Integer[Array, ""]
        Number of items recalled so far, excluding stop events.
    """

    is_active: Bool[Array, ""]
    recall_total: Integer[Array, ""]

    def experience(self, choice: Int_) -> "MemorySearch":
        """Advance the model by one study event.

        Parameters
        ----------
        choice : Int_
            One-indexed within-list item code; zero leaves the model unchanged.

        Returns
        -------
        MemorySearch
            Model after experiencing this item.
        """
        ...

    def start_retrieving(self) -> "MemorySearch":
        """Transition from study to retrieval.

        Returns
        -------
        MemorySearch
            Model ready for its first retrieval event.
        """
        ...

    def retrieve(self, choice: Int_) -> "MemorySearch":
        """Advance the model by one retrieval event.

        Parameters
        ----------
        choice : Int_
            One-indexed within-list item code, or zero to stop retrieval.

        Returns
        -------
        MemorySearch
            Model after recalling the item or stopping.
        """
        ...

    def outcome_probability(self, choice: Int_) -> Float[Array, ""]:
        """Calculate the probability of one retrieval event.

        Parameters
        ----------
        choice : Int_
            One-indexed within-list item code, or zero for a stop event.

        Returns
        -------
        Float[Array, ""]
            Event probability in the current state, without retrieving it.
        """
        ...

    def outcome_probabilities(self) -> Float[Array, " recall_outcomes"]:
        """Calculate probabilities for every retrieval event.

        Returns
        -------
        Float[Array, "recall_outcomes"]
            Stop probability at index zero, followed by probabilities for
            the corresponding one-indexed item codes. The model is unchanged.
        """
        ...


@runtime_checkable
class MemorySearchModelFactory(Protocol):
    """Prepare a dataset once, then construct a model for an individual trial.

    Simulation and losses receive a factory class, initialize it with their
    dataset and optional features, and call ``create_trial_model`` for each
    trial and parameter assignment. Trial-specific connections and other
    prepared inputs are the factory's responsibility.
    """

    def __init__(
        self,
        dataset: RecallDataset,
        features: Float[Array, " word_pool_items features_count"] | None,
    ) -> None:
        """Prepare the inputs shared across model constructions.

        Parameters
        ----------
        dataset : RecallDataset
            Trial data, including any fields required by the chosen model.
        features : Float[Array, "word_pool_items features_count"] or None
            Item features aligned to the model's vocabulary, when needed."""
        ...

    def create_trial_model(
        self,
        trial_index: Integer[Array, ""],
        parameters: Parameters,
    ) -> MemorySearch:
        """Construct a model ready to study the selected trial.

        Parameters
        ----------
        trial_index : Integer[Array, ""]
            Zero-based row in the dataset supplied to the factory.
        parameters : Parameters
            One complete assignment of model parameter values.

        Returns
        -------
        MemorySearch
            New model using the prepared inputs for this trial."""
        ...


@runtime_checkable
class TrialSimulator(Protocol):
    """Run a study and retrieval procedure for one trial.

    The procedure determines how observed recalls constrain simulation: it may
    ignore them, impose the first recall, or use their count or occupied slots.
    Dataset simulation translates recorded study positions into within-list
    item codes before calling this interface, and translates generated item
    codes back to study positions afterward."""

    def __call__(
        self,
        model: MemorySearch,
        present: Integer[Array, " study_events"],
        trial: Integer[Array, " recalls"],
        rng: PRNGKeyArray,
    ) -> tuple[MemorySearch, Integer[Array, " recall_events"]]:
        """Simulate one trial under the chosen experimental procedure.

        Parameters
        ----------
        model : MemorySearch
            Model initialized for the trial, before study begins.
        present : Integer[Array, "study_events"]
            One-indexed within-list item codes in study order; zero pads the list.
        trial : Integer[Array, "recalls"]
            Observed recalls expressed in the same item codes, with zero for unused
            slots. Their use depends on the chosen trial procedure.
        rng : PRNGKeyArray
            Random key for sampling responses.

        Returns
        -------
        tuple of MemorySearch and Integer[Array, "recall_events"]
            Final model and simulated item codes. Zero denotes an unused response
            slot; the procedure determines the output width and stopping behavior."""
        ...


@runtime_checkable
class LikelihoodMaskFn(Protocol):
    """Choose which recorded recall events contribute to a likelihood."""

    def __call__(
        self, recalls: Integer[Array, " recall_events"]
    ) -> Bool[Array, " recall_events"]:
        """Mark recall events to include in the objective.

        Parameters
        ----------
        recalls : Integer[Array, "recall_events"]
            Recorded recall sequence. Positive entries denote recalled items and
            zeros follow the final recall.

        Returns
        -------
        Bool[Array, "recall_events"]
            Keep-mask aligned to the supplied sequence. The chosen rule determines
            whether the first zero contributes as a stop event."""
        ...


@runtime_checkable
class LossFn(Protocol):
    """Score candidate parameter vectors on selected observations.

    Fitters initialize a loss with the model factory, dataset and features.
    Each call supplies selected trial rows, base parameters and a matrix of
    candidate values for the free parameters. Lower objective values are better;
    the chosen loss defines the measure, such as negative log likelihood or
    behavioral mean squared error."""

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Parameters,
        free_param_names: Iterable[str],
        x: Float[Array, " parameters n_samples"],
    ) -> Float[Array, " n_samples"]:
        """Evaluate each candidate on the selected observations.

        Parameters
        ----------
        trial_indices : Integer[Array, "trials"]
            Zero-based dataset rows contributing to the objective.
        base_params : Parameters
            Base values, overridden by each candidate's free-parameter values.
        free_param_names : Iterable[str]
            Names in the same order as the rows of x.
        x : Float[Array, "parameters n_samples"]
            Candidate values in physical parameter coordinates, one candidate per
            column and one free parameter per row.

        Returns
        -------
        Float[Array, "n_samples"]
            One objective value per candidate, in the supplied column order."""
        ...


class LossFactory(Protocol):
    """Construct a loss from a model factory, observations and features.

    Classes and configured callables such as functools.partial can implement
    this interface. The returned object supplies population scoring.
    """

    def __call__(
        self,
        model_factory_cls: type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Float[Array, " word_pool_items features_count"] | None,
    ) -> LossFn:
        """Prepare an ordinary loss for the supplied model and data."""
        ...


class FitResult(TypedDict):
    """Serialized parameter estimates and optimizer results, aligned by fit row.

    ``fitness``, ``nit``, ``converged`` and every column in ``fits`` refer to the
    same rows. One fit to pooled observations produces one row; participant-wise
    fitting produces one row per participant. Optimizer diagnostics are optional.
    The objective's meaning comes from the selected loss."""

    fixed: dict[str, float | int | bool]
    """Base parameter values preserving numeric and Boolean scalar types. Free-parameter defaults may
    appear here too; fitted values in fits override them."""

    free: dict[str, list[float]]
    """Free-parameter names mapped to [lower_bound, upper_bound]."""

    fitness: list[float]
    """Objective values, one per fit row. Their meaning is determined by the
    selected loss; likelihood examples store negative log likelihood."""

    nit: list[int]
    """Number of optimizer iterations or generations run for each selected fit."""

    converged: list[bool]
    """Whether each fit met the optimizer convergence criterion."""

    fits: dict[str, list[float | int | bool]]
    """Parameter columns aligned by fit row, including integer and Boolean values.
    The reserved subject column contains integer IDs, never floats or booleans.
    Participant rows have distinct IDs; a pooled fit has one row with ID -1."""

    hyperparameters: dict[str, Any]
    """Dictionary of hyperparameter names and their values used during fitting."""

    fit_time: float
    """Total time (in seconds) taken to perform the fitting."""

    diagnostics: NotRequired[list[dict[str, Any]]]
    """Optional diagnostics aligned with fit rows. Each entry includes subject,
    selected_restart and restart records, plus solver-specific details."""


@runtime_checkable
class FittingAlgorithm(Protocol):
    """Fit one pooled parameter set or separate parameter sets by participant.

    Construct an implementation with a dataset, model factory, loss class and
    optimizer settings. ``fit`` uses all selected observations in one objective;
    ``fit_subjects`` fits the selected participants separately. Both return a
    ``FitResult`` with parameter columns and optimizer outcomes aligned by row."""

    def __init__(
        self,
        dataset: RecallDataset,
        features: Float[Array, "word_pool_items features_count"] | None,
        base_params: Parameters,
        model_factory_cls: type[MemorySearchModelFactory],
        loss_fn_cls: LossFactory,
        hyperparams: dict[str, Any] | None = None,
    ) -> None:
        """Configure the fitting algorithm.

        Parameters
        ----------
        dataset : RecallDataset
            Observed trial data, including participant identities.
        features : Float[Array, "word_pool_items features_count"] or None
            Vocabulary-aligned features passed to the model factory when needed.
        base_params : Parameters
            Base parameter values. Candidate values override the free parameters.
        model_factory_cls : type[MemorySearchModelFactory]
            Factory class used to prepare trial-specific models.
        loss_fn_cls : LossFactory
            Loss class used to score candidate parameters.
        hyperparams : dict[str, Any] or None
            Optimizer-specific settings, including free-parameter bounds. Consult
            the chosen fitter for its supported settings and defaults."""
        ...

    def fit(
        self,
        trial_mask: Bool[Array, " trials"],
        subject_id: int = -1,
    ) -> FitResult:
        """Fit one parameter set to all selected observations.

        Parameters
        ----------
        trial_mask : Bool[Array, "trials"]
            Trials included in this fit.
        subject_id : int, optional
            Identity recorded for the fit row; -1 denotes a pooled fit.

        Returns
        -------
        FitResult
            One fit row and its optimizer results."""
        ...

    def fit_subjects(
        self,
        trial_mask: Bool[Array, " trials"],
    ) -> FitResult:
        """Fit each selected participant independently.

        Parameters
        ----------
        trial_mask : Bool[Array, "trials"]
            Trials included when forming participant-specific objectives.

        Returns
        -------
        FitResult
            Participant parameter columns and optimizer results, aligned by fit row."""
        ...
