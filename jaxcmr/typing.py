"""Type definitions and protocols for the jaxcmr package.

Defines array type aliases (via jaxtyping), model protocols
(``MemorySearch``, ``TrialSimulator``), dataset type dictionaries
(``RecallDataset``), and callable type aliases used throughout the
package.

"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    NotRequired,
    Protocol,
    runtime_checkable,
)

from jaxtyping import Array, ArrayLike, Bool, Float, Integer, PRNGKeyArray, Real, Shaped
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from jaxcmr.models.sequence import PreparedPrediction

Float_ = Float[Array, ""] | float | int
Int_ = Integer[Array, ""] | int
Bool_ = Bool[Array, ""] | bool

__all__ = [
    "Array",
    "ArrayLike",
    "BatchPreparableLossFn",
    "BatchSequencePredictionFactory",
    "Bool",
    "Bool_",
    "CVResult",
    "Context",
    "ContextCreateFn",
    "FitResult",
    "FittingAlgorithm",
    "Float",
    "Float_",
    "Int_",
    "Integer",
    "LikelihoodMaskFn",
    "LossFn",
    "Memory",
    "MemoryCreateFn",
    "MemorySearch",
    "MemorySearchCreateFn",
    "MemorySearchModelFactory",
    "PRNGKeyArray",
    "PreparableLossFn",
    "PreparedLossFn",
    "Real",
    "RecallAnalysisFn",
    "RecallDataset",
    "SequencePredictionFactory",
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
    such as valence or LPP measurements. A consumer that needs an optional or
    additional field must check that the dataset contains it. Every field must
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


class RecallAnalysisFn(Protocol):
    """Summarize observed or simulated recalls for selected trials.

    ``recalls`` already contains only the selected trials. Observations have
    one simulation per trial; model predictions can have many. ``dataset`` is
    the full dataset, so align any trial metadata with ``trial_indices``.
    Implementations aggregate the simulation dimension and return matching,
    fixed-shape arrays for observations and predictions. They must support
    JAX tracing; aggregation and recall-code interpretation are their explicit
    responsibility. Nonfinite observed bins can mark unavailable targets.
    """

    def __call__(
        self,
        recalls: Integer[Array, " trials simulations recall_positions"],
        *,
        list_length: int,
        dataset: Mapping[str, Array],
        trial_indices: Integer[Array, " trials"],
    ) -> Array: ...


@runtime_checkable
class Memory(Protocol):
    state: Float[Array, " input_size output_size"]

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
        """Return the updated memory after associating input and output patterns.

        Args:
            in_pattern: a feature pattern for an input.
            out_pattern: a feature pattern for an output.
            learning_rate: the learning rate parameter.
        """
        ...

    def probe(
        self,
        in_pattern: Float[Array, " input_size"],
    ) -> Float[Array, " output_size"]:
        """Return the output pattern associated with the input pattern in memory.

        Args:
            in_pattern: the input feature pattern.
        """
        ...


@runtime_checkable
class Context(Protocol):
    """Context representation for memory search models.

    Attributes:
        state: the current state of the context.
        initial_state: the initial state of the context.
    """

    state: Float[Array, " context_feature_units"]
    initial_state: Float[Array, " context_feature_units"]
    size: int

    def integrate(
        self,
        context_input: Float[Array, " context_feature_units"],
        drift_rate: Float_,
    ) -> "Context":
        """Returns context after integrating input representation.

        Args:
            context_input: the input representation to be integrated into the contextual state.
            drift_rate: The drift rate parameter.
        """
        ...

    def integrate_with_outlist(
        self,
        inlist_input: Float[Array, " context_feature_units"],
        ratio: Float_,
        drift_rate: Float_,
    ) -> "Context":
        """Return context after mixing in-list input with fresh out-of-list input.

        Requires sufficient out-of-list capacity for this update.

        Args:
            inlist_input: Input representation, normalized before mixing.
            ratio: Weight on the in-list input; out-of-list input receives 1 - ratio.
            drift_rate: Drift rate for integrating the mixed input.
        """
        ...


@runtime_checkable
class TerminationPolicy[ModelT](Protocol):
    """A stopping rule accepting the model or view described by ``ModelT``.

    Concrete policies declare their own structural input requirements. A policy
    need not accept every ``MemorySearch`` model, and its input need not be a
    complete memory-search model.
    """

    def stop_probability(self, model: ModelT) -> Float[Array, ""]:
        """Return stop probability for an object satisfying this rule's needs."""
        ...


@runtime_checkable
class MemorySearch(Protocol):
    """A model of memory search.

    Attributes:
        item_count: the number of item slots reserved in the model.
        is_active: indicates whether the model is active or not.
        studied: indicates whether each item has been studied.
        recallable: indicates whether each item can currently be recalled.
        recall_total: the number of recalled items so far.
        study_index: the number of items studied so far.
    """

    item_count: int
    is_active: Bool[Array, ""]
    studied: Bool[Array, " item_count"]
    recall_total: Integer[Array, ""]
    study_index: Integer[Array, ""]

    @property
    def recallable(self) -> Bool[Array, " item_count"]:
        """Indicate which candidates can currently be recalled."""
        ...

    def experience(self, choice: Int_) -> "MemorySearch":
        """Returns model after experiencing the specified study item.

        Args:
            choice: the index of the item to experience (1-indexed). 0 is ignored.
        """
        ...

    def start_retrieving(self) -> "MemorySearch":
        """Returns model after transitioning from study to retrieval mode."""
        ...

    def retrieve(self, choice: Int_) -> "MemorySearch":
        """Return model after simulating retrieval of the specified item or stopping.

        Args:
            choice: the index of the item to retrieve (1-indexed). 0 terminates retrieval.

        """
        ...

    def activations(self) -> Float[Array, " item_count"]:
        """Returns relative support for retrieval of each item given model state"""
        ...

    def outcome_probability(self, choice: Int_) -> Float[Array, ""]:
        """Return probability of the specified retrieval event.

        Args:
            choice: the index of the item to retrieve (1-indexed) or 0 to stop.
        """
        ...

    def outcome_probabilities(self) -> Float[Array, " recall_outcomes"]:
        """Return probabilities of all possible retrieval events."""
        ...


class ContextCreateFn(Protocol):
    """Callable that returns a context instance."""

    def __call__(self, item_count: int) -> Context:
        """Return a context instance for the given configuration."""
        ...


class MemoryCreateFn(Protocol):
    """Callable that returns the MFC and MCF memories."""

    def __call__(
        self,
        list_length: int,
        parameters: Mapping[str, Float_],
        context: Context,
    ) -> Memory:
        """Return the memories for the given configuration."""
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
        parameters: Mapping[str, Float_],
    ) -> TerminationPolicy[ModelT]:
        """Return a termination policy that can accept ``ModelT``."""
        ...


@runtime_checkable
class MemorySearchCreateFn(Protocol):
    """A factory for creating memory search models."""

    def __call__(
        self,
        list_length: int,
        parameters: Mapping[str, Float_],
        connections: Float[Array, " study_events study_events"] | None,
    ) -> MemorySearch:
        """Create a new memory search model with the specified parameters."""
        ...


@runtime_checkable
class LikelihoodMaskFn(Protocol):
    """Defines the callable signature for recall likelihood masking."""

    def __call__(
        self, recalls: Integer[Array, " recall_events"]
    ) -> Bool[Array, " recall_events"]:
        """Returns a keep-mask over recall events.

        Args:
          recalls: One-indexed recall events with zeros representing padding.
        """
        ...


@runtime_checkable
class MemorySearchModelFactory(Protocol):
    def __init__(
        self,
        dataset: RecallDataset,
        features: Float[Array, " word_pool_items features_count"] | None,
    ) -> None:
        """Initialize the factory with the specified trials and trial data."""
        ...

    def create_model(
        self,
        parameters: Mapping[str, Float_],
    ) -> MemorySearch:
        """Create a new memory search model with the specified parameters for the specified trial."""
        ...

    def create_trial_model(
        self,
        trial_index: Integer[Array, ""],
        parameters: Mapping[str, Float_],
    ) -> MemorySearch:
        """Create a new memory search model with the specified parameters for the specified trial."""
        ...


@runtime_checkable
class SequencePredictionFactory(MemorySearchModelFactory, Protocol):
    """Optional model-owned prediction of complete observed sequences.

    Masks specify which outputs the caller needs, not which events to process.
    Every real preceding event must still condition later predictions. A model
    may group/reorder blocks, but must preserve the requested observations and
    multiplicity, returning aligned masks in its PreparedPrediction. No loss
    reduction, parameter search, or assumptions about another model belong here.
    """

    def prepare_predictions(
        self,
        trial_indices: Integer[Array, " trials"],
        mask: Bool[Array, "trials recall_events"],
    ) -> "PreparedPrediction[Any]":
        """Prepare immutable observations before JIT; return model computation."""
        ...


@runtime_checkable
class BatchSequencePredictionFactory(SequencePredictionFactory, Protocol):
    """Optional model-owned layout planning across trial selections."""

    def prepare_prediction_selections(
        self,
        trial_indices: Iterable[Integer[Array, " trials"]],
        masks: Iterable[Bool[Array, "trials recall_events"]],
    ) -> Sequence["PreparedPrediction[Any]"]:
        """Return one independent prediction plan per requested selection."""
        ...


@runtime_checkable
class LossFn(Protocol):
    """Loss function for model fitting."""

    def __init__(
        self,
        model_factory_cls: type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Float[Array, " word_pool_items features_count"] | None,
    ) -> None:
        """Initialize with the specified trials and trial data."""

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Mapping[str, Float_],
        free_param_names: Iterable[str],
        x: Float[Array, " parameters n_samples"],
    ) -> Float[Array, " n_samples"]:
        """Return one loss per parameter vector."""
        ...


type PreparedLossFn = Callable[
    [Mapping[str, Float_], Iterable[str], Float[Array, "parameters n_samples"]],
    Float[Array, " n_samples"],
]


@runtime_checkable
class PreparableLossFn(LossFn, Protocol):
    """Optional loss capability for preprocessing a fixed trial selection.

    Preparation runs on the host before JIT compilation. The returned callable
    scores exactly the supplied trials (including their order/multiplicity) and
    accepts base parameters, free parameter names and a parameter matrix.
    It must retain its own selection rather than mutate another prepared loss.
    Ordinary LossFn implementations need not implement this capability.
    """

    def prepare_trials(
        self, trial_indices: Integer[Array, " trials"]
    ) -> PreparedLossFn:
        """Bind concrete trial indices and return a JAX-compatible loss."""
        ...


@runtime_checkable
class BatchPreparableLossFn(PreparableLossFn, Protocol):
    """Optional preparation of compatible layouts for multiple trial selections.

    Return one independent callable per selection, in the supplied order.
    Each must preserve that selection's trial multiplicity and scoring rules.
    Implementations may share layout metadata to permit compilation reuse;
    they must not change an already returned callable. A fitter can pass a
    jaxcmr.loss.prepared.PreparedLoss as dynamic data to JIT. Other returned
    callables retain the ordinary prepared-loss path.
    """

    def prepare_selections(
        self,
        trial_indices: Iterable[Integer[Array, " trials"]],
    ) -> Sequence[PreparedLossFn]:
        """Prepare explicit selections without compiling or fitting them."""
        ...


class FitResult(TypedDict):
    """Typed dict describing the results of a fitting procedure."""

    fixed: dict[str, float]
    """Dictionary of fixed parameters and their values."""

    free: dict[str, list[float]]
    """Dictionary of free parameters and their [lower_bound, upper_bound] or similar spec."""

    fitness: list[float]
    """Objective values, one per fit row. Their meaning is determined by the
    selected loss; likelihood examples store negative log likelihood."""

    nit: list[int]
    """Number of optimizer iterations or generations run for each selected fit."""

    converged: list[bool]
    """Whether each fit met the optimizer convergence criterion."""

    fits: dict[str, list[float]]
    """Dictionary of parameter names -> fitted/fixed values, aligned by fit row.
    The subject entry identifies each row; pooled fits conventionally use -1."""

    hyperparameters: dict[str, Any]
    """Dictionary of hyperparameter names and their values used during fitting."""

    fit_time: float
    """Total time (in seconds) taken to perform the fitting."""

    diagnostics: NotRequired[dict[str, Any]]
    """Optional optimizer diagnostics, such as per-restart evaluations and status."""


class CVResult(TypedDict):
    """Typed dict describing leave-one-fold-out cross-validation results."""

    n_folds: int
    """Number of CV folds."""

    fold_field: str
    """Dataset field used for fold splitting (e.g. 'list')."""

    fold_values: list[int]
    """Unique fold identifiers."""

    subjects: list[int]
    """Subject IDs in the same order as the fitness lists."""

    train_fitness: list[list[float]]
    """Per-fold list of per-subject training NLL. Shape: [n_folds][n_subjects]."""

    test_fitness: list[list[float]]
    """Per-fold list of per-subject held-out NLL. Shape: [n_folds][n_subjects]."""

    cv_fitness: list[float]
    """Aggregated held-out NLL per subject (sum across folds)."""

    fold_fits: list[FitResult]
    """Per-fold FitResult from training."""

    hyperparameters: dict[str, Any]
    """Hyperparameters used for fitting."""

    cv_time: float
    """Total wall-clock time for the CV procedure."""


@runtime_checkable
class FittingAlgorithm(Protocol):
    """Protocol describing a fitting algorithm for memory search models.

    Returned dicts should contain the following keys:
        - 'fixed': dict of fixed parameters and their values
        - 'free': dict of free parameters and their parameter bounds
        - 'fitness': fitness value(s) of the optimized parameters
        - 'nit': optimizer generations run for each fit
        - 'converged': convergence status for each fit
        - 'fits': dictionary of free parameters and their optimized value(s)
    """

    def __init__(
        self,
        dataset: RecallDataset,
        features: Float[Array, "word_pool_items features_count"] | None,
        base_params: Mapping[str, Float_],
        model_factory_cls: type[MemorySearchModelFactory],
        loss_fn_cls: type[LossFn],
        hyperparams: dict[str, Any] | None = None,
    ):
        """
        Configure the fitting algorithm.

        Args:
            dataset: The dataset containing trial data (including 'subject').
            features: Optional feature matrix aligned to the vocabulary.
            base_params: A dictionary of parameters that are held fixed.
            model_factory_cls: Class implementing MemorySearchModelFactory.
            loss_fn_cls: Class implementing LossFn.
            hyperparams: Optional dictionary of hyperparameters for the fitting routine.
                May include 'bounds' (dict[str, list[float]]) and other keys
                like 'num_steps', 'pop_size', etc.
        """
        ...

    def fit(
        self,
        trial_mask: Bool[Array, " trials"],
        subject_id: int = -1,
    ) -> FitResult:
        """Fit one parameter set to the trials selected by the mask."""
        ...

    def fit_subjects(
        self,
        trial_mask: Bool[Array, " trials"],
    ) -> FitResult:
        """Fit each subject independently and accumulate results."""
        ...


@runtime_checkable
class TrialSimulator(Protocol):
    """Returns model and recalled sequence for a single trial.

    Encapsulates a trial-level simulation step that consumes a study sequence
    and an initial recalls buffer, and produces the updated model alongside
    the simulated recall events.
    """

    def __call__(
        self,
        model: MemorySearch,
        present: Integer[Array, " study_events"],
        trial: Integer[Array, " recalls"],
        rng: PRNGKeyArray,
    ) -> tuple[MemorySearch, Integer[Array, " recall_events"]]:
        """Returns model and simulated recall sequence.

        Args:
          model: Memory search model to update during the trial.
          present: One-indexed study sequence for the trial.
          trial: One-indexed recall sequence for the trial (same indexing as present).
          rng: Random key.
        """
        ...


# Compatibility aliases. Do not use in new code.
LossFnGenerator = LossFn
