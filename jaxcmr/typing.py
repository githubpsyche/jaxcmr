"""Type definitions and protocols for the jaxcmr package.

Defines array type aliases (via jaxtyping), model protocols
(``MemorySearch``, ``TrialSimulator``), dataset type dictionaries
(``RecallDataset``), and callable type aliases used throughout the
package.

"""

from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    NotRequired,
    Optional,
    Protocol,
    Type,
    TypedDict,
    runtime_checkable,
)

import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, Integer, PRNGKeyArray, Real, Shaped

Float_ = Float[Array, ""] | float | int
Int_ = Integer[Array, ""] | int
Bool_ = Bool[Array, ""] | bool

__all__ = [
    "Array",
    "ArrayLike",
    "Bool",
    "Bool_",
    "Float",
    "Float_",
    "Int_",
    "Integer",
    "Real",
    "Shaped",
    "PRNGKeyArray",
    "RecallDataset",
    "MemorySearch",
    "MemorySearchCreateFn",
    "MemorySearchModelFactory",
    "Memory",
    "Context",
    "LossFnGenerator",
    "LikelihoodMaskFn",
    "FitResult",
    "CVResult",
    "FittingAlgorithm",
    "TrialSimulator",
    "TerminationPolicy",
    "ContextCreateFn",
    "MemoryCreateFn",
    "TerminationPolicyCreateFn",
]


class RecallDataset(TypedDict):
    """
    A typed dictionary representing a dataset for free or serial recall experiments.
    Each key maps to a 2D integer array of shape (n_trials, ?).
    Rows correspond to trials; columns vary by field.
    Zeros are used to indicate unused or padding entries, with values starting from 1.

    Required fields:

    - subject:       Subject IDs (one per trial).
    - listLength:    The length of the list presented in each trial.
    - pres_itemids:  Cross-list item IDs presented in each trial
                        (points to a global word pool).
    - pres_itemnos:  Within-list item numbers (1-based indices; 0 indicates padding).
    - rec_itemids:   Cross-list item IDs corresponding to items recalled.
    - recalls:       Within-list item numbers for recalled items
                        (1-based indices; 0 indicates padding).

    You can add as many as needed, with `NotRequired[...]`.
    """

    # REQUIRED FIELDS

    subject: Integer[Array, "n_trials 1"]
    """Subject ID for each trial (shape: [n_trials, 1])."""

    listLength: Integer[Array, "n_trials 1"]
    """List length for each trial (shape: [n_trials, 1])."""

    pres_itemnos: Integer[Array, "n_trials num_presented"]
    """Per-trial within-list item numbers (shape: [n_trials, num_presented]).
    1-based indices with 0 for unused/padding entries."""

    recalls: Integer[Array, "n_trials num_recalled"]
    """Within-list item numbers for recalled items (shape: [n_trials, num_recalled]).
    1-based indices with 0 for unused/padding entries."""

    # OPTIONAL FIELDS, REQUIRED FOR SEMANTIC ANALYSIS
    pres_itemids: Integer[Array, "n_trials num_presented"]
    """Per-trial cross-list item IDs (shape: [n_trials, num_presented]). 
    These IDs reference a global word pool and may repeat across trials."""

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
            memory: the current memory model.
            input_pattern: a feature pattern for an input.
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
            memory: the current memory state.
            in_pattern: the input feature pattern.
            activation_scale: the activation scaling factor.
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


@runtime_checkable
class TerminationPolicy(Protocol):
    """Define termination policies for memory search models."""

    def stop_probability(self, model: "MemorySearch") -> Float[Array, ""]:
        """Returns stop probability for the provided model.

        Args:
          model: Memory search model under evaluation.
        """
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
        context: the current context state.
    """

    item_count: int
    is_active: Bool[Array, ""]
    studied: Bool[Array, " item_count"]
    recallable: Bool[Array, " item_count"]
    recall_total: Integer[Array, ""]
    study_index: Integer[Array, ""]

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


class TerminationPolicyCreateFn(Protocol):
    """Callable that returns a termination policy instance."""

    def __call__(
        self,
        list_length: int,
        parameters: Mapping[str, Float_],
    ) -> TerminationPolicy:
        """Return a termination policy instance."""
        ...


@runtime_checkable
class MemorySearchCreateFn(Protocol):
    """A factory for creating memory search models."""

    def __call__(
        self,
        list_length: int,
        parameters: Mapping[str, Float_],
        connections: Optional[Float[Array, " study_events study_events"]],
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
        features: Optional[Float[Array, " word_pool_items features_count"]],
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
class LossFnGenerator(Protocol):
    """Generates loss function for model fitting."""

    def __init__(
        self,
        model_factory: Type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Optional[Float[Array, " word_pool_items features_count"]],
    ) -> None:
        """Initialize the factory with the specified trials and trial data."""

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Mapping[str, Float_],
        free_param_names: Iterable[str],
    ) -> Callable[[np.ndarray], Float[Array, ""]]:
        """Return the loss value for the specified model parameters."""
        ...


class FitResult(TypedDict):
    """Typed dict describing the results of a fitting procedure."""

    fixed: dict[str, float]
    """Dictionary of fixed parameters and their values."""

    free: dict[str, list[float]]
    """Dictionary of free parameters and their [lower_bound, upper_bound] or similar spec."""

    fitness: list[float]
    """List of one or more fitness values (e.g., for single-fit or per-subject fits)."""

    fits: dict[str, list[float]]
    """Dictionary of parameter names -> optimized values (one or many)."""

    hyperparameters: dict[str, Any]
    """Dictionary of hyperparameter names and their values used during fitting."""

    fit_time: float
    """Total time (in seconds) taken to perform the fitting."""


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
        - 'fits': dictionary of free parameters and their optimized value(s)
    """

    def __init__(
        self,
        dataset: RecallDataset,
        features: Optional[Float[Array, "word_pool_items features_count"]],
        base_params: Mapping[str, Float_],
        model_factory: Type[MemorySearchModelFactory],
        loss_fn_generator: Type[LossFnGenerator],
        hyperparams: Optional[dict[str, Any]] = None,
    ):
        """
        Configure the fitting algorithm.

        Args:
            dataset: The dataset containing trial data (including 'subject').
            features: Optional feature matrix aligned to the vocabulary.
            base_params: A dictionary of parameters that are held fixed.
            model_factory: Class implementing MemorySearchModelFactory.
            loss_fn_generator: Class implementing LossFnGenerator.
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
