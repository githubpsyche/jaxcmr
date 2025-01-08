from typing import Any, Callable, Mapping, Optional, Protocol, Type, TypedDict

import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, Integer, PRNGKeyArray, Real, Shaped

Float_ = Float[Array, ""] | float | int
Int_ = Integer[Array, ""] | int
Bool_ = Bool[Array, ""] | bool

__all__ = [
    "Array",
    "ArrayLike",
    "Bool",
    "Float",
    "Float_",
    "Int_",
    "Integer",
    "Real",
    "Shaped",
    "PRNGKeyArray",
    "MemorySearch",
    "MemorySearchCreateFn",
    "MemorySearchModelFactory",
    "Memory",
    "Context",
    "LossFnGenerator",
    "FitResult",
    "FittingAlgorithm",
]


class MemorySearch(Protocol):
    """A model of memory search.

    Attributes:
        item_count: the number of item slots reserved in the model.
        is_active: indicates whether the model is active or not.
    """

    item_count: int
    is_active: Bool[Array, ""]

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


class MemorySearchCreateFn(Protocol):
    """A factory for creating memory search models."""

    def __call__(
        self,
        list_length: int,
        parameters: Mapping[str, Float_],
    ) -> MemorySearch:
        """Create a new memory search model with the specified parameters."""
        ...


class MemorySearchModelFactory(Protocol):
    def __init__(
        self,
        dataset: dict[str, Integer[Array, " trials ?"]],
        connections: Optional[Integer[Array, " word_pool_items word_pool_items"]],
    ) -> None:
        """Initialize the factory with the specified trials and trial data."""
        ...

    def create_model(
        self,
        trial_index: Integer[Array, ""],
        parameters: Mapping[str, Float_],
    ) -> MemorySearch:
        """Create a new memory search model with the specified parameters for the specified trial."""
        ...


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


class LossFnGenerator(Protocol):
    """Generates loss function for model fitting."""

    def __init__(
        self,
        model_factory: Type[MemorySearchModelFactory],
        dataset: dict[str, Integer[Array, " trials ?"]],
        connections: Optional[Integer[Array, " word_pool_items word_pool_items"]],
    ) -> None:
        """Initialize the factory with the specified trials and trial data."""

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Mapping[str, Float_],
        free_params: Mapping[str, list[float]],
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
        dataset: dict[str, Integer[Array, " trials ?"]],
        connections: Optional[Integer[Array, " word_pool_items word_pool_items"]],
        base_params: Mapping[str, Float_],
        model_factory: Type["MemorySearchModelFactory"],
        loss_fn_generator: Type["LossFnGenerator"],
        hyperparams: Optional[dict[str, Any]] = None,
    ):
        """
        Configure the fitting algorithm.

        Args:
            dataset: The dataset containing trial data (including 'subject').
            connections: Optional connectivity matrix.
            base_params: A dictionary of parameters that are held fixed.
            model_factory: Class implementing MemorySearchModelFactory.
            loss_fn_generator: Class implementing LossFnGenerator.
            hyperparams: Optional dictionary of hyperparameters for the fitting routine.
                May include 'bounds' (dict[str, list[float]]) and other keys
                like 'num_steps', 'pop_size', etc.
        """
        ...

    def single_fit(
        self,
        trial_mask: Bool[Array, " trials"],
    ) -> FitResult:
        """Returns result of fitting the model to the trials specified by the mask."""
        ...

    def fit_to_subjects(
        self,
        trial_mask: Bool[Array, " trials"],
    ) -> FitResult:
        """Returns result of fitting the model separately to each subject present in the dataset."""
        ...

    def fit(
        self,
        trial_mask: Bool[Array, " trials"],
        fit_to_subjects: bool = True,
    ) -> FitResult:
        """Convenience wrapper for either single-fit or subject-by-subject fitting."""
        ...
