from typing import Any, Mapping, Optional, Type

import numpy as np
from scipy.optimize import differential_evolution
from tqdm import trange
from jax import numpy as jnp

from jaxcmr.typing import (
    Array,
    Float_,
    Bool,
    Integer,
    LossFnGenerator,
    MemorySearchModelFactory,
    FitResult
)


def make_subject_trial_masks(
    trial_mask: Bool[Array, " trials"], 
    subject_vector: Integer[Array, " trials"]
):
    """Returns a list of masks, one per unique subject, plus the list of unique subjects."""
    unique_subjects = np.unique(subject_vector)
    subject_masks = [
        (subject_vector == s) & trial_mask.astype(bool) for s in unique_subjects
    ]
    return subject_masks, unique_subjects


class ScipyDE:
    """A fitting class that uses SciPy's Differential Evolution algorithm."""

    def __init__(
        self,
        dataset: dict[str, Integer[Array, " trials ?"]],
        connections: Optional[Integer[Array, " word_pool_items word_pool_items"]],
        base_params: Mapping[str, Float_],
        model_factory: Type[MemorySearchModelFactory],
        loss_fn_generator: Type[LossFnGenerator],
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
        # Store essential data
        self.dataset = dataset
        self.connections = connections
        self.base_params = base_params

        # Extract hyperparams or use defaults
        if hyperparams is None:
            hyperparams = {}

        # Pull out free-parameter bounds from hyperparams (or default to empty dict)
        self.free_parameter_bounds = hyperparams.get("bounds", {})
        self.bounds = np.array(list(self.free_parameter_bounds.values()))

        self.num_steps = hyperparams.get("num_steps", 1000)
        self.pop_size = hyperparams.get("pop_size", 15)
        self.relative_tolerance = hyperparams.get("relative_tolerance", 0.001)
        self.cross_over_rate = hyperparams.get("cross_over_rate", 0.9)
        self.diff_w = hyperparams.get("diff_w", 0.85)
        self.progress_bar = hyperparams.get("progress_bar", True)
        self.display_iterations = hyperparams.get("display_iterations", False)

        # Instantiate the loss function generator
        self.loss_fn_generator = loss_fn_generator(model_factory, dataset, connections)

        # Subject IDs
        self.subjects = dataset["subject"].flatten()

    def single_fit(
        self,
        trial_mask: Bool[Array, " trials"],
    ) -> FitResult:
        """Returns result of fitting the model to the trials specified by the mask."""
        # Convert the mask to an array of trial indices
        trial_indices = jnp.where(trial_mask)[0]

        # Build a scalar loss function based on these trial indices
        loss_fn = self.loss_fn_generator(
            trial_indices, self.base_params, self.free_parameter_bounds
        )

        # Run differential evolution
        fit_result = differential_evolution(
            loss_fn,
            self.bounds,
            maxiter=self.num_steps,
            popsize=self.pop_size,
            vectorized=True,
            disp=self.display_iterations,
            tol=self.relative_tolerance,
            mutation=self.diff_w,
            recombination=self.cross_over_rate,
        )

        # Prepare output
        result: FitResult = {
            "fixed": {k: float(v) for k, v in self.base_params.items()},
            "free": {
                k: self.free_parameter_bounds[k] for k in self.free_parameter_bounds
            },
            "fitness": [float(fit_result.fun)],
            "fits": {
                # For each base param, we just repeat its original value
                **{
                    k: [float(v)]
                    for k, v in self.base_params.items()
                },
                # For each free param, we store the optimizer's best value
                **{
                    param_name: [float(fit_result.x[idx])]
                    for idx, param_name in enumerate(self.free_parameter_bounds)
                },
                # Subject is -1 if not subject-specific
                "subject": [-1],
            },
        }
        return result

    def fit_to_subjects(
        self,
        trial_mask: Bool[Array, " trials"],
    ) -> FitResult:
        """Returns result of fitting the model separately to each subject present in the dataset."""
        # Create one trial mask per subject
        subject_trial_masks, unique_subjects = make_subject_trial_masks(
            trial_mask, self.subjects
        )

        # Prepare a global result structure
        all_results: FitResult = {
            "fixed": {k: float(v) for k, v in self.base_params.items()},
            "free": {
                k: self.free_parameter_bounds[k] for k in self.free_parameter_bounds
            },
            "fitness": [],
            "fits": {
                **{k: [] for k in self.free_parameter_bounds},
                **{k: [] for k in self.base_params},
                "subject": [],
            },
        }

        # Optionally show progress bar
        subject_range = (
            trange(len(unique_subjects)) if self.progress_bar else range(len(unique_subjects))
        )

        for s in subject_range:
            # If no trials for this subject, skip
            if np.sum(subject_trial_masks[s]) == 0:
                continue

            # Single-fit on the subject-specific mask
            fit_result = self.single_fit(subject_trial_masks[s])
            all_results["fitness"] += fit_result["fitness"]

            # Show in tqdm progress bar
            if self.progress_bar:
                subject_range.set_description(  # type: ignore
                    f"Last fitness: {fit_result['fitness']} "
                    f"for subject {unique_subjects[s]}"
                )

            # Append subject ID
            all_results["fits"]["subject"].append(int(unique_subjects[s]))

            # Append param values
            for param_name, values_list in fit_result["fits"].items():
                # Skip 'subject' to avoid double-appending
                if param_name == "subject":
                    continue
                all_results["fits"][param_name] += values_list

        return all_results

    def fit(
        self,
        trial_mask: Bool[Array, " trials"],
        fit_to_subjects: bool = True,
    ) -> FitResult:
        """Convenience wrapper for either single-fit or subject-by-subject fitting."""
        if fit_to_subjects:
            return self.fit_to_subjects(trial_mask)
        else:
            return self.single_fit(trial_mask)
