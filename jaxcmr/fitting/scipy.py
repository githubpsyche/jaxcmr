"""SciPy differential-evolution fitting."""

import time
from typing import Any, Callable, Iterable, Mapping, Optional, Type

import numpy as np
from jax import jit
from jax import numpy as jnp
from scipy.optimize import differential_evolution
from tqdm import trange

from jaxcmr.helpers import make_subject_trial_masks
from jaxcmr.typing import (
    Array,
    Bool,
    FitResult,
    Float,
    Float_,
    Integer,
    LossFn,
    MemorySearchModelFactory,
    RecallDataset,
)


__all__ = [
    "ScipyDE",
    "make_scipy_loss_fn",
]


def make_scipy_loss_fn(
    loss_fn,
    trial_indices: Integer[Array, " trials"],
    base_params: Mapping[str, Float_],
    free_param_names: Iterable[str],
) -> Callable[[np.ndarray], Float[Array, ""]]:
    """Return a SciPy-style callable backed by a fitting loss."""
    free_param_names = tuple(free_param_names)

    @jit
    def single_param_loss(x: jnp.ndarray) -> Float[Array, ""]:
        """Returns loss for one parameter vector."""
        return loss_fn(trial_indices, base_params, free_param_names, x[:, None])[0]

    @jit
    def multi_param_loss(x: jnp.ndarray) -> Float[Array, " n_samples"]:
        """Returns one loss per parameter vector."""
        return loss_fn(trial_indices, base_params, free_param_names, x)

    return lambda x: multi_param_loss(x) if x.ndim > 1 else single_param_loss(x)


class ScipyDE:
    """A fitting class that uses SciPy's Differential Evolution algorithm."""

    def __init__(
        self,
        dataset: RecallDataset,
        features: Optional[Float[Array, "word_pool_items features_count"]],
        base_params: Mapping[str, Float_],
        model_factory_cls: Type[MemorySearchModelFactory],
        loss_fn_cls: Type[LossFn],
        hyperparams: Optional[dict[str, Any]] = None,
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
        # Store essential data
        self.dataset = dataset
        self.base_params = base_params
        self.subjects = dataset["subject"].flatten()

        # configure convenience features
        if hyperparams is None:
            hyperparams = {}
        self.progress_bar = hyperparams.get("progress_bar", True)
        self.display_iterations = hyperparams.get("display_iterations", False)

        # Extract bounds for free params; store all hyperparams for convenience
        self.free_parameter_bounds = hyperparams.get("bounds", {})
        self.bounds = np.array(list(self.free_parameter_bounds.values()))
        diff_w = hyperparams.get("diff_w", (0.5, 1.0))
        if not isinstance(diff_w, (tuple, list)):
            diff_w = (diff_w, diff_w)
        diff_w = tuple(diff_w)

        self.all_hyperparams = {
            "bounds": self.free_parameter_bounds,
            "num_steps": hyperparams.get("num_steps", 1000),
            "pop_size": hyperparams.get("pop_size", 15),
            "relative_tolerance": hyperparams.get("relative_tolerance", 0.001),
            "absolute_tolerance": hyperparams.get("absolute_tolerance", 0.0),
            "cross_over_rate": hyperparams.get("cross_over_rate", 0.9),
            "diff_w": diff_w,
            "best_of": hyperparams.get("best_of", 1),
            "init": hyperparams.get("init", "latinhypercube"),
            "polish": hyperparams.get("polish", True),
            "seed": int(hyperparams.get("seed", 0)),
        }
        self._fit_counter = 0

        self.loss_fn = loss_fn_cls(model_factory_cls, dataset, features)

    def fit(
        self, trial_mask: Bool[Array, " trials"], subject_id: int = -1
    ) -> FitResult:
        """Fit one parameter set to the trials selected by the mask.

        Parameters
        ----------
        trial_mask : Bool[Array, " trials"]
            Boolean mask selecting which trials to include in the fit.
        subject_id : int, optional
            Label stored in the result dict. Defaults to -1.

        Returns
        -------
        FitResult
            Fitted parameters, fitness value, and metadata.
        """
        t0 = time.perf_counter()

        # Convert the mask to an array of trial indices
        trial_indices = jnp.where(trial_mask)[0]

        # Build a scalar loss function based on these trial indices
        scipy_loss_fn = make_scipy_loss_fn(
            self.loss_fn,
            trial_indices, self.base_params, self.free_parameter_bounds
        )

        # Run differential evolution
        best_fitness = np.inf
        best_fit_result = None
        fit_counter = self._fit_counter
        self._fit_counter += 1
        for run_index in range(self.all_hyperparams["best_of"]):
            fit_result = differential_evolution(
                scipy_loss_fn,
                self.bounds,
                maxiter=self.all_hyperparams["num_steps"],
                popsize=self.all_hyperparams["pop_size"],
                vectorized=True,
                disp=self.display_iterations,
                tol=self.all_hyperparams["relative_tolerance"],
                atol=self.all_hyperparams["absolute_tolerance"],
                mutation=self.all_hyperparams["diff_w"],
                recombination=self.all_hyperparams["cross_over_rate"],
                init=self.all_hyperparams["init"],
                polish=self.all_hyperparams["polish"],
                rng=np.random.default_rng(
                    [self.all_hyperparams["seed"], fit_counter, run_index]
                ),
            )
            if fit_result.fun < best_fitness:
                best_fitness = fit_result.fun
                best_fit_result = fit_result

        assert best_fit_result is not None, "No fit result found"
        return {
            "fixed": {k: float(v) for k, v in self.base_params.items()},
            "free": {
                k: self.free_parameter_bounds[k] for k in self.free_parameter_bounds
            },
            "fitness": [float(best_fitness)],
            "nit": [int(best_fit_result.nit)],
            "converged": [bool(best_fit_result.success)],
            "fits": {
                # For each base param, we just repeat its original value
                **{k: [float(v)] for k, v in self.base_params.items()},
                # For each free param, we store the optimizer's best value
                **{
                    # Map each free param name to the best-fit value
                    param_name: [float(best_fit_result.x[idx])]
                    for idx, param_name in enumerate(self.free_parameter_bounds)
                },
                "subject": [subject_id],
            },
            "hyperparameters": self.all_hyperparams,
            "fit_time": time.perf_counter() - t0,
        }

    def fit_subjects(
        self, trial_mask: Bool[Array, " trials"]
    ) -> FitResult:
        """Fit each subject independently and accumulate results.

        Parameters
        ----------
        trial_mask : Bool[Array, " trials"]
            Boolean mask selecting which trials to include.

        Returns
        -------
        FitResult
            Combined results with per-subject fitted parameters.
        """
        t0 = time.perf_counter()

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
            "nit": [],
            "converged": [],
            "fits": {
                **{k: [] for k in self.free_parameter_bounds},
                **{k: [] for k in self.base_params},
                "subject": [],
            },
            "hyperparameters": self.all_hyperparams,
            "fit_time": 0.0,
        }

        # Optionally show progress bar
        subject_range = (
            trange(len(unique_subjects))
            if self.progress_bar
            else range(len(unique_subjects))
        )
        for s in subject_range:
            # If no trials for this subject, skip
            if np.sum(subject_trial_masks[s]) == 0:
                continue

            fit_result = self.fit(
                subject_trial_masks[s], int(unique_subjects[s])
            )
            all_results["fitness"] += fit_result["fitness"]
            all_results["nit"] += fit_result["nit"]
            all_results["converged"] += fit_result["converged"]

            # Show in tqdm progress bar
            if self.progress_bar:
                subject_range.set_description(  # type: ignore
                    f"Subject={unique_subjects[s]}, Fitness={fit_result['fitness'][0]}"
                )

            # Accumulate fitted parameters
            all_results["fits"]["subject"].append(int(unique_subjects[s]))

            # Append param values
            for param_name, values_list in fit_result["fits"].items():
                # Skip 'subject' to avoid double-appending
                if param_name == "subject":
                    continue
                all_results["fits"][param_name] += values_list

            # Bail out if we got a non-finite fitness
            if not jnp.isfinite(fit_result["fitness"][0]):
                raise ValueError(
                    f"Non-finite fitness for subject {unique_subjects[s]}", fit_result
                )

        all_results["fit_time"] = time.perf_counter() - t0
        return all_results
