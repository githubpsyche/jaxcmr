"""Leave-one-fold-out cross-validation for memory-search models.

Fits per-subject parameters on training folds and evaluates
held-out negative log-likelihood on the left-out fold, rotating
across all folds.

"""

import time
from typing import Mapping, Optional

import numpy as np
from jax import numpy as jnp
from tqdm import trange

from jaxcmr.fitting import ScipyDE
from jaxcmr.typing import (
    Array,
    Bool,
    CVResult,
    Float_,
    Integer,
)


__all__ = [
    "generate_fold_masks",
    "evaluate_held_out",
    "cross_validate",
]


def generate_fold_masks(
    data: dict,
    fold_field: str,
    fold_value: int,
    trial_mask: Bool[Array, " trial_count"],
) -> tuple[Bool[Array, " trial_count"], Bool[Array, " trial_count"]]:
    """Return train and test boolean masks for one CV fold.

    Parameters
    ----------
    data : dict
        Dataset with trial-indexed arrays.
    fold_field : str
        Name of the field to split on (e.g. 'list').
    fold_value : int
        The value identifying the held-out fold.
    trial_mask : Bool[Array, " trial_count"]
        Base trial mask applied before splitting.

    Returns
    -------
    tuple[Bool[Array, " trial_count"], Bool[Array, " trial_count"]]
        (train_mask, test_mask).

    """
    fold_vector = jnp.array(data[fold_field]).flatten()
    is_test = fold_vector == fold_value
    base = jnp.asarray(trial_mask).astype(bool)
    return base & ~is_test, base & is_test


def evaluate_held_out(
    loss_fn: object,
    test_trial_indices: Integer[Array, " test_trials"],
    parameters: Mapping[str, Float_],
) -> float:
    """Return held-out NLL for fitted parameters on test trials.

    Parameters
    ----------
    loss_fn : object
        Initialized loss function with a
        ``present_and_predict_trials_loss`` method.
    test_trial_indices : Integer[Array, " test_trials"]
        Indices of held-out trials.
    parameters : Mapping[str, Float_]
        Complete parameter dict (fixed + fitted free params).

    Returns
    -------
    float
        Negative log-likelihood on the held-out trials.

    """
    nll = loss_fn.present_and_predict_trials_loss(  # type: ignore[union-attr]
        test_trial_indices, parameters
    )
    return float(nll)


def cross_validate(
    fitter: ScipyDE,
    trial_mask: Bool[Array, " trial_count"],
    fold_field: str = "list",
    best_of: Optional[int] = None,
) -> CVResult:
    """Run leave-one-fold-out cross-validation.

    For each unique value in ``data[fold_field]``, fits per-subject on
    the remaining folds and evaluates held-out NLL on the left-out fold.

    Parameters
    ----------
    fitter : ScipyDE
        Configured fitter (stores dataset, loss_fn, params, bounds).
    trial_mask : Bool[Array, " trial_count"]
        Base trial mask applied before fold splitting.
    fold_field : str
        Dataset field for fold assignment. Default: ``'list'``.
    best_of : int, optional
        Override for ``best_of`` during CV fitting. If ``None``, uses
        the fitter's configured value.

    Returns
    -------
    CVResult
        Cross-validation results with per-fold and aggregated NLL.

    """
    t0 = time.perf_counter()
    data = fitter.dataset

    # Determine folds and subjects
    fold_values = sorted(int(v) for v in np.unique(np.array(data[fold_field])))
    subjects = sorted(int(v) for v in np.unique(np.array(data["subject"])))
    n_subjects = len(subjects)
    subject_index = {s: i for i, s in enumerate(subjects)}

    # Optionally override best_of
    original_best_of = fitter.all_hyperparams["best_of"]
    if best_of is not None:
        fitter.all_hyperparams["best_of"] = best_of

    # Accumulators
    all_train_fitness: list[list[float]] = []
    all_test_fitness: list[list[float]] = []
    all_fold_fits: list[dict] = []
    cv_accum = np.zeros(n_subjects, dtype=np.float64)

    subject_vector = np.array(data["subject"]).flatten()
    free_param_names = list(fitter.free_parameter_bounds.keys())

    try:
        for fold_idx, fold_val in enumerate(fold_values):
            print(f"\n--- Fold {fold_idx + 1}/{len(fold_values)} "
                  f"(held-out {fold_field}={fold_val}) ---")

            train_mask, test_mask = generate_fold_masks(
                data, fold_field, fold_val, trial_mask
            )

            # Fit on training folds
            fit_result = fitter.fit(train_mask)
            all_fold_fits.append(fit_result)
            all_train_fitness.append(fit_result["fitness"])

            # Evaluate on held-out fold per subject
            fold_test_nll: list[float] = []
            fitted_subjects = fit_result["fits"]["subject"]

            for j, subj_id in enumerate(fitted_subjects):
                # Reconstruct complete parameter dict for this subject
                params: dict[str, float] = {}
                params.update(fit_result["fixed"])
                for p in free_param_names:
                    params[p] = fit_result["fits"][p][j]

                # Get test trial indices for this subject
                subj_test_mask = (
                    jnp.asarray(test_mask).astype(bool)
                    & (jnp.array(subject_vector) == subj_id)
                )
                test_indices = jnp.where(subj_test_mask)[0]

                if test_indices.size == 0:
                    fold_test_nll.append(0.0)
                    continue

                nll = evaluate_held_out(fitter.loss_fn, test_indices, params)
                fold_test_nll.append(nll)

                # Accumulate into per-subject CV total
                idx = subject_index[int(subj_id)]
                cv_accum[idx] += nll

            all_test_fitness.append(fold_test_nll)
            mean_test = np.mean(fold_test_nll)
            print(f"  Mean held-out NLL: {mean_test:.2f}")

    finally:
        # Restore original best_of
        fitter.all_hyperparams["best_of"] = original_best_of

    cv_time = time.perf_counter() - t0

    return {
        "n_folds": len(fold_values),
        "fold_field": fold_field,
        "fold_values": fold_values,
        "subjects": subjects,
        "train_fitness": all_train_fitness,
        "test_fitness": all_test_fitness,
        "cv_fitness": cv_accum.tolist(),
        "fold_fits": all_fold_fits,
        "hyperparameters": fitter.all_hyperparams,
        "cv_time": cv_time,
    }
