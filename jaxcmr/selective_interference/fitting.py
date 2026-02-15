"""Fitted parameter loading and caching.

Selective interference simulations start from per-subject CMR
parameters fitted to a behavioural dataset (e.g. free recall of
film clips).  ``load_or_fit_params`` either loads a cached JSON fit
file or runs differential evolution fitting from scratch, then
optionally applies multiplicative scale factors (e.g. scaling
``stop_probability_scale`` to calibrate recall length) and clips
drift rates to [0, 1].

"""

import json
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp


_PARAM_UPPER_BOUNDS = {
    "encoding_drift_rate": 1.0,
    "start_drift_rate": 1.0,
    "recall_drift_rate": 1.0,
    "learning_rate": 1.0,
}


def load_or_fit_params(
    fit_path: str | Path,
    *,
    param_scales: Optional[dict[str, float]] = None,
    data=None,
    trial_mask: Optional[jax.Array] = None,
    model_factory=None,
    redo_fits: bool = False,
    best_of: int = 1,
) -> tuple[dict[str, jax.Array], int]:
    """Load cached fits or run DE fitting, then apply scales.

    Parameters
    ----------
    fit_path : str or Path
        Path to the JSON fit file.
    param_scales : dict[str, float], optional
        Multiplicative scale factors to apply to fitted values.
    data : RecallDataset, optional
        Required only when ``redo_fits=True``.
    trial_mask : jax.Array, optional
        Required only when ``redo_fits=True``.
    model_factory : callable, optional
        Required only when ``redo_fits=True``.
    redo_fits : bool
        If True, re-run fitting even if the file exists.
    best_of : int
        Number of fitting restarts.

    Returns
    -------
    tuple[dict[str, jax.Array], int]
        ``(params, n_subjects)``

    """
    fit_path = Path(fit_path)
    fit_path.parent.mkdir(parents=True, exist_ok=True)

    if fit_path.exists() and not redo_fits:
        with fit_path.open() as f:
            results = json.load(f)
        print(f"Loaded fits from {fit_path}")
    else:
        if data is None or model_factory is None:
            raise ValueError(
                "data and model_factory are required when redo_fits=True "
                "or fit file does not exist"
            )
        from jaxcmr.fitting import ScipyDE
        from jaxcmr.loss.sequence_likelihood import (
            MemorySearchLikelihoodFnGenerator,
        )

        free_bounds = {
            "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
            "start_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
            "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
            "shared_support": [2.220446049250313e-16, 99.9999999999999998],
            "item_support": [2.220446049250313e-16, 99.9999999999999998],
            "learning_rate": [2.220446049250313e-16, 0.9999999999999998],
            "primacy_scale": [2.220446049250313e-16, 99.9999999999999998],
            "primacy_decay": [2.220446049250313e-16, 99.9999999999999998],
            "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
            "stop_probability_scale": [2.22e-16, 5.0],
            "stop_probability_growth": [2.22e-16, 10.0],
        }
        fixed = {
            "allow_repeated_recalls": False,
            "learn_after_context_update": False,
        }
        fitter = ScipyDE(
            data,
            trial_mask,
            fixed,
            model_factory,
            MemorySearchLikelihoodFnGenerator,
            hyperparams={
                "bounds": free_bounds,
                "num_steps": 200,
                "pop_size": 15,
                "relative_tolerance": 0.001,
                "cross_over_rate": 0.9,
                "diff_w": 0.85,
                "best_of": best_of,
                "progress_bar": True,
            },
        )
        results = fitter.fit(trial_mask)
        with fit_path.open("w") as f:
            json.dump(results, f, indent=4)
        print(f"Fitted and saved to {fit_path}")

    params = {key: jnp.array(val) for key, val in results["fits"].items()}

    if param_scales:
        for name, scale in param_scales.items():
            if scale != 1.0 and name in params:
                params[name] = params[name] * scale
                if name in _PARAM_UPPER_BOUNDS:
                    hi = _PARAM_UPPER_BOUNDS[name]
                    n_clipped = int(jnp.sum(params[name] > hi))
                    params[name] = jnp.clip(params[name], 0.0, hi)
                    clip_msg = (
                        f" (clipped {n_clipped} subjects)" if n_clipped else ""
                    )
                    print(f"  Scaled {name} by {scale}{clip_msg}")
                else:
                    print(f"  Scaled {name} by {scale}")

    n_subjects = len(next(iter(params.values())))
    return params, n_subjects
