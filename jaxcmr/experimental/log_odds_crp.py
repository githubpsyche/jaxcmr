"""Compare recall at each lag with recall at a chosen reference lag.

``log_odds_crp`` first computes the ordinary lag-CRP across the supplied
trials. At each lag it subtracts the log odds at the reference lag from
the log odds at that lag. A positive contrast means greater odds than at
the reference. The calculation bounds probabilities away from zero and
one before converting to log odds. If the reference has no opportunities,
the entire contrast curve is undefined.

``plot_log_odds_crp`` computes the contrast separately for each participant
before averaging. Call ``log_odds_crp`` directly to transform a pooled curve.

This experimental analysis has a provisional interface and interpretation.
"""

__all__ = [
    "log_odds_crp",
    "plot_log_odds_crp",
]


from collections.abc import Sequence

from jax import jit
from jax import numpy as jnp
from matplotlib.axes import Axes

from jaxcmr.dataset import apply_by_subject

from ..analyses.crp import crp
from ..plotting import format_analysis_axes, plot_mean_curve, prepare_plot_inputs
from ..typing import Array, Bool, Float, RecallDataset


def log_odds_crp(
    dataset: RecallDataset,
    reference_lag: int = 10,
    epsilon: float = 1e-6,
    size: int = 3,
) -> Float[Array, " lags"]:
    """Lag-CRP log-odds relative to a reference lag.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.
    reference_lag : int
        Lag used as the zero-log-odds baseline.
    epsilon : float
        Probability clamp before the logit transform.
    size : int
        Maximum positions an item can occupy.

    Returns
    -------
    Float[Array, " lags"]
        Log-odds CRP relative to ``reference_lag``.

    """
    probabilities = crp(dataset, size=size)
    # Lags with zero availability stay NaN here so they never fabricate odds.
    clipped = jnp.clip(probabilities, epsilon, 1 - epsilon)
    # Clipping avoids ±inf when actual counts hit the bounds but keeps NaNs intact.
    logits = jnp.log(clipped) - jnp.log1p(-clipped)
    lag_range = dataset["pres_itemnos"].shape[1] - 1
    reference_index = reference_lag + lag_range
    reference_value = logits[reference_index]
    # Subtracting the reference lag removes subject-level baseline differences.
    return logits - reference_value


def plot_log_odds_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 5,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    reference_lag: int = 10,
    epsilon: float = 1e-6,
    size: int = 3,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot subject-wise Lag-CRP log-odds with error bounds.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        Recall datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean masks selecting trials per dataset.
    max_lag : int
        Maximum lag to plot.
    color_cycle : list[str], optional
        Colors for each dataset.
    labels : Sequence[str], optional
        Legend labels for ``datasets``.
    contrast_name : str, optional
        Legend title.
    axis : Axes, optional
        Existing Matplotlib Axes to plot on.
    reference_lag : int
        Lag defining the zero-log-odds baseline.
    epsilon : float
        Probability clamp for the logit transform.
    size : int
        Maximum positions an item can occupy.
    confidence_level : float
        Confidence level for error bounds.

    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.

    Returns
    -------
    Axes
        Axes with the Lag-CRP log-odds plot.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if labels is None:
        labels = [""] * len(datasets)

    lag_interval = jnp.arange(-max_lag, max_lag + 1, dtype=int)

    for data_index, data in enumerate(datasets):
        lag_range = (jnp.max(data["listLength"]) - 1).item()
        subject_values = apply_by_subject(
            data,
            trial_masks[data_index],
            jit(
                log_odds_crp,
                static_argnames=(
                    "reference_lag",
                    "size",
                ),
            ),
            reference_lag=reference_lag,
            epsilon=epsilon,
            size=size,
        )
        subject_values = jnp.vstack(subject_values)
        subject_values = subject_values[
            :, lag_range - max_lag : lag_range + max_lag + 1
        ]

        color = color_cycle[data_index % len(color_cycle)]
        plot_mean_curve(
            axis,
            lag_interval,
            subject_values,
            labels[data_index],
            color,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            bootstrap_seed=bootstrap_seed,
        )

    format_analysis_axes(
        axis,
        "Lag",
        f"Log-Odds CRP (vs lag {reference_lag})",
        contrast_name,
    )
    return axis
