"""
Lag-Conditional Response Probability (Lag-CRP).

Overview:
  Utilities to compute and plot Lag-CRP for free-recall data. CRP at serial lag
  ℓ is defined as the proportion of *actual* transitions at lag ℓ divided by the
  number of *available* transitions at lag ℓ.

Definition:
  CRP(ℓ) = actual_transitions(ℓ) / available_transitions(ℓ)

Serial lag:
  If the previously recalled item was studied at position X and the current item
  at position Y (1-indexed), then ℓ = Y - X. Negative lags move backward in
  study order; positive lags move forward.

Conventions:
  - list_length (L): Fixed study-list length within a call.
  - trials (recalls): int array [n_trials, n_recall_events] of serial positions
    in 1..L; 0 indicates padding after termination (ignored).
  - presentations: int array [n_trials, L] of item IDs at study positions;
    required only when items may repeat (0 permitted if your loader uses it).
  - size: Upper bound on how many distinct study positions a single item can
    occupy within a list (e.g., 3 → items may appear up to three times).
  - Lag axis indexing: All CRP outputs are length (2*L - 1). Index i corresponds
    to lag ℓ = i - (L - 1); the center (i = L - 1) is ℓ = 0.

Design decisions:
  - Padding & bounds: Zeros (pads) and out-of-range recalls are ignored via
    guards in the tabulators.
  - Division by zero: Lags with zero availability (often extreme lags) yield NaN.
  - Repeats policy: A recall of a repeated item is treated as recalling all of
    that item's study positions. For a transition from the previous item's study
    positions to the current item's positions, we accumulate the set-union of
    unique lags (each lag counted at most once per transition) to avoid
    multiplicity inflation. If you need every pairwise combination instead,
    consider a static `count_mode={"unique","pairwise"}` toggle.
  - First recall: The first non-zero recall marks that item's study positions as
    unavailable and defines the reference for the next transition lag.

JAX compilation:
  - All public functions are side-effect-free and JIT-safe.
  - Use `jit(crp, static_argnames=("size",))`; keep shapes (e.g., L, number of
    recall events) consistent within a compiled call to avoid recompiles.
  - Ensure `size` ≥ the true maximum per-item repetition in your data; compute
    it once outside JIT if needed.
"""

__all__ = [
    "set_false_at_index",
    "Tabulation",
    "tabulate_trial",
    "crp",
    "plot_crp",
]

from typing import Optional, Sequence

import numpy as np
from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib import rcParams  # type: ignore
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from simple_pytree import Pytree

from ..plotting import init_plot, plot_without_error_bars, set_plot_labels
from ..repetition import all_study_positions
from ..helpers import apply_by_subject
from ..typing import Array, Bool, Float, Int_, Bool_, Integer, Real, RecallDataset


def set_false_at_index(
    vec: Bool[Array, " positions"], i: Int_
) -> tuple[Bool[Array, " positions"], None]:
    """Set ``vec[i - 1]`` to ``False`` using 1-based indexing.

    Indices are 1-based; ``0`` is a no-op sentinel. Indices outside
    ``[1, vec.size]`` are ignored.

    Returns:
        Tuple of the (possibly updated) vector and ``None``.
    """

    should_update = (i > 0) & (i <= vec.size)
    return lax.cond(
        should_update, lambda: (vec.at[i - 1].set(False), None), lambda: (vec, None)
    )


class Tabulation(Pytree):
    """
    Maintains per-transition state for CRP with repeats.

    State:
        - previous_positions: study positions of previously recalled item
        - avail_recalls: boolean [L], study positions still available
        - actual_lags, avail_lags: int [2*L - 1], accumulated

    Update steps executed on each valid recall (in order):
        1) Previous item positions -> `previous_positions`.
        2) Available positions -> `avail_recalls = available_recalls_after(recall)`.
        3) Actual lags: union of unique lags from `previous_positions` to the
           current item's study positions (`tabulate_actual_lags`).
        4) Available lags: union of unique lags from each `previous_positions`
           to all currently available positions (`tabulate_available_lags`).

    Conventions:
        - Zeros in `recall_study_positions` are padding; safely ignored.
        - Indices are 1-based for positions; internal arrays use zero-based.
    """

    def __init__(
        self,
        presentation: Integer[Array, " study_events"],
        first_recall: Int_,
        size: int = 3,
    ):
        self.list_length = presentation.size
        self.lag_range = self.list_length - 1
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.size = size
        self.item_study_positions = lax.map(
            lambda i: all_study_positions(i, presentation, size),
            self.all_positions,
        )

        self.actual_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.avail_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)

        self.previous_positions = self.item_study_positions[first_recall - 1]
        self.avail_recalls = jnp.ones(self.list_length, dtype=bool)
        self.avail_recalls = self.available_recalls_after(first_recall)

    # for updating avail_recalls: study positions still available for retrieval
    def available_recalls_after(self, recall: Int_) -> Bool[Array, " positions"]:
        """
        Clear availability for all study positions of `recall`.
        Safe with padding: zeros are ignored.
        """
        study_positions = self.item_study_positions[recall - 1]
        return lax.scan(set_false_at_index, self.avail_recalls, study_positions)[0]

    # for updating actual_lags: lag-transitions actually made from the previous item
    def lags_from_previous(self, recall_pos: Int_) -> Bool[Array, " positions"]:
        """
        One-hot(-ish) vector of unique lags from each previous study position to `recall_pos`.
        Returns boolean union (unique lag set); use `count_mode="pairwise"` alternative
        (not implemented here) to count all pair combinations.
        """

        def f(prev):
            return lax.cond(
                (recall_pos * prev) == 0,
                lambda: self.base_lags,
                lambda: self.base_lags.at[recall_pos - prev + self.lag_range].add(1),
            )

        return lax.map(f, self.previous_positions).sum(0).astype(bool)

    def tabulate_actual_lags(self, recall: Int_) -> Integer[Array, " lags"]:
        "Tabulates the actual transition after a transition."
        recall_study_positions = self.item_study_positions[recall - 1]
        new_lags = (
            lax.map(self.lags_from_previous, recall_study_positions).sum(0).astype(bool)
        )
        return self.actual_lags + new_lags

    # for updating avail_lags: lag-transitions available from the previous item
    def available_lags_from(self, pos: Int_) -> Bool[Array, " lags"]:
        "Identifies recallable lag transitions from the specified study position."
        return lax.cond(
            pos == 0,
            lambda: self.base_lags,
            lambda: self.base_lags.at[self.all_positions - pos + self.lag_range].add(
                self.avail_recalls
            ),
        )

    def tabulate_available_lags(self) -> Integer[Array, " lags"]:
        "Union of lags from each previous position to all currently available positions."
        new_lags = (
            lax.map(self.available_lags_from, self.previous_positions)
            .sum(0)
            .astype(bool)
        )
        return self.avail_lags + new_lags

    # unifying tabulation of actual/avail lags, previous positions, and avail recalls
    def is_valid_recall(self, recall: Int_) -> Bool:
        """Return True when the recall maps to any still-available study position."""

        def _for_nonzero():
            recall_study_positions = self.item_study_positions[recall - 1]
            is_valid_study_position = recall_study_positions != 0
            is_available_study_position = self.avail_recalls[recall_study_positions - 1]
            return jnp.any(is_valid_study_position & is_available_study_position)

        return lax.cond(
            recall == 0,
            lambda: jnp.array(False),
            _for_nonzero,
        )

    def tabulate(self, recall: Int_, should_tabulate: Bool_) -> "Tabulation":
        """Update state and optionally count the transition into this recall."""

        def _update_state() -> "Tabulation":
            new_previous_positions = self.item_study_positions[recall - 1]
            new_avail_recalls = self.available_recalls_after(recall)

            def _with_counts() -> "Tabulation":
                return self.replace(
                    previous_positions=new_previous_positions,
                    avail_recalls=new_avail_recalls,
                    actual_lags=self.tabulate_actual_lags(recall),
                    avail_lags=self.tabulate_available_lags(),
                )

            def _without_counts() -> "Tabulation":
                return self.replace(
                    previous_positions=new_previous_positions,
                    avail_recalls=new_avail_recalls,
                )

            return lax.cond(should_tabulate, _with_counts, _without_counts)

        return lax.cond(self.is_valid_recall(recall), _update_state, lambda: self)


def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    should_tabulate: Bool[Array, " recall_events"],
    size: int = 3,
) -> tuple[Float[Array, " lags"], Float[Array, " lags"]]:
    """Tabulate actual and available lags for a single trial.

    Args:
        trial: Recall sequence encoded as within-list positions.
        presentation: Study presentation order for the trial.
        should_tabulate: Boolean mask aligned to recall events; True means count the
            transition into that recall.
        size: Maximum number of study positions an item can occupy.
    """

    init = Tabulation(presentation, trial[0], size)
    tab = lax.fori_loop(
        1, trial.size, lambda i, t: t.tabulate(trial[i], should_tabulate[i]), init
    )
    return tab.actual_lags, tab.avail_lags


def crp(
    dataset: RecallDataset,
    size: int = 3,
) -> Float[Array, " lags"]:
    """
    Lag-CRP with repeated items in the study list.

    Args:
        dataset: recall dataset containing at least ``recalls`` and ``pres_itemnos``,
            plus a boolean ``_should_tabulate`` field aligned to recall events.
        size: max # of study positions an item can occupy (compile-time constant).

    Repeats policy:
        - A recall of a repeated item is treated as recalling *all* of that
          item's study positions.
        - Actual and available lags are computed as **unique** lag set unions
          (boolean one-hot union per lag) to avoid over-counting.

    Returns:
        [2*L - 1] floats; NaN where denominator is zero.
    """
    should_tabulate = jnp.asarray(dataset["_should_tabulate"], dtype=bool) # type: ignore

    actual, possible = vmap(tabulate_trial, in_axes=(0, 0, 0, None))(
        dataset["recalls"],
        dataset["pres_itemnos"],
        should_tabulate,
        size,
    )
    return actual.sum(0) / possible.sum(0)


def _segment_by_nan(vector: jnp.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) segments split by NaN entries.

    Args:
        vector: Vector to segment.

    Returns:
        List of (start, end) index pairs.
    """
    segments: list[tuple[int, int]] = []
    start_index = 0
    for i in range(len(vector)):
        if bool(jnp.isnan(vector[i])):
            segments.append((start_index, i))
            start_index = i + 1
    if start_index < len(vector):
        segments.append((start_index, len(vector)))
    return segments


def _hierarchical_bootstrap_means(
    actual_by_trial: np.ndarray,
    possible_by_trial: np.ndarray,
    subject_ids: np.ndarray,
    trial_mask: Bool[Array, " trial_count"],
    n_resamples: int = 100,
    confidence_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return hierarchical bootstrap mean curves and confidence bounds.

    Args:
        actual_by_trial: Actual lag counts per trial (shape [trials, lags]).
        possible_by_trial: Available lag counts per trial (shape [trials, lags]).
        subject_ids: Subject id per trial (shape [trials]).
        trial_mask: Boolean mask selecting trials to include.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level for the bounds.
        rng: Optional NumPy random generator.

    Returns:
        Tuple of (mean_curves, ci_low, ci_high), each shaped [lags].
    """
    if rng is None:
        rng = np.random.default_rng()

    trial_mask_np = np.asarray(trial_mask, dtype=bool)
    trial_indices = np.nonzero(trial_mask_np)[0]
    if trial_indices.size == 0:
        raise ValueError("No trials selected by trial_mask.")

    subject_ids = np.asarray(subject_ids).reshape(-1)[trial_mask_np]
    actual_by_trial = np.asarray(actual_by_trial)[trial_mask_np]
    possible_by_trial = np.asarray(possible_by_trial)[trial_mask_np]

    unique_subjects = np.unique(subject_ids)
    indices_by_subject: dict[int, np.ndarray] = {}
    for subject in unique_subjects:
        indices_by_subject[int(subject)] = np.nonzero(subject_ids == subject)[0]

    boot_means: list[np.ndarray] = []
    for _ in range(n_resamples):
        sampled_subjects = rng.choice(unique_subjects, size=unique_subjects.size, replace=True)
        actual_sum = np.zeros(actual_by_trial.shape[1], dtype=float)
        possible_sum = np.zeros(possible_by_trial.shape[1], dtype=float)

        for subject in sampled_subjects:
            subject_indices = indices_by_subject[int(subject)]
            resampled_indices = rng.choice(
                subject_indices, size=subject_indices.size, replace=True
            )
            actual_sum += actual_by_trial[resampled_indices].sum(axis=0)
            possible_sum += possible_by_trial[resampled_indices].sum(axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            boot_means.append(actual_sum / possible_sum)

    mean_curves = np.vstack(boot_means)
    alpha = (1.0 - confidence_level) / 2.0
    ci_low = np.nanpercentile(mean_curves, 100 * alpha, axis=0)
    ci_high = np.nanpercentile(mean_curves, 100 * (1 - alpha), axis=0)
    return np.nanmean(mean_curves, axis=0), ci_low, ci_high


def _plot_with_error_bars(
    axis: Axes,
    x_values: Real[Array, " trial_count values"],
    y_mean: Real[Array, " values"],
    ci_low: Real[Array, " values"],
    ci_high: Real[Array, " values"],
    segments: list[tuple[int, int]],
    label: str,
    color: str,
) -> None:
    """Plot a line with hierarchical-bootstrap error bars.

    Args:
        axis: Axis to plot on.
        x_values: X-axis values.
        y_mean: Mean y values to plot.
        ci_low: Lower confidence bound for y values.
        ci_high: Upper confidence bound for y values.
        segments: Segments splitting NaN regions.
        label: Legend label for the line.
        color: Line and error bar color.
    """
    y_mean = np.asarray(y_mean)
    ci_low = np.asarray(ci_low)
    ci_high = np.asarray(ci_high)
    errors = np.vstack((y_mean - ci_low, ci_high - y_mean))

    for index, (start, end) in enumerate(segments):
        axis.errorbar(
            x_values[start:end],
            y_mean[start:end],
            errors[:, start:end],
            label=label if index == 0 else None,
            color=color,
            capsize=3,
            marker="o",
            markersize=5,
            linewidth=1.5,
        )


def plot_data(
    axis: Axes,
    x_values: Real[Array, " trial_count values"],
    y_values: Real[Array, " trial_count values"],
    label: str,
    color: str,
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    size: int,
    slice_start: int,
    slice_end: int,
) -> Axes:
    """Plot mean curves with hierarchical-bootstrap error bars.

    Args:
        axis: Axis to plot on.
        x_values: X-axis values.
        y_values: Subject-level curves stacked by row.
        label: Legend label for the line.
        color: Line and error bar color.
        dataset: Dataset used for bootstrap resampling.
        trial_mask: Trial mask applied before bootstrapping.
        size: Maximum number of study positions an item can occupy.
        slice_start: Starting lag index (inclusive) to plot.
        slice_end: Ending lag index (exclusive) to plot.
    """
    should_tabulate = jnp.asarray(dataset["_should_tabulate"], dtype=bool)
    actual_by_trial, possible_by_trial = vmap(tabulate_trial, in_axes=(0, 0, 0, None))(
        dataset["recalls"],
        dataset["pres_itemnos"],
        should_tabulate,
        size,
    )

    trial_mask_np = np.asarray(trial_mask, dtype=bool)
    actual_total = np.asarray(actual_by_trial)[trial_mask_np].sum(axis=0)
    possible_total = np.asarray(possible_by_trial)[trial_mask_np].sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        y_mean = actual_total / possible_total

    y_mean = np.asarray(y_mean)[slice_start:slice_end]
    segments = _segment_by_nan(jnp.asarray(y_mean))
    subject_ids = np.asarray(dataset["subject"]).reshape(-1)
    subject_count = np.unique(subject_ids[trial_mask_np]).size

    if subject_count > 1:
        _, ci_low, ci_high = _hierarchical_bootstrap_means(
            actual_by_trial=actual_by_trial,
            possible_by_trial=possible_by_trial,
            subject_ids=subject_ids,
            trial_mask=trial_mask,
        )
        ci_low = np.asarray(ci_low)[slice_start:slice_end]
        ci_high = np.asarray(ci_high)[slice_start:slice_end]
        _plot_with_error_bars(
            axis,
            x_values,
            y_mean,
            ci_low,
            ci_high,
            segments,
            label,
            color,
        )
    else:
        plot_without_error_bars(axis, x_values, y_mean, segments, label, color)

    x = jnp.asarray(x_values)
    x = x[jnp.isfinite(x)]
    if x.size and bool(jnp.allclose(x, jnp.round(x), atol=1e-9)):
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    return axis

def plot_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    should_tabulate: (
        Sequence[Bool[Array, " trial_count recall_events"]]
        | Bool[Array, " trial_count recall_events"]
    ),
    max_lag: int = 4,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    size: int = 3,
) -> Axes:
    """
    Plot subject-wise Lag-CRP and their mean ± error.

    Args:
        datasets: Datasets containing trial data to be plotted.
        trial_masks: Masks to filter trials in datasets.
        should_tabulate: Boolean masks aligned to recall events, one per dataset,
            indicating which transitions to include in tabulation.
        max_lag: Maximum lag to plot.
        color_cycle: List of colors for plotting each dataset.
        labels: Names for each dataset for legend, optional.
        contrast_name: Name of contrast for legend labeling, optional.
        axis: Existing matplotlib Axes to plot on, optional.
        size: Maximum number of study positions an item can be presented at.

    Returns:
        Matplotlib Axes with the Lag-CRP plot.

    Notes:
        - `datasets` must contain 'recalls', 'pres_itemnos', 'listLength'.
        - `trial_masks` filters trials; lengths must match datasets.
        - Color cycle wraps if more datasets than colors.
    """
    axis = init_plot(axis)

    if color_cycle is None:
        color_cycle = [each["color"] for each in rcParams["axes.prop_cycle"]]

    if labels is None:
        labels = [""] * len(datasets)

    if not isinstance(datasets, Sequence):
        datasets = [datasets]

    if not isinstance(should_tabulate, Sequence):
        should_tabulate = [jnp.array(should_tabulate)]

    if not isinstance(trial_masks, Sequence):
        trial_masks = [jnp.array(trial_masks)]

    lag_interval = jnp.arange(-max_lag, max_lag + 1, dtype=int)

    for data_index, data in enumerate(datasets):
        lag_range = (jnp.max(data["listLength"]) - 1).item()
        slice_start = lag_range - max_lag
        slice_end = lag_range + max_lag + 1
        data_with_mask = {
            **data,
            "_should_tabulate": should_tabulate[data_index],
        }
        subject_values = apply_by_subject(
            data_with_mask, # type: ignore
            trial_masks[data_index],
            jit(crp, static_argnames=("size")),
            size=size,
        )
        subject_values = jnp.vstack(subject_values)
        subject_values = subject_values[
            :, lag_range - max_lag : lag_range + max_lag + 1
        ]

        color = color_cycle.pop(0)
        plot_data(
            axis,
            lag_interval,
            subject_values,
            labels[data_index],
            color,
            dataset=data_with_mask,
            trial_mask=trial_masks[data_index],
            size=size,
            slice_start=slice_start,
            slice_end=slice_end,
        )

    set_plot_labels(axis, "Lag", "Conditional Resp. Prob.", contrast_name)
    return axis
