"""Selective interference simulation and analysis utilities."""

from .analysis import build_transition_masks, derive_cue_clips
from .context_tracking import track_context_trajectory
from .fitting import load_or_fit_params
from .paradigm import (
    Paradigm,
    compute_n_presented,
    make_extended_break,
    make_extended_filler,
    make_extended_interference,
)
from .preparation import prepare_all_subjects
from .remapping import (
    break_extended_remap,
    filler_extended_remap,
    interference_extended_remap,
    remap_recalls,
    standard_remap,
)
from .sweep import (
    batched_sweep,
    film_recalled_stats,
    run_sweep,
    sweep_defaults,
    sweep_rngs,
)
from .plotting import (
    add_filler_boundary,
    plot_context_trajectory,
    plot_interference_spc,
    plot_summary_dv,
    plot_sweep,
)

__all__ = [
    "build_transition_masks",
    "derive_cue_clips",
    "track_context_trajectory",
    "Paradigm",
    "prepare_all_subjects",
    "batched_sweep",
    "run_sweep",
    "sweep_rngs",
    "remap_recalls",
    "standard_remap",
    "interference_extended_remap",
    "filler_extended_remap",
    "break_extended_remap",
    "film_recalled_stats",
    "load_or_fit_params",
    "compute_n_presented",
    "make_extended_interference",
    "make_extended_break",
    "make_extended_filler",
    "sweep_defaults",
    "plot_interference_spc",
    "plot_context_trajectory",
    "plot_summary_dv",
    "plot_sweep",
    "add_filler_boundary",
]
