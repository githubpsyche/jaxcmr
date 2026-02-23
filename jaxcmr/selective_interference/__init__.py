"""Selective interference simulation and analysis utilities."""

from .analysis import build_transition_masks, derive_cue_clips
from .context_tracking import track_context_trajectory
from .cmr import make_factory
from .fitting import load_or_fit_params
from .paradigm import (
    Paradigm,
    compute_n_presented,
    make_extended_break,
    make_extended_filler,
    make_extended_interference,
    make_is_emotional,
)
from .pipeline import (
    PreparedSweep,
    batch_trial,
    configure_rates,
    film_recalled_stats,
    prepare_sweep,
    run_count_sweep,
    run_sweep,
    sweep_rngs,
)
from .remapping import (
    break_extended_remap,
    filler_extended_remap,
    interference_extended_remap,
    remap_recalls,
    standard_remap,
)
from .plotting import (
    add_filler_boundary,
    light_to_dark_colors,
    plot_context_trajectory,
    plot_interference_spc,
    plot_summary_dv,
)

__all__ = [
    "build_transition_masks",
    "derive_cue_clips",
    "track_context_trajectory",
    "Paradigm",
    "PreparedSweep",
    "prepare_sweep",
    "run_count_sweep",
    "run_sweep",
    "batch_trial",
    "configure_rates",
    "sweep_rngs",
    "remap_recalls",
    "standard_remap",
    "interference_extended_remap",
    "filler_extended_remap",
    "break_extended_remap",
    "film_recalled_stats",
    "load_or_fit_params",
    "make_factory",
    "compute_n_presented",
    "make_extended_interference",
    "make_extended_break",
    "make_extended_filler",
    "make_is_emotional",
    "plot_interference_spc",
    "plot_context_trajectory",
    "plot_summary_dv",
    "add_filler_boundary",
    "light_to_dark_colors",
]
