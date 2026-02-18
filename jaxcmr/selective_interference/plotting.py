"""Plotting helpers for selective interference simulations.

Provides convenience functions for visualizing context trajectories,
recall curves, and interference effects from selective interference
simulation results.

"""

from typing import Optional, Sequence, Union

import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.transforms import blended_transform_factory

from jaxcmr.plotting import init_plot, set_plot_labels
from jaxcmr.typing import Array, Float


__all__ = [
    "plot_interference_spc",
    "plot_context_trajectory",
    "plot_summary_dv",
    "add_filler_boundary",
    "light_to_dark_colors",
]


def light_to_dark_colors(n: int) -> list[str]:
    """Return *n* grey-scale hex codes from light to dark."""
    fracs = np.linspace(0.75, 0.10, n)
    return [f"#{int(f*255):02x}{int(f*255):02x}{int(f*255):02x}" for f in fracs]


def plot_interference_spc(
    spc_curves: Sequence[Float[Array, " list_length"]],
    labels: Optional[Sequence[str]] = None,
    n_film: int = 16,
    n_break: int = 0,
    n_presented: Optional[Union[int, Sequence[int]]] = None,
    color_cycle: Optional[list[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    ylabel: str = "Recall Probability",
) -> Axes:
    """Plot multi-line SPC with film / break / interference boundaries.

    Parameters
    ----------
    spc_curves : Sequence[Float[Array, " list_length"]]
        One recall-probability curve per condition.
    labels : Sequence[str], optional
        Legend label for each curve.
    n_film : int
        Number of film items (boundary position).
    n_break : int
        Number of break items shown (0 = no break zone).
    n_presented : int or Sequence[int], optional
        Number of valid study positions per curve. Positions beyond
        this are masked so the line terminates at the last real item.
        A single int applies to all curves.
    color_cycle : list[str], optional
        Colours for each curve; defaults to matplotlib colour cycle.
    contrast_name : str, optional
        Legend title passed to ``set_plot_labels``.
    axis : Axes, optional
        Matplotlib axes to plot on.
    ylabel : str
        Y-axis label.

    Returns
    -------
    Axes

    """
    axis = init_plot(axis)
    if labels is None:
        labels = [""] * len(spc_curves)
    if color_cycle is None:
        color_cycle = [each["color"] for each in rcParams["axes.prop_cycle"]]

    list_length = len(spc_curves[0])
    positions = np.arange(1, list_length + 1)

    for i, (curve, label, color) in enumerate(
        zip(spc_curves, labels, color_cycle)
    ):
        arr = np.asarray(curve, dtype=float)
        if n_presented is not None:
            n = n_presented[i] if not isinstance(n_presented, (int, np.integer)) else n_presented
            arr[n:] = np.nan
        arr[arr == 0.0] = np.nan
        axis.plot(positions, arr, color=color, label=label, linewidth=1.5)

    # Boundary lines
    film_end = n_film + 0.5
    break_end = n_film + n_break + 0.5
    axis.axvline(x=film_end, color="red", linewidth=1.5, linestyle="--", alpha=0.7)
    if n_break > 0:
        axis.axvline(x=break_end, color="red", linewidth=1.5, linestyle="--", alpha=0.7)

    # Zone labels (data x, axes y via blended transform)
    trans = blended_transform_factory(axis.transData, axis.transAxes)
    film_mid = (1 + n_film) / 2
    axis.text(film_mid, 1.02, "Film", ha="center", va="bottom",
            fontsize=10, fontweight="bold", transform=trans, clip_on=False)
    if n_break > 0:
        break_mid = n_film + (1 + n_break) / 2
        axis.text(break_mid, 1.02, "Break", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="gray",
                transform=trans, clip_on=False)
        interf_mid = n_film + n_break + (1 + (list_length - n_film - n_break)) / 2
    else:
        interf_mid = n_film + (1 + (list_length - n_film)) / 2
    axis.text(interf_mid, 1.02, "Interference", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="red",
            transform=trans, clip_on=False)

    # Reminder annotation along the pre-/post-reminder boundary
    reminder_x = n_film + n_break + 0.5
    reminder_label = "Reminder" if n_break > 0 else "Break + Reminder"
    axis.text(reminder_x, 0.97, reminder_label, rotation=90, ha="right", va="top",
              fontsize=10, fontstyle="italic", color="black",
              transform=trans, clip_on=True)

    set_plot_labels(axis, "Study Position", ylabel, contrast_name)
    return axis


def plot_context_trajectory(
    trajectories: Sequence[Float[Array, " steps"]],
    labels: Optional[Sequence[str]] = None,
    color_cycle: Optional[list[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
) -> Axes:
    """Plot context similarity over interference encoding steps.

    Parameters
    ----------
    trajectories : Sequence[Float[Array, " steps"]]
        Similarity values for each condition.
    labels : Sequence[str], optional
        Legend label for each trajectory.
    color_cycle : list[str], optional
        Colours; defaults to matplotlib colour cycle.
    contrast_name : str, optional
        Legend title passed to ``set_plot_labels``.
    axis : Axes, optional
        Matplotlib axes to plot on.

    Returns
    -------
    Axes

    """
    axis = init_plot(axis)
    if labels is None:
        labels = [""] * len(trajectories)
    if color_cycle is None:
        color_cycle = [each["color"] for each in rcParams["axes.prop_cycle"]]

    for traj, label, color in zip(trajectories, labels, color_cycle):
        steps = np.arange(1, len(traj) + 1)
        axis.plot(steps, np.asarray(traj), color=color, label=label, linewidth=1.5)

    set_plot_labels(axis, "Interference Encoding Step", "Similarity to Film-End Context",
                    contrast_name)
    axis.legend(loc="upper right", fontsize=12)
    return axis


def plot_summary_dv(
    x_values: Sequence[float],
    means: Sequence[float],
    ci_lower: Sequence[float],
    ci_upper: Sequence[float],
    xlabel: str = "Parameter Value",
    ylabel: str = "Film Items Recalled",
    axis: Optional[Axes] = None,
) -> Axes:
    """Plot swept parameter vs dependent variable with 95 % CI.

    Parameters
    ----------
    x_values : Sequence[float]
        Swept parameter values.
    means : Sequence[float]
        Mean DV at each value.
    ci_lower, ci_upper : Sequence[float]
        Lower and upper bounds of 95 % CI.
    xlabel, ylabel : str
        Axis labels.
    axis : Axes, optional
        Matplotlib axes.

    Returns
    -------
    Axes

    """
    axis = init_plot(axis)

    x = np.asarray(x_values)
    m = np.asarray(means)
    lo = np.asarray(ci_lower)
    hi = np.asarray(ci_upper)

    axis.fill_between(x, lo, hi, alpha=0.25, color="black")
    axis.plot(x, m, "o-", color="black", linewidth=1.5, markersize=5)

    set_plot_labels(axis, xlabel, ylabel)
    return axis


def add_filler_boundary(
    axis: Axes,
    n_film: int,
    n_interference: int,
    n_filler: int,
    n_presented: int,
    n_break: int = 0,
    show_fillers: bool = True,
) -> Axes:
    """Add filler zone boundary and relabel zones on an SPC axes.

    Parameters
    ----------
    axis : Axes
        Matplotlib axes with an existing SPC plot.
    n_film : int
        Number of film items.
    n_interference : int
        Number of interference items.
    n_filler : int
        Number of filler items.
    n_presented : int
        Total presented positions (after remapping).
    n_break : int
        Number of break items shown (0 = no break zone).
    show_fillers : bool
        Whether to draw the filler boundary.

    Returns
    -------
    Axes

    """
    if not show_fillers or n_filler <= 0:
        return axis
    filler_start = n_film + n_break + n_interference + 0.5
    axis.axvline(
        x=filler_start, color="blue", linewidth=1.5, linestyle=":", alpha=0.7
    )
    for txt in list(axis.texts):
        txt.remove()
    trans = blended_transform_factory(axis.transData, axis.transAxes)
    film_mid = (1 + n_film) / 2
    interf_mid = n_film + n_break + (1 + n_interference) / 2
    filler_mid = n_film + n_break + n_interference + (1 + n_filler) / 2
    axis.text(film_mid, 1.02, "Film", ha="center", va="bottom",
            fontsize=10, fontweight="bold", transform=trans, clip_on=False)
    if n_break > 0:
        break_mid = n_film + (1 + n_break) / 2
        axis.text(break_mid, 1.02, "Break", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="gray",
                transform=trans, clip_on=False)
    axis.text(interf_mid, 1.02, "Interference", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="red",
            transform=trans, clip_on=False)
    axis.text(filler_mid, 1.02, "Filler", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="blue",
            transform=trans, clip_on=False)

    # Reminder annotation along the pre-/post-reminder boundary
    reminder_x = n_film + n_break + 0.5
    reminder_label = "Reminder" if n_break > 0 else "Break + Reminder"
    axis.text(reminder_x, 0.97, reminder_label, rotation=90, ha="right", va="top",
              fontsize=10, fontstyle="italic", color="black",
              transform=trans, clip_on=True)
    return axis
