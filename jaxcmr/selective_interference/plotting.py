"""Plotting helpers for selective interference simulations.

Provides convenience functions for visualizing context trajectories,
recall curves, and interference effects from selective interference
simulation results.

"""

from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from jaxcmr.typing import Array, Float


__all__ = [
    "plot_interference_spc",
    "plot_context_trajectory",
    "plot_summary_dv",
]


def _light_to_dark_colors(n: int) -> list[str]:
    """Return *n* grey-scale hex codes from light to dark."""
    fracs = np.linspace(0.75, 0.10, n)
    return [f"#{int(f*255):02x}{int(f*255):02x}{int(f*255):02x}" for f in fracs]


def _style_ax(ax: Axes) -> None:
    """Apply standard axis styling matching jaxcmr.plotting conventions."""
    ax.tick_params(labelsize=14)
    for loc in ("top", "right"):
        ax.spines[loc].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)


def plot_interference_spc(
    spc_curves: Sequence[Float[Array, " list_length"]],
    labels: Sequence[str],
    n_film: int,
    *,
    n_break: int = 0,
    n_presented: Optional[Union[int, Sequence[int]]] = None,
    colors: Optional[Sequence[str]] = None,
    ax: Optional[Axes] = None,
    ylabel: str = "Recall Probability",
) -> tuple[Figure, Axes]:
    """Plot multi-line SPC with film / break / interference boundaries.

    Parameters
    ----------
    spc_curves : Sequence[Float[Array, " list_length"]]
        One recall-probability curve per condition.
    labels : Sequence[str]
        Legend label for each curve.
    n_film : int
        Number of film items (boundary position).
    n_break : int
        Number of break items shown (0 = no break zone).
    n_presented : int or Sequence[int], optional
        Number of valid study positions per curve. Positions beyond
        this are masked so the line terminates at the last real item.
        A single int applies to all curves.
    colors : Sequence[str], optional
        Colours for each curve; defaults to light-to-dark grey.
    ax : Axes, optional
        Matplotlib axes to plot on.
    ylabel : str
        Y-axis label.

    Returns
    -------
    tuple[Figure, Axes]

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
    if colors is None:
        colors = _light_to_dark_colors(len(spc_curves))

    list_length = len(spc_curves[0])
    positions = np.arange(1, list_length + 1)

    for i, (curve, label, color) in enumerate(zip(spc_curves, labels, colors)):
        arr = np.asarray(curve, dtype=float)
        if n_presented is not None:
            n = n_presented[i] if not isinstance(n_presented, (int, np.integer)) else n_presented
            arr[n:] = np.nan
        ax.plot(positions, arr, color=color, label=label, linewidth=1.5)

    # Boundary lines
    film_end = n_film + 0.5
    ax.axvline(x=film_end, color="red", linewidth=1.5, linestyle="--", alpha=0.7)
    if n_break > 0:
        break_end = n_film + n_break + 0.5
        ax.axvline(x=break_end, color="red", linewidth=1.5, linestyle="--", alpha=0.7)

    # Zone labels
    film_x = film_end / list_length / 2
    if n_break > 0:
        break_end_frac = break_end / list_length
        break_x = (film_end / list_length + break_end_frac) / 2
        interf_x = (break_end_frac + 1) / 2
        ax.text(film_x, 0.97, "Film", ha="center", va="top",
                fontsize=12, fontweight="bold", transform=ax.transAxes)
        ax.text(break_x, 0.97, "Break", ha="center", va="top",
                fontsize=12, fontweight="bold", color="gray",
                transform=ax.transAxes)
        ax.text(interf_x, 0.97, "Interference", ha="center",
                va="top", fontsize=12, fontweight="bold", color="red",
                transform=ax.transAxes)
    else:
        interf_x = (film_end / list_length + 1) / 2
        ax.text(film_x, 0.97, "Film", ha="center", va="top",
                fontsize=12, fontweight="bold", transform=ax.transAxes)
        ax.text(interf_x, 0.97, "Interference", ha="center",
                va="top", fontsize=12, fontweight="bold", color="red",
                transform=ax.transAxes)

    ax.set_xlabel("Study Position", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    _style_ax(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=12)
    fig.tight_layout()
    return fig, ax


def plot_context_trajectory(
    trajectories: Sequence[Float[Array, " steps"]],
    labels: Sequence[str],
    *,
    colors: Optional[Sequence[str]] = None,
    ax: Optional[Axes] = None,
) -> tuple[Figure, Axes]:
    """Plot context similarity over interference encoding steps.

    Parameters
    ----------
    trajectories : Sequence[Float[Array, " steps"]]
        Similarity values for each condition.
    labels : Sequence[str]
        Legend label for each trajectory.
    colors : Sequence[str], optional
        Colours; defaults to light-to-dark grey.
    ax : Axes, optional
        Matplotlib axes to plot on.

    Returns
    -------
    tuple[Figure, Axes]

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure
    if colors is None:
        colors = _light_to_dark_colors(len(trajectories))

    for traj, label, color in zip(trajectories, labels, colors):
        steps = np.arange(1, len(traj) + 1)
        ax.plot(steps, np.asarray(traj), color=color, label=label, linewidth=1.5)

    ax.set_xlabel("Interference Encoding Step", fontsize=16)
    ax.set_ylabel("Similarity to Film-End Context", fontsize=16)
    _style_ax(ax)
    ax.legend(loc="upper right", fontsize=12)
    fig.tight_layout()
    return fig, ax


def plot_summary_dv(
    x_values: ArrayLike,
    means: ArrayLike,
    ci_lower: ArrayLike,
    ci_upper: ArrayLike,
    *,
    xlabel: str = "Parameter Value",
    ylabel: str = "Film Items Recalled",
    ax: Optional[Axes] = None,
) -> tuple[Figure, Axes]:
    """Plot swept parameter vs dependent variable with 95 % CI.

    Parameters
    ----------
    x_values : ArrayLike
        Swept parameter values.
    means : ArrayLike
        Mean DV at each value.
    ci_lower, ci_upper : ArrayLike
        Lower and upper bounds of 95 % CI.
    xlabel, ylabel : str
        Axis labels.
    ax : Axes, optional
        Matplotlib axes.

    Returns
    -------
    tuple[Figure, Axes]

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    x = np.asarray(x_values)
    m = np.asarray(means)
    lo = np.asarray(ci_lower)
    hi = np.asarray(ci_upper)

    ax.fill_between(x, lo, hi, alpha=0.25, color="black")
    ax.plot(x, m, "o-", color="black", linewidth=1.5, markersize=5)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    _style_ax(ax)
    fig.tight_layout()
    return fig, ax
