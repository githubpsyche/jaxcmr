"""Check points drawn, selected trials, and labels, rather than Axes types."""

import jax.numpy as jnp
import numpy as np
import pytest
from jaxcmr.analyses import (
    spc,
    srac,
    relative_srac,
    termination_probability,
    distcrp,
    conditional_distcrp,
)
from jaxcmr.helpers import make_dataset


@pytest.mark.parametrize(
    "plot,kwargs,expected,xlabel,ylabel",
    [
        (spc.plot_spc, {}, [1, 1, 1], "Study Position", "Recall Rate"),
        (srac.plot_srac, {}, [1, 1, 1], "Study Position", "Serial Recall Accuracy"),
        (relative_srac.plot_relative_srac, {}, [1, 1, 1], "Study Position", "Recall Rate"),
        (
            termination_probability.plot_termination_probability,
            {"mode": "simple"},
            [0, 0, 1],
            "Recall Position",
            "P(Terminate)",
        ),
        (
            termination_probability.plot_termination_probability,
            {"mode": "conditional"},
            [0, 0, 1],
            "Recall Position",
            "P(Terminate | Reach)",
        ),
    ],
)
def test_recall_plots_draw_selected_values(plot, kwargs, expected, xlabel, ylabel):
    data = make_dataset(jnp.array([[1, 2, 3], [3, 0, 0]]), subject=jnp.array([2, 9]))
    axis = plot(data, jnp.array([True, False]), labels=["selected"], **kwargs)
    (line,) = [line for line in axis.lines if line.get_label() == "selected"]
    np.testing.assert_array_equal(line.get_xdata(), [1, 2, 3])
    np.testing.assert_allclose(line.get_ydata(), expected)
    assert axis.get_xlabel() == xlabel
    assert axis.get_ylabel() == ylabel


@pytest.mark.parametrize(
    "conditional,rule",
    [
        (False, "explicit"),
        (True, "explicit"),
        (False, "percentiles"),
        (True, "percentiles"),
        (False, "min_count"),
        (True, "min_count"),
    ],
)
def test_distance_plots_draw_counted_bins(conditional, rule):
    data = make_dataset(jnp.array([[1, 2, 0], [1, 3, 0]]), subject=jnp.array([2, 2]))
    data["pres_itemids"] = data["pres_itemnos"]
    distance = jnp.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    kwargs = dict(distance_matrix=distance, labels=["observed"])
    if rule == "explicit":
        kwargs.update(bin_edges=jnp.array([1.5]), bin_centers=jnp.array([1.0, 2.0]))
    elif rule == "percentiles":
        pass  # Exercise the actual default percentile rule.
    else:
        kwargs.update(bin_edges="min_count", min_transitions_per_subject=1)
    args = (data, jnp.array([True, True]))
    plot = distcrp.plot_dist_crp
    if conditional:
        args += (jnp.ones_like(data["recalls"], dtype=bool),)
        plot = conditional_distcrp.plot_dist_crp
    axis = plot(*args, **kwargs)
    lines = [line for line in axis.lines if line.get_marker() == "o"]
    assert "observed" in axis.get_legend_handles_labels()[1]
    values = np.concatenate([line.get_ydata() for line in lines])
    # At edge 1, both targets enter the upper bin: each event has that bin available.
    expected = [1.0] if rule == "min_count" else [0.5, 0.5]
    np.testing.assert_allclose(values, expected)
    positions = np.concatenate([line.get_xdata() for line in lines])
    assert np.all(np.diff(positions) > 0)
    assert "Distance" in axis.get_xlabel()
    assert axis.get_ylabel() == "Conditional Resp. Prob."


def test_conditional_distance_features_take_precedence_over_conflicting_matrix():
    data = make_dataset(jnp.array([[1, 2, 0], [1, 2, 0]]))
    data["pres_itemids"] = data["pres_itemnos"]
    mask = jnp.ones(2, dtype=bool)
    include = jnp.ones_like(data["recalls"], dtype=bool)
    # Items 1 and 2 coincide in feature space; supplied matrix says far apart.
    features = jnp.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    conflict = jnp.array([[0.0, 2.0, 0.0], [2.0, 0.0, 2.0], [0.0, 2.0, 0.0]])
    kwargs = dict(
        bin_edges=jnp.array([0.5]), bin_centers=jnp.array([0.25, 0.75]), labels=["features"]
    )
    axis = conditional_distcrp.plot_dist_crp(
        data, mask, include, features=features, distance_matrix=conflict, **kwargs
    )
    np.testing.assert_allclose(axis.lines[0].get_ydata(), [1.0, 0.0])
    with pytest.raises(ValueError):
        conditional_distcrp.plot_dist_crp(data, mask, include, **kwargs)
    with pytest.raises(ValueError, match="Exactly one"):
        distcrp.plot_dist_crp(data, mask, features=features, distance_matrix=conflict, **kwargs)
