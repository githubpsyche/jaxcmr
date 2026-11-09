"""Plot routing must use the selected trials and preserve the analysis values."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from jaxcmr.analyses import (
    crp,
    conditional_crp,
    repcrp,
    repaccesscrp,
    reproutecrp,
    repcuecrp,
    target_enrichment_crp,
    source_target_crp,
)
from jaxcmr.helpers import make_dataset


@pytest.mark.parametrize(
    "kind,measure,plot,options,labels",
    [
        ("ordinary", crp.crp, crp.plot_crp, {"size": 3}, ["selected"]),
        (
            "repeated",
            repcrp.repcrp,
            repcrp.plot_rep_crp,
            {"min_lag": 4, "size": 2},
            ["First", "Second"],
        ),
        (
            "access",
            repaccesscrp.repaccesscrp,
            repaccesscrp.plot_rep_access_crp,
            {"min_lag": 4, "size": 2},
            ["First", "Second"],
        ),
        (
            "route",
            reproutecrp.reproutecrp,
            reproutecrp.plot_rep_route_crp,
            {"min_lag": 4, "direction": "same", "use_lag2": True},
            ["selected"],
        ),
        (
            "cue",
            repcuecrp.repcuecrp,
            repcuecrp.plot_rep_cue_crp,
            {"min_lag": 4, "size": 2},
            ["First", "Second"],
        ),
        (
            "conditional",
            conditional_crp.crp,
            conditional_crp.plot_crp,
            {
                "size": 3,
                "source_field": "condition",
                "source_values": [0],
                "target_field": "condition",
                "target_values": [-1, 1],
            },
            ["Negative", "Positive"],
        ),
        (
            "enrichment",
            target_enrichment_crp.target_enrichment_crp,
            target_enrichment_crp.plot_target_enrichment_crp,
            {
                "size": 3,
                "source_field": "condition",
                "source_values": [0],
                "target_field": "condition",
                "target_value": -1,
                "target_values": [-1, 0, 1],
            },
            ["selected"],
        ),
    ],
)
def test_transition_plots_preserve_values_and_conditions(kind, measure, plot, options, labels):
    data = make_dataset(
        jnp.array([[2, 1, 3, 0, 0, 0, 0, 0], [7, 8, 0, 0, 0, 0, 0, 0]]),
        jnp.array([[1, 2, 3, 4, 5, 1, 7, 8]] * 2),
    )
    data["condition"] = jnp.array([[-1, 0, 1, 0, 0, -1, 0, 1]] * 2)
    if kind == "route":
        data["recalls"] = data["recalls"].at[0].set(jnp.array([5, 1, 7, 0, 0, 0, 0, 0]))
    selected = {key: value[:1] for key, value in data.items()}
    expected = np.asarray(measure(selected, **options))
    if kind == "cue":
        expected = expected[:, 8, :]  # cue offset +1 in the full 15-offset array
    axis = plot(data, jnp.array([True, False]), max_lag=7, labels=labels, **options)
    drawn = np.concatenate(
        [
            np.asarray(line.get_ydata(), dtype=float)
            for line in axis.lines
            if line.get_marker() == "o"
        ]
    )
    expected = expected[np.isfinite(expected)]
    assert expected.size > 0
    np.testing.assert_allclose(drawn, expected, atol=1e-7)
    assert set(labels) <= set(axis.get_legend_handles_labels()[1])
    assert "Lag" in axis.get_xlabel() or "Offset" in axis.get_xlabel()
    assert axis.get_ylabel()
    plt.close(axis.figure)


def test_source_target_plot_counts_available_items_and_labels_dataset():
    data = make_dataset(jnp.array([[1, 3, 0], [2, 1, 0]]))
    data["valence"] = jnp.array([[-1, 1, 1]] * 2)
    axis = source_target_crp.plot_source_target_crp(
        data,
        jnp.array([True, False]),
        source_field="valence",
        source_values=[-1],
        target_field="valence",
        target_values=[1],
        source_labels=["Negative"],
        labels=["selected"],
        size=3,
    )
    drawn = np.concatenate(
        [
            np.asarray(line.get_ydata(), dtype=float)
            for line in axis.lines
            if line.get_marker() == "o"
        ]
    )
    np.testing.assert_allclose(drawn, [0.5])
    assert "selected" in axis.get_legend_handles_labels()[1]
    assert axis.get_ylabel()
    plt.close(axis.figure)


@pytest.mark.parametrize("plot", [repcrp.plot_first_rep_crp, repcrp.plot_second_rep_crp])
def test_repetition_wrappers_keep_parameter_sweep_labels(plot):
    data = make_dataset(
        jnp.array([[1, 3, 2, 4, 0, 0, 0, 0]]), jnp.array([[1, 2, 3, 4, 5, 1, 2, 8]])
    )
    mask = jnp.array([True])
    axis = plot([data, data], [mask, mask], max_lag=3, min_lag=4, size=2, labels=["0.0", "1.0"])
    assert {"0.0", "1.0"} <= set(axis.get_legend_handles_labels()[1])
    assert any(np.isfinite(np.asarray(line.get_ydata(), dtype=float)).any() for line in axis.lines)
