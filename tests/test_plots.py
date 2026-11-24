"""Check points drawn, selected trials, and labels, rather than Axes types."""

from copy import deepcopy

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import Normalize

from jaxcmr.analyses import (
    conditional_distcrp,
    distcrp,
    relative_srac,
    spc,
    srac,
    termination_probability,
)
from jaxcmr.dataset import make_dataset
from jaxcmr.plotting import (
    comparison_panels,
    iter_analysis_figures,
    matrix_heatmap,
    plot_mean_curve,
    save_figure,
)


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
    lines = [line for line in axis.lines if line.get_label() == "selected"]
    assert len(lines) == 1, "Selected curve must be drawn"
    line = lines[0]
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
    np.testing.assert_allclose(values[np.isfinite(values)], expected)
    positions = np.concatenate([line.get_xdata() for line in lines])
    assert np.all(np.diff(positions[np.isfinite(values)]) > 0)
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



@pytest.mark.parametrize("draw_ci", [False, True])
def test_mean_curve_preserves_missing_bins_and_one_legend_entry(draw_ci):
    values = np.array([
        [np.nan, 1, 3, np.nan, 5, 7, np.nan],
        [np.nan, 2, 4, np.nan, 6, 8, np.nan],
        [np.nan, 3, 5, np.nan, 7, 9, np.nan],
    ])
    original = values.copy()
    _, axis = plt.subplots()
    plot_mean_curve(axis, np.arange(7), values, "Participants", "blue", draw_ci=draw_ci)
    line = next(line for line in axis.lines if line.get_marker() == "o")
    np.testing.assert_allclose(
        line.get_ydata(orig=False), [np.nan, 2, 4, np.nan, 6, 8, np.nan], equal_nan=True
    )
    # A missing bin breaks the line rather than joining positions 2 and 4.
    segments = list(line.get_path().iter_segments())
    assert [code for _, code in segments] == [1, 2, 1, 2]  # MOVETO, LINETO, ...
    np.testing.assert_allclose([point for point, _ in segments], [[1, 2], [2, 4], [4, 6], [5, 8]])
    assert axis.get_legend_handles_labels()[1] == ["Participants"]
    if draw_ci:
        bars = [bar for bar in axis.collections[0].get_segments() if len(bar)]
        np.testing.assert_array_equal([bar[0, 0] for bar in bars], [1, 2, 4, 5])
    else:
        assert not axis.collections
    np.testing.assert_array_equal(values, original)


def sample_data():
    data = make_dataset(jnp.ones((4, 2), dtype=int))
    data["condition"] = jnp.array([1, 1, 2, 2])
    return data


def test_panels_use_explicit_mode_even_with_one_dataset():
    data=sample_data();masks=[data['condition'] == value for value in (1, 2)]
    shared=( [data], [masks], ['Observed'], ['First','Second'] )
    conditions=comparison_panels(*shared,mode='conditions',selection_label='Comparison')
    datasets=comparison_panels(*shared,mode='datasets')
    assert len(conditions)==1 and conditions[0]['labels']==['First','Second']
    assert conditions[0]['title']=='Comparison' and conditions[0]['datasets']==[data,data]
    assert len(datasets)==2 and [p['title'] for p in datasets]==['First','Second']
    assert datasets[0]['labels']==['Observed']
    second = comparison_panels(
        [data, data], [masks, [None, masks[0]]], ["Observed", "Model"], ["First", "Second"],
        mode="datasets",
    )
    np.testing.assert_array_equal(second[0]["trial_masks"][1], np.ones(4, dtype=bool))
    np.testing.assert_array_equal(second[1]["trial_masks"][1], masks[0])
    assert second[1]['datasets'][1] is data
    np.testing.assert_array_equal(second[1]['trial_masks'][0],masks[1])
    second[0]['labels'].append('extra')
    assert second[1]['labels']==['Observed','Model']


@pytest.mark.parametrize(
    "mode, masks, dataset_labels, condition_labels",
    [
        ("datasets", [[None, None]], ["Observed"], ["A"]),
        ("conditions", [[None]], ["Observed"], ["A", "B"]),
        ("datasets", [[None]], [], ["A"]),
        ("datasets", [], ["Observed"], ["A"]),
    ],
)
def test_comparison_panels_reject_misaligned_inputs(mode, masks, dataset_labels, condition_labels):
    with pytest.raises(ValueError):
        comparison_panels([sample_data()], masks, dataset_labels, condition_labels, mode=mode)


def test_condition_overlay_requires_one_dataset():
    data = sample_data()
    with pytest.raises(ValueError, match="one response dataset"):
        comparison_panels([data, data], [[None], [None]], ["A", "B"], ["All"], mode="conditions")


PLOT_CALLS = []


def plot_probe(datasets,masks,**options):
    PLOT_CALLS.append((datasets,[np.asarray(m).copy() for m in masks],dict(options)))
    options['axis'].plot([0,1],[0,1])


def test_figure_iteration_masks_options_suffixes_and_laziness():
    PLOT_CALLS.clear();data=sample_data()
    configs=[
        {'target':plot_probe,'title':'First','figsize':(3,2),'ylim':(-1,2),
         'labels':['Override'],'kwargs':{'confidence_level':.8,'color_cycle':['red']},'color_cycle':['blue']},
        {'target':plot_probe,'trial_query':"data['condition'] == 2"},
        {'target':plot_probe,'figure_suffix':'named'},
        {'target':__name__+'.plot_probe','figure_suffix':'imported'},
    ]
    original=deepcopy(configs)
    iterator=iter_analysis_figures([data],configs,['Default'],base_masks=[jnp.array([True,False,True,False])],suffix='_single',n_resamples=399,bootstrap_seed=17)
    assert not PLOT_CALLS
    first,suffix=next(iterator)
    assert suffix=='probe_single_1' and len(PLOT_CALLS)==1
    np.testing.assert_array_equal(PLOT_CALLS[0][1][0],[True,False,True,False])
    assert PLOT_CALLS[0][2]['confidence_level']==.8
    assert PLOT_CALLS[0][2]['n_resamples']==399
    assert PLOT_CALLS[0][2]['bootstrap_seed']==17
    assert PLOT_CALLS[0][2]['color_cycle']==['blue']
    assert PLOT_CALLS[0][2]['labels']==['Override']
    assert first.axes[0].get_title()=='First' and first.axes[0].get_ylim()==(-1,2)
    np.testing.assert_allclose(first.get_size_inches(),[3,2])
    rest=list(iterator)
    assert [s for _,s in rest]==['probe_single_2','named_single','imported_single']
    np.testing.assert_array_equal(PLOT_CALLS[1][1][0],[False,False,True,False])
    assert configs==original
    assert PLOT_CALLS[0][0][0] is data



def test_figure_iteration_draws_on_its_own_axes():
    _, unrelated_axis = plt.subplots()
    config = {"target": spc.plot_spc, "kwargs": {"axis": unrelated_axis}}
    data = make_dataset(jnp.array([[1, 2]]))
    figure, _ = next(iter_analysis_figures([data], [config], ["Observed"]))

    line = next(line for line in figure.axes[0].lines if line.get_marker() == "o")
    np.testing.assert_allclose(line.get_ydata(), [1, 1])
    assert not unrelated_axis.lines
    assert config["kwargs"]["axis"] is unrelated_axis


def test_save_explicit_figure_even_when_another_is_current(tmp_path, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: pytest.fail("Saving must not display figures"))
    first, axis = plt.subplots()
    axis.plot([1, 2], [3, 4])
    other, _ = plt.subplots()
    saved = []
    real_save = first.savefig
    monkeypatch.setattr(first, "savefig", lambda *args, **kwargs: (
        saved.append(args[0]), real_save(*args, **kwargs)
    ))
    monkeypatch.setattr(other, "savefig", lambda *args, **kwargs: pytest.fail("Saved the wrong figure"))
    save_figure(tmp_path, "chosen", figure=first, dpi=40, formats=["png", "svg"])
    assert len(saved) == 2 and (tmp_path / "chosen.png").exists()
    assert plt.gcf() is other
    assert plt.fignum_exists(first.number)
    plt.close(first)
    plt.close(other)


def test_figure_formats(tmp_path):
    figure, axis = plt.subplots()
    axis.plot([1, 2], [0.2, 0.5])
    save_figure(tmp_path, "plot", formats=["png", "svg", "png"], figure=figure)
    assert (tmp_path / "plot.png").stat().st_size > 0
    assert "<svg" in (tmp_path / "plot.svg").read_text()
    assert len(list(tmp_path.iterdir())) == 2
    plt.close(figure)


@pytest.mark.parametrize(
    "matrix, expected_annotations",
    [
        (
            jnp.array([-1.0, 0.0, 0.124, 0.25, 1.0]),
            {(0.5, 0.5): "-1.00", (3.5, 0.5): "0.25", (4.5, 0.5): "1.00"},
        ),
        (
            np.array([[-1.0, -0.125, 0.124], [0.125, 1.0, np.nan]]),
            {(0.5, 0.5): "-1.00", (1.5, 0.5): "-0.12", (0.5, 1.5): "0.12", (1.5, 1.5): "1.00"},
        ),
    ],
)
def test_matrix_heatmap_preserves_values_annotations_and_supplied_axes(
    matrix, expected_annotations
):
    original = np.array(matrix, copy=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    try:
        returned_figure, returned_axis = matrix_heatmap(
            matrix, axis=axis, print_threshold=0.125,
            title="Memory", xlabel="Features", ylabel="Updates",
            label_font_size=18, annot_font_size=12,
        )
        assert returned_figure is figure and returned_axis is axis
        mesh = axis.collections[0]
        np.testing.assert_allclose(
            np.ma.filled(mesh.get_array(), np.nan), np.atleast_2d(original), equal_nan=True
        )
        np.testing.assert_allclose(np.asarray(matrix), original, equal_nan=True)
        assert axis.get_ylim() == (np.atleast_2d(original).shape[0], 0)
        assert mesh.cmap.name == "magma"
        assert mesh.colorbar.ax in figure.axes and len(figure.axes) == 2
        assert {text.get_position(): text.get_text() for text in axis.texts} == expected_annotations
        assert all(text.get_fontsize() == 12 for text in axis.texts)
        colours = {text.get_text(): text.get_color() for text in axis.texts}
        assert colours["-1.00"] == "white" and colours["1.00"] == "black"
        assert (axis.get_title(), axis.get_xlabel(), axis.get_ylabel()) == (
            "Memory", "Features", "Updates"
        )
        assert axis.xaxis.label.get_fontsize() == 18
    finally:
        plt.close(figure)


def test_heatmaps_share_a_color_scale_without_colorbars_and_keep_cell_labels():
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    scale = Normalize(vmin=0, vmax=2)
    try:
        for axis, values in zip(axes, ([0.5, 1.0], [0.5, 2.0])):
            matrix_heatmap(
                np.array(values), axis=axis, norm=scale, colorbar=False,
                row_labels=["Trace 7"], column_labels=["Input 0", "Output 0"],
            )
            assert [tick.get_text() for tick in axis.get_xticklabels()] == ["Input 0", "Output 0"]
            assert [tick.get_text() for tick in axis.get_yticklabels()] == ["Trace 7"]
            assert axis.collections[0].get_clim() == (0, 2)
            assert axis.collections[0].colorbar is None
        assert len(figure.axes) == 2
        meshes = [axis.collections[0] for axis in axes]
        np.testing.assert_allclose(meshes[0].to_rgba(0.5), meshes[1].to_rgba(0.5))
    finally:
        plt.close(figure)
