"""Check points drawn, selected trials, and labels, rather than Axes types."""

from copy import deepcopy

import matplotlib.pyplot as plt
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
from jaxcmr.dataset import make_dataset
from jaxcmr.plotting import comparison_panels, iter_analysis_figures, save_figure


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
    second=comparison_panels([data,data],[masks,masks],['Observed','Model'],['First','Second'],mode='datasets')
    assert second[1]['datasets'][1] is data
    np.testing.assert_array_equal(second[1]['trial_masks'][0],masks[1])
    second[0]['labels'].append('extra')
    assert second[1]['labels']==['Observed','Model']


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
    iterator=iter_analysis_figures([data],configs,['Default'],base_masks=[jnp.array([True,False,True,False])],suffix='_single')
    assert not PLOT_CALLS
    first,suffix=next(iterator)
    assert suffix=='probe_single_1' and len(PLOT_CALLS)==1
    np.testing.assert_array_equal(PLOT_CALLS[0][1][0],[True,False,True,False])
    assert PLOT_CALLS[0][2]['confidence_level']==.8
    assert PLOT_CALLS[0][2]['color_cycle']==['blue']
    assert PLOT_CALLS[0][2]['labels']==['Override']
    assert first.axes[0].get_title()=='First' and first.axes[0].get_ylim()==(-1,2)
    np.testing.assert_allclose(first.get_size_inches(),[3,2])
    rest=list(iterator)
    assert [s for _,s in rest]==['probe_single_2','named_single','imported_single']
    np.testing.assert_array_equal(PLOT_CALLS[1][1][0],[False,False,True,False])
    assert configs==original
    assert PLOT_CALLS[0][0][0] is data


def test_save_explicit_figure_even_when_another_is_current(tmp_path,monkeypatch):
    monkeypatch.setattr(plt,'show',lambda:None)
    first,ax=plt.subplots();ax.plot([1,2],[3,4])
    other,_=plt.subplots()
    saved=[];real=first.savefig
    monkeypatch.setattr(first,'savefig',lambda *a,**kw:(saved.append(a[0]),real(*a,**kw)))
    monkeypatch.setattr(other,'savefig',lambda *a,**kw:pytest.fail('Saved the wrong figure'))
    save_figure(str(tmp_path),'chosen',figure=first,dpi=40,formats=['png','svg'])
    assert len(saved)==2 and (tmp_path/'chosen.png').exists()
    save_figure(str(tmp_path),'',figure=first)
    assert len(saved)==2


def test_figure_formats(tmp_path, monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    fig, ax = plt.subplots()
    ax.plot([1, 2], [0.2, 0.5])
    save_figure(str(tmp_path), 'plot', formats=['png', 'svg'])
    assert (tmp_path/'plot.png').stat().st_size > 0
    assert '<svg' in (tmp_path/'plot.svg').read_text()
    plt.close(fig)
