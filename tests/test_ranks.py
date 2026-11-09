"""Ranks use remaining choices, midpoint ties, and distinct repeat occurrences."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from jaxcmr.analyses import lagrank, distrank, replagrank, serialreplagrank
from jaxcmr.helpers import make_dataset


@pytest.mark.parametrize(
    "measure,choice,pool,answer",
    [
        (lagrank.percentile_rank, 1, [1, 2, 3, 4], 1.0),
        (lagrank.percentile_rank, 4, [1, 2, 3, 4], 0.0),
        (lagrank.percentile_rank, 2, [1, 2, 2, 3], 0.5),
        (lagrank.percentile_rank, 5, [5], np.nan),
        (lagrank.percentile_rank, 1, [-1, -1, 1, 3], 1.0),
        (distrank.percentile_rank, 2, [1.0, 2.0, 2.0, 3.0], 0.5),
        (distrank.percentile_rank, 5, [5.0], np.nan),
        (distrank.percentile_rank, 1, [np.inf, np.inf, 1.0, 3.0], 1.0),
    ],
)
def test_percentile_rank_known_choices(measure, choice, pool, answer):
    np.testing.assert_allclose(measure(jnp.int32(choice), jnp.array(pool)), answer, equal_nan=True)


@pytest.mark.parametrize(
    "measure,plot,present,recalls,kwargs,answer",
    [
        (lagrank.lagrank, lagrank.plot_lagrank, [1, 2, 3, 4], [4, 3, 2, 1], {"size": 1}, [1.0]),
        (lagrank.lagrank, lagrank.plot_lagrank, [1, 2, 3, 4], [1, 2, 2, 3], {"size": 1}, [1.0]),
        (
            replagrank.replagrank,
            replagrank.plot_rep_lagrank,
            [1, 2, 3, 4, 5, 1, 7, 8],
            [2, 3, 0, 0],
            {"min_lag": 4, "size": 2},
            [[np.nan, np.nan]],
        ),
        (
            serialreplagrank.serialreplagrank,
            serialreplagrank.plot_serial_rep_lagrank,
            [1, 2, 3, 4, 5, 1, 2, 8],
            [3, 1, 2, 5],
            {"min_lag": 2, "size": 2},
            [[np.nan, np.nan]],
        ),
        (
            lagrank.lagrank,
            lagrank.plot_lagrank,
            [1, 2, 3, 4, 5],
            [3, 5, 0, 0, 0],
            {"size": 1},
            [0.0],
        ),
        (
            replagrank.replagrank,
            replagrank.plot_rep_lagrank,
            [1, 2, 3, 4, 5, 1, 7, 8],
            [1, 2, 0, 0],
            {"min_lag": 4, "size": 2},
            [[1.0, 0.0]],
        ),
        (
            serialreplagrank.serialreplagrank,
            serialreplagrank.plot_serial_rep_lagrank,
            [1, 2, 3, 4, 5, 1, 7, 8],
            [1, 2, 0, 0],
            {"min_lag": 4, "size": 2},
            [[1.0, 0.0]],
        ),
    ],
)
def test_rank_values_survive_compilation_and_plotting(
    measure, plot, present, recalls, kwargs, answer
):
    data = make_dataset(jnp.array([recalls]), jnp.array([present]))
    np.testing.assert_allclose(jax.jit(lambda d: measure(d, **kwargs))(data), answer, atol=1e-6)
    labels = ["First", "Second"] if len(np.shape(answer)) == 2 else ["selected"]
    axis = plot(data, jnp.array([True]), labels=labels, **kwargs)
    segments = [line.get_ydata() for line in axis.lines if line.get_marker() == "o"]
    points = np.concatenate(segments) if segments else np.array([])
    expected = np.ravel(answer)
    np.testing.assert_allclose(points.astype(float), expected, equal_nan=True)
    assert axis.get_ylabel()
    plt.close(axis.figure)


@pytest.mark.parametrize("source", ["matrix", "features"])
@pytest.mark.parametrize("second,answer", [(2, 1.0), (3, 0.0)])
def test_distance_rank_counts_nearest_and_furthest_remaining_items(source, second, answer):
    distance = jnp.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    data = make_dataset(jnp.array([[1, second, 0]]))
    data["pres_itemids"] = data["pres_itemnos"]
    np.testing.assert_allclose(distrank.distrank(data, distance), [answer])
    inputs = (
        dict(distance_matrix=distance)
        if source == "matrix"
        else dict(features=jnp.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]))
    )
    axis = distrank.plot_distrank(data, jnp.array([True]), labels=["selected"], **inputs)
    assert axis.get_ylabel() == "Organization Score"
    assert [label.get_text() for label in axis.get_xticklabels()] == ["selected"]
    points = np.concatenate([line.get_ydata() for line in axis.lines if line.get_marker() == "o"])
    np.testing.assert_allclose(points.astype(float), [answer])
    plt.close(axis.figure)


@pytest.mark.parametrize(
    "cue,include,expected", [(1, True, 1.0), (0, True, np.nan), (1, False, np.nan)]
)
def test_cue_rank_requires_an_included_valid_cue(cue, include, expected):
    from jaxcmr.analyses.cue_centered_lagrank import cue_centered_lagrank, plot_cue_centered_lagrank

    data = make_dataset(jnp.array([[1, 2, 0]]), jnp.array([[1, 2, 3, 4]]))
    data.update(
        cue_clips=jnp.array([[0, cue, 0]]), _should_tabulate=jnp.array([[True, include, False]])
    )
    np.testing.assert_allclose(
        jax.jit(lambda d: cue_centered_lagrank(d, size=1))(data), [expected], equal_nan=True
    )
    if include and cue:
        axis = plot_cue_centered_lagrank(
            data, jnp.array([True]), data["_should_tabulate"], size=1, labels=["cued"]
        )
        points = np.concatenate(
            [line.get_ydata() for line in axis.lines if line.get_marker() == "o"]
        )
        np.testing.assert_allclose(points.astype(float), [expected])
        assert axis.get_ylabel() == "Organization Score"
        assert [label.get_text() for label in axis.get_xticklabels()] == ["cued"]
