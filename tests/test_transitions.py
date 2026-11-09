"""Hand-counted transition opportunities, including exclusions and padding."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxcmr.analyses import crp, conditional_crp, source_target_crp, target_enrichment_crp
from jaxcmr.helpers import make_dataset


@pytest.mark.parametrize(
    "recalls,presentations,expected",
    [
        ([1, 3, 2, 0], [1, 2, 3, 4], [np.nan, np.nan, 1, np.nan, 0, 1, 0]),
        ([1, 2, 1, 3], [1, 2, 3], [np.nan, np.nan, np.nan, 1, 0]),
        ([0, 0, 0, 0], [1, 2, 3, 0], [np.nan] * 7),
    ],
)
def test_lag_probabilities_ignore_repeats_and_padding(recalls, presentations, expected):
    data = make_dataset(jnp.array([recalls]), jnp.array([presentations]))
    np.testing.assert_allclose(crp.crp(data, size=3), expected, equal_nan=True)
    data["_should_tabulate"] = jnp.ones_like(data["recalls"], dtype=bool)
    np.testing.assert_allclose(conditional_crp.crp(data, size=3), expected, equal_nan=True)


@pytest.mark.parametrize(
    "source,target,expected",
    [
        (
            [0, -1],
            None,
            [
                [np.nan, np.nan, 1, np.nan, 0, 0, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, 0, 1],
            ],
        ),
        (
            None,
            [-1, 0],
            [
                [np.nan, np.nan, 1, np.nan, 0, 0, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, 0, 1],
            ],
        ),
        (
            [0],
            [-1, 0],
            [
                [np.nan, np.nan, 1, np.nan, 0, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan],
            ],
        ),
    ],
)
def test_conditional_lag_denominators_and_jit(source, target, expected):
    data = make_dataset(jnp.array([[2, 1, 4, 0]]))
    data["condition"] = jnp.array([[-1, 0, -1, 0]])
    kwargs = {"size": 3}
    if source is not None:
        kwargs.update(source_field="condition", source_values=jnp.array(source))
    if target is not None:
        kwargs.update(target_field="condition", target_values=jnp.array(target))
    run = jax.jit(conditional_crp.crp, static_argnames=("source_field", "target_field", "size"))
    np.testing.assert_allclose(run(data, **kwargs), expected, equal_nan=True)
    data["_should_tabulate"] = jnp.zeros_like(data["recalls"], dtype=bool)
    assert np.isnan(run(data, **kwargs)).all()


@pytest.mark.parametrize(
    "recalls,conditions,include,expected",
    [
        ([1, 3, 2, 4, 0], [-1, 0, 1, 1], [True] * 5, [[np.nan, 0.5], [np.nan, 0]]),
        ([1, 2, 3, 0], [-1, 0, 1], [True, False, True, True], [[np.nan, np.nan], [np.nan, 1]]),
    ],
)
def test_source_target_counts_items_and_updates_excluded_events(
    recalls, conditions, include, expected
):
    data = make_dataset(jnp.array([recalls]), jnp.arange(1, len(conditions) + 1)[None])
    data.update(valence=jnp.array([conditions]), _should_tabulate=jnp.array([include]))
    sources = [-1, 1] if len(conditions) == 4 else [-1, 0]
    result = source_target_crp.source_target_crp(
        data,
        source_field="valence",
        source_values=sources,
        target_field="valence",
        target_values=[-1, 1],
        size=3,
    )
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_target_enrichment_subtracts_available_fraction():
    data = make_dataset(jnp.array([[2, 3, 0], [2, 1, 0]]))
    data["condition"] = jnp.array([[0, 0, -1], [0, 0, 0]])
    run = jax.jit(
        target_enrichment_crp.target_enrichment_crp,
        static_argnames=("target_field", "target_value", "source_field", "size"),
    )
    kwargs = dict(
        target_field="condition",
        target_value=-1,
        target_values=jnp.array([-1, 0]),
        source_field="condition",
        source_values=jnp.array([0]),
        size=3,
    )
    np.testing.assert_allclose(
        run(data, **kwargs), [[np.nan, 0, np.nan, 0.5, np.nan]], equal_nan=True
    )
    data["_should_tabulate"] = jnp.zeros_like(data["recalls"], dtype=bool)
    assert np.isnan(run(data, **kwargs)).all()


@pytest.mark.parametrize(
    "include,expected",
    [
        ([[True, True, True]] * 2, [0.75, 0.5]),
        ([[True, False, True]] * 2, [1.0, np.nan]),
        ([[False, False, False]] * 2, [np.nan, np.nan]),
    ],
)
def test_conditional_distance_masks_update_the_recall_state(include, expected):
    from jaxcmr.analyses.conditional_distcrp import dist_crp

    data = make_dataset(jnp.array([[1, 2, 3], [1, 3, 2]]))
    data.update(pres_itemids=data["pres_itemnos"], _should_tabulate=jnp.array(include))
    distances = jnp.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    np.testing.assert_allclose(
        dist_crp(data, distances, jnp.array([1.5])), expected, equal_nan=True
    )


@pytest.mark.parametrize(
    "distances,percentiles,expected",
    [
        (
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 1.0, 2.0],
                [2.0, 1.0, 0.0, 1.0],
                [3.0, 2.0, 1.0, 0.0],
            ],
            [33.0, 66.0],
            [1.0, 2.0],
        ),
        (
            [
                [0.0, 1.0, np.nan, 3.0],
                [1.0, 0.0, 2.0, np.nan],
                [np.nan, 2.0, 0.0, 4.0],
                [3.0, np.nan, 4.0, 0.0],
            ],
            [50.0],
            [2.5],
        ),
    ],
)
def test_distance_bin_edges_use_finite_upper_triangle(distances, percentiles, expected):
    from jaxcmr.analyses.distcrp import compute_distance_bins_percentiles

    edges, centers = compute_distance_bins_percentiles(jnp.array(distances), jnp.array(percentiles))
    np.testing.assert_allclose(edges, expected)
    assert np.all(np.diff(centers) > 0)
