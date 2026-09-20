"""Binning, eligibility, aggregation, and LPP compatibility contracts."""

import importlib

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from jaxcmr.analyses.cat_recall_by_factor import (
    binned_recall_counts,
    cat_recall_by_factor,
    category_factor_recall_histogram,
    compute_factor_bin_edges,
    plot_cat_recall_by_factor,
    subject_cat_recall_by_factor,
)
from jaxcmr.analyses.cat_recall_by_lpp import (
    cat_recall_by_lpp,
    category_lpp_recall_histogram,
    category_recall_counts,
    plot_cat_recall_by_lpp,
)
from jaxcmr.typing import RecallDataset


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _dataset() -> RecallDataset:
    return {
        "subject": jnp.array([[9], [2]]),
        "listLength": jnp.array([[3], [1]]),
        "pres_itemnos": jnp.array([[1, 2, 3], [1, 0, 0]]),
        "recalls": jnp.array([[0, 0, 0], [1, 0, 0]]),
        "category": jnp.ones((2, 3), dtype=int),
        "score": jnp.ones((2, 3)),
        "LateLPP": jnp.ones((2, 3)),
    }


@pytest.mark.parametrize("compile", [False, True])
def test_counts_exclude_missing_and_overflow_and_include_final_edge(compile):
    fn = jax.jit(binned_recall_counts) if compile else binned_recall_counts
    values = jnp.array([-1., 0., 2., 5., 10., 11., jnp.nan, jnp.inf, -jnp.inf])
    recalled = jnp.array([True, True, False, True, True, True, True, True, True])
    numerator, denominator = fn(
        recalled, values, jnp.ones(values.shape, dtype=bool), jnp.array([0., 5., 10.])
    )
    # 0 and 2 enter the first bin; 5 and the final edge 10 enter the second.
    np.testing.assert_array_equal(numerator, [1, 2])
    np.testing.assert_array_equal(denominator, [2, 2])


def test_explicit_inclusion_mask_removes_both_recall_and_exposure():
    numerator, denominator = binned_recall_counts(
        jnp.array([True, False, True]), jnp.array([1, 2, 3]),
        jnp.array([False, True, True]), jnp.array([0, 4]),
    )
    np.testing.assert_array_equal(numerator, [1])
    np.testing.assert_array_equal(denominator, [2])


def test_duplicate_and_invalid_recall_codes_do_not_inflate_probabilities():
    recalls = jnp.array([[1, 1, 0, -1, 9]])
    values = jnp.array([[2., 8.]])
    categories = jnp.ones((1, 2), dtype=int)
    edges = jnp.array([0., 5., 10.])
    rates = jax.jit(category_factor_recall_histogram)(recalls, values, categories, 1, edges)
    np.testing.assert_array_equal(rates, [1, 0])
    np.testing.assert_array_equal(
        category_lpp_recall_histogram(recalls, values, categories, 1, edges, 2), [1, 0]
    )
    np.testing.assert_array_equal(category_recall_counts(recalls[0], categories[0], 1, 2), [1, 0])


def test_lpp_missing_and_outside_values_leave_empty_bins_undefined():
    edges = jnp.array([0., 5., 10.])
    categories = jnp.ones((1, 2), dtype=int)
    missing = category_lpp_recall_histogram(
        jnp.array([[2, 0]]), jnp.array([[2., jnp.nan]]), categories, 1, edges, 2
    )
    np.testing.assert_allclose(missing, [0, np.nan], equal_nan=True)
    outside = category_lpp_recall_histogram(
        jnp.array([[1, 2]]), jnp.array([[-10., 100.]]), categories, 1, edges, 2
    )
    assert np.all(np.isnan(outside))


def test_reference_edges_use_only_finite_included_values_under_jit():
    fn = jax.jit(compute_factor_bin_edges, static_argnames=("bin_count",))
    values = jnp.array([1., 3., -100., jnp.nan, jnp.inf])
    valid = jnp.array([True, True, False, True, True])
    np.testing.assert_array_equal(fn(values, valid, bin_count=2), [1, 2, 3])
    assert np.all(np.isnan(fn(values, jnp.zeros_like(valid), bin_count=2)))
    assert np.all(np.isnan(fn(jnp.array([]), jnp.array([], dtype=bool), bin_count=2)))


@pytest.mark.parametrize("value", [0., -3., 1e10])
def test_constant_factor_gets_distinct_edges_and_one_occupied_bin(value):
    values = jnp.full(4, value)
    valid = jnp.ones(4, dtype=bool)
    edges = compute_factor_bin_edges(values, valid, 4)
    assert np.all(np.diff(edges) > 0)
    assert float(edges[0]) < value < float(edges[-1])
    recalled, eligible = binned_recall_counts(valid, values, valid, edges)
    assert int(jnp.sum(eligible)) == 4
    assert int(jnp.count_nonzero(eligible)) == 1
    np.testing.assert_array_equal(recalled, eligible)


def test_dataset_excludes_padding_short_lists_and_quality_failures():
    data = _dataset()
    # Positive presentation IDs past listLength must still be excluded.
    data["pres_itemnos"] = jnp.array([[1, 2, 3], [1, 2, 0]])
    data["score"] = jnp.array([[1., 1., 1000.], [1., -1000., -999.]])
    data["quality"] = jnp.array([[1., 1., 0.], [1., 1., 1.]])
    fn = jax.jit(cat_recall_by_factor, static_argnames=(
        "category_field", "factor_field", "valid_field", "bin_count",
    ))
    # Two eligible unrecalled positions and one eligible recalled position.
    np.testing.assert_allclose(
        fn(data, "category", 1, "score", jnp.array([0., 2.]), valid_field="quality"),
        [1 / 3], rtol=1e-6,
    )
    auto = fn(data, "category", 1, "score", bin_count=2, valid_field="quality")
    np.testing.assert_allclose(auto, [np.nan, 1 / 3], equal_nan=True, rtol=1e-6)


def test_automatic_edges_are_independent_of_the_selected_category():
    data: RecallDataset = {
        "subject": jnp.array([[1]]),
        "listLength": jnp.array([[3]]),
        "pres_itemnos": jnp.array([[1, 2, 3]]),
        "recalls": jnp.array([[1, 2, 0]]),
        "category": jnp.array([[1, 2, 1]]),
        "score": jnp.array([[0., 4., 8.]]),
    }
    fn = jax.jit(cat_recall_by_factor, static_argnames=(
        "category_field", "factor_field", "bin_count",
    ))
    np.testing.assert_allclose(fn(data, "category", 1, "score", bin_count=2), [1, 0])
    np.testing.assert_allclose(
        fn(data, "category", 2, "score", bin_count=2), [np.nan, 1], equal_nan=True
    )


def test_subject_rates_and_pooled_rates_retain_distinct_weighting():
    data = _dataset()
    edges = jnp.array([0., 2.])
    pooled = cat_recall_by_factor(data, "category", 1, "score", edges)
    subjects = subject_cat_recall_by_factor(
        data, jnp.array([True, True]), "category", [1, 8], "score", edges
    )
    np.testing.assert_allclose(pooled, [0.25])
    assert subjects.shape == (2, 2, 1)
    # Subject 2 (one recalled position) precedes subject 9 (three unrecalled).
    np.testing.assert_array_equal(subjects[:, 0, 0], [1, 0])
    assert np.all(np.isnan(subjects[:, 1, :]))
    np.testing.assert_allclose(jnp.mean(subjects[:, 0, :], axis=0), [0.5])
    selected = subject_cat_recall_by_factor(
        data, jnp.array([True, False]), "category", [1], "score", edges
    )
    np.testing.assert_array_equal(selected, [[[0]]])
    empty = subject_cat_recall_by_factor(
        data, jnp.array([False, False]), "category", [1], "score", edges
    )
    assert empty.shape == (0, 1, 1)


def test_lpp_dataset_wrapper_matches_factor_analysis_with_and_without_explicit_edges():
    data = _dataset()
    fn = jax.jit(cat_recall_by_lpp, static_argnames=("category_field", "bin_count"))
    np.testing.assert_allclose(fn(data, "category", 1, bin_edges=jnp.array([0., 2.])), [0.25])
    np.testing.assert_allclose(
        fn(data, "category", 1, bin_count=2),
        cat_recall_by_factor(data, "category", 1, "LateLPP", bin_count=2),
        equal_nan=True,
    )


def test_plot_shares_bins_across_datasets_categories_and_subjects(monkeypatch):
    module = importlib.import_module("jaxcmr.analyses.cat_recall_by_factor")
    seen = []

    def capture(axis, x, values, label, color, **kwargs):
        seen.append((np.asarray(x), np.asarray(values), label))
        return axis

    monkeypatch.setattr(module, "plot_data", capture)
    first, second = _dataset(), _dataset()
    first["score"] = jnp.array([[1., 2., 3.], [1000., 0., 0.]])
    second["score"] = jnp.array([[7., 8., 9.], [-1000., 0., 0.]])
    first["category"] = second["category"] = jnp.array([[1, 2, 1], [1, 0, 0]])
    axis = plot_cat_recall_by_factor(
        [first, second], [jnp.array([True, False])] * 2,
        "category", [1, 2], "score", bin_count=2,
        labels=["observed", "model"], category_labels=["A", "B"],
    )
    assert axis.get_xlabel() == "score"
    assert len(seen) == 4
    for x, values, _ in seen:
        # Only selected trials define [1, 5, 9]; every curve uses those bins.
        np.testing.assert_array_equal(x, [3, 7])
        assert values.shape == (1, 2)
    assert [entry[2] for entry in seen] == [
        "observed / A", "observed / B", "model / A", "model / B",
    ]
    np.testing.assert_allclose(seen[0][1], [[0, np.nan]], equal_nan=True)
    np.testing.assert_allclose(seen[2][1], [[np.nan, 0]], equal_nan=True)


def test_real_plot_uses_lpp_units_and_retains_legacy_labels():
    axis = plot_cat_recall_by_lpp(
        _dataset(), jnp.array([True, False]), "category", [1],
        bin_edges=jnp.array([0., 2.]), labels=["observed"],
    )
    assert axis.get_xlabel() == "LateLPP (uV)"
    assert axis.get_ylabel() == "Recall Rate"
    np.testing.assert_array_equal(axis.lines[0].get_ydata(), [0])
    assert axis.lines[0].get_label() == "observed"


@pytest.mark.parametrize("edges", [jnp.array([0., 0., 1.]), jnp.array([2., 1.]),
                                  jnp.array([0., jnp.nan])])
def test_invalid_dynamic_edges_yield_undefined_rates_and_plotting_rejects_them(edges):
    result = jax.jit(cat_recall_by_factor, static_argnames=("category_field", "factor_field"))(
        _dataset(), "category", 1, "score", edges
    )
    assert np.all(np.isnan(result))
    with pytest.raises(ValueError, match="strictly increasing"):
        plot_cat_recall_by_factor(
            _dataset(), jnp.array([True, True]), "category", [1], "score", edges
        )


def test_no_reference_values_are_undefined_numerically_and_clear_in_plotting():
    data = _dataset()
    data["score"] = jnp.full((2, 3), jnp.nan)
    assert np.all(np.isnan(cat_recall_by_factor(data, "category", 1, "score")))
    with pytest.raises(ValueError, match="finite eligible"):
        plot_cat_recall_by_factor(data, jnp.array([True, True]), "category", [1], "score")


def test_configuration_shape_and_field_errors_are_clear():
    data = _dataset()
    with pytest.raises(ValueError, match="requires dataset"):
        cat_recall_by_factor(data, "category", 1, "missing")
    data["score"] = jnp.ones((2, 2))
    with pytest.raises(ValueError, match="pres_itemnos shape"):
        cat_recall_by_factor(data, "category", 1, "score")
    with pytest.raises(ValueError, match="positive integer"):
        compute_factor_bin_edges(jnp.array([1.]), jnp.array([True]), bin_count=0)
    with pytest.raises(ValueError, match="matching shapes"):
        binned_recall_counts(jnp.array([True]), jnp.array([1., 2.]),
                             jnp.array([True]), jnp.array([0., 2.]))
    with pytest.raises(ValueError, match="at least two edges"):
        binned_recall_counts(jnp.array([True]), jnp.array([1.]),
                             jnp.array([True]), jnp.array([0.]))
    with pytest.raises(ValueError, match="Boolean vector"):
        subject_cat_recall_by_factor(
            _dataset(), jnp.array([1, 0]), "category", [1], "score", jnp.array([0., 2.])
        )


def test_plot_rejects_inconsistent_labels_and_masks():
    data = _dataset()
    with pytest.raises(ValueError, match="one trial mask"):
        plot_cat_recall_by_factor([data, data], [jnp.array([True, True])],
                                  "category", [1], "score")
    with pytest.raises(ValueError, match="labels must match"):
        plot_cat_recall_by_factor(data, jnp.array([True, True]), "category", [1],
                                  "score", labels=["a", "b"])
