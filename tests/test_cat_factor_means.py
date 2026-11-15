"""Factor-mean eligibility, recall coding, aggregation, and LPP compatibility."""

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest
from matplotlib.container import ErrorbarContainer

matplotlib.use("Agg", force=True)
from jaxcmr.analyses.cat_factor_by_recall import (
    cat_factor_by_recall,
    plot_cat_factor_by_recall,
    subject_cat_factor_by_recall,
)
from jaxcmr.analyses.cat_factor_spc import (
    cat_factor_spc,
    category_factor_values,
    factor_sums_and_counts,
    plot_cat_factor_spc,
    subject_cat_factor_spc,
)
from jaxcmr.analyses.cat_lpp_by_recall import (
    expand_categories_by_recall,
    plot_cat_lpp_by_recall,
)
from jaxcmr.analyses.cat_lpp_spc import (
    cat_lpp_spc,
    category_lpp_values,
    plot_cat_lpp_spc,
)
from jaxcmr.typing import RecallDataset


def _dataset() -> RecallDataset:
    return {
        "subject": jnp.array([[9], [2]]),
        "listLength": jnp.array([[3], [3]]),
        "pres_itemnos": jnp.array([[3, 1, 2], [3, 1, 2]]),
        "recalls": jnp.array([[1, 1, 0], [2, 0, 0]]),
        "category": jnp.ones((2, 3), dtype=int),
        "score": jnp.array([[4.0, 6.0, 8.0], [10.0, 20.0, 30.0]]),
        "EarlyLPP": jnp.full((2, 3), 2.0),
        "LateLPP": jnp.full((2, 3), 7.0),
    }


@pytest.mark.parametrize("compile", [False, True])
def test_means_exclude_nonfinite_and_masked_values_from_both_sums_and_counts(compile):
    fn = jax.jit(factor_sums_and_counts) if compile else factor_sums_and_counts
    values = jnp.array([[4.0, jnp.inf, 9.0, jnp.nan], [jnp.nan, 6.0, 50.0, -jnp.inf]])
    include = jnp.array([[True, True, True, True], [True, True, False, True]])
    sums, counts = fn(values, include)
    np.testing.assert_array_equal(sums, [4, 6, 9, 0])
    np.testing.assert_array_equal(counts, [1, 1, 1, 0])
    means = category_factor_values(values, jnp.ones(values.shape, dtype=int), 1, valid=include)
    np.testing.assert_allclose(means, [4, 6, 9, np.nan], equal_nan=True)
    np.testing.assert_allclose(
        category_lpp_values(values[:, :2], jnp.ones((2, 2), dtype=int), 1), [4, 6]
    )
    integer_sums, _ = fn(jnp.array([[1], [2]]), jnp.ones((2, 1), dtype=bool))
    assert jnp.issubdtype(integer_sums.dtype, jnp.floating)


def test_dataset_eligibility_and_dynamic_categories_work_under_jit():
    data = _dataset()
    data["listLength"] = jnp.array([[3], [1]])
    data["pres_itemnos"] = jnp.array([[3, 1, 0], [3, 1, 2]])
    data["category"] = jnp.array([[0, 0, 0], [1, 1, 1]])
    data["quality"] = jnp.array([[1.0, jnp.nan, 1.0], [1.0, 1.0, 1.0]])
    fn = jax.jit(cat_factor_spc, static_argnames=("category_field", "factor_field", "valid_field"))
    np.testing.assert_allclose(
        fn(data, "category", 0, "score", valid_field="quality"), [4, np.nan, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(
        fn(data, "category", 1, "score", valid_field="quality"),
        [10, np.nan, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(fn(data, "category", 0, "score"), [4, 6, np.nan], equal_nan=True)


def test_recall_split_uses_positions_in_reordered_presentations_under_jit():
    data = _dataset()
    fn = jax.jit(cat_factor_by_recall, static_argnames=("category_field", "factor_field"))
    expected = [[4, 20, np.nan], [10, 6, 19]]
    np.testing.assert_allclose(fn(data, "category", 1, "score"), expected, equal_nan=True)
    np.testing.assert_array_equal(
        expand_categories_by_recall(data, "category"), [[2, 1, 1], [1, 2, 1]]
    )
    assert np.all(np.isnan(fn(data, "category", 2, "score")))


def test_invalid_recalls_and_padding_do_not_mark_valid_item_occurrences():
    data = _dataset()
    data["pres_itemnos"] = jnp.array([[3, 1, 0], [3, 1, 3]])
    data["listLength"] = jnp.array([[3], [2]])
    data["recalls"] = jnp.array([[3, 0, -1, 9], [3, 0, -1, 9]])
    result = cat_factor_by_recall(data, "category", 1, "score", expand_repetitions=True)
    np.testing.assert_allclose(result, [[np.nan] * 3, [7, 13, np.nan]], equal_nan=True)
    np.testing.assert_array_equal(expand_categories_by_recall(data, "category"), [[1, 1, 0]] * 2)


@pytest.mark.parametrize("recalled_position", [1, 3])
def test_repetition_expansion_is_explicit_and_independent_of_measurement_quality(recalled_position):
    data: RecallDataset = {
        "subject": jnp.array([[1]]),
        "listLength": jnp.array([[4]]),
        "pres_itemnos": jnp.array([[7, 2, 7, 4]]),
        "recalls": jnp.array([[recalled_position, recalled_position, 0]]),
        "category": jnp.ones((1, 4), dtype=int),
        "score": jnp.array([[2.0, 4.0, 6.0, 8.0]]),
        "quality": jnp.ones((1, 4)),
    }
    fn = jax.jit(
        cat_factor_by_recall,
        static_argnames=("category_field", "factor_field", "valid_field", "expand_repetitions"),
    )
    direct = fn(data, "category", 1, "score")
    expected_direct = np.full((2, 4), np.nan)
    expected_direct[1] = [2, 4, 6, 8]
    expected_direct[0, recalled_position - 1] = recalled_position * 2
    expected_direct[1, recalled_position - 1] = np.nan
    np.testing.assert_allclose(direct, expected_direct, equal_nan=True)
    np.testing.assert_allclose(
        fn(data, "category", 1, "score", expand_repetitions=True),
        [[2, np.nan, 6, np.nan], [np.nan, 4, np.nan, 8]],
        equal_nan=True,
    )
    data["score"] = data["score"].at[0, recalled_position - 1].set(jnp.nan)
    data["quality"] = data["quality"].at[0, recalled_position - 1].set(0)
    expanded = fn(data, "category", 1, "score", valid_field="quality", expand_repetitions=True)
    other_position = 2 if recalled_position == 1 else 0
    assert expanded[0, other_position] == (other_position + 1) * 2
    assert jnp.isnan(expanded[1, other_position])


def _unequal_subjects() -> RecallDataset:
    return {
        "subject": jnp.array([[9], [9], [9], [2]]),
        "listLength": jnp.ones((4, 1), dtype=int),
        "pres_itemnos": jnp.ones((4, 1), dtype=int),
        "recalls": jnp.ones((4, 1), dtype=int),
        "category": jnp.ones((4, 1), dtype=int),
        "LateLPP": jnp.array([[0.0], [0.0], [0.0], [10.0]]),
    }


def test_subject_summaries_preserve_subject_and_category_order_and_weighting():
    data = _unequal_subjects()
    mask = jnp.ones(4, dtype=bool)
    np.testing.assert_allclose(cat_factor_spc(data, "category", 1, "LateLPP"), [2.5])
    subjects = subject_cat_factor_spc(data, mask, "category", [2, 1], "LateLPP")
    assert subjects.shape == (2, 2, 1)
    np.testing.assert_array_equal(subjects[:, 1, 0], [10, 0])
    assert np.all(np.isnan(subjects[:, 0, :]))
    recalled = subject_cat_factor_by_recall(data, mask, "category", [2, 1], "LateLPP")
    assert recalled.shape == (2, 2, 2, 1)
    np.testing.assert_allclose(recalled[:, :, 0, :], subjects, equal_nan=True)
    assert np.all(np.isnan(recalled[:, :, 1, :]))
    selected = subject_cat_factor_spc(
        data, jnp.array([True, True, True, False]), "category", [1], "LateLPP"
    )
    np.testing.assert_array_equal(selected, [[[0]]])


def test_hiding_intervals_keeps_real_plot_at_equal_subject_mean_and_does_not_mutate_data():
    data = _unequal_subjects()
    before = {key: np.array(value) for key, value in data.items()}
    for exclude_ci in (False, True):
        axis = plot_cat_lpp_by_recall(
            data, jnp.ones(4, dtype=bool), "category", [2], exclude_ci=exclude_ci
        )
        # Participants contribute means 0 and 10, despite having three and one trials.
        curves = [line for line in axis.lines if line.get_marker() == "o"]
        assert len(curves) == 1
        np.testing.assert_array_equal(
            curves[0].get_ydata(), [5], err_msg="Plot means must weight participants equally"
        )
        assert axis.get_ylabel() == "LateLPP (uV)"
        intervals = [item for item in axis.containers if isinstance(item, ErrorbarContainer)]
        assert len(intervals) == (0 if exclude_ci else 1)
    for key, value in before.items():
        np.testing.assert_array_equal(data[key], value)


def _plotted_curves(axis):
    """Match visible line segments to their legend colour, ignoring empty groups."""
    result = {}
    for handle, label in zip(*axis.get_legend_handles_labels(), strict=True):
        lines = [line for line in axis.lines
                 if line.get_color() == handle.get_color() and len(line.get_xdata())]
        if lines:
            result[label] = (np.concatenate([line.get_xdata() for line in lines]),
                             np.concatenate([line.get_ydata() for line in lines]))
    return result


def test_plots_label_each_dimension_and_allow_different_position_widths():
    first, second = (_dataset(), _unequal_subjects())
    axis = plot_cat_factor_by_recall(
        [first, second],
        [jnp.ones(2, dtype=bool), jnp.ones(4, dtype=bool)],
        "category", [1], "LateLPP",
        labels=["observed", "model"], category_labels=["A"],
        recall_labels=["yes", "no"], draw_ci=False,
    )
    expected = {
        "observed / A / yes": ([1, 2], [7, 7]),
        "observed / A / no": ([1, 2, 3], [7, 7, 7]),
        "model / A / yes": ([1], [5]),
    }
    curves = _plotted_curves(axis)
    assert list(curves) == list(expected)
    for label, (positions, values) in expected.items():
        np.testing.assert_array_equal(curves[label][0], positions)
        np.testing.assert_array_equal(curves[label][1], values)
    assert not any(isinstance(item, ErrorbarContainer) for item in axis.containers)


def test_legacy_defaults_labels_and_expanded_code_sequences_are_preserved():
    data = _dataset()
    np.testing.assert_array_equal(cat_lpp_spc(data, "category", 1), [2, 2, 2])
    axis = plot_cat_lpp_spc(
        data, jnp.ones(2, dtype=bool), "category", [1], labels=["LPP"], draw_ci=False
    )
    np.testing.assert_array_equal(axis.lines[0].get_xdata(), [1, 2, 3])
    np.testing.assert_array_equal(axis.lines[0].get_ydata(), [7, 7, 7])
    assert axis.get_ylabel() == "LateLPP (uV)"
    assert axis.get_legend_handles_labels()[1] == ["LPP"]
    data["category"] = jnp.array([[2, 1, 1], [2, 1, 1]])
    labels = ["category 2 recalled", "category 1 missed"]
    axis = plot_cat_lpp_by_recall(
        data, jnp.ones(2, dtype=bool), "category", (4, 1), lpp_field="score",
        labels=labels, exclude_ci=True,
    )
    assert axis.get_legend_handles_labels()[1] == labels
    curves = _plotted_curves(axis)
    assert list(curves) == labels
    for label, positions, values in zip(labels, [[1], [2, 3]], [[4], [6, 19]], strict=True):
        np.testing.assert_array_equal(curves[label][0], positions)
        np.testing.assert_array_equal(curves[label][1], values)
    axis = plot_cat_lpp_by_recall(
        data, jnp.ones(2, dtype=bool), "category", 2, lpp_field="score", exclude_ci=True
    )
    # Scalar category 2 expands to recalled then missed curves.
    curves = _plotted_curves(axis)
    assert len(curves) == 2
    for (positions, values), value in zip(curves.values(), [4, 10], strict=True):
        np.testing.assert_array_equal(positions, [1])
        np.testing.assert_array_equal(values, [value])


def test_empty_subject_selections_are_shaped_and_skipped_in_plots():
    data = _dataset()
    empty = jnp.zeros(2, dtype=bool)
    assert subject_cat_factor_spc(data, empty, "category", [1], "score").shape == (0, 1, 3)
    assert subject_cat_factor_by_recall(data, empty, "category", [1], "score").shape == (0, 1, 2, 3)
    assert not plot_cat_factor_spc(data, empty, "category", [1], "score").lines
    assert not plot_cat_factor_by_recall(data, empty, "category", [1], "score").lines
    assert not plot_cat_lpp_by_recall(data, empty, "category", 1).lines
