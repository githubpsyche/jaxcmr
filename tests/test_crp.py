import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pytest
from jaxcmr.analyses import crp
from jaxcmr.typing import RecallDataset


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _lag_index(lags_len: int, lag: int) -> int:
    """Map a signed lag value to an index in a lag vector.

    Args:
        lags_len: Total length of the lag vector (odd; center is zero-lag).
        lag: Signed lag where negative is backward and positive is forward.

    Returns:
        Index in the lag vector corresponding to ``lag``.
    """
    center = lags_len // 2
    return center + lag


# -----------------------------------------------------------------------------
# set_false_at_index tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vec,idx,expected",
    [
        (jnp.array([True, True, True, True], dtype=bool), 3, [True, True, False, True]),
        (jnp.array([True, True, True, True], dtype=bool), 0, [True, True, True, True]),
    ],
)
def test_flips_element_to_false_when_idx_positive(vec, idx, expected):
    """Behavior: Set element to ``False`` only for positive indices.

    Given:
      - A boolean vector.
      - An index where ``0`` acts as a sentinel.
    When:
      - ``set_false_at_index`` is invoked.
    Then:
      - The element becomes ``False`` only if ``idx`` > 0.
    Why this matters:
      - Preserves the sentinel semantics for downstream logic.
    """
    # Arrange / Given
    # Parameters ``vec`` and ``idx`` supply the setup

    # Act / When
    updated, _ = crp.set_false_at_index(vec, idx)

    # Assert / Then
    assert jnp.array_equal(updated, jnp.array(expected, dtype=bool)).item()


def test_returns_tuple_and_none_when_updating_index():
    """Behavior: Provide ``(updated_vec, None)`` on success.

    Given:
      - A boolean vector and a valid index.
    When:
      - ``set_false_at_index`` is called.
    Then:
      - A tuple containing the updated vector and ``None`` flag is returned.
    Why this matters:
      - Confirms the function's API contract.
    """
    # Arrange / Given
    vec = jnp.array([True, True], dtype=bool)

    # Act / When
    updated, flag = crp.set_false_at_index(vec, 2)

    # Assert / Then
    assert isinstance(updated, jnp.ndarray)
    assert flag is None
    assert jnp.array_equal(updated, jnp.array([True, False], dtype=bool)).item()


def test_set_false_at_index_out_of_range_is_noop():
    """Behavior: Out-of-range indices leave vector unchanged.

    Given:
      - A boolean vector.
    When:
      - ``set_false_at_index`` is called with negative or too-large index.
    Then:
      - The vector is unchanged and ``None`` is returned.
    Why this matters:
      - Prevents unintended updates for invalid positions.
    """
    # Arrange / Given
    vec = jnp.array([True, True], dtype=bool)

    # Act / When
    updated_large, flag_large = crp.set_false_at_index(vec, 99)
    updated_neg, flag_neg = crp.set_false_at_index(vec, -5)

    # Assert / Then
    assert updated_large.tolist() == vec.tolist()
    assert updated_neg.tolist() == vec.tolist()
    assert flag_large is None
    assert flag_neg is None


# -----------------------------------------------------------------------------
# Tabulation: sentinel and validity tests
# -----------------------------------------------------------------------------


def test_leaves_lags_unchanged_when_tabulating_zero():
    """Behavior: ``tabulate(0)`` does not modify lag counts.

    Given:
      - A ``Tabulation`` initialized with a study list.
    When:
      - ``tabulate`` is called with ``0``.
    Then:
      - ``actual_lags`` and ``avail_lags`` remain unchanged.
    Why this matters:
      - Validates the sentinel ``0`` as a no-op.
    """
    # Arrange / Given
    presentation = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    tab = crp.Tabulation(presentation=presentation, first_recall=1, size=2)
    before_actual = list(map(int, tab.actual_lags))
    before_avail = list(map(int, tab.avail_lags))

    # Sentinel check
    assert hasattr(tab, "should_tabulate"), (
        "Missing 'should_tabulate' method on Tabulation"
    )
    assert tab.should_tabulate(0).item() is False, (
        "Expected tab.should_tabulate(0) to be False for 0 sentinel"
    )

    # Act / When
    tab_after = tab.tabulate(0)

    # Assert / Then
    after_actual = list(map(int, tab_after.actual_lags))
    after_avail = list(map(int, tab_after.avail_lags))
    assert before_actual == after_actual, (
        f"Calling tabulate(0) should not change actual_lags.\n"
        f"before={before_actual}\nafter={after_actual}"
    )
    assert before_avail == after_avail, (
        f"Calling tabulate(0) should not change avail_lags.\n"
        f"before={before_avail}\nafter={after_avail}"
    )


def test_flags_position_available_when_valid_and_unrecalled():
    """Behavior: Report availability only for valid, unrecalled positions.

    Given:
      - A ``Tabulation`` built from a study list.
    When:
      - Querying ``should_tabulate`` for several positions.
    Then:
      - Only valid and unrecalled positions return ``True``.
    Why this matters:
      - Prevents counting transitions for invalid or repeated recalls.
    """
    # Arrange / Given
    presentation = jnp.array([1, 2, 3], dtype=jnp.int32)
    tab = crp.Tabulation(presentation=presentation, first_recall=1, size=1)

    # Act / When
    tab2 = tab.tabulate(2)

    # Assert / Then
    assert tab.should_tabulate(2).item() is True
    assert tab.should_tabulate(1).item() is False
    assert tab.should_tabulate(0).item() is False
    assert tab2.should_tabulate(2).item() is False


# -----------------------------------------------------------------------------
# Tabulation: transition counting and invariants
# -----------------------------------------------------------------------------


def test_counts_only_first_transition_when_recall_repeated():
    """Behavior: Ignore repeated recalls in transition counts.

    Given:
      - A ``Tabulation`` with an initial recall.
    When:
      - The same item is recalled again.
    Then:
      - Only the first valid transition increments the count.
    Why this matters:
      - Ensures repeated recalls do not inflate transition metrics.
    """
    # Arrange / Given
    presentation = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    tab = crp.Tabulation(presentation=presentation, first_recall=3, size=1)

    # Act / When
    tab = tab.tabulate(3)  # ignored
    tab = tab.tabulate(4)  # valid

    # Assert / Then
    total = jnp.sum(tab.actual_lags).item()
    assert total == 1, f"Expected exactly 1 transition, got {total}"


def test_preserves_prior_transitions_when_recall_repeated():
    """Behavior: Repeats do not overwrite earlier transition bins.

    Given:
      - A sequence of recalls containing a repeated item.
    When:
      - ``tabulate`` processes each recall.
    Then:
      - Earlier transition counts remain unchanged.
    Why this matters:
      - Guarantees idempotence when recalls repeat.
    """
    # Arrange / Given
    presentation = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    recalls = [3, 1, 3, 4]
    tab = crp.Tabulation(presentation=presentation, first_recall=recalls[0], size=1)

    # Act / When
    for r in recalls[1:]:
        tab = tab.tabulate(int(r))

    # Assert / Then
    final_lags = list(map(int, tab.actual_lags))
    assert sum(final_lags) == 2, (
        f"Expected two total transitions, got {sum(final_lags)}"
    )
    neg2_idx = _lag_index(len(final_lags), -2)
    plus3_idx = _lag_index(len(final_lags), +3)
    plus1_idx = _lag_index(len(final_lags), +1)
    assert final_lags[neg2_idx] == 1, f"Expected 1 at lag -2 (3→1), got {final_lags}"
    assert final_lags[plus3_idx] == 1, f"Expected 1 at lag +3 (1→4), got {final_lags}"
    assert final_lags[plus1_idx] == 0, (
        f"Did not expect transition at +1 (3→4). Got {final_lags}"
    )


def test_ignores_padded_positions_when_recalled():
    """Behavior: Zero-padded study positions are treated as invalid.

    Given:
      - A ``Tabulation`` with padded study list entries.
    When:
      - A recall targets a padded position.
    Then:
      - The transition is ignored.
    Why this matters:
      - Avoids indexing artifacts from padding values.
    """
    # Arrange / Given
    presentation = jnp.array([10, 20, 30, 40], dtype=jnp.int32)
    tab = crp.Tabulation(presentation=presentation, first_recall=1, size=3)

    # Act / When
    tab = tab.tabulate(2)

    # Assert / Then
    total = jnp.sum(tab.actual_lags).item()
    assert total == 1, f"Expected exactly one valid transition (1→2), got {total}"


def test_marks_single_lag_bin_when_from_previous():
    """Behavior: ``lags_from_previous`` marks exactly one lag bin.

    Given:
      - A ``Tabulation`` on a short study list.
    When:
      - ``lags_from_previous`` is invoked for a position.
    Then:
      - Exactly one lag bin is ``True``.
    Why this matters:
      - Validates correct lag indexing.
    """
    # Arrange / Given
    presentation = jnp.array([1, 2, 3], dtype=jnp.int32)
    tab = crp.Tabulation(presentation=presentation, first_recall=1, size=1)

    # Act / When
    mask = tab.lags_from_previous(2)

    # Assert / Then
    assert mask.dtype == bool
    assert mask.sum().item() == 1
    expected_idx = _lag_index(mask.size, +1)
    assert bool(mask[expected_idx])


def test_returns_empty_lags_when_position_zero():
    """Behavior: Zero position yields no available lags.

    Given:
      - A ``Tabulation`` object.
    When:
      - ``available_lags_from`` is called for zero and nonzero positions.
    Then:
      - Position ``0`` yields no lags, others yield some.
    Why this matters:
      - Ensures sentinel positions don't contribute to availability.
    """
    # Arrange / Given
    presentation = jnp.array([1, 2, 3], dtype=jnp.int32)
    tab = crp.Tabulation(presentation=presentation, first_recall=1, size=1)

    # Act / When
    base = tab.available_lags_from(0)
    nonzero = tab.available_lags_from(2)

    # Assert / Then
    assert base.sum().item() == 0
    assert nonzero.sum().item() > 0


def test_returns_new_tabulation_when_position_valid():
    """Behavior: ``tabulate`` returns new object and updates counts.

    Given:
      - An initial ``Tabulation``.
    When:
      - ``tabulate`` is called with a valid position.
    Then:
      - A new object is returned and counts increase.
    Why this matters:
      - Confirms immutability and proper counting.
    """
    # Arrange / Given
    presentation = jnp.array([1, 2, 3], dtype=jnp.int32)
    tab = crp.Tabulation(presentation=presentation, first_recall=1, size=1)
    before_actual = jnp.sum(tab.actual_lags).item()
    before_avail = jnp.sum(tab.avail_lags).item()

    # Act / When
    tab2 = tab.tabulate(2)

    # Assert / Then
    assert tab2 is not tab
    assert jnp.sum(tab2.actual_lags).item() == before_actual + 1
    assert jnp.sum(tab2.avail_lags).item() >= before_avail


def test_counts_transitions_when_zeros_in_trial():
    """Behavior: ``tabulate_trial`` ignores zero recalls.

    Given:
      - A trial sequence containing zeros.
    When:
      - ``tabulate_trial`` processes the trial.
    Then:
      - Zero entries do not contribute to transition count.
    Why this matters:
      - Ensures padding zeros do not affect metrics.
    """
    # Arrange / Given
    trial = jnp.array([1, 0, 2, 0, 3], dtype=jnp.int32)
    presentation = jnp.array([1, 2, 3], dtype=jnp.int32)

    # Act / When
    actual, _ = crp.tabulate_trial(trial, presentation, size=1)

    # Assert / Then
    assert jnp.sum(actual).item() == 2


def test_matches_total_transitions_when_using_simple_or_full():
    """Behavior: Simple and full tabulators agree on totals.

    Given:
      - Matching trial and presentation sequences.
    When:
      - Both tabulation methods are applied.
    Then:
      - The total transitions are identical.
    Why this matters:
      - Confirms equivalence of tabulation approaches.
    """
    # Arrange / Given
    trial = jnp.array([1, 2, 3], dtype=jnp.int32)
    presentation = jnp.array([1, 2, 3], dtype=jnp.int32)

    # Act / When
    simple = crp.simple_tabulate_trial(trial, list_length=3)
    simple_count = jnp.sum(simple.actual_transitions).item()
    full_actual, _ = crp.tabulate_trial(trial, presentation, size=1)
    full_count = jnp.sum(full_actual).item()

    # Assert / Then
    assert simple_count == 2
    assert full_count == 2
    assert simple_count == full_count


def test_produces_same_crp_when_repeats_removed():
    """Behavior: Repeated recalls do not alter Lag-CRP outcome.

    Given:
      - A trial with a repeated item and an equivalent trial without it.
    When:
      - ``crp`` processes both trials.
    Then:
      - The resulting Lag-CRP arrays match.
    Why this matters:
      - Ensures repeat handling does not affect analysis.
    """
    # Arrange / Given
    trials_with_repeat = jnp.array([[2, 3, 2, 4]], dtype=jnp.int32)
    trials_without_repeat = jnp.array([[2, 3, 4, 0]], dtype=jnp.int32)
    presentations = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)

    # Act / When
    with_repeat = crp.crp(trials_with_repeat, presentations, list_length=4, size=1)
    without_repeat = crp.crp(
        trials_without_repeat, presentations, list_length=4, size=1
    )

    # Assert / Then
    assert jnp.allclose(with_repeat, without_repeat, equal_nan=True)


def test_returns_array_when_single_trial_item():
    """Behavior: ``crp`` handles single-item trials.

    Given:
      - A trial and presentation with one item.
    When:
      - ``crp`` is computed.
    Then:
      - A JAX array with one row is produced.
    Why this matters:
      - Confirms edge-case handling for minimal input.
    """
    # Arrange / Given
    trials = jnp.array([[1]], dtype=jnp.int32)
    presentations = jnp.array([[1]], dtype=jnp.int32)

    # Act / When
    out = crp.crp(trials, presentations, list_length=1, size=1)

    # Assert / Then
    assert isinstance(out, jnp.ndarray)
    assert out.shape[0] == 1


def test_matches_uncompiled_result_when_jitted_with_static_argnames():
    """Behavior: JIT compilation with static args preserves results.

    Given:
      - Trials and presentations for two lists.
    When:
      - ``crp`` is JIT-compiled with ``static_argnames``.
    Then:
      - Compiled and uncompiled outputs are equal.
    Why this matters:
      - Validates compatibility with JAX JIT.
    """
    # Arrange / Given
    trials = jnp.array([[1, 2, 3], [1, 3, 2]], dtype=jnp.int32)
    presentations = jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32)
    list_length = 3
    size = 1
    expected = crp.crp(trials, presentations, list_length, size)

    # Act / When
    jitted_crp = jax.jit(crp.crp, static_argnames=("size", "list_length"))
    result = jitted_crp(trials, presentations, list_length, size)

    # Assert / Then
    assert result.shape == expected.shape
    assert jnp.allclose(result, expected, equal_nan=True)


def test_runs_with_larger_size_when_jitted():
    """Behavior: JIT-compiled ``crp`` handles larger ``size``.

    Given:
      - Trials and presentations for one list.
    When:
      - ``crp`` is JIT-compiled with a larger ``size``.
    Then:
      - Output matches the uncompiled result.
    Why this matters:
      - Demonstrates JIT flexibility with size parameter.
    """
    # Arrange / Given
    trials = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    presentations = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    list_length = 3
    size = 2
    expected = crp.crp(trials, presentations, list_length, size)

    # Act / When
    jitted_crp = jax.jit(crp.crp, static_argnames=("size", "list_length"))
    result = jitted_crp(trials, presentations, list_length, size)

    # Assert / Then
    assert result.shape == expected.shape
    assert jnp.allclose(result, expected, equal_nan=True)


def test_returns_axes_object_when_plotting_crp():
    """Behavior: ``plot_crp`` provides a Matplotlib ``Axes``.

    Given:
      - A minimal ``RecallDataset``.
    When:
      - ``plot_crp`` is called.
    Then:
      - A Matplotlib ``Axes`` with a ``Figure`` is returned.
    Why this matters:
      - Ensures visualization utility returns standard objects.
    """
    # Arrange / Given
    dataset: RecallDataset = {
        "subject": jnp.array([[1], [1]], dtype=jnp.int32),
        "listLength": jnp.array([[3], [3]], dtype=jnp.int32),
        "pres_itemnos": jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32),
        "recalls": jnp.array([[1, 2, 0], [2, 3, 0]], dtype=jnp.int32),
        "pres_itemids": jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32),
    }
    trial_mask = jnp.array([True, True], dtype=bool)

    # Act / When
    axis = crp.plot_crp(dataset, trial_mask, max_lag=1)

    # Assert / Then
    assert isinstance(axis, Axes)
    fig = axis.figure
    assert isinstance(fig, Figure)
    plt.close(fig)
