import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
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
def test_set_false_at_index_behaves_for_zero_and_nonzero(vec, idx, expected):
    """Flip an index to ``False`` only when ``idx`` is greater than zero.

    Given:
        A boolean vector and an index where ``0`` is a sentinel.
    When:
        ``set_false_at_index`` is called.
    Then:
        The element at ``idx`` becomes ``False`` only if ``idx`` > 0.
    """
    updated, _ = crp.set_false_at_index(vec, idx)
    assert updated.tolist() == expected


def test_set_false_at_index_returns_tuple_and_none_flag():
    """Return a tuple of the updated vector and ``None`` flag.

    Given:
        A boolean vector and an index.
    When:
        ``set_false_at_index`` is invoked.
    Then:
        The function returns ``(updated_vec, None)``.
    """
    vec = jnp.array([True, True], dtype=bool)
    updated, flag = crp.set_false_at_index(vec, 2)
    assert isinstance(updated, jnp.ndarray)
    assert flag is None
    assert updated.tolist() == [True, False]


# -----------------------------------------------------------------------------
# Tabulation: sentinel and validity tests
# -----------------------------------------------------------------------------


def test_tabulate_zero_is_noop_for_actual_and_avail_lags():
    """``tabulate(0)`` leaves ``actual_lags`` and ``avail_lags`` unchanged.

    Given:
        A ``Tabulation`` initialized with a presentation sequence.
    When:
        ``tabulate(0)`` is called.
    Then:
        ``actual_lags`` and ``avail_lags`` remain unchanged.
    """
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

    tab_after = tab.tabulate(0)

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


def test_should_tabulate_validity_and_availability():
    """Respect validity and availability of study positions.

    Given:
        A ``Tabulation`` with a study list.
    When:
        ``should_tabulate`` is queried for various positions.
    Then:
        Only valid, previously unrecalled positions return ``True``.
    """
    presentation = jnp.array([1, 2, 3], dtype=jnp.int32)
    tab = crp.Tabulation(presentation=presentation, first_recall=1, size=1)

    assert tab.should_tabulate(2).item() is True
    assert tab.should_tabulate(1).item() is False
    assert tab.should_tabulate(0).item() is False

    tab2 = tab.tabulate(2)
    assert tab2.should_tabulate(2).item() is False


# -----------------------------------------------------------------------------
# Tabulation: transition counting and invariants
# -----------------------------------------------------------------------------


def test_repeated_recall_is_ignored_counts_one_transition():
    """Ignore repeated recalls and count only valid transitions.

    Given:
        A ``Tabulation`` with an initial recall.
    When:
        The same item is recalled again.
    Then:
        Only the first valid transition is counted.
    """
    presentation = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    tab = crp.Tabulation(presentation=presentation, first_recall=3, size=1)
    tab = tab.tabulate(3)  # ignored
    tab = tab.tabulate(4)  # valid
    total = jnp.sum(tab.actual_lags).item()
    assert total == 1, f"Expected exactly 1 transition, got {total}"


def test_repeat_does_not_change_previous_positions_transitions():
    """Ensure repeats do not rewrite earlier transitions.

    Given:
        A sequence of recalls with a repeated item.
    When:
        ``tabulate`` processes each recall.
    Then:
        Earlier transition counts remain unchanged.
    """
    presentation = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    recalls = [3, 1, 3, 4]
    tab = crp.Tabulation(presentation=presentation, first_recall=recalls[0], size=1)
    for r in recalls[1:]:
        tab = tab.tabulate(int(r))

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


def test_zero_padded_study_positions_not_indexed():
    """Ignore zero-padded study positions as invalid.

    Given:
        A ``Tabulation`` with a padded study list.
    When:
        A recall targets a padded position.
    Then:
        The transition is ignored as invalid.
    """
    presentation = jnp.array([10, 20, 30, 40], dtype=jnp.int32)
    tab = crp.Tabulation(presentation=presentation, first_recall=1, size=3)
    tab = tab.tabulate(2)
    total = jnp.sum(tab.actual_lags).item()
    assert total == 1, f"Expected exactly one valid transition (1→2), got {total}"


def test_lags_from_previous_single_lag_location():
    """Mark exactly one lag bin from the previous position.

    Given:
        A ``Tabulation`` on a short study list.
    When:
        ``lags_from_previous`` is called with a position.
    Then:
        Exactly one lag bin is set to ``True``.
    """
    presentation = jnp.array([1, 2, 3], dtype=jnp.int32)
    tab = crp.Tabulation(presentation=presentation, first_recall=1, size=1)
    mask = tab.lags_from_previous(2)
    assert mask.dtype == bool
    assert mask.sum().item() == 1
    expected_idx = _lag_index(mask.size, +1)
    assert bool(mask[expected_idx])


def test_available_lags_from_zero_and_nonzero():
    """Compute available lags for zero and nonzero positions.

    Given:
        A ``Tabulation`` object.
    When:
        ``available_lags_from`` is called for zero and nonzero positions.
    Then:
        The zero position yields no lags and the other yields at least one.
    """
    presentation = jnp.array([1, 2, 3], dtype=jnp.int32)
    tab = crp.Tabulation(presentation=presentation, first_recall=1, size=1)
    base = tab.available_lags_from(0)
    nonzero = tab.available_lags_from(2)
    assert base.sum().item() == 0
    assert nonzero.sum().item() > 0


def test_tabulate_returns_new_object_and_updates_counts():
    """Return a new object and update counts when tabulating.

    Given:
        An initial ``Tabulation``.
    When:
        ``tabulate`` is called with a valid position.
    Then:
        A new object is returned and transition counts increase.
    """
    presentation = jnp.array([1, 2, 3], dtype=jnp.int32)
    tab = crp.Tabulation(presentation=presentation, first_recall=1, size=1)

    before_actual = jnp.sum(tab.actual_lags).item()
    before_avail = jnp.sum(tab.avail_lags).item()

    tab2 = tab.tabulate(2)

    assert tab2 is not tab
    assert jnp.sum(tab2.actual_lags).item() == before_actual + 1
    assert jnp.sum(tab2.avail_lags).item() >= before_avail


def test_tabulate_trial_counts_transitions_ignoring_zeros():
    """Ignore zero recalls when counting trial transitions.

    Given:
        A trial sequence containing zeros.
    When:
        ``tabulate_trial`` processes the trial.
    Then:
        Zero entries are ignored in the transition count.
    """
    trial = jnp.array([1, 0, 2, 0, 3], dtype=jnp.int32)
    presentation = jnp.array([1, 2, 3], dtype=jnp.int32)
    actual, _ = crp.tabulate_trial(trial, presentation, size=1)
    assert jnp.sum(actual).item() == 2


def test_simple_and_full_tabulation_agree_on_total_actual_transitions():
    """Simple and full tabulators should agree on total transitions.

    Given:
        Matching trial and presentation sequences.
    When:
        Both simple and full tabulation methods are used.
    Then:
        The total number of transitions is identical.
    """
    trial = jnp.array([1, 2, 3], dtype=jnp.int32)
    presentation = jnp.array([1, 2, 3], dtype=jnp.int32)

    simple = crp.simple_tabulate_trial(trial, list_length=3)
    simple_count = jnp.sum(simple.actual_transitions).item()

    full_actual, _ = crp.tabulate_trial(trial, presentation, size=1)
    full_count = jnp.sum(full_actual).item()

    assert simple_count == 2
    assert full_count == 2
    assert simple_count == full_count


def test_crp_outcome_identical_with_and_without_repeats():
    """Lag-CRP outcomes are unaffected by repeated recalls.

    Given:
        A recall trial containing a repeated item and the same trial without the
        repeat.
    When:
        ``crp`` tabulates both trials.
    Then:
        The resulting Lag-CRP arrays are identical.
    """

    trials_with_repeat = jnp.array([[2, 3, 2, 4]], dtype=jnp.int32)
    trials_without_repeat = jnp.array([[2, 3, 4, 0]], dtype=jnp.int32)
    presentations = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)

    with_repeat = crp.crp(trials_with_repeat, presentations, list_length=4, size=1)
    without_repeat = crp.crp(
        trials_without_repeat, presentations, list_length=4, size=1
    )

    assert jnp.allclose(with_repeat, without_repeat, equal_nan=True)


def test_crp_handles_single_item_without_crashing():
    """Handle one-item trials without raising an error.

    Given:
        A single-item trial and presentation.
    When:
        ``crp`` is computed.
    Then:
        The result is a valid JAX array with one row.
    """
    trials = jnp.array([[1]], dtype=jnp.int32)
    presentations = jnp.array([[1]], dtype=jnp.int32)
    out = crp.crp(trials, presentations, list_length=1, size=1)
    assert isinstance(out, jnp.ndarray)
    assert out.shape[0] == 1


def test_crp_jit_with_static_argnames():
    """Run ``crp`` jitted with ``static_argnames`` and compare results.

    Given:
        Trials and presentations for two lists.
    When:
        ``crp`` is JIT-compiled with ``static_argnames``.
    Then:
        The compiled and uncompiled results are equal.
    """
    trials = jnp.array([[1, 2, 3], [1, 3, 2]], dtype=jnp.int32)
    presentations = jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32)
    list_length = 3
    size = 1

    expected = crp.crp(trials, presentations, list_length, size)
    jitted_crp = jax.jit(crp.crp, static_argnames=("size", "list_length"))
    result = jitted_crp(trials, presentations, list_length, size)

    assert result.shape == expected.shape
    assert jnp.allclose(result, expected, equal_nan=True)


def test_crp_jit_with_different_size_compiles_and_runs():
    """Compile and run ``crp`` with a different ``size`` value.

    Given:
        Trials and presentations for one list.
    When:
        ``crp`` is JIT-compiled with a larger ``size``.
    Then:
        The compiled output matches the uncompiled result.
    """
    trials = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    presentations = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    list_length = 3
    size = 2

    expected = crp.crp(trials, presentations, list_length, size)
    jitted_crp = jax.jit(crp.crp, static_argnames=("size", "list_length"))
    result = jitted_crp(trials, presentations, list_length, size)

    assert result.shape == expected.shape
    assert jnp.allclose(result, expected, equal_nan=True)


def test_plot_crp_returns_axes():
    """Return a Matplotlib ``Axes`` instance from ``plot_crp``.

    Given:
        A minimal ``RecallDataset``.
    When:
        ``plot_crp`` is called.
    Then:
        A Matplotlib ``Axes`` with a ``Figure`` is returned.
    """
    dataset: RecallDataset = {
        "subject": jnp.array([[1], [1]], dtype=jnp.int32),
        "listLength": jnp.array([[3], [3]], dtype=jnp.int32),
        "pres_itemnos": jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32),
        "recalls": jnp.array([[1, 2, 0], [2, 3, 0]], dtype=jnp.int32),
        "pres_itemids": jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32),
    }
    trial_mask = jnp.array([True, True], dtype=bool)
    axis = crp.plot_crp(dataset, trial_mask, max_lag=1)
    assert isinstance(axis, Axes)
    fig = axis.figure
    assert isinstance(fig, Figure)
    plt.close(fig)
