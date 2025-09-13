import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pytest
from jaxcmr.analyses import crp


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _lag_index(lags_len: int, lag: int) -> int:
    """Helper: map a signed lag value to an index in a lag vector.

    Arrange:
    - `lags_len`: total length of the lag vector (odd; center is zero-lag).
    - `lag`: signed lag (negative = backward, positive = forward).

    Act:
    - Compute `center = lags_len // 2` and return `center + lag`.

    Assert:
    - Returns an integer index in `[0, lags_len - 1]` if `lag` is within range.

    Why this matters:
    - Keeps lag→index mapping explicit so tests remain readable and robust.
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
    """Behavior: correctly flips an index to False only when `idx > 0`.

    Arrange:
    - A boolean vector and an index `idx` (0 = sentinel / no-op).

    Act:
    - Call `set_false_at_index(vec, idx)`.

    Assert:
    - Returned first element is a JAX array with values matching `expected`.

    Why this matters:
    - Guards the sentinel convention and avoids accidental masking when `idx == 0`.
    """
    updated, _ = crp.set_false_at_index(vec, idx)
    assert updated.tolist() == expected


def test_set_false_at_index_returns_tuple_and_none_flag():
    """Behavior: returns a tuple `(updated_vec, None)`.

    Arrange:
    - A simple boolean vector and an in-range index.

    Act:
    - Call `set_false_at_index(vec, 2)`.

    Assert:
    - First element is a JAX array, second is `None`, and value at index 1 becomes False.

    Why this matters:
    - Confirms the public API shape and discourages callers from relying on a non-`None` flag.
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
    """Behavior: `tab.tabulate(0)` is a no-op for both `actual_lags` and `avail_lags`.

    Arrange:
    - Presentation `[1,2,3,4]`, `first_recall=1`, `size=2`.

    Act:
    - Capture `actual_lags` and `avail_lags` before calling `tab.tabulate(0)`.
    - Also require `tab.should_tabulate(0)` to be `False`.

    Assert:
    - `actual_lags` and `avail_lags` are identical before vs after.

    Why this matters:
    - Enforces the sentinel convention for `0` to avoid accidental state mutation.
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
    """Behavior: `should_tabulate` respects validity and availability of study positions.

    Arrange:
    - Presentation `[1,2,3]`, `first_recall=1`, `size=1`.

    Act:
    - Query `should_tabulate` for items 2 (available), 1 (already consumed), and 0 (sentinel).
    - Then consume item 2 and query again for 2.

    Assert:
    - True for 2 initially; False for 1 and 0; False for 2 after consuming it.

    Why this matters:
    - Protects the main gate that determines whether a recall event mutates state and counts.
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
    """Behavior: repeated recalls are ignored; only valid transitions are counted.

    Arrange:
    - Presentation `[1,2,3,4]`, start at item 3.

    Act:
    - Attempt to recall 3 again (ignored), then recall 4 (valid 3→4 transition).

    Assert:
    - Sum of `actual_lags` equals 1.

    Why this matters:
    - Detects accidental double-counting when an already recalled item is repeated.
    """
    presentation = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    tab = crp.Tabulation(presentation=presentation, first_recall=3, size=1)
    tab = tab.tabulate(3)  # ignored
    tab = tab.tabulate(4)  # valid
    total = jnp.sum(tab.actual_lags).item()
    assert total == 1, f"Expected exactly 1 transition, got {total}"


def test_repeat_does_not_change_previous_positions_transitions():
    """Behavior: an ignored repeat must not rewrite earlier transitions.

    Arrange:
    - Presentation `[1,2,3,4]`, recalls `[3,1,3,4]`.

    Act:
    - Valid transitions: 3→1 (lag -2) and 1→4 (lag +3). The middle `3` is a repeat and ignored.

    Assert:
    - Exactly two total transitions; 1 count each at lags -2 and +3; 0 at +1 (no 3→4 from the repeat).

    Why this matters:
    - Guards against logic that retroactively mutates `previous_positions` or accepts repeats.
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
    """Behavior: zero-padded study positions are ignored as invalid.

    Arrange:
    - Presentation `[10,20,30,40]`, `first_recall=1`, `size=3` (induces internal zero padding).

    Act:
    - Recall item 2.

    Assert:
    - Exactly one valid transition (1→2).

    Why this matters:
    - Prevents zeros used for padding from being misinterpreted as real study positions.
    """
    presentation = jnp.array([10, 20, 30, 40], dtype=jnp.int32)
    tab = crp.Tabulation(presentation=presentation, first_recall=1, size=3)
    tab = tab.tabulate(2)
    total = jnp.sum(tab.actual_lags).item()
    assert total == 1, f"Expected exactly one valid transition (1→2), got {total}"


def test_lags_from_previous_single_lag_location():
    """Behavior: `lags_from_previous` marks exactly the correct lag bin.

    Arrange:
    - Presentation `[1,2,3]`, `first_recall=1`, `size=1`.

    Act:
    - Compute mask for `recall_pos=2`.

    Assert:
    - The mask is boolean with exactly one True at the index for lag +1.

    Why this matters:
    - Ensures the core lag-to-index mapping inside the tabulation logic is correct.
    """
    presentation = jnp.array([1, 2, 3], dtype=jnp.int32)
    tab = crp.Tabulation(presentation=presentation, first_recall=1, size=1)
    mask = tab.lags_from_previous(2)
    assert mask.dtype == bool
    assert mask.sum().item() == 1
    expected_idx = _lag_index(mask.size, +1)
    assert bool(mask[expected_idx])


def test_available_lags_from_zero_and_nonzero():
    """Behavior: `available_lags_from(0)` yields no lags; nonzero positions yield some.

    Arrange:
    - Presentation `[1,2,3]`, `first_recall=1`, `size=1`.

    Act:
    - Query `available_lags_from(0)` and `available_lags_from(2)`.

    Assert:
    - Sum is 0 for position 0; strictly greater than 0 for a valid nonzero position.

    Why this matters:
    - Confirms the special-case for `pos == 0` and the basic availability computation.
    """
    presentation = jnp.array([1, 2, 3], dtype=jnp.int32)
    tab = crp.Tabulation(presentation=presentation, first_recall=1, size=1)
    base = tab.available_lags_from(0)
    nonzero = tab.available_lags_from(2)
    assert base.sum().item() == 0
    assert nonzero.sum().item() > 0


def test_tabulate_returns_new_object_and_updates_counts():
    """Behavior: `tabulate` is persistent (returns new) and updates counts.

    Arrange:
    - Presentation `[1,2,3]`, `first_recall=1`, `size=1`.

    Act:
    - Call `tab.tabulate(2)`.

    Assert:
    - Returns a *new* object; `actual_lags` increases by 1; `avail_lags` does not decrease.

    Why this matters:
    - Confirms the functional update style and a basic counting invariant.
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
    """Behavior: `tabulate_trial` ignores zero recalls when counting transitions.

    Arrange:
    - Trial `[1,0,2,0,3]`, Presentation `[1,2,3]`, `size=1`.

    Act:
    - Run `tabulate_trial`.

    Assert:
    - Sum of `actual_lags` equals 2 (transitions 1→2 and 2→3 only).

    Why this matters:
    - Ensures trial-level API respects the zero sentinel contract.
    """
    trial = jnp.array([1, 0, 2, 0, 3], dtype=jnp.int32)
    presentation = jnp.array([1, 2, 3], dtype=jnp.int32)
    actual, _ = crp.tabulate_trial(trial, presentation, size=1)
    assert jnp.sum(actual).item() == 2


def test_simple_and_full_tabulation_agree_on_total_actual_transitions():
    """Behavior: simple and full tabulators agree on total transition count (basic case).

    Arrange:
    - Trial `[1,2,3]`, Presentation `[1,2,3]`, `size=1`.

    Act:
    - Run `simple_tabulate_trial` and `tabulate_trial`.

    Assert:
    - Sums of actual-transition vectors match and equal 2.

    Why this matters:
    - Protects against divergence between the simple and full implementations.
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


def test_crp_handles_single_item_without_crashing():
    """Behavior: `crp` handles degenerate one-item trials without raising.

    Arrange:
    - Trials `[[1]]`, Presentations `[[1]]`, `size=1`.

    Act:
    - Call `crp`.

    Assert:
    - Returns a JAX array of length 1 (no exception raised).

    Why this matters:
    - Ensures graceful behavior when possible transitions are zero.
    """
    trials = jnp.array([[1]], dtype=jnp.int32)
    presentations = jnp.array([[1]], dtype=jnp.int32)
    out = crp.crp(trials, presentations, list_length=1, size=1)
    assert isinstance(out, jnp.ndarray)
    assert out.shape[0] == 1


def test_crp_jit_with_static_argnames():
    """Ensure `crp` works when jitted with static_argnames=("size","list_length").

    We compute the non-jitted result and compare to the jitted result using
    `jax.jit(..., static_argnames=("size","list_length"))` to ensure the
    compiled version returns the same values and shape.
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
    """Sanity check that jitting with a different `size` value also runs.

    This test compiles with a non-default `size` and verifies execution.
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
    """`plot_crp` should return a Matplotlib ``Axes`` instance."""
    dataset = {
        "subject": jnp.array([[1], [1]], dtype=jnp.int32),
        "listLength": jnp.array([[3], [3]], dtype=jnp.int32),
        "pres_itemnos": jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32),
        "recalls": jnp.array([[1, 2, 0], [2, 3, 0]], dtype=jnp.int32),
        "pres_itemids": jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32),
    }
    trial_mask = jnp.array([True, True], dtype=bool)
    axis = crp.plot_crp(dataset, trial_mask, max_lag=1)
    assert isinstance(axis, Axes)
    plt.close(axis.figure)

