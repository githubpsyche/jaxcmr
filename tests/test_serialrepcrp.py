import jax.numpy as jnp

from jaxcmr.analyses.serialrepcrp import tabulate_trial


def test_correct_runup_from_first_occurrence_counts_same_successor():
    """Behavior: correct serial run-up to first occurrence counts R -> i+1."""
    pres = jnp.array([1, 2, 3, 1, 4, 5, 6, 7], dtype=jnp.int32)
    trial = jnp.array([1, 2, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    lag_range = pres.size - 1

    actual, possible = tabulate_trial(trial, pres, min_lag=2, size=2)

    assert actual.shape == (2, 15)
    assert possible.shape == (2, 15)
    assert int(actual[0, lag_range + 1]) == 1
    assert int(actual[1, lag_range - 2]) == 1
    assert int(actual.sum()) == 2


def test_transition_to_second_occurrence_successor_is_tabulated_before_error_blocks():
    """Behavior: R -> j+1 is tabulated even though that response breaks order."""
    pres = jnp.array([1, 2, 3, 1, 4, 5, 6, 7], dtype=jnp.int32)
    trial = jnp.array([1, 5, 2, 0, 0, 0, 0, 0], dtype=jnp.int32)
    lag_range = pres.size - 1

    actual, _ = tabulate_trial(trial, pres, min_lag=2, size=2)

    assert int(actual[0, lag_range + 4]) == 1
    assert int(actual[1, lag_range + 1]) == 1
    assert int(actual.sum()) == 2


def test_broken_runup_before_repeater_blocks_tabulation():
    """Behavior: an out-of-order prefix prevents later repeated-item counts."""
    pres = jnp.array([1, 2, 3, 1, 4, 5, 6, 7], dtype=jnp.int32)
    trial = jnp.array([2, 1, 4, 0, 0, 0, 0, 0], dtype=jnp.int32)

    actual, possible = tabulate_trial(trial, pres, min_lag=2, size=2)

    assert int(actual.sum()) == 0
    assert int(possible.sum()) == 0


def test_current_availability_denominator_keeps_recalled_repeater_positions():
    """Behavior: serialrepcrp currently does not clear recalled-item availability."""
    pres = jnp.array([1, 2, 3, 1, 4, 5, 6, 7], dtype=jnp.int32)
    trial = jnp.array([1, 2, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    lag_range = pres.size - 1

    _, possible = tabulate_trial(trial, pres, min_lag=2, size=2)

    assert int(possible[0, lag_range + 0]) == 1
    assert int(possible[0, lag_range + 3]) == 1
    assert int(possible[1, lag_range - 3]) == 1
    assert int(possible[1, lag_range + 0]) == 1
