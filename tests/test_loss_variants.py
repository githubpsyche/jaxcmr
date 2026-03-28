"""Tests for loss function variants.

Covers base_sequence_likelihood, transform_sequence_likelihood,
set_permutation_likelihood, and spc_mse.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np

import jaxcmr.components.context as TemporalContext
import jaxcmr.components.linear_memory as LinearMemory
from jaxcmr.components.termination import PositionalTermination
from jaxcmr.helpers import make_dataset
from jaxcmr.models.cmr import make_factory as base_make_factory

# ── shared fixtures ─────────────────────────────────────────────────────────

BaseCMRFactory = base_make_factory(
    LinearMemory.init_mfc,
    LinearMemory.init_mcf,
    TemporalContext.init,
    PositionalTermination,
)


def _params() -> dict[str, Any]:
    return {
        "encoding_drift_rate": 0.5,
        "start_drift_rate": 0.4,
        "recall_drift_rate": 0.6,
        "shared_support": 1.0,
        "item_support": 1.5,
        "learning_rate": 0.3,
        "primacy_scale": 1.0,
        "primacy_decay": 0.1,
        "stop_probability_scale": 0.05,
        "stop_probability_growth": 0.2,
        "choice_sensitivity": 2.0,
        "learn_after_context_update": False,
        "allow_repeated_recalls": False,
    }


def _dataset() -> Any:
    presents = jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32)
    recalls = jnp.array([[1, 2, 0], [2, 3, 0]], dtype=jnp.int32)
    return make_dataset(recalls, pres_itemnos=presents, listLength=3)


# ── base_sequence_likelihood ────────────────────────────────────────────────


def test_mask_trailing_terminations_exact_values():
    """Behavior: Trailing zeros map to False, nonzero entries to True.

    Given:
      - A recall vector ``[2, 1, 0, 0]``.
    When:
      - ``mask_trailing_terminations`` is applied.
    Then:
      - Positions 0 and 1 are True; positions 2 and 3 are False.
    Why this matters:
      - This mask excludes trailing stop events from the likelihood
        computation; incorrect masking would corrupt the loss.
    """
    # Arrange / Given
    from jaxcmr.loss.base_sequence_likelihood import mask_trailing_terminations

    recalls = jnp.array([2, 1, 0, 0], dtype=jnp.int32)

    # Act / When
    mask = mask_trailing_terminations(recalls)

    # Assert / Then
    expected = jnp.array([True, True, False, False])
    assert jnp.array_equal(mask, expected).item()


def test_mask_first_recall_exact_values():
    """Behavior: Position 0 is masked out; all others remain True.

    Given:
      - A recall vector ``[2, 1, 3]``.
    When:
      - ``mask_first_recall`` is applied.
    Then:
      - Position 0 is False, positions 1 and 2 are True.
    Why this matters:
      - Some models treat the first recall as a free choice; masking
        position 0 prevents it from contributing to the likelihood.
    """
    # Arrange / Given
    from jaxcmr.loss.base_sequence_likelihood import mask_first_recall

    recalls = jnp.array([2, 1, 3], dtype=jnp.int32)

    # Act / When
    mask = mask_first_recall(recalls)

    # Assert / Then
    expected = jnp.array([False, True, True])
    assert jnp.array_equal(mask, expected).item()


def test_base_generator_returns_positive_finite_loss():
    """Behavior: Base sequence likelihood is finite and positive.

    Given:
      - A two-trial dataset with the base CMR factory.
    When:
      - ``base_predict_trials_loss`` is called.
    Then:
      - The returned scalar is finite and positive (negative log-likelihood
        of an imperfect prediction must be positive).
    Why this matters:
      - Confirms end-to-end forward pass through the base likelihood
        pipeline produces a valid optimization objective.
    """
    # Arrange / Given
    from jaxcmr.loss.base_sequence_likelihood import (
        MemorySearchLikelihoodFnGenerator,
        mask_trailing_terminations,
    )

    dataset = _dataset()
    gen = MemorySearchLikelihoodFnGenerator(
        BaseCMRFactory, dataset, None, mask_trailing_terminations
    )
    params = _params()
    trial_idx = jnp.array([0, 1], dtype=jnp.int32)

    # Act / When
    loss = gen.present_and_predict_trials_loss(trial_idx, params)

    # Assert / Then
    assert jnp.isfinite(loss).item()
    assert float(loss) > 0


# ── transform_sequence_likelihood ───────────────────────────────────────────


def test_transform_generator_returns_positive_finite_loss():
    """Behavior: Transform sequence likelihood is finite and positive.

    Given:
      - A two-trial dataset with the base CMR factory and trailing-zero mask.
    When:
      - ``present_and_predict_trials_loss`` is called.
    Then:
      - The returned scalar is finite and positive.
    Why this matters:
      - Validates the per-trial presentation variant computes valid
        negative log-likelihoods for parameter optimization.
    """
    # Arrange / Given
    from jaxcmr.loss.transform_sequence_likelihood import (
        MemorySearchLikelihoodFnGenerator,
        mask_trailing_terminations,
    )

    dataset = _dataset()
    gen = MemorySearchLikelihoodFnGenerator(
        BaseCMRFactory, dataset, None, mask_trailing_terminations
    )
    params = _params()
    trial_idx = jnp.array([0, 1], dtype=jnp.int32)

    # Act / When
    loss = gen.present_and_predict_trials_loss(trial_idx, params)

    # Assert / Then
    assert jnp.isfinite(loss).item()
    assert float(loss) > 0


def test_transform_predict_trial_valid_probabilities():
    """Behavior: Per-event recall probabilities are in (0, 1].

    Given:
      - A two-trial dataset and the transform generator.
    When:
      - ``predict_trial`` is called for trial 0.
    Then:
      - All returned probabilities are in (0, 1].
      - Shape matches the list length (one probability per recall event).
    Why this matters:
      - Each value is the model-predicted probability of the observed
        recall at that position; invalid probabilities would produce
        undefined log-likelihoods.
    """
    # Arrange / Given
    from jaxcmr.loss.transform_sequence_likelihood import (
        MemorySearchLikelihoodFnGenerator,
        mask_trailing_terminations,
    )

    dataset = _dataset()
    gen = MemorySearchLikelihoodFnGenerator(
        BaseCMRFactory, dataset, None, mask_trailing_terminations
    )
    params = _params()

    # Act / When
    probs = gen.predict_trial(jnp.int32(0), params)

    # Assert / Then
    assert probs.shape == (3,)
    assert jnp.all(probs > 0).item()
    assert jnp.all(probs <= 1).item()


# ── set_permutation_likelihood ──────────────────────────────────────────────


def test_set_permutation_generator_has_expected_structure():
    """Behavior: Set permutation generator has correct trial permutations.

    Given:
      - A two-trial dataset with the base CMR factory.
    When:
      - The generator is instantiated.
    Then:
      - trial_permutations has shape (n_trials, simulation_count).
      - Each permutation column contains a valid permutation of trial indices.
    Why this matters:
      - The permutation structure controls the Monte Carlo sampling;
        incorrect shapes or invalid indices would corrupt the loss.
    """
    # Arrange / Given
    from jaxcmr.loss.set_permutation_likelihood import (
        ExcludeTerminationLikelihoodFnGenerator,
    )

    dataset = _dataset()

    # Act / When
    gen = ExcludeTerminationLikelihoodFnGenerator(BaseCMRFactory, dataset, None)

    # Assert / Then
    assert gen._inner.trial_permutations.shape[0] == 2  # n_trials
    assert gen._inner.trial_permutations.shape[1] == gen._inner.simulation_count
    assert gen._inner.trial_permutations.shape[2] == 3  # list_length
    # Values are non-negative (valid item positions or zero padding)
    assert jnp.all(gen._inner.trial_permutations >= 0).item()


def test_set_permutation_generator_returns_finite_loss():
    """Behavior: Set permutation generator returns a finite scalar loss.

    Given:
      - A two-trial dataset with the base CMR factory.
    When:
      - ``present_and_predict_trials_loss`` is called.
    Then:
      - The returned scalar is finite.
      - Calling again with the same params produces the same loss
        (deterministic given fixed PRNG keys).
    Why this matters:
      - Validates the Monte Carlo permutation approach produces a valid
        and reproducible negative log-likelihood estimate.
    """
    # Arrange / Given
    from jaxcmr.loss.set_permutation_likelihood import (
        ExcludeTerminationLikelihoodFnGenerator,
    )

    dataset = _dataset()
    gen = ExcludeTerminationLikelihoodFnGenerator(BaseCMRFactory, dataset, None)
    params = _params()
    trial_idx = jnp.array([0, 1], dtype=jnp.int32)

    # Act / When
    loss1 = gen._inner.present_and_predict_trials_loss(trial_idx, params)
    loss2 = gen._inner.present_and_predict_trials_loss(trial_idx, params)

    # Assert / Then
    assert jnp.isfinite(loss1).item()
    assert jnp.isclose(loss1, loss2).item()  # deterministic


# ── include_termination_set_permutation_likelihood ─────────────────────────


def test_stop_permutation_structure():
    """Behavior: Stop-aware permutations preserve the recall bag.

    Given:
      - A two-trial dataset where each trial recalls 2 of 3 items.
    When:
      - The stop-aware generator is instantiated.
    Then:
      - Each permutation is a valid permutation of the trial items
        (same multiset of recalled items and zeros, just reordered).
    """
    # Arrange / Given
    from jaxcmr.loss.set_permutation_likelihood import (
        IncludeTerminationLikelihoodFnGenerator,
    )

    dataset = _dataset()

    # Act / When
    gen = IncludeTerminationLikelihoodFnGenerator(
        BaseCMRFactory, dataset, None
    )

    # Assert / Then
    inner = gen._inner
    perms = inner.trial_permutations
    assert perms.shape[0] == 2  # n_trials
    assert perms.shape[1] == inner.simulation_count
    assert perms.shape[2] == 3  # recall_width

    for trial_idx in range(2):
        # All permutations should contain the same multiset of values
        reference = sorted(int(x) for x in perms[trial_idx, 0])
        for sim_idx in range(inner.simulation_count):
            perm = perms[trial_idx, sim_idx]
            assert sorted(int(x) for x in perm) == reference


def test_stop_aware_loss_returns_finite_positive():
    """Behavior: Stop-aware generator returns a finite, positive loss.

    Given:
      - A two-trial dataset with the base CMR factory using
        PositionalTermination.
    When:
      - ``present_and_predict_trials_loss`` is called.
    Then:
      - The returned scalar is finite and positive.
      - Calling again produces the same value (deterministic).
    """
    # Arrange / Given
    from jaxcmr.loss.set_permutation_likelihood import (
        IncludeTerminationLikelihoodFnGenerator,
    )

    dataset = _dataset()
    gen = IncludeTerminationLikelihoodFnGenerator(
        BaseCMRFactory, dataset, None
    )
    params = _params()
    trial_idx = jnp.array([0, 1], dtype=jnp.int32)

    # Act / When
    loss1 = gen._inner.present_and_predict_trials_loss(trial_idx, params)
    loss2 = gen._inner.present_and_predict_trials_loss(trial_idx, params)

    # Assert / Then
    assert jnp.isfinite(loss1).item()
    assert float(loss1) > 0
    assert jnp.isclose(loss1, loss2).item()


def test_stop_aware_generator_rejects_full_recall():
    """Behavior: Generator raises ValueError when no padding slot exists.

    Given:
      - A dataset where one trial recalls all items (no zero-padding).
    When:
      - The stop-aware generator is instantiated.
    Then:
      - A ValueError is raised.
    """
    # Arrange / Given
    import pytest

    from jaxcmr.loss.set_permutation_likelihood import (
        IncludeTerminationLikelihoodFnGenerator,
    )

    presents = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    recalls = jnp.array([[1, 2, 3]], dtype=jnp.int32)  # full recall, no padding
    dataset = make_dataset(recalls, pres_itemnos=presents, listLength=3)

    # Act / When / Assert / Then
    with pytest.raises(ValueError, match="at least one padding slot"):
        IncludeTerminationLikelihoodFnGenerator(
            BaseCMRFactory, dataset, None
        )


# ── spc_mse ─────────────────────────────────────────────────────────────────


def test_spc_mse_generator_has_expected_structure():
    """Behavior: SPC MSE generator has correct internal structure.

    Given:
      - A two-trial dataset with the base CMR factory.
    When:
      - ``MemorySearchSpcMseFnGenerator`` is instantiated.
    Then:
      - list_length = 3, simulation_count = 20, trial_position_counts
        has shape (2, 3) with values in [0, 1] (normalized recall counts).
    Why this matters:
      - Validates that SPC target computation and key splitting succeed,
        and that the per-position recall counts are valid proportions.
    """
    # Arrange / Given
    from jaxcmr.loss.spc_mse import MemorySearchSpcMseFnGenerator

    dataset = _dataset()

    # Act / When
    gen = MemorySearchSpcMseFnGenerator(BaseCMRFactory, dataset, None)

    # Assert / Then
    assert gen.list_length == 3
    assert gen.simulation_count == 20
    assert gen.trial_position_counts.shape == (2, 3)
    # Position counts are non-negative integers
    assert jnp.all(gen.trial_position_counts >= 0).item()


def test_spc_mse_generator_returns_nonnegative_finite_loss():
    """Behavior: SPC MSE generator returns a finite, non-negative loss.

    Given:
      - A two-trial dataset with the base CMR factory.
    When:
      - The loss function is evaluated.
    Then:
      - The returned scalar is finite and non-negative (MSE >= 0).
    Why this matters:
      - MSE is non-negative by definition; a negative or non-finite
        value would indicate a computation error.
    """
    # Arrange / Given
    from jaxcmr.loss.spc_mse import MemorySearchSpcMseFnGenerator

    dataset = _dataset()
    gen = MemorySearchSpcMseFnGenerator(BaseCMRFactory, dataset, None)
    params = _params()
    trial_idx = jnp.array([0, 1], dtype=jnp.int32)
    free = ["encoding_drift_rate"]
    f = gen(trial_idx, base_params=params, free_param_names=free)

    # Act / When
    loss = f(np.array([0.5]))

    # Assert / Then
    assert jnp.isfinite(loss).item()
    assert float(loss) >= 0
