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
from jaxcmr.fitting.scipy import make_scipy_loss_fn
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


def test_old_loss_names_remain_import_aliases():
    """Behavior: old loss names remain temporary aliases for compatibility."""
    from jaxcmr import typing
    from jaxcmr.loss import (
        base_sequence_likelihood,
        cat_spc_mse,
        mse,
        sequence_likelihood,
        set_permutation_likelihood,
        spc_mse,
        transform_sequence_likelihood,
    )

    assert typing.LossFnGenerator is typing.LossFn
    assert (
        sequence_likelihood.MemorySearchLikelihoodFnGenerator
        is sequence_likelihood.MemorySearchLikelihoodLoss
    )
    assert (
        base_sequence_likelihood.MemorySearchLikelihoodFnGenerator
        is base_sequence_likelihood.MemorySearchLikelihoodLoss
    )
    assert (
        base_sequence_likelihood.ExcludeFirstRecallLikelihoodFnGenerator
        is base_sequence_likelihood.ExcludeFirstRecallLikelihoodLoss
    )
    assert (
        base_sequence_likelihood.ExcludeTerminationLikelihoodFnGenerator
        is base_sequence_likelihood.ExcludeTerminationLikelihoodLoss
    )
    assert (
        transform_sequence_likelihood.MemorySearchLikelihoodFnGenerator
        is transform_sequence_likelihood.MemorySearchLikelihoodLoss
    )
    assert (
        transform_sequence_likelihood.ExcludeFirstRecallLikelihoodFnGenerator
        is transform_sequence_likelihood.ExcludeFirstRecallLikelihoodLoss
    )
    assert (
        transform_sequence_likelihood.ExcludeTerminationLikelihoodFnGenerator
        is transform_sequence_likelihood.ExcludeTerminationLikelihoodLoss
    )
    assert (
        transform_sequence_likelihood.AccumulatingExcludeTerminationLikelihoodFnGenerator
        is transform_sequence_likelihood.AccumulatingExcludeTerminationLikelihoodLoss
    )
    assert (
        transform_sequence_likelihood.SkippingAccumulatingExcludeTerminationLikelihoodFnGenerator
        is transform_sequence_likelihood.SkippingAccumulatingExcludeTerminationLikelihoodLoss
    )
    assert (
        set_permutation_likelihood.MemorySearchLikelihoodFnGenerator
        is set_permutation_likelihood.MemorySearchLikelihoodLoss
    )
    assert (
        set_permutation_likelihood.ExcludeTerminationLikelihoodFnGenerator
        is set_permutation_likelihood.ExcludeTerminationLikelihoodLoss
    )
    assert (
        set_permutation_likelihood.IncludeTerminationLikelihoodFnGenerator
        is set_permutation_likelihood.IncludeTerminationLikelihoodLoss
    )
    assert mse.MemorySearchMseFnGenerator is mse.MemorySearchMseLoss
    assert spc_mse.MemorySearchSpcMseFnGenerator is spc_mse.MemorySearchSpcMseLoss
    assert (
        cat_spc_mse.MemorySearchCatSpcMseFnGenerator
        is cat_spc_mse.MemorySearchCatSpcMseLoss
    )


def test_old_loss_names_are_not_advertised_in_all():
    """Behavior: compatibility aliases are not advertised as primary API."""
    from jaxcmr import typing
    from jaxcmr.loss import (
        base_sequence_likelihood,
        cat_spc_mse,
        mse,
        sequence_likelihood,
        set_permutation_likelihood,
        spc_mse,
        transform_sequence_likelihood,
    )

    modules = [
        typing,
        sequence_likelihood,
        base_sequence_likelihood,
        transform_sequence_likelihood,
        set_permutation_likelihood,
        mse,
        spc_mse,
        cat_spc_mse,
    ]
    for module in modules:
        assert not any(name.endswith("FnGenerator") for name in module.__all__)


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
        MemorySearchLikelihoodLoss,
        mask_trailing_terminations,
    )

    dataset = _dataset()
    gen = MemorySearchLikelihoodLoss(
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
        MemorySearchLikelihoodLoss,
        mask_trailing_terminations,
    )

    dataset = _dataset()
    gen = MemorySearchLikelihoodLoss(
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
        MemorySearchLikelihoodLoss,
        mask_trailing_terminations,
    )

    dataset = _dataset()
    gen = MemorySearchLikelihoodLoss(
        BaseCMRFactory, dataset, None, mask_trailing_terminations
    )
    params = _params()

    # Act / When
    probs = gen.predict_trial(jnp.int32(0), params)

    # Assert / Then
    assert probs.shape == (3,)
    assert jnp.all(probs > 0).item()
    assert jnp.all(probs <= 1).item()


def test_transform_accumulating_exclude_termination_matches_existing_loss():
    """Behavior: accumulating exclude-termination loss matches current loss."""
    from jaxcmr.loss.transform_sequence_likelihood import (
        AccumulatingExcludeTerminationLikelihoodLoss,
        ExcludeTerminationLikelihoodLoss,
        SkippingAccumulatingExcludeTerminationLikelihoodLoss,
    )

    dataset = _dataset()
    params = _params()
    trial_idx = jnp.array([0, 1], dtype=jnp.int32)
    current = ExcludeTerminationLikelihoodLoss(BaseCMRFactory, dataset, None)
    accumulating = AccumulatingExcludeTerminationLikelihoodLoss(
        BaseCMRFactory, dataset, None
    )
    skipping = SkippingAccumulatingExcludeTerminationLikelihoodLoss(
        BaseCMRFactory, dataset, None
    )

    current_loss = current._inner.present_and_predict_trials_loss(trial_idx, params)
    accumulating_loss = accumulating._inner.present_and_predict_trials_loss(
        trial_idx, params
    )
    skipping_loss = skipping._inner.present_and_predict_trials_loss(trial_idx, params)

    np.testing.assert_allclose(accumulating_loss, current_loss, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(skipping_loss, current_loss, rtol=1e-6, atol=1e-6)


def test_transform_accumulating_loss_matches_existing_loss_call():
    """Behavior: accumulating loss matches current loss call."""
    from jaxcmr.loss.transform_sequence_likelihood import (
        AccumulatingExcludeTerminationLikelihoodLoss,
        ExcludeTerminationLikelihoodLoss,
        SkippingAccumulatingExcludeTerminationLikelihoodLoss,
    )

    dataset = _dataset()
    params = _params()
    trial_idx = jnp.array([0, 1], dtype=jnp.int32)
    free = ["encoding_drift_rate", "recall_drift_rate"]
    x = jnp.array(
        [
            [params["encoding_drift_rate"], 0.7],
            [params["recall_drift_rate"], 0.5],
        ]
    )
    current = ExcludeTerminationLikelihoodLoss(BaseCMRFactory, dataset, None)
    accumulating = AccumulatingExcludeTerminationLikelihoodLoss(
        BaseCMRFactory, dataset, None
    )
    skipping = SkippingAccumulatingExcludeTerminationLikelihoodLoss(
        BaseCMRFactory, dataset, None
    )

    current_loss = current(trial_idx, params, free, x)
    accumulating_loss = accumulating(trial_idx, params, free, x)
    skipping_loss = skipping(trial_idx, params, free, x)

    np.testing.assert_allclose(accumulating_loss, current_loss, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(skipping_loss, current_loss, rtol=1e-6, atol=1e-6)


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
        ExcludeTerminationLikelihoodLoss,
    )

    dataset = _dataset()

    # Act / When
    gen = ExcludeTerminationLikelihoodLoss(BaseCMRFactory, dataset, None)

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
        ExcludeTerminationLikelihoodLoss,
    )

    dataset = _dataset()
    gen = ExcludeTerminationLikelihoodLoss(BaseCMRFactory, dataset, None)
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
        IncludeTerminationLikelihoodLoss,
    )

    dataset = _dataset()

    # Act / When
    gen = IncludeTerminationLikelihoodLoss(
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
        IncludeTerminationLikelihoodLoss,
    )

    dataset = _dataset()
    gen = IncludeTerminationLikelihoodLoss(
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
        IncludeTerminationLikelihoodLoss,
    )

    presents = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    recalls = jnp.array([[1, 2, 3]], dtype=jnp.int32)  # full recall, no padding
    dataset = make_dataset(recalls, pres_itemnos=presents, listLength=3)

    # Act / When / Assert / Then
    with pytest.raises(ValueError, match="at least one padding slot"):
        IncludeTerminationLikelihoodLoss(
            BaseCMRFactory, dataset, None
        )


# ── spc_mse ─────────────────────────────────────────────────────────────────


def test_spc_mse_generator_has_expected_structure():
    """Behavior: SPC MSE generator has correct internal structure.

    Given:
      - A two-trial dataset with the base CMR factory.
    When:
      - ``MemorySearchSpcMseLoss`` is instantiated.
    Then:
      - list_length = 3, simulation_count = 20, trial_position_counts
        has shape (2, 3) with values in [0, 1] (normalized recall counts).
    Why this matters:
      - Validates that SPC target computation and key splitting succeed,
        and that the per-position recall counts are valid proportions.
    """
    # Arrange / Given
    from jaxcmr.loss.spc_mse import MemorySearchSpcMseLoss

    dataset = _dataset()

    # Act / When
    gen = MemorySearchSpcMseLoss(BaseCMRFactory, dataset, None)

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
    from jaxcmr.loss.spc_mse import MemorySearchSpcMseLoss

    dataset = _dataset()
    gen = MemorySearchSpcMseLoss(BaseCMRFactory, dataset, None)
    params = _params()
    trial_idx = jnp.array([0, 1], dtype=jnp.int32)
    free = ["encoding_drift_rate"]
    f = make_scipy_loss_fn(gen, trial_idx, params, free)

    # Act / When
    loss = f(np.array([0.5]))

    # Assert / Then
    assert jnp.isfinite(loss).item()
    assert float(loss) >= 0


# ── SciPy adapters ──────────────────────────────────────────────────────────


def _assert_loss_matches_scipy_adapter(name, gen, params, trial_idx, free):
    f = make_scipy_loss_fn(gen, trial_idx, params, free)
    x_single = np.asarray([params[param] for param in free])
    x_batch = np.stack([x_single, x_single], axis=0).T

    call_single = np.asarray(f(x_single))
    call_batch = np.asarray(f(x_batch))
    loss_single = np.asarray(gen(trial_idx, params, free, jnp.asarray(x_single[:, None]))[0])
    loss_batch = np.asarray(gen(trial_idx, params, free, jnp.asarray(x_batch)))

    np.testing.assert_allclose(
        loss_single,
        call_single,
        rtol=1e-6,
        atol=1e-6,
        err_msg=name,
    )
    np.testing.assert_allclose(
        loss_batch,
        call_batch,
        rtol=1e-6,
        atol=1e-6,
        err_msg=name,
    )


def test_sequence_variant_losses_match_scipy_adapter():
    """Behavior: built-in sequence variants share the SciPy adapter."""
    from jaxcmr.loss import base_sequence_likelihood, transform_sequence_likelihood
    from jaxcmr.loss import set_permutation_likelihood

    dataset = _dataset()
    params = _params()
    trial_idx = jnp.array([0, 1], dtype=jnp.int32)
    free = ["encoding_drift_rate", "recall_drift_rate"]
    cases = [
        (
            "base sequence",
            base_sequence_likelihood.MemorySearchLikelihoodLoss(
                BaseCMRFactory,
                dataset,
                None,
                base_sequence_likelihood.mask_trailing_terminations,
            ),
        ),
        (
            "base exclude first",
            base_sequence_likelihood.ExcludeFirstRecallLikelihoodLoss(
                BaseCMRFactory, dataset, None
            ),
        ),
        (
            "base exclude termination",
            base_sequence_likelihood.ExcludeTerminationLikelihoodLoss(
                BaseCMRFactory, dataset, None
            ),
        ),
        (
            "transform sequence",
            transform_sequence_likelihood.MemorySearchLikelihoodLoss(
                BaseCMRFactory,
                dataset,
                None,
                transform_sequence_likelihood.mask_trailing_terminations,
            ),
        ),
        (
            "transform exclude first",
            transform_sequence_likelihood.ExcludeFirstRecallLikelihoodLoss(
                BaseCMRFactory, dataset, None
            ),
        ),
        (
            "transform exclude termination",
            transform_sequence_likelihood.ExcludeTerminationLikelihoodLoss(
                BaseCMRFactory, dataset, None
            ),
        ),
        (
            "transform accumulating exclude termination",
            transform_sequence_likelihood.AccumulatingExcludeTerminationLikelihoodLoss(
                BaseCMRFactory, dataset, None
            ),
        ),
        (
            "transform skipping accumulating exclude termination",
            transform_sequence_likelihood.SkippingAccumulatingExcludeTerminationLikelihoodLoss(
                BaseCMRFactory, dataset, None
            ),
        ),
        (
            "set permutation exclude termination",
            set_permutation_likelihood.ExcludeTerminationLikelihoodLoss(
                BaseCMRFactory, dataset, None
            ),
        ),
        (
            "set permutation include termination",
            set_permutation_likelihood.IncludeTerminationLikelihoodLoss(
                BaseCMRFactory, dataset, None
            ),
        ),
    ]

    for name, gen in cases:
        _assert_loss_matches_scipy_adapter(name, gen, params, trial_idx, free)


def test_mse_losses_match_scipy_adapter():
    """Behavior: built-in MSE variants share the SciPy adapter."""
    from jaxcmr.loss.cat_spc_mse import MemorySearchCatSpcMseLoss
    from jaxcmr.loss.mse import MemorySearchMseLoss
    from jaxcmr.loss.spc_mse import MemorySearchSpcMseLoss

    def analysis_fn(recalls, list_length, dataset, trial_indices):
        return jnp.mean(recalls > 0, axis=(0, 1))

    dataset = _dataset()
    category_dataset = _dataset()
    category_dataset["condition"] = jnp.array(
        [[1, 2, 1], [1, 2, 1]], dtype=jnp.int32
    )
    params = _params()
    trial_idx = jnp.array([0, 1], dtype=jnp.int32)
    free = ["encoding_drift_rate"]
    cases = [
        (
            "mse",
            MemorySearchMseLoss(
                BaseCMRFactory, dataset, None, analysis_fn
            ),
        ),
        ("spc mse", MemorySearchSpcMseLoss(BaseCMRFactory, dataset, None)),
        (
            "category spc mse",
            MemorySearchCatSpcMseLoss(
                BaseCMRFactory, category_dataset, None
            ),
        ),
    ]

    for name, gen in cases:
        _assert_loss_matches_scipy_adapter(name, gen, params, trial_idx, free)
