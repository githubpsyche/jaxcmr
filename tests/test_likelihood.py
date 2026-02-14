import numpy as np
import jax.numpy as jnp

import jaxcmr.components.context as TemporalContext
import jaxcmr.components.linear_memory as LinearMemory
from jaxcmr.components.termination import PositionalTermination
from jaxcmr.loss.sequence_likelihood import MemorySearchLikelihoodFnGenerator
from jaxcmr.models.cmr import make_factory
from jaxcmr.helpers import make_dataset

BaseCMRFactory = make_factory(
    LinearMemory.init_mfc,
    LinearMemory.init_mcf,
    TemporalContext.init,
    PositionalTermination,
)


def _params():
    """Returns a small, valid parameter set for BaseCMRFactory models."""
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


# -----------------------------------------------------------------------------
# Reindexing tests
# -----------------------------------------------------------------------------


def test_preserves_recalls_when_item_ids_match_canonical_positions():
    """Behavior: Reindexing keeps canonical item IDs unchanged.

    Given:
      - A single trial with canonical presentation IDs ``[1, 2, 3]``.
    When:
      - The generator reindexes the recall sequence ``[1, 3, 2]``.
    Then:
      - The stored recalls equal the original input.
    Why this matters:
      - Confirms canonical presentations survive preprocessing unchanged.
    """
    # Arrange / Given
    dataset = make_dataset(
        recalls=[[1, 3, 2]],  # canonical item IDs == serial positions
        pres_itemnos=[[1, 2, 3]],
        listLength=3,
    )

    # Act / When
    gen = MemorySearchLikelihoodFnGenerator(BaseCMRFactory, dataset, None)

    # Assert / Then
    expected = jnp.array([[1, 3, 2]], dtype=jnp.int32)
    assert jnp.array_equal(gen.trials, expected).item(), (
        f"Expected positions unchanged {expected.tolist()}, got {gen.trials.tolist()}"
    )


def test_preserves_recalls_when_already_serial_positions():
    """Behavior: Reindexing keeps serial-position recalls unchanged.

    Given:
      - A canonical presentation ``[1, 2, 3]`` with recall positions ``[2, 1, 3]``.
    When:
      - The generator performs its reindexing step.
    Then:
      - The stored recalls match the serial-position input.
    Why this matters:
      - Ensures position-based inputs persist through preprocessing.
    """
    # Arrange / Given
    dataset = make_dataset(
        recalls=[[2, 1, 3]],  # serial positions
        pres_itemnos=[[1, 2, 3]],
        listLength=3,
    )

    # Act / When
    gen = MemorySearchLikelihoodFnGenerator(BaseCMRFactory, dataset, None)

    # Assert / Then
    expected = jnp.array([[2, 1, 3]], dtype=jnp.int32)
    assert jnp.array_equal(gen.trials, expected).item(), (
        f"Expected positions unchanged {expected.tolist()}, got {gen.trials.tolist()}"
    )


# -----------------------------------------------------------------------------
# Likelihood tests
# -----------------------------------------------------------------------------


def test_predict_trial_returns_positive_probabilities_when_single_trial():
    """Behavior: Trial simulation yields probabilistic outcomes.

    Given:
      - One trial with canonical presentations and recalls.
    When:
      - `predict_trial` simulates recall probabilities.
    Then:
      - Each probability lies within ``(0, 1]``.
    Why this matters:
      - Ensures trial-level predictions remain valid probabilities.
    """
    # Arrange / Given
    dataset = make_dataset(
        recalls=[[1, 2, 0]],
        pres_itemnos=[[1, 2, 3]],
        listLength=3,
    )
    gen = MemorySearchLikelihoodFnGenerator(BaseCMRFactory, dataset, None)
    params = _params()

    # Act / When
    probabilities = gen.predict_trial(0, params)

    # Assert / Then
    assert probabilities.shape == (3,)
    assert jnp.all(probabilities > 0).item()
    assert jnp.all(probabilities <= 1).item()


def test_present_and_predict_loss_matches_manual_when_lists_identical():
    """Behavior: Loss helper matches manual log-likelihood.

    Given:
      - Two trials with identical presentation lists.
    When:
      - Loss is computed via the helper and by manual aggregation.
    Then:
      - The results are numerically equivalent.
    Why this matters:
      - Validates the helper after removal of the base path.
    """
    # Arrange / Given
    presents = [[1, 2, 3], [1, 2, 3]]
    recalls = [[1, 2, 0], [2, 3, 0]]
    dataset = make_dataset(recalls, pres_itemnos=presents, listLength=3)
    gen = MemorySearchLikelihoodFnGenerator(BaseCMRFactory, dataset, None)
    params = _params()
    trial_idx = jnp.array([0, 1], dtype=jnp.int32)

    # Act / When
    helper = getattr(gen, "present_and_predict_trials_loss", gen.present_and_predict_trials_loss)
    helper_loss = float(helper(trial_idx, params))
    manual_loss = float(
        -jnp.sum(
            jnp.log(
                jnp.stack(
                    [gen.predict_trial(int(i), params) for i in trial_idx.tolist()],
                    axis=0,
                )
            )
        )
    )

    # Assert / Then
    assert np.isclose(helper_loss, manual_loss), (
        f"Expected helper loss {manual_loss}, got {helper_loss}"
    )


def test_vectorized_and_scalar_loss_agree_when_batching_params():
    """Behavior: Vectorized evaluation equals repeated scalar loss.

    Given:
      - Identical presentations across two trials.
      - Two identical parameter vectors supplied in batch format.
    When:
      - The specialized loss function is called for scalar and batched inputs.
    Then:
      - Each batched entry equals the scalar loss.
    Why this matters:
      - Confirms compatibility with vectorized differential evolution APIs.
    """
    # Arrange / Given
    presents = [[1, 2, 3], [1, 2, 3]]
    recalls = [[1, 2, 0], [2, 3, 0]]
    dataset = make_dataset(recalls, pres_itemnos=presents, listLength=3)
    gen = MemorySearchLikelihoodFnGenerator(BaseCMRFactory, dataset, None)
    params = _params()
    trial_idx = jnp.array([0, 1], dtype=jnp.int32)
    free = ["encoding_drift_rate", "recall_drift_rate"]
    f = gen(trial_idx, base_params=params, free_param_names=free)

    x_single = np.array([params["encoding_drift_rate"], params["recall_drift_rate"]])
    # SciPy's vectorized DE calls expect shape (n_params, n_samples)
    x_batch = np.stack([x_single, x_single], axis=0).T  # (n_params, n_samples)

    # Act / When
    scalar_val = float(f(x_single))
    batched_vals = np.array(f(x_batch))
    # Assert / Then
    assert batched_vals.shape == (2,)
    assert np.allclose(batched_vals, scalar_val)
