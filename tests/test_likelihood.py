import numpy as np
import jax.numpy as jnp

from jaxcmr.likelihood import MemorySearchLikelihoodFnGenerator
from jaxcmr.models_repfr.cmr import BaseCMRFactory
from jaxcmr.typing import RecallDataset


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _make_dataset(presents: list[list[int]], recalls: list[list[int]], list_length: int) -> RecallDataset:
    """Returns a minimal RecallDataset-like dict for likelihood tests."""
    n_trials = len(presents)
    return {
        "pres_itemnos": jnp.array(presents, dtype=jnp.int32),
        "recalls": jnp.array(recalls, dtype=jnp.int32),
        "listLength": jnp.array([list_length] * n_trials, dtype=jnp.int32),
        # Subject vector is unused by the likelihood generator but present for completeness
        "subject": jnp.arange(n_trials, dtype=jnp.int32).reshape(-1, 1),
    } # type: ignore


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
        # Optional for cmr.CMR, but include to be explicit across variants
        "allow_repeated_recalls": False,
    }


# -----------------------------------------------------------------------------
# Reindexing tests
# -----------------------------------------------------------------------------


def test_preserves_recalls_when_item_ids_match_canonical_positions():
    """Behavior: Recalls remain unchanged when item IDs are canonical positions.

    Given:
      - A presentation that assigns item IDs canonically: ``[1, 2, 3]``.
      - Recalls provided as those item IDs ``[1, 3, 2]`` (i.e., serial positions).
    When:
      - The likelihood generator initializes and performs its reindex step.
    Then:
      - Recalls are unchanged: ``[1, 3, 2]``.
    Why this matters:
      - Presentations follow the invariant (start at 1, then 2, ... with repeats allowed);
        thus item IDs equal serial positions and reindexing is a no-op.
    """
    # Arrange / Given
    dataset = _make_dataset(
        presents=[[1, 2, 3]],
        recalls=[[1, 3, 2]],  # canonical item IDs == serial positions
        list_length=3,
    )

    # Act / When
    gen = MemorySearchLikelihoodFnGenerator(BaseCMRFactory, dataset, connections=None)

    # Assert / Then
    expected = jnp.array([[1, 3, 2]], dtype=jnp.int32)
    assert jnp.array_equal(gen.trials, expected).item(), (
        f"Expected positions unchanged {expected.tolist()}, got {gen.trials.tolist()}"
    )


def test_preserves_recalls_when_already_serial_positions():
    """Behavior: Keep recalls unchanged when already serial positions.

    Given:
      - A trial with canonical presentation ``[1, 2, 3]`` and recalls by position ``[2, 1, 3]``.
    When:
      - The likelihood generator initializes.
    Then:
      - Recalls remain ``[2, 1, 3]``.
    Why this matters:
      - Ensures the reindexing step is a no-op when input is positions.
    """
    # Arrange / Given
    dataset = _make_dataset(
        presents=[[1, 2, 3]],
        recalls=[[2, 1, 3]],  # serial positions
        list_length=3,
    )

    # Act / When
    gen = MemorySearchLikelihoodFnGenerator(BaseCMRFactory, dataset, connections=None)

    # Assert / Then
    expected = jnp.array([[2, 1, 3]], dtype=jnp.int32)
    assert jnp.array_equal(gen.trials, expected).item(), (
        f"Expected positions unchanged {expected.tolist()}, got {gen.trials.tolist()}"
    )


# -----------------------------------------------------------------------------
# Path selection and equivalence tests
# -----------------------------------------------------------------------------


def test_base_equals_present_and_predict_when_presentations_identical():
    """Behavior: Base and present-and-predict losses are equal when lists match.

    Given:
      - Two trials with identical presentation lists.
    When:
      - Loss is computed via both base and present-and-predict paths.
    Then:
      - The negative log-likelihoods match.
    Why this matters:
      - Confirms branch equivalence under identical present lists.
    """
    # Arrange / Given
    presents = [[1, 2, 3], [1, 2, 3]]
    # Serial-position recalls (1-based)
    recalls = [[1, 2, 0], [2, 3, 0]]
    dataset = _make_dataset(presents, recalls, list_length=3)
    gen = MemorySearchLikelihoodFnGenerator(BaseCMRFactory, dataset, None)
    params = _params()
    idx = jnp.array([0, 1], dtype=jnp.int32)

    # Act / When
    base_ll = gen.base_predict_trials_loss(idx, params)
    pap_ll = gen.present_and_predict_trials_loss(idx, params)

    # Assert / Then
    assert jnp.allclose(base_ll, pap_ll).item(), (
        f"Expected equal losses, base={float(base_ll)}, present+predict={float(pap_ll)}"
    )


def test_call_uses_base_path_when_presentations_identical():
    """Behavior: __call__ selects base path when present lists match.

    Given:
      - Two trials with identical presentations and a free parameter.
    When:
      - The returned loss function is evaluated.
    Then:
      - Its value equals ``base_predict_trials_loss``.
    Why this matters:
      - Verifies branch selection in the loss generator.
    """
    # Arrange / Given
    presents = [[1, 2, 3], [1, 2, 3]]
    recalls = [[1, 2, 0], [2, 3, 0]]
    dataset = _make_dataset(presents, recalls, list_length=3)
    gen = MemorySearchLikelihoodFnGenerator(BaseCMRFactory, dataset, None)
    params = _params()
    trial_idx = jnp.array([0, 1], dtype=jnp.int32)
    free = ["encoding_drift_rate"]
    f = gen(trial_idx, base_params=params, free_params=free)

    # Act / When
    val = f(np.array([params["encoding_drift_rate"]], dtype=float))
    expected = gen.base_predict_trials_loss(trial_idx, params)

    # Assert / Then
    assert np.isclose(float(val), float(expected)), (
        f"Expected base path loss {float(expected)}, got {float(val)}"
    )


def test_call_uses_present_and_predict_when_presentations_differ():
    """Behavior: __call__ selects present-and-predict when lists differ.

    Given:
      - Two trials with differing presentations and a free parameter.
    When:
      - The returned loss function is evaluated.
    Then:
      - Its value equals ``present_and_predict_trials_loss``.
    Why this matters:
      - Ensures correct branch selection for non-identical lists.
    """
    # Arrange / Given
    presents = [[1, 2, 3], [1, 2, 1]]  # differing but canonical
    recalls = [[1, 3, 0], [3, 1, 0]]  # positions per trial
    dataset = _make_dataset(presents, recalls, list_length=3)
    gen = MemorySearchLikelihoodFnGenerator(BaseCMRFactory, dataset, None)
    params = _params()
    trial_idx = jnp.array([0, 1], dtype=jnp.int32)
    free = ["encoding_drift_rate"]
    f = gen(trial_idx, base_params=params, free_params=free)

    # Act / When
    val = f(np.array([params["encoding_drift_rate"]], dtype=float))
    expected = gen.present_and_predict_trials_loss(trial_idx, params)

    # Assert / Then
    assert np.isclose(float(val), float(expected)), (
        f"Expected present+predict loss {float(expected)}, got {float(val)}"
    )


def test_vectorized_and_scalar_loss_agree_when_batching_params():
    """Behavior: Batched loss equals repeated scalar loss.

    Given:
      - Identical presentations across trials and two identical param vectors.
    When:
      - The loss function is evaluated for a batch and a scalar input.
    Then:
      - Each batch entry equals the scalar loss value.
    Why this matters:
      - Validates the vectorized DE interface (`vectorized=True`).
    """
    # Arrange / Given
    presents = [[1, 2, 3], [1, 2, 3]]
    recalls = [[1, 2, 0], [2, 3, 0]]
    dataset = _make_dataset(presents, recalls, list_length=3)
    gen = MemorySearchLikelihoodFnGenerator(BaseCMRFactory, dataset, None)
    params = _params()
    trial_idx = jnp.array([0, 1], dtype=jnp.int32)
    free = ["encoding_drift_rate", "recall_drift_rate"]
    f = gen(trial_idx, base_params=params, free_params=free)

    x_single = np.array([params["encoding_drift_rate"], params["recall_drift_rate"]])
    # SciPy's vectorized DE calls expect shape (n_params, n_samples)
    x_batch = np.stack([x_single, x_single], axis=0).T  # (n_params, n_samples)

    # Act / When
    scalar_val = float(f(x_single))
    batched_vals = np.array(f(x_batch))

    # Assert / Then
    assert batched_vals.shape == (2,)
    assert np.allclose(batched_vals, scalar_val)
