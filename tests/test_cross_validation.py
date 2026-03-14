import numpy as np
import pytest
from jax import numpy as jnp

from jaxcmr.cross_validation import generate_fold_masks


@pytest.mark.parametrize("held_out", [1, 2, 3])
def test_masks_are_disjoint_when_fold_specified(held_out):
    """Behavior: train and test masks have no overlapping True entries.

    Given:
      - A dataset with 3 folds and an all-True trial mask.
    When:
      - ``generate_fold_masks`` is called for a specific fold value.
    Then:
      - No trial index is True in both masks.
    Why this matters:
      - Overlap would leak test data into training, invalidating CV.
    """
    # Arrange / Given
    data = {"fold": np.array([[1], [1], [2], [2], [3], [3]])}
    trial_mask = jnp.ones(6, dtype=bool)

    # Act / When
    train_mask, test_mask = generate_fold_masks(data, "fold", held_out, trial_mask)

    # Assert / Then
    assert not np.any(np.array(train_mask) & np.array(test_mask))


@pytest.mark.parametrize("held_out", [1, 2, 3])
def test_masks_are_exhaustive_when_fold_specified(held_out):
    """Behavior: train and test masks together cover all base-mask trials.

    Given:
      - A dataset with 3 folds and an all-True trial mask.
    When:
      - ``generate_fold_masks`` is called for a specific fold value.
    Then:
      - The union of train and test masks equals the original trial mask.
    Why this matters:
      - Missing trials would discard data and bias the CV estimate.
    """
    # Arrange / Given
    data = {"fold": np.array([[1], [1], [2], [2], [3], [3]])}
    trial_mask = jnp.ones(6, dtype=bool)

    # Act / When
    train_mask, test_mask = generate_fold_masks(data, "fold", held_out, trial_mask)

    # Assert / Then
    np.testing.assert_array_equal(
        np.array(train_mask) | np.array(test_mask), np.array(trial_mask)
    )


def test_test_mask_selects_correct_fold_when_fold_value_given():
    """Behavior: test mask selects exactly the trials matching the fold value.

    Given:
      - A dataset with folds [1, 1, 2, 2, 3, 3] and an all-True trial mask.
    When:
      - ``generate_fold_masks`` is called with fold_value=2.
    Then:
      - Test mask is True only at indices 2 and 3.
    Why this matters:
      - Correct fold assignment is the foundation of the CV split.
    """
    # Arrange / Given
    data = {"fold": np.array([[1], [1], [2], [2], [3], [3]])}
    trial_mask = jnp.ones(6, dtype=bool)

    # Act / When
    _, test_mask = generate_fold_masks(data, "fold", 2, trial_mask)

    # Assert / Then
    np.testing.assert_array_equal(
        np.array(test_mask), [False, False, True, True, False, False]
    )


def test_base_mask_exclusions_propagate_when_trials_excluded():
    """Behavior: excluded trials in the base mask stay excluded in both outputs.

    Given:
      - A dataset with 6 trials and a base mask that excludes trial 0.
    When:
      - ``generate_fold_masks`` is called with fold_value=1.
    Then:
      - Trial 0 is False in both train and test masks.
    Why this matters:
      - Pre-existing exclusions (e.g. bad subjects) must be respected.
    """
    # Arrange / Given
    data = {"fold": np.array([[1], [1], [2], [2], [3], [3]])}
    trial_mask = jnp.array([False, True, True, True, True, True])

    # Act / When
    train_mask, test_mask = generate_fold_masks(data, "fold", 1, trial_mask)

    # Assert / Then
    assert not train_mask[0]
    assert not test_mask[0]
