import numpy as np

from jaxcmr.fitting import make_subject_trial_masks


def test_single_subject_returns_one_mask():
    """Behavior: ``make_subject_trial_masks`` returns one mask for a single subject.

    Given:
      - A trial mask of all True and a subject vector with one unique subject.
    When:
      - ``make_subject_trial_masks`` is called.
    Then:
      - Exactly one mask is returned, and it matches the input trial mask.
    Why this matters:
      - The single-subject case is the simplest partition; the mask
        should pass through unchanged.
    """
    # Arrange / Given
    trial_mask = np.array([True, True, True])
    subject_vector = np.array([1, 1, 1])

    # Act / When
    masks, unique_subjects = make_subject_trial_masks(trial_mask, subject_vector)  # type: ignore[arg-type]

    # Assert / Then
    assert len(masks) == 1
    assert len(unique_subjects) == 1
    np.testing.assert_array_equal(masks[0], [True, True, True])


def test_two_subjects_partition_trials():
    """Behavior: ``make_subject_trial_masks`` partitions trials by subject.

    Given:
      - A trial mask of all True and a subject vector with two subjects.
    When:
      - ``make_subject_trial_masks`` is called.
    Then:
      - Two masks are returned, each selecting only its subject's trials,
        and their union covers all trials.
    Why this matters:
      - Per-subject fitting depends on correct partitioning so each
        subject's data is isolated.
    """
    # Arrange / Given
    trial_mask = np.array([True, True, True, True])
    subject_vector = np.array([1, 1, 2, 2])

    # Act / When
    masks, unique_subjects = make_subject_trial_masks(trial_mask, subject_vector)  # type: ignore[arg-type]

    # Assert / Then
    assert len(masks) == 2
    np.testing.assert_array_equal(unique_subjects, [1, 2])
    np.testing.assert_array_equal(masks[0], [True, True, False, False])
    np.testing.assert_array_equal(masks[1], [False, False, True, True])


def test_false_trials_excluded_from_all_masks():
    """Behavior: ``make_subject_trial_masks`` respects False in the trial mask.

    Given:
      - A trial mask with some False entries and two subjects.
    When:
      - ``make_subject_trial_masks`` is called.
    Then:
      - Trials marked False are excluded from every subject mask.
    Why this matters:
      - Condition filtering via the trial mask must propagate into
        per-subject masks to avoid fitting on excluded trials.
    """
    # Arrange / Given
    trial_mask = np.array([True, False, True, False])
    subject_vector = np.array([1, 1, 2, 2])

    # Act / When
    masks, unique_subjects = make_subject_trial_masks(trial_mask, subject_vector)  # type: ignore[arg-type]

    # Assert / Then
    assert len(masks) == 2
    np.testing.assert_array_equal(masks[0], [True, False, False, False])
    np.testing.assert_array_equal(masks[1], [False, False, True, False])
