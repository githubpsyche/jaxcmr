import jax.numpy as jnp

from jaxcmr.helpers import (
    have_common_nonzero,
    all_rows_identical,
    log_likelihood,
    format_floats,
    find_max_list_length,
    apply_by_subject,
    has_repeats_per_row,
    make_dataset,
    limit_to_first_subjects,
)


def test_have_common_nonzero_returns_true_when_overlap_exists():
    """Behavior: ``have_common_nonzero`` detects shared nonzero values.

    Given:
      - Two arrays sharing a nonzero value.
    When:
      - ``have_common_nonzero`` is called.
    Then:
      - Returns True.
    Why this matters:
      - Used to detect repeated items across study positions.
    """
    # Arrange / Given
    a = jnp.array([1, 2, 0])
    b = jnp.array([0, 2, 3])

    # Act / When
    result = have_common_nonzero(a, b)

    # Assert / Then
    assert bool(result) is True


def test_have_common_nonzero_returns_false_when_only_zeros_overlap():
    """Behavior: ``have_common_nonzero`` ignores zeros.

    Given:
      - Two arrays that only share the value 0.
    When:
      - ``have_common_nonzero`` is called.
    Then:
      - Returns False.
    Why this matters:
      - Zero is padding and should not count as a shared item.
    """
    # Arrange / Given
    a = jnp.array([0, 1, 2])
    b = jnp.array([0, 3, 4])

    # Act / When
    result = have_common_nonzero(a, b)

    # Assert / Then
    assert bool(result) is False


def test_all_rows_identical_returns_true_for_uniform_array():
    """Behavior: ``all_rows_identical`` detects uniform rows.

    Given:
      - A 2D array where every row is the same.
    When:
      - ``all_rows_identical`` is called.
    Then:
      - Returns True.
    Why this matters:
      - Used to check if presentation lists are homogeneous.
    """
    # Arrange / Given
    arr = jnp.array([[1, 2, 3], [1, 2, 3]])

    # Act / When
    result = all_rows_identical(arr)

    # Assert / Then
    assert result is True


def test_all_rows_identical_returns_false_for_different_rows():
    """Behavior: ``all_rows_identical`` detects heterogeneous rows.

    Given:
      - A 2D array with differing rows.
    When:
      - ``all_rows_identical`` is called.
    Then:
      - Returns False.
    Why this matters:
      - Distinguishes fixed from variable presentation orders.
    """
    # Arrange / Given
    arr = jnp.array([[1, 2, 3], [3, 2, 1]])

    # Act / When
    result = all_rows_identical(arr)

    # Assert / Then
    assert result is False


def test_log_likelihood_returns_negative_sum_of_logs():
    """Behavior: ``log_likelihood`` computes negative summed log.

    Given:
      - A vector of probability values.
    When:
      - ``log_likelihood`` is called.
    Then:
      - The result equals ``-sum(log(values))``.
    Why this matters:
      - Core loss computation for model fitting.
    """
    # Arrange / Given
    probs = jnp.array([0.5, 0.25])

    # Act / When
    result = log_likelihood(probs)

    # Assert / Then
    expected = -jnp.sum(jnp.log(probs))
    assert jnp.isclose(result, expected).item()


def test_format_floats_rounds_to_precision():
    """Behavior: ``format_floats`` formats numbers to specified decimal places.

    Given:
      - A list of floats and precision = 2.
    When:
      - ``format_floats`` is called.
    Then:
      - Each float is formatted as a string with 2 decimal places.
    Why this matters:
      - Used for display in summary tables.
    """
    # Arrange / Given
    values = [1.234, 5.678]

    # Act / When
    result = format_floats(values, precision=2)

    # Assert / Then
    assert result == ["1.23", "5.68"]


def test_find_max_list_length_returns_maximum_across_datasets():
    """Behavior: ``find_max_list_length`` finds the highest list length.

    Given:
      - Two datasets with different list lengths and full masks.
    When:
      - ``find_max_list_length`` is called.
    Then:
      - Returns the maximum list length.
    Why this matters:
      - Determines output array dimensions for plotting.
    """
    # Arrange / Given
    ds1 = make_dataset(recalls=jnp.array([[1, 0, 0]]), listLength=3)
    ds2 = make_dataset(recalls=jnp.array([[1, 2, 0, 0]]), listLength=4)
    masks = [jnp.array([True]), jnp.array([True])]

    # Act / When
    result = find_max_list_length([ds1, ds2], masks)

    # Assert / Then
    assert result == 4


def test_apply_by_subject_returns_one_result_per_subject():
    """Behavior: ``apply_by_subject`` splits data by subject.

    Given:
      - A dataset with two subjects.
    When:
      - ``apply_by_subject`` is called with a simple function.
    Then:
      - The function is called once per subject.
    Why this matters:
      - Per-subject analysis is the primary aggregation pattern.
    """
    # Arrange / Given
    ds = make_dataset(
        recalls=jnp.array([[1, 0], [2, 0]]),
        subject=jnp.array([[1], [2]]),
    )
    mask = jnp.array([True, True])

    # Act / When
    results = apply_by_subject(ds, mask, lambda d: jnp.array(d["recalls"].shape[0]))

    # Assert / Then
    assert len(results) == 2
    assert all(int(r) == 1 for r in results)


def test_apply_by_subject_respects_trial_mask():
    """Behavior: ``apply_by_subject`` filters by mask before splitting.

    Given:
      - A dataset with one subject and a mask excluding one trial.
    When:
      - ``apply_by_subject`` is called.
    Then:
      - Only the masked trial reaches the function.
    Why this matters:
      - Ensures condition filtering works with subject splitting.
    """
    # Arrange / Given
    ds = make_dataset(
        recalls=jnp.array([[1, 0], [2, 0]]),
        subject=jnp.array([[1], [1]]),
    )
    mask = jnp.array([True, False])

    # Act / When
    results = apply_by_subject(ds, mask, lambda d: jnp.array(d["recalls"].shape[0]))

    # Assert / Then
    assert len(results) == 1
    assert int(results[0]) == 1


def test_has_repeats_per_row_flags_repeated_nonzero():
    """Behavior: ``has_repeats_per_row`` detects repeated nonzero values.

    Given:
      - Rows with and without repeated nonzero values.
    When:
      - ``has_repeats_per_row`` is called.
    Then:
      - Only the row with repeats is flagged True.
    Why this matters:
      - Used to filter trials with repeated item recalls.
    """
    # Arrange / Given
    arr = jnp.array([[1, 2, 1], [1, 2, 3]], dtype=jnp.int32)

    # Act / When
    result = has_repeats_per_row(arr)

    # Assert / Then
    assert result[0].item() is True
    assert result[1].item() is False


def test_has_repeats_per_row_ignores_repeated_zeros():
    """Behavior: ``has_repeats_per_row`` treats zeros as padding.

    Given:
      - A row with repeated zeros but no repeated nonzero values.
    When:
      - ``has_repeats_per_row`` is called.
    Then:
      - Returns False for that row.
    Why this matters:
      - Zero is padding, not a repeated recall.
    """
    # Arrange / Given
    arr = jnp.array([[1, 0, 0]], dtype=jnp.int32)

    # Act / When
    result = has_repeats_per_row(arr)

    # Assert / Then
    assert result[0].item() is False


def test_make_dataset_minimal_call():
    """Behavior: ``make_dataset`` constructs a dataset from recalls alone.

    Given:
      - A single-trial recalls array.
    When:
      - ``make_dataset`` is called with only recalls.
    Then:
      - All required fields are populated with consistent shapes.
    Why this matters:
      - Validates the most common test helper usage pattern.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 2, 0]])

    # Act / When
    ds = make_dataset(recalls)

    # Assert / Then
    assert "recalls" in ds
    assert "pres_itemnos" in ds
    assert "listLength" in ds
    assert "subject" in ds
    assert ds["recalls"].shape == (1, 3)
    assert ds["pres_itemnos"].shape[0] == 1


def test_make_dataset_with_custom_subjects():
    """Behavior: ``make_dataset`` accepts custom subject identifiers.

    Given:
      - Recalls and a multi-subject array.
    When:
      - ``make_dataset`` is called with custom subjects.
    Then:
      - Subject identifiers are preserved.
    Why this matters:
      - Per-subject analyses depend on correct subject labeling.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 0], [2, 0]])
    subjects = jnp.array([[5], [10]])

    # Act / When
    ds = make_dataset(recalls, subject=subjects)

    # Assert / Then
    assert jnp.array_equal(ds["subject"], subjects).item()


def test_make_dataset_reps_tiles_trials():
    """Behavior: ``make_dataset`` tiles trials when ``reps > 1``.

    Given:
      - A single trial with reps = 3.
    When:
      - ``make_dataset`` is called.
    Then:
      - The output has 3 copies of the trial.
    Why this matters:
      - Supports bootstrapped or repeated-measure dataset construction.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 2, 0]])

    # Act / When
    ds = make_dataset(recalls, reps=3)

    # Assert / Then
    assert ds["recalls"].shape[0] == 3
    assert jnp.array_equal(ds["recalls"][0], ds["recalls"][2]).item()


def test_limit_to_first_subjects_restricts_output():
    """Behavior: ``limit_to_first_subjects`` keeps only the first N subjects.

    Given:
      - A dataset with 3 subjects.
    When:
      - ``limit_to_first_subjects`` is called with max_subjects = 2.
    Then:
      - Only trials from the first 2 subjects remain.
    Why this matters:
      - Used to limit dataset size for quick debugging.
    """
    # Arrange / Given
    ds = make_dataset(
        recalls=jnp.array([[1, 0], [2, 0], [3, 0]]),
        subject=jnp.array([[1], [2], [3]]),
    )

    # Act / When
    limited = limit_to_first_subjects(ds, max_subjects=2)

    # Assert / Then
    assert limited["recalls"].shape[0] == 2
    assert jnp.max(limited["subject"]).item() <= 2
