import jax.numpy as jnp
import pytest

from jaxcmr.repetition import (
    item_to_study_positions,
    all_study_positions,
    filter_repeated_recalls,
    relabel_trial_to_firstpos,
    make_control_dataset,
)
from jaxcmr.helpers import has_repeats_per_row


# -----------------------------------------------------------------------------
# item_to_study_positions
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "item,presentation,size,expected",
    [
        (5, jnp.array([5, 6, 5, 7], dtype=jnp.int32), 3, [1, 3, 0]),
        (0, jnp.array([5, 6, 5, 7], dtype=jnp.int32), 2, [0, 0]),
        (8, jnp.array([5, 6, 5, 7], dtype=jnp.int32), 2, [0, 0]),
    ],
)
def test_returns_positions_when_item_repeats(item, presentation, size, expected):
    """Behavior: Return one-indexed study positions of the item.

    Given:
      - A 1D presentation with possible item repetitions.
    When:
      - Querying positions for a specific item.
    Then:
      - The returned vector lists positions (padded with 0s).
    Why this matters:
      - Enables mapping recalls to all valid study positions.
    """
    # Arrange / Given is parameterized

    # Act / When
    result = item_to_study_positions(item, presentation, size)

    # Assert / Then
    assert result.dtype in [jnp.int32, jnp.int64]
    assert result.tolist() == expected


# -----------------------------------------------------------------------------
# all_study_positions
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "study_position,presentation,size,expected",
    [
        (3, jnp.array([5, 6, 5, 7], dtype=jnp.int32), 3, [1, 3, 0]),  # item at pos 3 is 5
        (0, jnp.array([5, 6, 5, 7], dtype=jnp.int32), 2, [0, 0]),
    ],
)
def test_maps_from_position_to_all_positions(study_position, presentation, size, expected):
    """Behavior: Return all positions for the item shown at the study position.

    Given:
      - A 1D presentation with possible repeated items.
    When:
      - Converting a study position to all valid positions for that item.
    Then:
      - The result lists the item's positions, padded with 0s.
    Why this matters:
      - Supports analyses that treat repeats as belonging to the same item.
    """
    # Arrange / Given is parameterized

    # Act / When
    result = all_study_positions(study_position, presentation, size)

    # Assert / Then
    assert result.tolist() == expected


# -----------------------------------------------------------------------------
# filter_repeated_recalls
# -----------------------------------------------------------------------------


def test_keeps_only_first_occurrence_within_each_trial():
    """Behavior: Set repeated recalls to 0 while retaining first occurrences.

    Given:
      - A 2D recall matrix with repeated nonzero values per row.
    When:
      - Filtering repeated recalls.
    Then:
      - Only the first occurrence of each item remains nonzero in each row.
    Why this matters:
      - Downstream analyses should not double-count repeated recalls.
    """
    # Arrange / Given
    recalls = jnp.array(
        [
            [1, 2, 2, 3, 1, 0],  # repeats of 2 and 1
            [4, 4, 4, 0, 0, 0],  # repeats of 4
        ],
        dtype=jnp.int32,
    )

    # Act / When
    filtered = filter_repeated_recalls(recalls)

    # Assert / Then
    expected = jnp.array(
        [
            [1, 2, 0, 3, 0, 0],
            [4, 0, 0, 0, 0, 0],
        ],
        dtype=jnp.int32,
    )
    assert jnp.array_equal(filtered, expected).item()


# -----------------------------------------------------------------------------
# relabel_trial_to_firstpos
# -----------------------------------------------------------------------------


def test_relabels_recalls_to_first_occurrence_in_presentation():
    """Behavior: Map recall positions to the first occurrence of that item.

    Given:
      - A presentation row with an item that repeats.
      - A recall row referencing positions in that presentation.
    When:
      - Relabeling to the first occurrence positions.
    Then:
      - Each recalled position is replaced by the item's first study position.
    Why this matters:
      - Control analyses treat duplicates as a single item with one canonical position.
    """
    # Arrange / Given
    pres_row = jnp.array([7, 8, 7, 9], dtype=jnp.int32)
    rec_row = jnp.array([3, 1, 0, 4], dtype=jnp.int32)

    # Act / When
    relabeled = relabel_trial_to_firstpos(rec_row, pres_row)

    # Assert / Then
    assert relabeled.tolist() == [1, 1, 0, 4]


# -----------------------------------------------------------------------------
# make_control_dataset
# -----------------------------------------------------------------------------


def _toy_dataset_for_controls():
    # Two subjects; each has one pure (list_type=0) and one mixed (list_type=1) trial
    L = 4
    subject = jnp.array([[1], [1], [2], [2]], dtype=jnp.int32)
    listLength = jnp.array([[L], [L], [L], [L]], dtype=jnp.int32)

    # Pure trials (unique items), Mixed trials (item 1 repeats)
    pres_itemnos = jnp.array(
        [
            [1, 2, 3, 4],  # subj 1 pure
            [1, 2, 1, 4],  # subj 1 mixed (1 repeats)
            [1, 2, 3, 4],  # subj 2 pure
            [2, 1, 2, 3],  # subj 2 mixed (2 repeats)
        ],
        dtype=jnp.int32,
    )

    # Recalls (include an intra-trial repeat to exercise remove_repeats)
    recalls = jnp.array(
        [
            [1, 2, 3, 0],  # subj 1 pure
            [3, 1, 1, 0],  # subj 1 mixed
            [1, 1, 2, 0],  # subj 2 pure (repeat 1)
            [2, 3, 0, 0],  # subj 2 mixed
        ],
        dtype=jnp.int32,
    )

    list_type = jnp.array([[0], [1], [0], [1]], dtype=jnp.int32)

    return {
        "subject": subject,
        "listLength": listLength,
        "pres_itemnos": pres_itemnos,
        "recalls": recalls,
        "list_type": list_type,
    }


def test_builds_rows_aligned_to_mixed_trials_per_subject():
    """Behavior: Construct a shuffled-control dataset aligned to mixed presentations.

    Given:
      - A dataset where each subject has both mixed and pure trials.
    When:
      - Building a control dataset with one shuffle per subject.
    Then:
      - Output contains aligned ``recalls``/``pres_itemnos`` with consistent shapes.
    Why this matters:
      - Ensures control construction is structurally sound for analysis.
    """
    # Arrange / Given
    data = _toy_dataset_for_controls()

    # Act / When
    ctrl = make_control_dataset(
        data=data,
        mixed_query="data['list_type'].flatten() == 1",
        control_query="data['list_type'].flatten() == 0",
        n_shuffles=1,
        remove_repeats=True,
        seed=0,
    )

    # Assert / Then
    n_rows = int(ctrl["recalls"].shape[0])
    assert n_rows > 0
    assert ctrl["recalls"].shape[0] == ctrl["pres_itemnos"].shape[0]
    assert ctrl["subject"].shape[0] == ctrl["recalls"].shape[0]


@pytest.mark.parametrize("remove", [True, False])
def test_toggles_repeated_recalls_removal(remove: bool):
    """Behavior: ``remove_repeats`` controls whether within-trial repeats are zeroed.

    Given:
      - A dataset containing recall rows with repeated items.
    When:
      - Building the control dataset with and without repeat removal.
    Then:
      - Rows contain no repeats only when removal is enabled.
    Why this matters:
      - Confirms that the flag enforces a clear API guarantee.
    """
    # Arrange / Given
    data = _toy_dataset_for_controls()

    # Act / When
    ctrl = make_control_dataset(
        data=data,
        mixed_query="data['list_type'].flatten() == 1",
        control_query="data['list_type'].flatten() == 0",
        n_shuffles=2,
        remove_repeats=remove,
        seed=0,
    )

    # Assert / Then
    has_repeats = has_repeats_per_row(ctrl["recalls"]).any().item()
    assert has_repeats is (not remove)
