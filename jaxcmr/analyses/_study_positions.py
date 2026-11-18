"""Internal trial-by-study-position masks for measurement analyses.

A study position is a presentation slot; recall codes name these positions.
Within-list item codes in ``pres_itemnos`` identify repeated presentations of
one item. These are distinct from global vocabulary IDs in ``pres_itemids``.

Presentation inclusion and recall classification are separate calculations.
Missing measurements or quality failures exclude values from an average, but
must not change whether their item was recalled. Callers supply aligned arrays;
these functions compute masks rather than validate the dataset.
"""

from jax import numpy as jnp

from ..typing import Array, Bool, Integer, RecallDataset


def study_position_mask(
    dataset: RecallDataset,
    valid_field: str | None = None,
) -> Bool[Array, " trials study_positions"]:
    """Return True at eligible presentation slots in each trial.

    Include the first ``listLength`` nonzero presentations, skipping empty slots.
    When ``valid_field`` names an aligned quality array, also require its flag
    to be finite and positive. Measurement values and recalls are not examined.
    Callers apply category selection and exclude nonfinite measurements later.

    For presentations [7, 2, 7, 0] and length 3, return [True, True, True, False].
    A quality flag of zero at position 3 would additionally exclude that slot.
    Under JIT, ``valid_field`` must be static.
    """
    presentations = dataset["pres_itemnos"]
    lengths = dataset["listLength"].reshape(-1)
    presented = presentations > 0
    included = presented & (jnp.cumsum(presented, axis=1) <= lengths[:, None])
    if valid_field is not None:
        flags = dataset[valid_field]
        included = included & jnp.isfinite(flags) & (flags > 0)
    return included


def last_study_position(
    dataset: RecallDataset, trial_mask: Bool[Array, " trials"]
) -> int:
    """Last occupied study slot across the selected trials, counting from one."""
    included = study_position_mask(dataset) & trial_mask[:, None]
    positions = jnp.arange(1, included.shape[1] + 1)
    return int(jnp.max(jnp.where(included.any(axis=0), positions, 0), initial=0))


def recalled_positions(
    recalls: Integer[Array, " trials recall_events"],
    position_count: int,
) -> Bool[Array, " trials study_positions"]:
    """Mark positions named by at least one response, once per trial.

    Positive recall codes are one-based study positions. For recalls [1, 1, 0]
    and ``position_count=4``, return [True, False, False, False]. Codes outside
    1..position_count match nothing; repeated responses do not add counts.

    This array operation knows the storage width, not which positions were
    actually presented. ``recall_status`` additionally excludes unused slots
    using the dataset. Under JIT, ``position_count`` must be static.
    """
    study_positions = jnp.arange(1, position_count + 1)
    response_matches_position = recalls[..., None] == study_positions
    # Compare every response to every study position, then reduce responses.
    return jnp.any(response_matches_position, axis=1)


def recall_status(dataset: RecallDataset) -> Bool[Array, " trials study_positions"]:
    """Mark all eligible presentations of every recalled item, once per trial.

    Response codes address study positions, not individual remembered
    occurrences. For presentations [7, 2, 7, 0], length 3 and recalls [1, 1, 0],
    return [True, False, True, False]. A response at position 3 gives the same
    result. Padding and invalid response positions add no recalled items.
    Measurement values and quality flags do not affect item identity.
    """
    presentations = dataset["pres_itemnos"]
    included = study_position_mask(dataset)
    direct_recall = (
        recalled_positions(dataset["recalls"], presentations.shape[1]) & included
    )
    same_item_at_positions = presentations[:, :, None] == presentations[:, None, :]
    recalled_occurrences = same_item_at_positions & direct_recall[:, None, :]
    return jnp.any(recalled_occurrences, axis=2) & included
