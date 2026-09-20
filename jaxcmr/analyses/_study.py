"""Shared study-field validity and position-coded recall indicators.

These helpers keep metadata eligibility separate from recall status. A missing
factor measurement does not make an otherwise valid recall disappear.
"""

from jax import numpy as jnp

from ..typing import Array, Bool, Integer, RecallDataset


def study_field(dataset: RecallDataset, name: str) -> Array:
    """Return a field aligned with the presentation matrix."""
    presentations = dataset["pres_itemnos"]
    if presentations.ndim != 2:
        raise ValueError("pres_itemnos must be a trial-by-position array.")
    if name not in dataset:
        raise ValueError(f"Study-factor analysis requires dataset[{name!r}].")
    values = dataset[name]
    if values.shape != presentations.shape:
        raise ValueError(f"Field {name!r} must match the pres_itemnos shape.")
    return values


def study_validity(
    dataset: RecallDataset, valid_field: str | None = None,
) -> Bool[Array, " trials study_positions"]:
    """Exclude padding, positions past listLength, and optional invalid flags."""
    presentations = study_field(dataset, "pres_itemnos")
    lengths = dataset["listLength"].reshape(-1)
    if lengths.size != presentations.shape[0]:
        raise ValueError("listLength must have one value per trial.")
    valid = (presentations > 0) & (
        jnp.arange(presentations.shape[1])[None, :] < lengths[:, None]
    )
    if valid_field is not None:
        flags = study_field(dataset, valid_field)
        valid = valid & jnp.isfinite(flags) & (flags > 0)
    return valid


def factor_inputs(
    dataset: RecallDataset,
    category_field: str,
    factor_field: str,
    valid_field: str | None,
) -> tuple[Array, Array, Array]:
    """Return factor values, categories, and study eligibility."""
    return (
        study_field(dataset, factor_field),
        study_field(dataset, category_field),
        study_validity(dataset, valid_field),
    )


def recalled_positions(
    recalls: Integer[Array, " trials recall_events"],
    position_count: int,
) -> Bool[Array, " trials study_positions"]:
    """Mark direct positional hits once, ignoring zero/invalid recall codes."""
    if recalls.ndim != 2:
        raise ValueError("recalls must be a trial-by-recall array.")
    return jnp.any(recalls[..., None] == jnp.arange(1, position_count + 1), axis=1)


def recall_status(
    dataset: RecallDataset, *, expand_repetitions: bool = False,
) -> Bool[Array, " trials study_positions"]:
    """Mark recalled positions, optionally expanding to the item's occurrences.

    Recalls always encode study positions, not item IDs. Expansion first finds
    the directly recalled positions and then marks valid positions with the
    same presentation item ID. Repeated responses never count more than once.
    The expansion choice is static under JIT. Factor/quality missingness does
    not affect these flags; it is applied separately when averaging values.
    """
    presentations = study_field(dataset, "pres_itemnos")
    recalls = dataset["recalls"]
    if recalls.ndim != 2 or recalls.shape[0] != presentations.shape[0]:
        raise ValueError("recalls must have one row per presentation trial.")
    valid = study_validity(dataset)
    hits = recalled_positions(recalls, presentations.shape[1]) & valid
    if expand_repetitions:
        same_item = presentations[:, :, None] == presentations[:, None, :]
        hits = jnp.any(same_item & hits[:, None, :], axis=2) & valid
    return hits


def validate_trial_mask(dataset: RecallDataset, mask: Array) -> None:
    if mask.shape != (dataset["pres_itemnos"].shape[0],) or mask.dtype != jnp.bool_:
        raise ValueError("trial_mask must be a Boolean vector with one entry per trial.")
