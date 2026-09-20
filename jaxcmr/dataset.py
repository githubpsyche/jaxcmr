"""Operations that preserve or replace trial-indexed recall datasets."""

from collections.abc import Callable, Collection

from jaxcmr.typing import Array, Integer, RecallDataset

_REQUIRED_FIELDS = frozenset({"subject", "listLength", "pres_itemnos", "recalls"})
_RECALL_EVENT_FIELDS = frozenset(
    {"rec_itemids", "irt", "cue_clips", "_should_tabulate"}
)


def _validate_trial_fields(data: RecallDataset) -> None:
    """Require each dataset array to have one row per recall trial."""
    trial_count = data["recalls"].shape[0]
    for field, values in data.items():
        if values.ndim == 0 or values.shape[0] != trial_count:
            raise ValueError(
                f"RecallDataset field {field!r} must have {trial_count} trial rows."
            )


def _transform_trial_rows(
    data: RecallDataset, transform: Callable[[Array], Array]
) -> RecallDataset:
    """Transform every trial array while retaining the required keys."""
    _validate_trial_fields(data)
    transformed = data.copy()
    for field, values in data.items():
        transformed[field] = transform(values)
    return transformed


def _invalidated_recall_fields(additional_fields: Collection[str]) -> set[str]:
    """Fields that cannot describe a newly generated recall sequence."""
    additional = set(additional_fields)
    if invalid := additional & _REQUIRED_FIELDS:
        raise ValueError(
            f"Required RecallDataset fields cannot be invalidated: {sorted(invalid)}."
        )
    return set(_RECALL_EVENT_FIELDS) | additional


def _replace_recalls(
    data: RecallDataset,
    recalls: Integer[Array, "n_trials num_recalled"],
    additional_recall_fields: Collection[str] = (),
) -> RecallDataset:
    """Set generated recalls and remove metadata tied to the old recall events."""
    trial_count = data["recalls"].shape[0]
    if recalls.shape[0] != trial_count:
        raise ValueError(f"Generated recalls must have {trial_count} trial rows.")
    replaced = data.copy()
    for field in _invalidated_recall_fields(additional_recall_fields):
        replaced.pop(field, None)
    replaced["recalls"] = recalls
    return replaced
