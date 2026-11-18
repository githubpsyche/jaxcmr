"""Selection of promising, separated starts from a scored candidate bank."""

import numpy as np

from jaxcmr.typing import Float, Integer

__all__ = ["select_starts"]


def select_starts(
    candidates: Float[np.ndarray, " candidates parameters"],
    scores: Float[np.ndarray, " candidates"],
    count: int,
    min_distance: float = 0.1,
) -> Integer[np.ndarray, " selected_candidates"]:
    """Choose low-loss candidates separated in sampling coordinates.

    Stable score ordering resolves ties in input order. Separation is the root
    mean square coordinate difference; it discourages redundant starts without
    claiming to identify distinct attraction basins.

    Parameters
    ----------
    candidates : Float[ArrayLike, "candidates parameters"]
        Finite unit-interval coordinates, one candidate per row.
    scores : Float[ArrayLike, "candidates"]
        One objective per candidate; nonfinite scores are excluded.
    count : int
        Positive maximum number of starts to select.
    min_distance : float
        Minimum RMS separation, in [0, 1]. Zero selects by score alone.

    Returns
    -------
    Integer[np.ndarray, "selected"]
        Candidate indices in score order. Fewer than count may remain after
        excluding unusable or insufficiently separated candidates.
    """
    candidates, scores = (
        np.asarray(candidates, dtype=float),
        np.asarray(scores, dtype=float),
    )
    selected_indices: list[int] = []
    for candidate_index in np.argsort(
        np.where(np.isfinite(scores), scores, np.inf), kind="stable"
    ):
        if not np.isfinite(scores[candidate_index]):
            break
        if selected_indices:
            selected_distances = np.sqrt(
                np.mean(
                    (candidates[selected_indices] - candidates[candidate_index]) ** 2,
                    axis=1,
                )
            )
            if np.any(selected_distances < min_distance):
                continue
        selected_indices.append(int(candidate_index))
        if len(selected_indices) == count:
            break
    return np.asarray(selected_indices, dtype=int)
