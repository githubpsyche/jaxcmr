"""Selection of promising, separated starts from a scored candidate bank."""

import numpy as np

__all__ = ["select_starts"]


def select_starts(
    candidates: np.ndarray,
    scores: np.ndarray,
    count: int,
    min_distance: float = 0.1,
) -> np.ndarray:
    """Return up to ``count`` candidate indices, ordered by increasing score.

    Candidates have shape (candidates, parameters) in unit-interval fitting
    coordinates. Greedy selection keeps a point only if its root-mean-square
    coordinate distance from every selected point is at least ``min_distance``.
    This discourages redundant starts; it does not identify attraction basins.
    Nonfinite scores are excluded. Set ``min_distance=0`` to use score alone.

    Args:
        candidates: Finite unit-interval coordinates, with one candidate per row.
        scores: One loss per candidate; lower is better.
        count: Positive maximum number of starts to select.
        min_distance: Minimum RMS separation in these coordinates, in [0, 1].

    Returns:
        Candidate row indices in score order, resolving equal scores in input
        order. May contain fewer than count entries if too few finite, separated
        candidates remain; no finite scores produces an empty array.

    Raises:
        ValueError: Shapes, coordinates, count or minimum distance are invalid.
    """
    candidates, scores = (
        np.asarray(candidates, dtype=float),
        np.asarray(scores, dtype=float),
    )
    if (
        candidates.ndim != 2
        or candidates.shape[1] == 0
        or scores.shape != (candidates.shape[0],)
        or not np.all(np.isfinite(candidates))
        or np.any((candidates < 0) | (candidates > 1))
    ):
        raise ValueError(
            "Provide finite unit-interval candidate rows and one score per row."
        )
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer.")
    if not np.isfinite(min_distance) or not 0 <= min_distance <= 1:
        raise ValueError("min_distance must lie between zero and one.")
    selected: list[int] = []
    for index in np.argsort(
        np.where(np.isfinite(scores), scores, np.inf), kind="stable"
    ):
        if not np.isfinite(scores[index]):
            break
        if selected:
            distances = np.sqrt(
                np.mean((candidates[selected] - candidates[index]) ** 2, axis=1)
            )
            if np.any(distances < min_distance):
                continue
        selected.append(int(index))
        if len(selected) == count:
            break
    return np.asarray(selected, dtype=int)
