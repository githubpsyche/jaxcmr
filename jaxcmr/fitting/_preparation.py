"""Adapt ordinary and optionally prepared losses to one fitter-facing form.

Fitters always evaluate PreparedLoss objects. This boundary handles optional
preparation, callable adaptation and selection caching; losses still control
their observations and array layouts. Host preparation never compiles or fits.
"""

from collections.abc import Iterable, Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np

from jaxcmr.loss.prepared import PreparedLoss
from jaxcmr.typing import (
    Array,
    BatchPreparableLossFn,
    Bool,
    Float,
    Float_,
    Integer,
    LossFn,
    PreparableLossFn,
    PreparedLossFn,
)


class LossPreparation:
    """Own one loss's current selection and any explicitly planned selections.

    The host cache retains only the current unplanned loss. An explicit plan
    retains all requested selections until replaced. Returned objects keep
    their own observations; selecting another subject cannot change them.

    Ordinary losses use trial indices as dynamic data. Explicit PreparedLoss
    objects pass through unchanged. Plain prepared callables become static
    evaluators with no dynamic data, preserving their per-callable compilation.
    JAX can retain those static closures in its compilation cache; explicit
    PreparedLoss data avoid that per-selection compilation and retention cost.
    """

    def __init__(self, loss: LossFn) -> None:
        """Retain the loss and create one stable evaluator for ordinary calls."""
        self.loss = loss
        self._plan: dict[tuple[int, ...], PreparedLoss[Any]] = {}
        self._selection: tuple[int, ...] | None = None
        self._current: PreparedLoss[Any] | None = None

        def evaluate(indices, base, names, parameters):
            return loss(indices, base, names, parameters)

        # A function is hashable JIT metadata even when the loss object is not.
        self._evaluate_indices = evaluate

    @property
    def prepared_trial_count(self) -> int | None:
        """Number of currently prepared trials, or None for an ordinary loss."""
        return None if self._selection is None else len(self._selection)

    def ordinary(self, indices: Integer[Array, " trials"]) -> PreparedLoss[Array]:
        """Bind dynamic indices without host preparation, including inside JIT.

        Scan/map subject drivers deliberately use this route: their indices
        are tracers, whereas optional data preparation needs concrete values.
        """
        return PreparedLoss(indices, self._evaluate_indices)

    def select(self, indices: Integer[Array, " trials"]) -> PreparedLoss[Any]:
        """Return a loss for concrete indices, reusing a plan or current selection.

        Row order and multiplicity are preserved. The returned loss always has
        the same call signature, whether or not the source supports preparation.
        """
        if not isinstance(self.loss, PreparableLossFn):
            return self.ordinary(indices)
        selection = tuple(np.asarray(indices).tolist())
        if selection != self._selection or self._current is None:
            prepared = self._plan.get(selection)
            if prepared is None:
                prepared = _as_prepared_loss(self.loss.prepare_trials(indices))
            self._current, self._selection = prepared, selection
        return self._current

    def prepare_selections(
        self,
        trial_masks: Iterable[Bool[Array, " trials"]],
        trial_shape: tuple[int, ...],
    ) -> None:
        """Replace the explicit plan after validating and preparing every mask.

        Requires batch preparation. Duplicate masks are prepared once, empty
        input clears the plan, and failure preserves the previous plan/cache.
        No compilation occurs; unplanned selections remain available via select.
        """
        prepared = prepare_losses_for_selections(self.loss, trial_masks, trial_shape)
        plan = {rows: _as_prepared_loss(loss) for rows, loss in prepared.items()}
        self._plan = plan
        self._selection = self._current = None


def _as_prepared_loss(loss: PreparedLossFn) -> PreparedLoss[Any]:
    """Keep explicit data, or adapt a callable with its own static evaluator.

    Captured observations cannot be made dynamic automatically. This adapter
    therefore preserves a plain callable's compilation boundary; it does not
    promise compilation reuse across independently prepared closures.
    """
    if isinstance(loss, PreparedLoss):
        return loss

    def evaluate(
        _data: tuple[()],
        base: Mapping[str, Float_],
        names: Iterable[str],
        parameters: Float[Array, "parameters n_samples"],
    ) -> Float[Array, " n_samples"]:
        return loss(base, names, parameters)

    return PreparedLoss((), evaluate)


def prepare_losses_for_selections(
    loss: LossFn,
    trial_masks: Iterable[Bool[Array, " trials"]],
    trial_shape: tuple[int, ...],
) -> dict[tuple[int, ...], PreparedLossFn]:
    """Return prepared losses keyed by the dataset rows selected for each fit.

    Preparing selections together lets a loss coordinate array layouts for
    compilation reuse. Fitters install the returned mapping only after this
    function succeeds, so failed preparation leaves their previous plan intact.

    Args:
        loss: Loss supporting the optional BatchPreparableLossFn capability.
        trial_masks: Concrete Boolean masks with one entry per dataset trial.
            Each mask must select at least one trial. Identical selections are
            prepared once, in the order of their first appearance.
        trial_shape: Expected mask shape, normally (number_of_trials,).

    Returns:
        Mapping from tuples of ascending, zero-based dataset row indices to
        prepared loss callables. An empty mask collection returns an empty
        mapping, which the fitter uses to clear its existing plan.

    Raises:
        TypeError: The loss does not support preparing multiple selections.
        ValueError: A mask has the wrong shape or dtype, selects no trials, or
            the loss does not return exactly one callable per unique selection.
    """
    if not isinstance(loss, BatchPreparableLossFn):
        raise TypeError("The loss does not support preparing multiple selections.")
    selections: dict[tuple[int, ...], Array] = {}
    for mask in trial_masks:
        values = np.asarray(mask)
        if (
            values.shape != trial_shape
            or values.dtype != np.bool_
            or not np.any(values)
        ):
            raise ValueError(
                "Each trial mask must be a nonempty Boolean selection with one entry per trial."
            )
        indices = np.flatnonzero(values).astype(np.int32)
        # The same row tuple is used by fitters to look up this prepared loss.
        selections.setdefault(tuple(indices.tolist()), jnp.asarray(indices))
    prepared = tuple(loss.prepare_selections(selections.values()))
    if len(prepared) != len(selections) or any(
        not callable(value) for value in prepared
    ):
        raise ValueError(
            "prepare_selections must return one callable per supplied selection."
        )
    return dict(zip(selections, prepared, strict=True))
