"""Model-owned predictions for fixed observed sequences.

The optional factory capability is independent of any memory representation.
A factory may group trials and omit unrequested outputs, but must condition on
every real preceding event, including events whose scoring mask is false.
Returned blocks and masks are aligned; masked entries may use a filler value.
No model equation or loss reduction belongs in this container.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import jax

from jaxcmr.typing import Array, Bool, Float_, Integer, MemorySearchModelFactory

type PredictionFn[DataT] = Callable[[DataT, Mapping[str, Float_]], tuple[Array, ...]]
type _SequentialData = tuple[
    Integer[Array, " trials"],
    Integer[Array, "trials study_positions"],
    Integer[Array, "trials recall_outputs"],
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PreparedPrediction[DataT]:
    """Immutable prediction plan with dynamic observations and static code.

    ``evaluate`` accepts data and a complete model parameter mapping and returns
    probability blocks. ``masks`` contains the caller's scoring requests in the
    same block layout. Reordering or padding must preserve observation order
    and multiplicity. Preparation snapshots source data before compilation.

    Attributes
    ----------
    data : DataT
        Model-specific observation arrays in a JAX-compatible pytree. These
        remain dynamic arguments, permitting compilation reuse across datasets.
    masks : tuple[Array, ...]
        Boolean scoring requests, with one mask per returned probability block.
    evaluate : PredictionFn[DataT]
        Pure model calculation, stored as static JIT metadata. Keep observations
        in data rather than its closure when compilation should be reusable.

    """

    data: DataT
    masks: tuple[Bool[Array, "..."], ...]
    evaluate: PredictionFn[DataT] = field(metadata={"static": True})

    def probabilities(self, parameters: Mapping[str, Float_]) -> tuple[Array, ...]:
        """Predict requested events, retaining the full conditioning history.

        Parameters
        ----------
        parameters : Mapping[str, Float_]
            Complete model parameter mapping, including fixed configuration.

        Returns
        -------
        tuple[Array, ...]
            Probability blocks with the same shapes and ordering as masks.
            Unrequested entries may contain fillers; consumers must mask them
            before a transformation such as log that excludes those fillers.

        """
        return self.evaluate(self.data, parameters)


def make_sequential_evaluator(
    factory: MemorySearchModelFactory,
) -> PredictionFn[_SequentialData]:
    """Bind one stable evaluator to a factory's ordinary event methods.

    Create this once per loss instance and reuse it in PreparedPrediction
    objects. Observations are explicit data, allowing selections with matching
    shapes and dtypes to reuse compilation. The closure has its own identity,
    so the factory itself need not be hashable or participate in a global cache.

    Args:
        factory: Factory whose event methods define the model calculation.
            Treat its configuration as fixed for the evaluator's lifetime.

    Returns:
        Function accepting (dataset row indices, study lists, observed outcomes)
        and a complete parameter mapping. Outcomes must already be converted
        from dataset position codes to one-indexed item labels, with zero for
        STOP. It returns one trial-by-output probability block. Every response
        advances the model, even if a caller excludes its probability from scoring.
    """
    create_model = factory.create_trial_model

    def evaluate(
        data: _SequentialData, parameters: Mapping[str, Float_]
    ) -> tuple[Array, ...]:
        selected, study, recalls = data

        def trial(index, presented, observed):
            model = create_model(index, parameters)
            model = jax.lax.scan(
                lambda m, item: (m.experience(item), None), model, presented
            )[0]
            model = model.start_retrieving()
            return jax.lax.scan(
                lambda m, item: (m.retrieve(item), m.outcome_probability(item)),
                model,
                observed,
            )[1]

        return (jax.vmap(trial)(selected, study, recalls),)

    return evaluate
