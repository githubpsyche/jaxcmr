"""Tests for loss function variants.

Covers base_sequence_likelihood, transform_sequence_likelihood,
set_permutation_likelihood, and spc_mse.
"""

import jaxcmr.components.context as TemporalContext
import jaxcmr.components.linear_memory as LinearMemory
from jaxcmr.components.termination import NoStopTermination, PositionalTermination
from jaxcmr.models.cmr import make_factory as base_make_factory

BaseCMRFactory = base_make_factory(
    LinearMemory.init_mfc, LinearMemory.init_mcf, TemporalContext.init, PositionalTermination
)
NoStopCMRFactory = base_make_factory(
    LinearMemory.init_mfc, LinearMemory.init_mcf, TemporalContext.init, NoStopTermination
)


def test_old_loss_names_remain_import_aliases():
    from jaxcmr.fitting import BatchEvosaxDE, ScanEvosaxDE

    assert BatchEvosaxDE is ScanEvosaxDE
    """old loss names remain temporary aliases for compatibility."""
    from jaxcmr import typing
    from jaxcmr.loss import (
        base_sequence_likelihood,
        cat_spc_mse,
        mse,
        sequence_likelihood,
        set_permutation_likelihood,
        spc_mse,
        transform_sequence_likelihood,
    )

    assert typing.LossFnGenerator is typing.LossFn
    assert (
        sequence_likelihood.MemorySearchLikelihoodFnGenerator
        is sequence_likelihood.MemorySearchLikelihoodLoss
    )
    assert (
        base_sequence_likelihood.MemorySearchLikelihoodFnGenerator
        is base_sequence_likelihood.MemorySearchLikelihoodLoss
    )
    assert (
        base_sequence_likelihood.ExcludeFirstRecallLikelihoodFnGenerator
        is base_sequence_likelihood.ExcludeFirstRecallLikelihoodLoss
    )
    assert (
        base_sequence_likelihood.ExcludeTerminationLikelihoodFnGenerator
        is base_sequence_likelihood.ExcludeTerminationLikelihoodLoss
    )
    assert (
        transform_sequence_likelihood.MemorySearchLikelihoodFnGenerator
        is transform_sequence_likelihood.MemorySearchLikelihoodLoss
    )
    assert (
        transform_sequence_likelihood.ExcludeFirstRecallLikelihoodFnGenerator
        is transform_sequence_likelihood.ExcludeFirstRecallLikelihoodLoss
    )
    assert (
        transform_sequence_likelihood.ExcludeTerminationLikelihoodFnGenerator
        is transform_sequence_likelihood.ExcludeTerminationLikelihoodLoss
    )
    assert (
        set_permutation_likelihood.MemorySearchLikelihoodFnGenerator
        is set_permutation_likelihood.MemorySearchLikelihoodLoss
    )
    assert (
        set_permutation_likelihood.ExcludeTerminationLikelihoodFnGenerator
        is set_permutation_likelihood.ExcludeTerminationLikelihoodLoss
    )
    assert (
        set_permutation_likelihood.IncludeTerminationLikelihoodFnGenerator
        is set_permutation_likelihood.IncludeTerminationLikelihoodLoss
    )
    assert mse.MemorySearchMseFnGenerator is mse.MemorySearchMseLoss
    assert spc_mse.MemorySearchSpcMseFnGenerator is spc_mse.MemorySearchSpcMseLoss
    assert cat_spc_mse.MemorySearchCatSpcMseFnGenerator is cat_spc_mse.MemorySearchCatSpcMseLoss
