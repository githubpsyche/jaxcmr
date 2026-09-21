"""Compatibility name for the generic model-owned sequence likelihood.

Select jaxcmr.models.optimized_cmr.ModelFactory for efficient regular CMR.
The supplied factory always determines predictions. Ordinary or modified model
factories use their own event methods; this module contains no CMR equations
and does not substitute an implementation based on component identities.
"""

from jaxcmr.loss.transform_sequence_likelihood import ExcludeTerminationLikelihoodLoss

FastCMRLikelihoodLoss = ExcludeTerminationLikelihoodLoss

__all__ = ["FastCMRLikelihoodLoss"]
