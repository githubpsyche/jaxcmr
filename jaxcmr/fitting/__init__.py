"""Estimate model parameters from a loss evaluated on selected trials.

The loss determines what counts as a good fit. For example, a likelihood
loss scores the observed recall sequence, while an SPC loss compares
observed and simulated serial position curves. The fitter searches within
the free-parameter bounds for values that minimize the chosen loss.

``ScipyDE`` and the Evosax differential-evolution fitters compare populations
of candidates. ``ScipyLBFGSB`` and ``ScipyTrustConstr`` use loss gradients
and can search from several starting points. ``ParameterSpace`` translates
between search coordinates and physical parameter values. Results report
parameters in the units used by the model.

A pooled fit estimates one parameter set from all selected trials.
Participant-wise fitting estimates a set for each participant. In a
``FitResult``, each row links a parameter set to its objective value and
optimizer results. ``parameters_for_subjects`` uses those rows to assign a
pooled estimate or matching participant estimates for simulation.
"""

from jaxcmr.fitting.evosax import (
    EvosaxDE,
    MapEvosaxDE,
    ScanEvosaxDE,
)
from jaxcmr.fitting.gradient import ScipyLBFGSB, ScipyTrustConstr
from jaxcmr.fitting.initialization import select_starts
from jaxcmr.fitting.parameters import ParameterSpace, ParameterTransform
from jaxcmr.fitting.records import (
    combine_participant_parameters,
    parameters_for_subjects,
)
from jaxcmr.fitting.scipy import ScipyDE, make_scipy_loss_fn

__all__ = [
    "EvosaxDE",
    "MapEvosaxDE",
    "ParameterSpace",
    "ParameterTransform",
    "ScanEvosaxDE",
    "ScipyDE",
    "ScipyLBFGSB",
    "ScipyTrustConstr",
    "combine_participant_parameters",
    "make_scipy_loss_fn",
    "parameters_for_subjects",
    "select_starts",
]
