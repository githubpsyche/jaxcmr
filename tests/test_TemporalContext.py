"""
Tests for TemporalContext type and functions
"""

# %% Setup

from jaxcmr.context import (
    Context, TemporalContext, integrate, integrate_outlist_context, integrate_start_context
)
from compmempy.models.context import TemporalContext as numba_TemporalContext
from jax import jit, numpy as jnp
import numpy as np
import json


def integrate_drift_f():
    item_count = 5
    context = TemporalContext.create(item_count)
    drift_rate = 0.5
    return integrate_outlist_context(context, drift_rate)


class TestTemporalContext:

    # comparison stucture from numba implementation
    with open('D:/data/base_cmr_parameters.json') as f:
        parameters = json.load(f)['fixed']
    item_count = 16
    items = np.eye(item_count)
    numba_context = numba_TemporalContext(items, parameters)

    context = TemporalContext.create(item_count)

    # mfc activations decide context_input

    context_input = np.array([0., 0.89544394, 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0.])

    # there's also a normalization step
    context_input = context_input / np.sqrt(np.sum(np.square(context_input)))

    # context_input is integrated into context,
    numba_context.integrate(context_input, parameters['encoding_drift_rate'])

    numba_context.state

