"""Participant parameter rows can be combined and assigned without notebooks."""
from copy import deepcopy

import jax.numpy as jnp
import numpy as np

from jaxcmr.fitting import combine_participant_parameters, parameters_for_subjects


def test_combine_sorts_every_column_and_preserves_booleans_without_mutation():
    parts = [dict(subject=[9, 2], rate=[.9, .2], enabled=[True, False]),
             dict(subject=[4], rate=[.4], enabled=[True])]
    before = deepcopy(parts)
    result = combine_participant_parameters(parts)
    assert result == dict(subject=[2,4,9], rate=[.2,.4,.9], enabled=[False,True,True])
    assert all(isinstance(v,bool) for v in result['enabled'])
    result['rate'][0] = 99
    assert parts == before


def test_combine_accepts_numpy_columns():
    result = combine_participant_parameters([{'subject':np.array([2,1]), 'rate':np.array([.2,.1])}])
    assert result['subject'] == [1,2]
    assert all(isinstance(value, int) for value in result['subject'])
    assert result['rate'] == [.1,.2]


def test_parameter_assignment_matches_identities_and_restores_boolean_values():
    record=dict(fits={'subject':[8,3,12], 'rate':[.8,.3,.12], 'enabled':[1.,0.,1.]},
                fixed={'enabled':False},fit_configuration={'parameters':{'fixed':{'enabled':True}}})
    before=deepcopy(record)
    result=parameters_for_subjects(record,[12,8])
    np.testing.assert_array_equal(result['subject'],[12,8])
    assert jnp.issubdtype(result['subject'].dtype, jnp.integer)
    np.testing.assert_allclose(result['rate'],[.12,.8])
    assert result['enabled'].dtype==jnp.bool_
    np.testing.assert_array_equal(result['enabled'],[True,True])
    assert record==before
    pooled=dict(fits={'subject':[-1],'rate':[.4],'enabled':[0.]},fixed={'enabled':False})
    result=parameters_for_subjects(pooled,[8,3])
    np.testing.assert_array_equal(result['subject'],[8,3])
    np.testing.assert_allclose(result['rate'],[.4,.4])
    np.testing.assert_array_equal(result['enabled'],[False,False])
