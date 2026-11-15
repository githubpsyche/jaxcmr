"""Saved-fit compatibility and participant rows are useful without notebooks."""
from copy import deepcopy

import jax.numpy as jnp
import numpy as np
import pytest

from jaxcmr.fitting import combine_participant_parameters, parameters_for_subjects, validate_fit_compatibility


def saved_fit():
    configuration = dict(
        parameters={"fixed": {"enabled": True}, "free": {"rate": [0., 1.]}},
        search_settings={"num_steps": 10, "progress_bar": False},
        selected_trial_indices=[0, 1], model_factory="example.model",
        components={"memory": "example.memory"}, loss_function="example.loss",
        data_path="data/input.h5", feature_column=6, pooled=True, jax_enable_x64=True,
    )
    record = dict(
        free={"rate": [0., 1.]}, fits={"subject": [-1], "rate": [.4], "enabled": [True]},
        fit_configuration=deepcopy(configuration), selected_trial_indices=[0, 1],
        model_factory="example.model", components={"memory": "example.memory"},
        loss_function="example.loss", data_path="data/input.h5",
    )
    return record, configuration


def test_compatible_record_and_display_only_options_are_not_mutated():
    record, configuration = saved_fit()
    configuration['search_settings'].update(progress_bar=True, display_iterations=True)
    before = deepcopy((record, configuration))
    assert validate_fit_compatibility(record, configuration) is None
    assert (record, configuration) == before


def test_legacy_missing_metadata_is_accepted_without_backfilling():
    record, configuration = saved_fit()
    record['fit_configuration'].update(source_sha256='old source', data_sha256='old data', feature_sha256=None)
    configuration.update(source_sha256='different source', data_sha256='different data')
    record['fit_configuration'].pop('selected_trial_indices')
    record['fit_configuration'].pop('jax_enable_x64')
    before = deepcopy(record)
    validate_fit_compatibility(record, configuration)
    assert record == before
    record.pop('fit_configuration')
    record.pop('data_path')
    validate_fit_compatibility(record, configuration)


@pytest.mark.parametrize('field,value', [
    ('data_path','other.h5'), ('feature_column',7), ('model_factory','other'),
    ('components',{}), ('loss_function','other'), ('pooled',False),
    ('jax_enable_x64',False), ('selected_trial_indices',[1]),
    ('search_settings',{'num_steps':20}),
    ('parameters',{'fixed':{'enabled':True},'free':{'rate':[0.,2.]}}),
])
def test_recorded_configuration_changes_block_reuse(field, value):
    record, configuration = saved_fit()
    configuration[field] = value
    with pytest.raises(ValueError):
        validate_fit_compatibility(record, configuration)


@pytest.mark.parametrize('change', ['bounds','fixed','rows','model','input','components','loss'])
def test_legacy_records_still_check_the_evidence_they_have(change):
    record, configuration = saved_fit()
    record.pop('fit_configuration')
    if change == 'bounds': record['free']['rate'] = [0.,2.]
    elif change == 'fixed': record['fits']['enabled'] = [False]
    elif change == 'rows': record['selected_trial_indices'] = [1]
    else: record[{'model':'model_factory','input':'data_path','components':'components','loss':'loss_function'}[change]] = 'other'
    with pytest.raises(ValueError): validate_fit_compatibility(record, configuration)


def test_free_parameters_take_precedence_over_fixed_values():
    record, configuration = saved_fit()
    configuration['parameters']['fixed']['rate'] = .7
    record.pop('fit_configuration')
    validate_fit_compatibility(record, configuration)


def test_combine_sorts_every_column_and_preserves_booleans_without_mutation():
    parts = [dict(subject=[9, 2], rate=[.9, .2], enabled=[True, False]),
             dict(subject=[4], rate=[.4], enabled=[True])]
    before = deepcopy(parts)
    result = combine_participant_parameters(parts)
    assert result == dict(subject=[2,4,9], rate=[.2,.4,.9], enabled=[False,True,True])
    assert all(isinstance(v,bool) for v in result['enabled'])
    result['rate'][0] = 99
    assert parts == before


@pytest.mark.parametrize('parts', [
    [], [{}], [{'subject':[]}], [{'subject':[-1], 'rate':[.2]}],
    [{'subject':[1,1], 'rate':[.2,.3]}],
    [{'subject':[1], 'rate':[.2]}, {'subject':[1], 'rate':[.3]}],
    [{'subject':[1], 'rate':[.2]}, {'subject':[2], 'other':[.3]}],
    [{'subject':[1,2], 'rate':[.2]}], [{'subject':[1.5], 'rate':[.2]}],
    [{'subject':[True], 'rate':[.2]}], [{'subject':[1], 'rate':[[.2]]}],
])
def test_combine_rejects_ambiguous_or_misaligned_rows(parts):
    with pytest.raises(ValueError): combine_participant_parameters(parts)


def test_combine_accepts_numpy_columns():
    result = combine_participant_parameters([{'subject':np.array([2.,1.]), 'rate':np.array([.2,.1])}])
    assert result['subject'] == [1,2]
    assert all(isinstance(value, int) for value in result['subject'])
    assert result['rate'] == [.1,.2]


def test_parameter_assignment_matches_identities_and_restores_boolean_values():
    record=dict(fits={'subject':[8,3,12], 'rate':[.8,.3,.12], 'enabled':[1.,0.,1.]},
                fixed={'enabled':False},fit_configuration={'parameters':{'fixed':{'enabled':True}}})
    before=deepcopy(record)
    result=parameters_for_subjects(record,[12.,8.])
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


@pytest.mark.parametrize('record,ids',[
    ({'fits':{'subject':[1],'rate':[.3]}},[2]),
    ({'fits':{'subject':[1,1],'rate':[.3,.4]}},[1]),
    ({'fits':{'subject':[1],'rate':[.3,.4]}},[1]),
    ({'fits':{'subject':[1],'rate':[.3]}},[1,1]),
    ({'fits':{'subject':[1],'rate':[.3]}},[-1]),
    ({'fits':{'subject':[1],'rate':[.3]}},[1.5]),
    ({'fits':{'subject':[-1],'rate':[.3]},'pooled':False},[1]),
    ({'fits':{'subject':[1],'rate':[.3]},'pooled':True},[1]),
])
def test_parameter_assignment_rejects_ambiguous_or_missing_rows(record,ids):
    with pytest.raises(ValueError):parameters_for_subjects(record,ids)
