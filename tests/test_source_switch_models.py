"""Tests for source-switch eCMR and CMR3 variants."""

import jax.numpy as jnp
import numpy as np

import jaxcmr.components.context as TemporalContext
from jaxcmr.helpers import make_dataset
from jaxcmr.models.cmr3 import CMR3
from jaxcmr.models.cmr3_source_switch import CMR3SourceSwitch
from jaxcmr.models.cmr3_source_switch import (
    make_factory as make_cmr3_source_switch_factory,
)
from jaxcmr.models.ecmr import eCMR
from jaxcmr.models.ecmr_source_switch import eCMRSourceSwitch
from jaxcmr.models.ecmr_source_switch import (
    make_factory as make_ecmr_source_switch_factory,
)


_PARAMS = {
    "encoding_drift_rate": 0.5,
    "start_drift_rate": 0.4,
    "recall_drift_rate": 0.6,
    "emotion_encoding_drift_rate": 1.0,
    "emotion_recall_drift_rate": 0.6,
    "shared_support": 1.0,
    "item_support": 1.5,
    "learning_rate": 0.3,
    "primacy_scale": 1.0,
    "primacy_decay": 0.1,
    "stop_probability_scale": 0.05,
    "stop_probability_growth": 0.2,
    "choice_sensitivity": 2.0,
    "modulate_emotion_by_primacy": False,
    "emotion_scale": 1.0,
    "phi_emot_modulates_temporal": False,
    "learn_after_context_update": False,
    "allow_repeated_recalls": False,
}


def _study_all(model, choices):
    for choice in choices:
        model = model.experience(jnp.int32(choice))
    return model


def test_ecmr_source_switch_drift_defaults_to_neutral():
    """Omitting source_switch_drift_rate leaves source switches inactive."""
    model = eCMRSourceSwitch(
        4,
        _PARAMS,
        jnp.array([False, True, True, False]),
    )
    first_outlist_unit = int(model.context.next_outlist_unit)

    model = _study_all(model, [1, 2, 3, 4])

    assert int(model.context.next_outlist_unit) == first_outlist_unit


def test_ecmr_source_switch_consumes_outlist_only_on_category_changes():
    """Emotional/neutral switches consume fresh outlist units when enabled."""
    model = eCMRSourceSwitch(
        4,
        {**_PARAMS, "source_switch_drift_rate": 0.3},
        jnp.array([False, True, True, False]),
    )
    first_outlist_unit = int(model.context.next_outlist_unit)

    model = model.experience(jnp.int32(1))
    assert int(model.context.next_outlist_unit) == first_outlist_unit

    model = model.experience(jnp.int32(2))
    assert int(model.context.next_outlist_unit) == first_outlist_unit + 1

    model = model.experience(jnp.int32(3))
    assert int(model.context.next_outlist_unit) == first_outlist_unit + 1

    model = model.experience(jnp.int32(4))
    assert int(model.context.next_outlist_unit) == first_outlist_unit + 2
    assert int(model.study_index) == 4


def test_ecmr_source_switch_same_category_repeats_do_not_consume_outlist():
    """Same emotional category repeats do not create pseudo-events."""
    model = eCMRSourceSwitch(
        4,
        {**_PARAMS, "source_switch_drift_rate": 0.3},
        jnp.array([True, True, True, True]),
    )
    first_outlist_unit = int(model.context.next_outlist_unit)

    model = _study_all(model, [1, 2, 3, 4])

    assert int(model.context.next_outlist_unit) == first_outlist_unit


def test_cmr3_source_switch_consumes_outlist_on_valence_category_changes():
    """Negative/positive/neutral switches consume fresh outlist units."""
    model = CMR3SourceSwitch(
        4,
        {**_PARAMS, "source_switch_drift_rate": 0.3},
        is_negative=jnp.array([True, False, False, False]),
        is_positive=jnp.array([False, True, False, False]),
    )
    first_outlist_unit = int(model.context.next_outlist_unit)

    model = model.experience(jnp.int32(1))
    assert int(model.context.next_outlist_unit) == first_outlist_unit

    model = model.experience(jnp.int32(2))
    assert int(model.context.next_outlist_unit) == first_outlist_unit + 1

    model = model.experience(jnp.int32(3))
    assert int(model.context.next_outlist_unit) == first_outlist_unit + 2

    model = model.experience(jnp.int32(4))
    assert int(model.context.next_outlist_unit) == first_outlist_unit + 2
    assert int(model.study_index) == 4


def test_cmr3_source_switch_same_category_repeats_do_not_consume_outlist():
    """Same valence category repeats do not create pseudo-events."""
    model = CMR3SourceSwitch(
        4,
        {**_PARAMS, "source_switch_drift_rate": 0.3},
        is_negative=jnp.array([False, False, False, False]),
        is_positive=jnp.array([True, True, True, True]),
    )
    first_outlist_unit = int(model.context.next_outlist_unit)

    model = _study_all(model, [1, 2, 3, 4])

    assert int(model.context.next_outlist_unit) == first_outlist_unit


def test_source_switch_variants_zero_initial_temporal_mcf_outlist_rows():
    """Fresh disruptor units start with no context-to-item support."""
    ecmr_model = eCMRSourceSwitch(
        4,
        {**_PARAMS, "source_switch_drift_rate": 0.3},
        jnp.array([False, True, True, False]),
    )
    cmr3_model = CMR3SourceSwitch(
        4,
        {**_PARAMS, "source_switch_drift_rate": 0.3},
        is_negative=jnp.array([True, False, False, False]),
        is_positive=jnp.array([False, True, False, False]),
    )

    assert jnp.all(ecmr_model.mcf.state[ecmr_model.item_count + 1 :] == 0).item()
    assert jnp.all(cmr3_model.mcf.state[cmr3_model.item_count + 1 :] == 0).item()


def _custom_expanded_context(item_count):
    return TemporalContext.TemporalContext(item_count, (item_count * 3) + 1)


def test_source_switch_variants_default_to_expanded_temporal_memories():
    """Default source-switch variants expand MFC/MCF for outlist units."""
    ecmr_model = eCMRSourceSwitch(
        4,
        _PARAMS,
        jnp.array([False, True, True, False]),
    )
    cmr3_model = CMR3SourceSwitch(
        4,
        _PARAMS,
        is_negative=jnp.array([True, False, False, False]),
        is_positive=jnp.array([False, True, False, False]),
    )

    assert ecmr_model.context.size == 9
    assert ecmr_model.mfc.output_size == ecmr_model.context.size
    assert ecmr_model.mcf.input_size == ecmr_model.context.size
    assert cmr3_model.context.size == 9
    assert cmr3_model.mfc.output_size == cmr3_model.context.size
    assert cmr3_model.mcf.input_size == cmr3_model.context.size


def test_source_switch_variants_honor_custom_expanded_context_builders():
    """Custom expanded context builders remain available through the factory API."""
    ecmr_model = eCMRSourceSwitch(
        4,
        _PARAMS,
        jnp.array([False, True, True, False]),
        context_create_fn=_custom_expanded_context,
    )
    cmr3_model = CMR3SourceSwitch(
        4,
        _PARAMS,
        is_negative=jnp.array([True, False, False, False]),
        is_positive=jnp.array([False, True, False, False]),
        context_create_fn=_custom_expanded_context,
    )

    assert ecmr_model.context.size == 13
    assert ecmr_model.mfc.output_size == ecmr_model.context.size
    assert ecmr_model.mcf.input_size == ecmr_model.context.size
    assert cmr3_model.context.size == 13
    assert cmr3_model.mfc.output_size == cmr3_model.context.size
    assert cmr3_model.mcf.input_size == cmr3_model.context.size


def test_source_switch_factories_create_expanded_models():
    """Public factory paths create source-switch models with expanded context."""
    dataset = make_dataset(jnp.array([[1, 2, 3, 4]], dtype=jnp.int32))
    dataset["pres_itemids"] = dataset["pres_itemnos"]
    dataset["valence"] = jnp.array([[0, -1, 1, 0]], dtype=jnp.int32)

    ecmr_factory_cls = make_ecmr_source_switch_factory()
    cmr3_factory_cls = make_cmr3_source_switch_factory()

    ecmr_model = ecmr_factory_cls(dataset, None).create_model(_PARAMS)
    cmr3_model = cmr3_factory_cls(dataset, None).create_model(_PARAMS)

    assert isinstance(ecmr_model, eCMRSourceSwitch)
    assert isinstance(cmr3_model, CMR3SourceSwitch)
    assert ecmr_model.context.size == 9
    assert cmr3_model.context.size == 9


def test_ecmr_source_switch_zero_rate_matches_base_activations():
    """Zero drift rate preserves base eCMR behavior aside from expanded state."""
    params = {**_PARAMS, "source_switch_drift_rate": 0.0}
    is_emotional = jnp.array([False, True, True, False])

    base = eCMR(4, _PARAMS, is_emotional)
    variant = eCMRSourceSwitch(4, params, is_emotional)

    base = _study_all(base, [1, 2, 3, 4]).start_retrieving()
    variant = _study_all(variant, [1, 2, 3, 4]).start_retrieving()

    np.testing.assert_allclose(variant.activations(), base.activations(), rtol=1e-6)


def test_cmr3_source_switch_zero_rate_matches_base_activations():
    """Zero drift rate preserves base CMR3 behavior aside from expanded state."""
    params = {**_PARAMS, "source_switch_drift_rate": 0.0}
    is_negative = jnp.array([True, False, False, False])
    is_positive = jnp.array([False, True, False, False])

    base = CMR3(4, _PARAMS, is_negative, is_positive)
    variant = CMR3SourceSwitch(4, params, is_negative, is_positive)

    base = _study_all(base, [1, 2, 3, 4]).start_retrieving()
    variant = _study_all(variant, [1, 2, 3, 4]).start_retrieving()

    np.testing.assert_allclose(variant.activations(), base.activations(), rtol=1e-6)
