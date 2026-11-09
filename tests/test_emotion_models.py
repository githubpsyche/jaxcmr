"""Separate temporal and emotional learning; source changes are pseudo-events."""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxcmr.components.context import TemporalContext
from jaxcmr.helpers import make_dataset
from jaxcmr.models.ecmr import eCMR
from jaxcmr.models.cmr3 import CMR3
from jaxcmr.models.ecmr_source_switch import make_factory as ecmr_switch
from jaxcmr.models.cmr3_source_switch import make_factory as cmr3_switch

PARAMS = dict(
    encoding_drift_rate=0.5,
    start_drift_rate=0.4,
    recall_drift_rate=0.6,
    shared_support=1.0,
    item_support=1.5,
    learning_rate=0.3,
    primacy_scale=1.0,
    primacy_decay=0.1,
    stop_probability_scale=0.05,
    stop_probability_growth=0.2,
    choice_sensitivity=2.0,
    modulate_emotion_by_primacy=True,
    emotion_scale=1.0,
    phi_emot_modulates_temporal=False,
    learn_after_context_update=False,
    allow_repeated_recalls=False,
)


def emotional_model(kind, params, emotional=True):
    negative = jnp.array([emotional, False, False])
    return (
        eCMR(3, params, negative)
        if kind == "ecmr"
        else CMR3(3, params, negative, jnp.array([False, True, False]))
    )


@pytest.mark.parametrize("kind", ["ecmr", "cmr3"])
@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, [1.0, 0.6]),
        ({"emotion_encoding_drift_rate": 0.25, "emotion_recall_drift_rate": 0.35}, [0.25, 0.35]),
        (
            {
                "emotion_drift_rate": 0.45,
                "emotion_encoding_drift_rate": 0.25,
                "emotion_recall_drift_rate": 0.35,
            },
            [0.45, 0.45],
        ),
    ],
)
def test_emotional_drift_defaults_separation_and_precedence(kind, overrides, expected):
    model = emotional_model(kind, PARAMS | overrides)
    np.testing.assert_allclose(
        [model.emotion_encoding_drift_rate, model.emotion_recall_drift_rate], expected
    )


@pytest.mark.parametrize(
    "kind,emotional,source_scale,temporal_scale,boost",
    [
        ("ecmr", False, 1.0, 1.0, False),
        ("ecmr", True, 1.0, 1.0, True),
        ("ecmr", True, 1.0, 0.0, False),
        ("ecmr", True, 0.0, 1.0, True),
        ("cmr3", True, 1.0, 1.0, True),
    ],
)
def test_emotional_learning_channels(kind, emotional, source_scale, temporal_scale, boost):
    params = PARAMS | {"emotion_scale": source_scale}
    source = emotional_model(kind, params, emotional).experience(jnp.int32(1))
    broad_params = params | dict(
        phi_emot_modulates_temporal=True,
        modulate_temporal_emotion_by_primacy=True,
        temporal_emotion_scale=temporal_scale,
    )
    broad = emotional_model(kind, broad_params, emotional).experience(jnp.int32(1))
    if boost:
        assert jnp.sum(broad.mcf.state - source.mcf.state) > 0
    else:
        np.testing.assert_allclose(broad.mcf.state, source.mcf.state)
    np.testing.assert_allclose(broad.emotion_mcf.state, source.emotion_mcf.state)
    if source_scale == 0:
        np.testing.assert_array_equal(broad.emotion_mcf.state, 0)
    if source_scale == temporal_scale:
        del broad_params["temporal_emotion_scale"]
        default = emotional_model(kind, broad_params, emotional).experience(jnp.int32(1))
        np.testing.assert_allclose(default.mcf.state, broad.mcf.state)
        np.testing.assert_allclose(default.emotion_mcf.state, broad.emotion_mcf.state)


@pytest.mark.parametrize(
    "factory,base,categories,rate,expected,context_size",
    [
        (ecmr_switch, eCMR, [0, -1, -1, 0], 0.3, [0, 1, 1, 2], 9),
        (cmr3_switch, CMR3, [-1, 1, 0, 0], 0.3, [0, 1, 2, 2], 9),
        (ecmr_switch, eCMR, [1, 1, 1, 1], 0.3, [0, 0, 0, 0], 9),
        (cmr3_switch, CMR3, [1, 1, 1, 1], 0.3, [0, 0, 0, 0], 9),
        (ecmr_switch, eCMR, [0, -1, -1, 0], None, [0, 0, 0, 0], 13),
        (cmr3_switch, CMR3, [-1, 1, 0, 0], 0.0, [0, 0, 0, 0], 13),
    ],
)
def test_source_switch_allocation_and_neutral_equivalence(
    factory, base, categories, rate, expected, context_size
):
    data = make_dataset(jnp.array([[1, 2, 3, 4]]))
    data.update(pres_itemids=data["pres_itemnos"], valence=jnp.array([categories]))
    params = PARAMS if rate is None else PARAMS | {"source_switch_drift_rate": rate}
    config = (
        {} if context_size == 9 else {"context_create_fn": lambda n: TemporalContext(n, 3 * n + 1)}
    )
    model = factory(**config)(data, None).create_trial_model(jnp.int32(0), params)
    assert model.context.size == model.mfc.output_size == model.mcf.input_size == context_size
    np.testing.assert_array_equal(model.mcf.state[model.item_count + 1 :], 0)
    start = int(model.context.next_outlist_unit)
    observed = []
    # Consecutive steps of one study sequence, not independent scenarios.
    for choice in range(1, 5):
        model = model.experience(jnp.int32(choice))
        observed.append(int(model.context.next_outlist_unit) - start)
    assert observed == expected
    assert model.study_index == 4
    if not rate:
        categories = jnp.array(categories)
        reference = (
            base(4, PARAMS, categories != 0)
            if base is eCMR
            else base(4, PARAMS, categories < 0, categories > 0)
        )
        for choice in range(1, 5):
            reference = reference.experience(jnp.int32(choice))
        np.testing.assert_allclose(
            model.start_retrieving().activations(),
            reference.start_retrieving().activations(),
            rtol=1e-6,
        )
