"""Tests for eCMR emotional-context drift parameters."""

import jax.numpy as jnp

from jaxcmr.models.cmr3 import CMR3
from jaxcmr.models.ecmr import eCMR


_PARAMS = {
    "encoding_drift_rate": 0.5,
    "start_drift_rate": 0.4,
    "recall_drift_rate": 0.6,
    "shared_support": 1.0,
    "item_support": 1.5,
    "learning_rate": 0.3,
    "primacy_scale": 1.0,
    "primacy_decay": 0.1,
    "stop_probability_scale": 0.05,
    "stop_probability_growth": 0.2,
    "choice_sensitivity": 2.0,
    "modulate_emotion_by_primacy": True,
    "emotion_scale": 1.0,
    "phi_emot_modulates_temporal": False,
    "learn_after_context_update": False,
    "allow_repeated_recalls": False,
}


def test_ecmr_uses_named_emotion_context_drift_rates():
    """New parameter names separately control emotional encoding and recall drift."""
    model = eCMR(
        3,
        {
            **_PARAMS,
            "emotion_encoding_drift_rate": 0.25,
            "emotion_recall_drift_rate": 0.35,
        },
        jnp.array([False, True, False]),
    )

    assert float(model.emotion_encoding_drift_rate) == 0.25
    assert float(model.emotion_recall_drift_rate) == 0.35


def test_ecmr_emotion_encoding_drift_defaults_to_one():
    """Omitting emotion_encoding_drift_rate uses the emotional-context default."""
    model = eCMR(
        3,
        _PARAMS,
        jnp.array([False, True, False]),
    )

    assert float(model.emotion_encoding_drift_rate) == 1.0
    assert float(model.emotion_recall_drift_rate) == float(_PARAMS["recall_drift_rate"])


def test_ecmr_shared_emotion_drift_rate_controls_encoding_and_recall():
    """Shared emotion_drift_rate sets both emotional context drift rates."""
    model = eCMR(
        3,
        {**_PARAMS, "emotion_drift_rate": 0.45},
        jnp.array([False, True, False]),
    )

    assert float(model.emotion_encoding_drift_rate) == 0.45
    assert float(model.emotion_recall_drift_rate) == 0.45


def test_ecmr_shared_emotion_drift_rate_takes_precedence():
    """Shared emotion_drift_rate overrides separate emotional drift values."""
    model = eCMR(
        3,
        {
            **_PARAMS,
            "emotion_drift_rate": 0.45,
            "emotion_encoding_drift_rate": 0.25,
            "emotion_recall_drift_rate": 0.35,
        },
        jnp.array([False, True, False]),
    )

    assert float(model.emotion_encoding_drift_rate) == 0.45
    assert float(model.emotion_recall_drift_rate) == 0.45


def test_ecmr_temporal_multiplicative_broad_preserves_neutral_learning():
    """Neutral items keep primacy-only temporal MCF learning in multiplicative broad."""
    source = eCMR(
        3,
        {**_PARAMS, "phi_emot_modulates_temporal": False},
        jnp.array([False, True, False]),
    ).experience(jnp.int32(1))
    broad = eCMR(
        3,
        {
            **_PARAMS,
            "phi_emot_modulates_temporal": True,
            "modulate_temporal_emotion_by_primacy": True,
        },
        jnp.array([False, True, False]),
    ).experience(jnp.int32(1))

    assert jnp.allclose(source.mcf.state, broad.mcf.state).item()


def test_ecmr_temporal_multiplicative_broad_boosts_emotional_learning():
    """Emotional items get stronger temporal MCF learning in multiplicative broad."""
    source = eCMR(
        3,
        {**_PARAMS, "phi_emot_modulates_temporal": False},
        jnp.array([True, False, False]),
    ).experience(jnp.int32(1))
    broad = eCMR(
        3,
        {
            **_PARAMS,
            "phi_emot_modulates_temporal": True,
            "modulate_temporal_emotion_by_primacy": True,
        },
        jnp.array([True, False, False]),
    ).experience(jnp.int32(1))

    assert jnp.sum(broad.mcf.state - source.mcf.state) > 0


def test_ecmr_temporal_emotion_scale_defaults_to_emotion_scale():
    """Omitting temporal_emotion_scale preserves tied broad temporal learning."""
    tied = eCMR(
        3,
        {
            **_PARAMS,
            "phi_emot_modulates_temporal": True,
            "modulate_temporal_emotion_by_primacy": True,
        },
        jnp.array([True, False, False]),
    ).experience(jnp.int32(1))
    explicit = eCMR(
        3,
        {
            **_PARAMS,
            "temporal_emotion_scale": _PARAMS["emotion_scale"],
            "phi_emot_modulates_temporal": True,
            "modulate_temporal_emotion_by_primacy": True,
        },
        jnp.array([True, False, False]),
    ).experience(jnp.int32(1))

    assert jnp.allclose(tied.mcf.state, explicit.mcf.state).item()
    assert jnp.allclose(tied.emotion_mcf.state, explicit.emotion_mcf.state).item()


def test_ecmr_zero_temporal_emotion_scale_removes_temporal_boost():
    """Temporal boost can be disabled while keeping source emotion learning."""
    source = eCMR(
        3,
        {**_PARAMS, "phi_emot_modulates_temporal": False},
        jnp.array([True, False, False]),
    ).experience(jnp.int32(1))
    temporal_off = eCMR(
        3,
        {
            **_PARAMS,
            "temporal_emotion_scale": 0.0,
            "phi_emot_modulates_temporal": True,
            "modulate_temporal_emotion_by_primacy": True,
        },
        jnp.array([True, False, False]),
    ).experience(jnp.int32(1))

    assert jnp.allclose(source.mcf.state, temporal_off.mcf.state).item()
    assert jnp.sum(temporal_off.emotion_mcf.state) > 0


def test_ecmr_temporal_only_emotion_modulates_temporal_not_source():
    """emotion_scale=0 and temporal_emotion_scale>0 creates temporal-only emotion."""
    no_emotion = eCMR(
        3,
        {
            **_PARAMS,
            "emotion_scale": 0.0,
            "phi_emot_modulates_temporal": False,
        },
        jnp.array([True, False, False]),
    ).experience(jnp.int32(1))
    temporal_only = eCMR(
        3,
        {
            **_PARAMS,
            "emotion_scale": 0.0,
            "temporal_emotion_scale": 1.0,
            "phi_emot_modulates_temporal": True,
            "modulate_temporal_emotion_by_primacy": True,
        },
        jnp.array([True, False, False]),
    ).experience(jnp.int32(1))

    assert jnp.sum(temporal_only.mcf.state - no_emotion.mcf.state) > 0
    assert jnp.allclose(temporal_only.emotion_mcf.state, 0.0).item()


def test_cmr3_uses_named_emotion_context_drift_rates():
    """CMR3 mirrors eCMR emotional-context drift parameter handling."""
    model = CMR3(
        3,
        {
            **_PARAMS,
            "emotion_encoding_drift_rate": 0.2,
            "emotion_recall_drift_rate": 0.3,
        },
        jnp.array([False, True, False]),
        jnp.array([False, False, True]),
    )

    assert float(model.emotion_encoding_drift_rate) == 0.2
    assert float(model.emotion_recall_drift_rate) == 0.3


def test_cmr3_emotion_recall_drift_defaults_to_recall_drift():
    """Omitting emotion_recall_drift_rate preserves shared recall drift behavior."""
    model = CMR3(
        3,
        {**_PARAMS, "emotion_encoding_drift_rate": 0.4},
        jnp.array([False, True, False]),
        jnp.array([False, False, True]),
    )

    assert float(model.emotion_encoding_drift_rate) == 0.4
    assert float(model.emotion_recall_drift_rate) == float(_PARAMS["recall_drift_rate"])


def test_cmr3_shared_emotion_drift_rate_controls_encoding_and_recall():
    """CMR3 supports shared emotion_drift_rate for both emotional drift rates."""
    model = CMR3(
        3,
        {**_PARAMS, "emotion_drift_rate": 0.45},
        jnp.array([False, True, False]),
        jnp.array([False, False, True]),
    )

    assert float(model.emotion_encoding_drift_rate) == 0.45
    assert float(model.emotion_recall_drift_rate) == 0.45


def test_cmr3_temporal_emotion_scale_defaults_to_emotion_scale():
    """CMR3 also preserves tied broad behavior when temporal scale is absent."""
    tied = CMR3(
        3,
        {
            **_PARAMS,
            "phi_emot_modulates_temporal": True,
            "modulate_temporal_emotion_by_primacy": True,
        },
        jnp.array([True, False, False]),
        jnp.array([False, False, False]),
    ).experience(jnp.int32(1))
    explicit = CMR3(
        3,
        {
            **_PARAMS,
            "temporal_emotion_scale": _PARAMS["emotion_scale"],
            "phi_emot_modulates_temporal": True,
            "modulate_temporal_emotion_by_primacy": True,
        },
        jnp.array([True, False, False]),
        jnp.array([False, False, False]),
    ).experience(jnp.int32(1))

    assert jnp.allclose(tied.mcf.state, explicit.mcf.state).item()
    assert jnp.allclose(tied.emotion_mcf.state, explicit.emotion_mcf.state).item()
