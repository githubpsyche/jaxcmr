"""Whole-response rules at the simulation and likelihood boundary."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm
from support import CMR_PARAMS, recall_fixture

from jaxcmr.components.response import (
    ExponentialEvidenceSupportRatio,
    ExponentialStopCompetition,
    LearnedStopCompetition,
    PowerLawStopCompetition,
    PositionalStop,
    RecallUntilExhausted,
    RelativePowerSupportRatio,
)
from jaxcmr.components.support_ratio import (
    RawSupportRatioRelativeChoice,
    RawSupportRatioSoftmaxChoice,
    raw_support_ratio_stop_probability,
)
from jaxcmr.experimental.finite_window_race import (
    FiniteWindowRace,
    finite_window_probabilities,
)
from jaxcmr.loss.sequence_likelihood import sequence_probabilities
from jaxcmr.models import model_factory_from_paths, recorded_policy_path
from jaxcmr.models.cmr import CMR, make_factory
from jaxcmr.models.terminal_item_cmr import TerminalItemCMR
from jaxcmr.models.terminal_item_cmr import make_factory as make_terminal_item_factory
from jaxcmr.simulation import simulate_free_recall, simulate_study

PARAMS = dict(
    CMR_PARAMS,
    stop_probability_scale=0.05,
    stop_probability_growth=0.3,
    stop_log_weight=-1.2,
    stop_weight_growth=0.4,
    race_boundary=1.4,
    race_drift_scale=1.7,
)


def test_saved_cmr_and_ecmr_records_reconstruct_their_policies():
    fits = Path(__file__).resolve().parents[1] / "data/fits"
    ordinary = json.loads((fits / "Dupertuys2026_CMR_demo.json").read_text())
    emotional = json.loads((fits / "Dupertuys2026_eCMR_source_only.json").read_text())
    current = dict(ordinary["components"])
    del current["termination_policy_create_fn"]
    current["response_policy_create_fn"] = "jaxcmr.components.response.RecallUntilExhausted"
    assert recorded_policy_path(ordinary["components"]) == recorded_policy_path(current)
    assert recorded_policy_path(emotional["components"]) == recorded_policy_path(current)

    for components in (ordinary["components"], current):
        factory = model_factory_from_paths(ordinary["model_factory"], components)
        model = factory(recall_fixture(), None).create_trial_model(jnp.array(0), PARAMS)
        assert isinstance(model.response_policy, RecallUntilExhausted)
    assert "termination_policy_create_fn" in ordinary["components"]
    # The same historical field reconstructs the emotional model.
    factory = model_factory_from_paths(emotional["model_factory"], emotional["components"])
    parameters = {name: column[0] for name, column in emotional["fits"].items() if name != "subject"}
    data = recall_fixture()
    data["pres_itemids"] = data["pres_itemnos"]
    data["valence"] = jnp.zeros_like(data["pres_itemnos"])
    model = factory(data, None).create_trial_model(jnp.array(0), parameters)
    assert isinstance(model.response_policy, RecallUntilExhausted)


def test_positional_response_pools_repeated_codes_and_excludes_padding():
    from jaxcmr.models.positional_cmr import CMR as PositionalCMR

    parameters = PARAMS | {"mfc_sensitivity": 1.0}
    model = PositionalCMR(4, parameters, response_policy_create_fn=RecallUntilExhausted)
    state = simulate_study(model, jnp.array([3, 1, 3, 0])).start_retrieving()
    positions = np.asarray(state.position_activations())
    expected = np.array([0., positions[1], 0., positions[0] + positions[2], 0.])
    np.testing.assert_allclose(state.outcome_probabilities(), expected / expected.sum())
    after = state.retrieve(jnp.int32(3))
    np.testing.assert_allclose(after.outcome_probabilities(), [0., 1., 0., 0., 0.])
    np.testing.assert_allclose(np.asarray(after.candidate_weights(~after.recallable, 1.))[[0, 1, 3]], 0.)


@pytest.mark.parametrize(
    "rule", [RecallUntilExhausted, PositionalStop, RelativePowerSupportRatio]
)
def test_relative_choice_policies_match_their_equations(rule):
    model = simulate_study(
        CMR(4, PARAMS, response_policy_create_fn=rule), jnp.array([3, 1, 3, 2])
    ).start_retrieving().retrieve(3)
    evidence = np.asarray(model.item_evidence())
    available = np.asarray(model.recallable)
    eps = np.finfo(np.float32).eps

    def weights_for(mask):
        selected = evidence[mask]
        return (selected / selected.max()) ** PARAMS["choice_sensitivity"] + eps

    weights = np.zeros_like(evidence)
    weights[available] = weights_for(available)
    if rule is RelativePowerSupportRatio:
        unavailable = np.asarray(model.studied) & ~available
        ratio = weights.sum() / weights_for(unavailable).sum()
        stop = min(PARAMS["stop_probability_scale"] + np.exp(-PARAMS["stop_probability_growth"] * ratio),
                   1 - eps * available.sum())
    elif rule is PositionalStop:
        stop = PARAMS["stop_probability_scale"] * np.exp(PARAMS["stop_probability_growth"])
    else:
        stop = 0.
    expected = jnp.append(stop, (1 - stop) * weights / weights.sum())
    np.testing.assert_allclose(model.outcome_probabilities(), expected, rtol=1e-6)


@pytest.mark.parametrize(
    "policy, schedule",
    [(ExponentialStopCompetition, lambda k: k), (PowerLawStopCompetition, np.log1p)],
)
def test_scheduled_stop_competes_on_absolute_evidence(policy, schedule):
    model = simulate_study(
        CMR(3, {k: v for k, v in PARAMS.items() if k != "choice_sensitivity"},
            response_policy_create_fn=policy), jnp.array([1, 2, 3])
    )
    model = model.start_retrieving().retrieve(2)
    weights = np.exp(np.asarray(model.item_evidence())) * np.asarray(model.recallable)
    stop = np.exp(
        PARAMS["stop_log_weight"] + PARAMS["stop_weight_growth"] * schedule(1)
    )
    expected = np.r_[stop, weights] / (stop + weights.sum())
    np.testing.assert_allclose(model.outcome_probabilities(), expected, rtol=3e-6)


@pytest.mark.parametrize("x64", [False, True])
def test_raw_ratio_uses_one_support_scale_and_excludes_padding(x64):
    studied = jnp.array([True, True, True, False])
    recallable = jnp.array([True, False, True, False])

    def stop(evidence):
        return raw_support_ratio_stop_probability(
            evidence, studied, recallable, scale=0.001, growth=0.3, is_active=True
        )

    with jax.enable_x64(x64):
        evidence = jnp.array([4., 1., -2., 50.])
        score = jax.jit(jax.value_and_grad(stop))
        for strength in (4., 8.):
            value, grad = score(evidence.at[0].set(strength))
            ratio = (strength + 1e-6) / 1.
            decay = np.exp(-0.3 * ratio)
            np.testing.assert_allclose(value, 0.001 + decay, rtol=1e-6)
            np.testing.assert_allclose(grad, [-0.3 * decay, 0.3 * ratio * decay, 0., 0.], rtol=1e-6)


def test_saved_ratio_paths_preserve_the_separately_weighted_rule():
    historical = {"termination_policy_create_fn": "jaxcmr.components.termination.SupportRatioTermination"}
    weighted = {"response_policy_create_fn": "jaxcmr.components.response.RelativePowerSupportRatio"}
    raw = {"response_policy_create_fn": "jaxcmr.components.support_ratio.RawSupportRatioRelativeChoice"}
    assert recorded_policy_path(historical) == recorded_policy_path(weighted)
    assert recorded_policy_path(raw) != recorded_policy_path(historical)
    for components, policy in ((historical, RelativePowerSupportRatio), (raw, RawSupportRatioRelativeChoice)):
        factory = model_factory_from_paths("jaxcmr.models.cmr.make_factory", components)
        model = factory(recall_fixture(), None).create_trial_model(jnp.array(0), PARAMS)
        assert isinstance(model.response_policy, policy)


def test_ratio_and_item_choice_use_the_same_evidence():
    model = simulate_study(
        CMR(4, PARAMS, response_policy_create_fn=ExponentialEvidenceSupportRatio),
        jnp.array([3, 1, 2]),
    ).start_retrieving()
    assert model.outcome_probability(0) == pytest.approx(PARAMS["stop_probability_scale"])
    state = model.retrieve(2)
    weights = np.exp(np.asarray(state.item_evidence()))
    ratio = (weights[0] + weights[2]) / weights[1]
    stop = PARAMS["stop_probability_scale"] + np.exp(
        -PARAMS["stop_probability_growth"] * ratio
    )
    stop = min(stop, 1.0 - 2 * np.finfo(np.float32).eps)
    expected = np.array([weights[0], 0, weights[2], 0])
    np.testing.assert_allclose(
        state.outcome_probabilities(),
        np.r_[stop, (1 - stop) * expected / expected.sum()],
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "policy",
    [ExponentialEvidenceSupportRatio, RelativePowerSupportRatio,
     RawSupportRatioSoftmaxChoice, RawSupportRatioRelativeChoice],
)
def test_first_stop_has_zero_growth_gradient(policy):
    def first_stop(growth):
        params = {**PARAMS, "stop_probability_growth": growth}
        model = CMR(2, params, response_policy_create_fn=policy)
        return (
            simulate_study(model, jnp.array([1, 2]))
            .start_retrieving()
            .outcome_probability(0)
        )

    value, gradient = jax.jit(jax.value_and_grad(first_stop))(0.3)
    assert value == pytest.approx(PARAMS["stop_probability_scale"])
    assert gradient == 0.0


def test_terminal_item_learns_end_context_and_has_no_recall_code():
    model = TerminalItemCMR(3, PARAMS)
    ordinary = CMR(3, PARAMS)
    assert model.items.shape == (4, 4)
    assert ordinary.items.shape == (3, 3)
    assert model.context.size == ordinary.context.size
    assert model.recallable.size == ordinary.recallable.size == 3
    studied = simulate_study(model, jnp.array([1, 2, 3]))
    ready = studied.start_retrieving()
    increment = ready.mcf.state[:, -1] - studied.mcf.state[:, -1]
    np.testing.assert_allclose(
        increment, studied.context.state * studied.mfc_learning_rate, atol=2e-7
    )
    np.testing.assert_allclose(
        ready.mfc.state[-1] - studied.mfc.state[-1],
        studied.context.state * studied.mfc_learning_rate,
        atol=2e-7,
    )

    # Same recall count and eligible items. Changing the end association changes STOP.
    unlearned = ready.replace(mcf=studied.mcf)
    assert ready.outcome_probability(0) > unlearned.outcome_probability(0)
    for state in (ready, ready.retrieve(2)):
        raw = state.mcf.probe(state.context.state)
        eligible = np.r_[state.recallable, True]
        weights = np.array(raw) * eligible
        weights = (
            (weights / weights.max()) ** PARAMS["choice_sensitivity"]
            + np.finfo(np.float32).eps
        ) * eligible
        expected = np.r_[weights[-1], weights[:-1]] / weights.sum()
        np.testing.assert_allclose(state.outcome_probabilities(), expected, rtol=3e-6)


@pytest.mark.parametrize(
    "factory,policy",
    [
        (make_factory, RelativePowerSupportRatio),
        (make_factory, RawSupportRatioSoftmaxChoice),
        (make_factory, RawSupportRatioRelativeChoice),
        (make_factory, PositionalStop),
        (make_factory, RecallUntilExhausted),
        (make_factory, ExponentialEvidenceSupportRatio),
        (make_factory, ExponentialStopCompetition),
        (make_factory, PowerLawStopCompetition),
        (make_factory, FiniteWindowRace),
        (make_terminal_item_factory, LearnedStopCompetition),
        (make_terminal_item_factory, PowerLawStopCompetition),
    ],
)
def test_simulation_and_likelihood_use_the_same_response_distribution(factory, policy):
    model = factory(response_policy_create_fn=policy)(
        recall_fixture(), None
    ).create_trial_model(jnp.array(0), PARAMS)
    ready = simulate_study(model, jnp.array([3, 1, 3, 2])).start_retrieving()
    _, recalls = jax.jit(simulate_free_recall, static_argnums=1)(
        ready, 5, jax.random.key(28)
    )
    actual = jax.jit(sequence_probabilities)(ready, recalls)
    state = ready
    for item, probability in zip(recalls, actual, strict=True):
        expected = state.outcome_probabilities()
        np.testing.assert_allclose(expected.sum(), 1.0, atol=2e-6)
        np.testing.assert_allclose(probability, expected[item], rtol=3e-6)
        scalar_scores = jax.vmap(state.outcome_probability)(jnp.arange(expected.size))
        np.testing.assert_allclose(scalar_scores, expected, rtol=3e-6)
        assert probability > 0
        state = state.retrieve(item)
    for state in (ready.retrieve(0), ready.retrieve(3).retrieve(1).retrieve(2)):
        expected = state.outcome_probabilities()
        np.testing.assert_array_equal(expected[0], 1)
        np.testing.assert_array_equal(expected[1:], 0)


@pytest.mark.parametrize("x64", [False, True])
def test_deadline_race_matches_first_passage_integrals_and_derivatives(x64):
    boundary = 1.25
    drifts = np.array([-0.25, 0.7, 2.4])
    eligible = jnp.array([True, True, False])

    def reference(values, boundary=boundary):
        def survival(t, v):
            return norm.cdf((boundary - v * t) / np.sqrt(t)) - np.exp(
                2 * boundary * v
            ) * norm.cdf(-(boundary + v * t) / np.sqrt(t))

        def winner(t, v, other):
            return (
                boundary
                * np.exp(-((boundary - v * t) ** 2) / (2 * t))
                / np.sqrt(2 * np.pi * t**3)
                * survival(t, other)
            )

        first, second = values[:2]
        return np.array(
            [
                survival(1, first) * survival(1, second),
                quad(winner, 0, 1, args=(first, second))[0],
                quad(winner, 0, 1, args=(second, first))[0],
                0,
            ]
        )

    with jax.enable_x64(x64):
        score = jax.jit(lambda v: finite_window_probabilities(v, eligible, boundary))
        np.testing.assert_array_equal(
            finite_window_probabilities(
                jnp.array([]), jnp.array([], dtype=bool), boundary
            ),
            [1.0],
        )
        np.testing.assert_allclose(
            score(jnp.asarray(drifts)), reference(drifts), atol=2e-7
        )
        derivative = jax.jit(jax.jacrev(score))(jnp.asarray(drifts))
        step = np.eye(3) * 1e-5
        numeric = np.column_stack(
            [(reference(drifts + d) - reference(drifts - d)) / 2e-5 for d in step]
        )
        np.testing.assert_allclose(derivative, numeric, atol=3e-6)
        boundary_gradient = jax.jacrev(
            lambda a: finite_window_probabilities(jnp.asarray(drifts), eligible, a)
        )(boundary)
        expected_gradient = (
            reference(drifts, boundary + 1e-5) - reference(drifts, boundary - 1e-5)
        ) / 2e-5
        np.testing.assert_allclose(boundary_gradient, expected_gradient, atol=3e-6)
