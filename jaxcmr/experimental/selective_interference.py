"""Simulation helpers for selective interference experiments.

Provides composable building blocks to model trauma-film + reminder +
interference paradigms with any memory-search model that exposes context
integration and cue-driven retrieval.
"""

from typing import Callable

import jax.numpy as jnp
from jax import lax, random

from jaxcmr.simulation import simulate_free_recall
from jaxcmr.typing import Array, Float_, Int_, Integer, MemorySearch, PRNGKeyArray


def drift_context(
    model: MemorySearch, drift_rate: Float_, steps: Int_
) -> MemorySearch:
    """Advance model state via repeated drift without item input.

    Args:
      model: Memory search model with a context attribute.
      drift_rate: Drift rate applied per step.
      steps: Number of drift steps to apply.
    """

    def step(_: Int_, state: MemorySearch) -> MemorySearch:
        context = state.context.integrate(state.context.initial_state, drift_rate)
        return state.replace(context=context)

    return lax.fori_loop(0, steps, step, model)


def cue_context(
    model: MemorySearch, cue_id: Int_, drift_rate: Float_, probe_fn: Callable
) -> MemorySearch:
    """Integrate a cue into the model state.

    Args:
      model: Memory search model with context and memory associations.
      cue_id: One-indexed cue item used to reactivate film context.
      drift_rate: Drift rate used for cue integration.
      probe_fn: Callable mapping cue ids to feature vectors.
    """
    cue_features = probe_fn(cue_id)
    cue_input = model.mfc.probe(cue_features)
    context = model.context.integrate(cue_input, drift_rate)
    return model.replace(context=context)


def apply_reminder_and_interference(
    model: MemorySearch,
    reminder_id: Int_,
    reminder_drift_rate: Float_,
    interference_steps: Int_,
    interference_drift_rate: Float_,
    probe_fn: Callable,
) -> tuple[MemorySearch, MemorySearch]:
    """Returns control and interference models after reminder and drift phases.

    Args:
      model: Memory search model after film encoding.
      reminder_id: One-indexed reminder cue.
      reminder_drift_rate: Drift rate for integrating the reminder.
      interference_steps: Drift steps applied only in the interference branch.
      interference_drift_rate: Drift rate for the interference drift.
      probe_fn: Callable mapping cue ids to feature vectors.
    """
    reminded = cue_context(model, reminder_id, reminder_drift_rate, probe_fn)
    interfered = drift_context(reminded, interference_drift_rate, interference_steps)
    return reminded, interfered


def simulate_vigilance_block(
    model: MemorySearch,
    cue_ids: Integer[Array, " trials"],
    cue_drift_rate: Float_,
    max_intrusions: Int_,
    rng: PRNGKeyArray,
    probe_fn: Callable,
) -> Integer[Array, " trials recall_events"]:
    """Sample intrusions across a block of VIT cues.

    Args:
      model: Memory search model representing post-manipulation state.
      cue_ids: One-indexed cues for each VIT trial.
      cue_drift_rate: Drift rate used to integrate cues.
      max_intrusions: Maximum intrusion events to sample per cue.
      rng: Random key for sampling.
      probe_fn: Callable mapping cue ids to feature vectors.
    """
    recalls = jnp.zeros((cue_ids.size, max_intrusions), dtype=int)

    def body(
        i: Int_,
        carry: tuple[Integer[Array, " trials recall_events"], PRNGKeyArray],
    ) -> tuple[Integer[Array, " trials recall_events"], PRNGKeyArray]:
        buffer, key = carry
        trial_key, key = random.split(key)
        cued = cue_context(model, cue_ids[i], cue_drift_rate, probe_fn)
        cued = cued.start_retrieving()
        _, trial_recalls = simulate_free_recall(cued, max_intrusions, trial_key)
        return buffer.at[i].set(trial_recalls), key

    recalls, _ = lax.fori_loop(0, cue_ids.size, body, (recalls, rng))
    return recalls


def simulate_recognition_block(
    model: MemorySearch,
    cue_ids: Integer[Array, " trials"],
    target_ids: Integer[Array, " trials"],
    cue_drift_rate: Float_,
    decision_threshold: Float_,
    rng: PRNGKeyArray,
    probe_fn: Callable,
) -> Integer[Array, " trials"]:
    """Sample binary old/new decisions across a block of recognition trials.

    Args:
      model: Memory search model representing post-manipulation state.
      cue_ids: One-indexed cues for each recognition trial.
      target_ids: One-indexed targets paired with each cue.
      cue_drift_rate: Drift rate used to integrate cues.
      decision_threshold: Criterion for endorsing targets as old.
      rng: Random key for sampling.
      probe_fn: Callable mapping cue ids to feature vectors.
    """
    decisions = jnp.zeros(cue_ids.size, dtype=int)

    def body(
        i: Int_, carry: tuple[Integer[Array, " trials"], PRNGKeyArray]
    ) -> tuple[Integer[Array, " trials"], PRNGKeyArray]:
        buffer, key = carry
        trial_key, key = random.split(key)
        cued = cue_context(model, cue_ids[i], cue_drift_rate, probe_fn)
        cued = cued.start_retrieving()
        target_probability = cued.outcome_probability(target_ids[i])
        decision = jnp.where(target_probability >= decision_threshold, 1, 0)
        return buffer.at[i].set(decision), key

    decisions, _ = lax.fori_loop(0, cue_ids.size, body, (decisions, rng))
    return decisions
