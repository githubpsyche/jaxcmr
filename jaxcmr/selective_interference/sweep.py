"""Parameter sweep execution and analysis.

A parameter sweep varies one simulation parameter (e.g. interference
MCF scale) across a set of values while holding everything else
constant, then collects serial position curves (SPCs) and film-recall
statistics at each value.

The sweep pipeline has three layers:

1. **Single trial** — ``run_from_cache`` takes a pre-built model
   (from ``preparation.prepare_all_subjects``), encodes interference
   and filler items with scaled drift/learning rates, and runs free
   recall.  Returns raw recall sequences.

2. **Batched trials** — ``batched_sweep`` wraps ``run_from_cache``
   with ``jit(vmap(vmap(...)))`` to run all subjects x replications
   in one compiled call.

3. **Sweep loop** — ``run_sweep`` iterates over sweep values, calls
   ``batched_sweep`` at each value, remaps the raw recalls (via
   ``remapping.standard_remap`` by default), computes an SPC and
   film-recall statistics, and returns the collected results.

``sweep_defaults`` provides the baseline argument dict for
``batched_sweep``; ``sweep_rngs`` splits a PRNG key into the
``(n_subjects, experiment_count)`` shape that ``batched_sweep``
expects; ``film_recalled_stats`` computes mean and 95% CI of film
items recalled across subjects.

"""

from typing import Optional

import jax
import jax.numpy as jnp
from jax import jit, lax, random, vmap

from jaxcmr.analyses.spc import fixed_pres_spc
from jaxcmr.simulation import simulate_free_recall

from .paradigm import Paradigm
from .remapping import standard_remap


def sweep_defaults(paradigm: Paradigm) -> dict:
    """Return default ``batched_sweep`` arguments.

    Parameters
    ----------
    paradigm : Paradigm
        Paradigm geometry.

    Returns
    -------
    dict

    """
    return dict(
        interference_items=paradigm.interference_items,
        filler_items=paradigm.filler_items,
        interference_drift_scale=jnp.float32(1.0),
        interference_mcf_scale=jnp.float32(1.0),
        filler_drift_scale=jnp.float32(1.0),
        filler_mcf_scale=jnp.float32(1.0),
        start_drift_scale=jnp.float32(1.0),
        tau_scale=jnp.float32(1.0),
    )


def run_from_cache(
    cached_model,
    rng: jax.Array,
    interference_items: jax.Array,
    filler_items: jax.Array,
    interference_drift_scale: float,
    interference_mcf_scale: float,
    filler_drift_scale: float,
    filler_mcf_scale: float,
    start_drift_scale: float,
    tau_scale: float,
    max_recall: int,
) -> jax.Array:
    """Encode interference and fillers, then recall.

    Parameters
    ----------
    cached_model : MemorySearch
        Post-reminder model state from ``prepare_single_subject``.
    rng : jax.Array
        PRNG key for this trial.
    interference_items : jax.Array
        Item IDs for the interference phase.
    filler_items : jax.Array
        Item IDs for the post-interference filler phase.
    interference_drift_scale : float
        Multiplier on encoding drift rate during interference.
    interference_mcf_scale : float
        Multiplier on MCF learning rate during interference.
    filler_drift_scale : float
        Multiplier on encoding drift rate during fillers.
    filler_mcf_scale : float
        Multiplier on MCF learning rate during fillers.
    start_drift_scale : float
        Multiplier on start drift rate at recall onset.
    tau_scale : float
        Multiplier on choice sensitivity (tau) at recall.
    max_recall : int
        Maximum number of recall attempts.

    Returns
    -------
    jax.Array

    """
    original_drift = cached_model.encoding_drift_rate
    original_mcf_lr = cached_model._mcf_learning_rate

    # Interference
    interference_drift = jnp.clip(
        interference_drift_scale * original_drift, 0.0, 1.0
    )
    model = cached_model.replace(
        encoding_drift_rate=interference_drift,
        _mcf_learning_rate=interference_mcf_scale * original_mcf_lr,
    )
    model = lax.fori_loop(
        0, interference_items.size,
        lambda i, m: m.experience(interference_items[i]), model,
    )

    # Filler
    filler_drift = jnp.clip(filler_drift_scale * original_drift, 0.0, 1.0)
    model = model.replace(
        encoding_drift_rate=filler_drift,
        _mcf_learning_rate=filler_mcf_scale * original_mcf_lr,
    )
    model = lax.fori_loop(
        0, filler_items.size,
        lambda i, m: m.experience(filler_items[i]), model,
    )

    # Recall
    scaled_start_drift = jnp.clip(
        start_drift_scale * cached_model.start_drift_rate, 0.0, 1.0
    )
    model = model.replace(
        start_drift_rate=scaled_start_drift,
        mcf_sensitivity=tau_scale * cached_model.mcf_sensitivity,
    )
    model = model.start_retrieving()
    _, recalls = simulate_free_recall(model, max_recall, rng)
    return recalls


batched_sweep = jit(
    vmap(vmap(
        run_from_cache,
        in_axes=(None, 0, None, None, None, None, None, None, None, None, None),
    ), in_axes=(0, 0, None, None, None, None, None, None, None, None, None)),
    static_argnums=(10,),
)


def sweep_rngs(
    rng: jax.Array,
    n_subjects: int,
    experiment_count: int,
) -> tuple[jax.Array, jax.Array]:
    """Split RNG into per-subject, per-replication keys.

    Parameters
    ----------
    rng : jax.Array
        Parent PRNG key.
    n_subjects : int
        Number of subjects.
    experiment_count : int
        Replications per subject.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``(rngs_2d, next_rng)`` where ``rngs_2d`` has shape
        ``(n_subjects, experiment_count, 2)``.

    """
    rng, sub_rng = random.split(rng)
    rngs = random.split(sub_rng, n_subjects * experiment_count)
    return rngs.reshape(n_subjects, experiment_count, -1), rng


def film_recalled_stats(
    recalls: jax.Array,
    paradigm: Paradigm,
    n_subjects: int,
) -> tuple[float, float, float]:
    """Compute mean and 95% CI of film items recalled.

    Parameters
    ----------
    recalls : jax.Array
        Recall sequences, shape ``(n_trials, max_recall)``.
    paradigm : Paradigm
        Paradigm geometry.
    n_subjects : int
        Number of subjects.

    Returns
    -------
    tuple[float, float, float]
        ``(mean, ci_lower, ci_upper)``

    """
    film_mask = (recalls >= 1) & (recalls <= paradigm.n_film)
    per_trial = jnp.sum(film_mask, axis=1).astype(float)
    per_sub = per_trial.reshape(n_subjects, paradigm.experiment_count)
    sub_means = jnp.mean(per_sub, axis=1)
    mu = float(jnp.mean(sub_means))
    se = float(jnp.std(sub_means) / jnp.sqrt(n_subjects))
    return mu, mu - 1.96 * se, mu + 1.96 * se


_SWEEP_ARG_NAMES = [
    "interference_items", "filler_items",
    "interference_drift_scale", "interference_mcf_scale",
    "filler_drift_scale", "filler_mcf_scale",
    "start_drift_scale", "tau_scale",
]


def run_sweep(
    cached_models: jax.Array,
    rng: jax.Array,
    sweep_values: list,
    swept_param: str,
    *,
    paradigm: Paradigm,
    n_subjects: int,
    overrides: Optional[dict] = None,
    remap_fn=standard_remap,
    n_presented: Optional[int] = None,
) -> tuple[list, list, jax.Array]:
    """Run a parameter sweep and compute SPCs and stats.

    Parameters
    ----------
    cached_models : jax.Array
        Pre-computed post-reminder model states.
    rng : jax.Array
        PRNG key.
    sweep_values : list
        Values to sweep over.
    swept_param : str
        Name of the ``batched_sweep`` argument to vary.
    paradigm : Paradigm
        Paradigm geometry.
    n_subjects : int
        Number of subjects.
    overrides : dict, optional
        Additional ``batched_sweep`` argument overrides.
    remap_fn : callable
        Recall remapping function. Must accept
        ``(recalls, paradigm, **kwargs)``.
    n_presented : int, optional
        Number of presented positions for SPC. Defaults to
        ``paradigm.list_length``.

    Returns
    -------
    tuple[list, list, jax.Array]
        ``(spcs, stats, next_rng)``

    """
    if n_presented is None:
        n_presented = paradigm.list_length
    args = sweep_defaults(paradigm)
    if overrides:
        args.update(overrides)

    spcs, stats = [], []
    for val in sweep_values:
        rngs_2d, rng = sweep_rngs(rng, n_subjects, paradigm.experiment_count)
        args[swept_param] = jnp.float32(val)
        ordered = [args[k] for k in _SWEEP_ARG_NAMES]
        recalls_3d = batched_sweep(
            cached_models, rngs_2d, *ordered, paradigm.max_recall
        )
        recalls = recalls_3d.reshape(-1, recalls_3d.shape[-1])
        recalls = remap_fn(recalls, paradigm)
        spcs.append(fixed_pres_spc(recalls, n_presented))
        stats.append(film_recalled_stats(recalls, paradigm, n_subjects))
    return spcs, stats, rng
