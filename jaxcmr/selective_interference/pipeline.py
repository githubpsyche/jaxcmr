"""Simulation pipeline for selective interference experiments.

The selective interference paradigm encodes items across six phases::

    ⓪ create → ① film → ② break → ③ reminder → ④ interference → ⑤ filler → recall

Any phase boundary (⓪–⑤) is a valid cache point.  The high-level
API — ``prepare_sweep`` and ``run_sweep`` — automates caching and
batching for parameter sweeps.

The module is organized in layers that build on each other:

**Layer 1 — Rate configuration** (``configure_rates``).  Takes
batched models (leading subject dimension) and keyword scale
factors.  Returns a new Pytree with rates adjusted via
``.replace()``; the original models are not mutated.  Drift-rate
scales multiply the base rate and clip to [0, 1].  MCF scales and
``tau_scale`` multiply their base values directly.
``primacy_scale`` and ``primacy_decay`` replace model values
directly (not multiplicative).

**Layer 2 — Phase table and trial generation** (``_PHASES``,
``_CACHE_POINTS``, ``_encode_phase``, ``_make_trial_fn``).
``_PHASES`` lists each phase as
``(name, method, paradigm_attr, pre_step)`` where ``method`` is a
``PhasedMemorySearch`` protocol method name and ``pre_step`` is an
optional method called before the item loop (used only by the
reminder phase for ``start_reminders``).  ``_encode_phase`` loops
items via ``lax.fori_loop`` using ``getattr``.  A cache point at
index *i* means phases ``[:i]`` are pre-encoded by
``_encode_prefix`` and phases ``[i:]`` are encoded inside the trial
function built by ``_make_trial_fn``.

**Layer 3 — Batching** (``batch_trial``).  Wraps any trial
function for ``(subjects × replications)`` execution.  Uses the
explicit ``n_args`` count (or falls back to ``inspect.signature``)
to build ``vmap`` ``in_axes``: model is vmapped over subjects
(outer) only, RNG over both subjects (outer) and reps (inner),
item arrays are broadcast (``None``).  Trailing arguments
(``n_static``) are passed to ``jit`` via ``static_argnums``.

**Layer 4 — Sweep utilities** (``sweep_rngs``,
``film_recalled_stats``).  ``sweep_rngs`` splits a parent RNG into
a ``(n_subjects, experiment_count)`` key array and an advanced
``rng``.  ``film_recalled_stats`` computes mean and 95% CI of film
items recalled across subjects.

**Layer 5 — High-level sweep API** (``PreparedSweep``,
``prepare_sweep``, ``run_sweep``).

``prepare_sweep`` is a one-time setup that:

1. Creates one model per subject via ``vmap(model_factory)``.
2. Applies any fixed pre-cache scales via ``configure_rates``.
3. Encodes phases up to the chosen ``cache_after`` boundary
   using ``_encode_prefix`` with the ``_PHASES[:idx]`` slice.
4. Generates a trial function for the remaining phases via
   ``_make_trial_fn(_PHASES[idx:])`` and collects the
   deduplicated item arrays into ``item_args``.
5. Wraps the trial function with ``batch_trial``.
6. Returns a ``PreparedSweep`` holding the cached models, the
   batched callable, item args, and dimension counts.

``run_sweep`` iterates over values of a single swept parameter:

1. Separates ``**scales`` into one list-valued key (the swept
   parameter) and any scalar-valued keys (held fixed).
2. For each swept value, calls ``configure_rates`` on the cached
   models — this returns a new Pytree with adjusted rates without
   mutating the cache.
3. Splits fresh RNG keys via ``sweep_rngs``.
4. Calls ``prepared.batched(ready, rngs, *prepared.item_args)``
   to run all subjects × replications.
5. Stacks results into shape
   ``(n_values, n_subjects, experiment_count, max_recall)``
   and returns it with the advanced ``rng``.

The same ``PreparedSweep`` can be reused across multiple
``run_sweep`` calls with different swept parameters, since the
cached models are never mutated.

"""

from typing import Callable, Mapping, NamedTuple, Sequence

import jax.numpy as jnp
from jax import jit, lax, random, vmap

from jaxcmr.simulation import simulate_free_recall
from jaxcmr.typing import Array, Float, Integer, PRNGKeyArray

from .paradigm import (
    Paradigm,
    make_extended_break,
    make_extended_filler,
    make_extended_interference,
)
from .typing import PhasedMemorySearch, PhasedMemorySearchCreateFn


# ── Layer 1: Rate configuration ───────────────────────────────────


def configure_rates(
    models: PhasedMemorySearch,
    **scales: float,
) -> PhasedMemorySearch:
    """Apply sweep scale factors to per-subject base rates.

    Parameters
    ----------
    models : PhasedMemorySearch
        Batched models with leading subject dimension.
    **scales : float
        Scale factors keyed by name: ``encoding_drift_scale``,
        ``break_drift_scale``, ``break_mcf_scale``,
        ``reminder_start_drift_scale``, ``reminder_drift_scale``,
        ``interference_drift_scale``, ``interference_mcf_scale``,
        ``filler_drift_scale``, ``filler_mcf_scale``,
        ``start_drift_scale``, ``tau_scale``.
        Unspecified scales default to 1.0.
        ``primacy_scale`` and ``primacy_decay`` replace the
        model values directly when provided.

    Returns
    -------
    PhasedMemorySearch

    """
    def _apply(model):
        encoding_drift = model.encoding_drift_rate
        start_drift = model.start_drift_rate
        return model.replace(
            encoding_drift_rate=jnp.clip(
                scales.get("encoding_drift_scale", 1.0)
                * encoding_drift, 0.0, 1.0,
            ),
            break_drift_rate=jnp.clip(
                scales.get("break_drift_scale", 1.0)
                * encoding_drift, 0.0, 1.0,
            ),
            break_mcf_scale=scales.get("break_mcf_scale", 1.0),
            reminder_start_drift_rate=jnp.clip(
                scales.get("reminder_start_drift_scale", 1.0)
                * start_drift, 0.0, 1.0,
            ),
            reminder_drift_rate=jnp.clip(
                scales.get("reminder_drift_scale", 1.0)
                * encoding_drift, 0.0, 1.0,
            ),
            interference_drift_rate=jnp.clip(
                scales.get("interference_drift_scale", 1.0)
                * encoding_drift, 0.0, 1.0,
            ),
            interference_mcf_scale=scales.get("interference_mcf_scale", 1.0),
            filler_drift_rate=jnp.clip(
                scales.get("filler_drift_scale", 1.0)
                * encoding_drift, 0.0, 1.0,
            ),
            filler_mcf_scale=scales.get("filler_mcf_scale", 1.0),
            start_drift_rate=jnp.clip(
                scales.get("start_drift_scale", 1.0)
                * start_drift, 0.0, 1.0,
            ),
            mcf_sensitivity=(
                scales.get("tau_scale", 1.0)
                * model.mcf_sensitivity
            ),
            primacy_scale=scales.get("primacy_scale", model.primacy_scale),
            primacy_decay=scales.get("primacy_decay", model.primacy_decay),
        )
    return vmap(_apply)(models)


# ── Layer 2: Phase table + trial generation ──────────────────────

_PHASES = [
    ("film",         "experience_film",         "film_items",  None),
    ("break",        "experience_break",        "break_items", None),
    ("reminder",     "remind",                  "film_items",  "start_reminders"),
    ("interference", "experience_interference", "interference_items", None),
    ("filler",       "experience_filler",       "filler_items", None),
]

_CACHE_POINTS = {
    "creation": 0,
    "film": 1,
    "break": 2,
    "reminder": 3,
    "interference": 4,
    "filler": 5,
}

_TIER_LIST_LENGTH = {
    "standard": "list_length",
    "break_extended": "break_extended_list_length",
    "interference_extended": "interference_extended_list_length",
    "filler_extended": "filler_extended_list_length",
}

_TIER_ATTR_OVERRIDES: dict[str, dict[str, str]] = {
    "standard": {},
    "break_extended": {
        "break_items": "break_extended_break_items",
        "interference_items": "break_extended_interference_items",
        "filler_items": "break_extended_filler_items",
    },
    "interference_extended": {
        "interference_items": "interference_extended_interference_items",
        "filler_items": "interference_extended_filler_items",
    },
    "filler_extended": {
        "filler_items": "filler_extended_filler_items",
    },
}


def _encode_phase(
    model: PhasedMemorySearch,
    items: Integer[Array, " n_items"],
    method: str,
    pre_step: str | None = None,
) -> PhasedMemorySearch:
    """Encode items via a named protocol method.

    Parameters
    ----------
    model : PhasedMemorySearch
        Model state before encoding.
    items : Integer[Array, " n_items"]
        Item IDs (1-indexed, 0 skipped).
    method : str
        Protocol method name (e.g. ``"experience_film"``).
    pre_step : str or None
        Optional method to call before the loop
        (e.g. ``"start_reminders"``).

    Returns
    -------
    PhasedMemorySearch

    """
    if pre_step is not None:
        model = getattr(model, pre_step)()
    if items.size == 0:
        return model
    return lax.fori_loop(
        0, items.size,
        lambda i, m: getattr(m, method)(items[i]), model,
    )


def _encode_prefix(
    model: PhasedMemorySearch,
    phases: list[tuple[str, str, str, str | None]],
    paradigm: Paradigm,
    overrides: dict[str, str] | None = None,
) -> PhasedMemorySearch:
    """Encode pre-cache phases sequentially (unrolls at trace time).

    Parameters
    ----------
    model : PhasedMemorySearch
        Model state before encoding.
    phases : list[tuple[str, str, str, str | None]]
        Phase specs ``(name, method, paradigm_attr, pre_step)``.
    paradigm : Paradigm
        Paradigm geometry (supplies item arrays via ``getattr``).
    overrides : dict[str, str] or None
        Attribute name overrides from tier selection.

    Returns
    -------
    PhasedMemorySearch

    """
    if overrides is None:
        overrides = {}
    for _, method, attr, pre in phases:
        actual_attr = overrides.get(attr, attr)
        model = _encode_phase(model, getattr(paradigm, actual_attr), method, pre)
    return model


def _make_trial_fn(
    post_phases: list[tuple[str, str, str, str | None]],
) -> tuple[Callable, int]:
    """Build a trial function for the given remaining phases.

    Parameters
    ----------
    post_phases : list[tuple[str, str, str, str | None]]
        Phase specs ``(name, method, paradigm_attr, pre_step)``
        for phases remaining after the cache point.

    Returns
    -------
    tuple[Callable, int]
        ``(trial_fn, n_item_args)`` where ``n_item_args`` is the
        number of unique item arrays the closure expects.

    """
    seen: dict[str, int] = {}
    phase_to_arg: list[int] = []
    for _, _, attr, _ in post_phases:
        if attr not in seen:
            seen[attr] = len(seen)
        phase_to_arg.append(seen[attr])
    n_item_args = len(seen)

    def trial(model, rng, *args):
        for (_, method, _, pre), idx in zip(post_phases, phase_to_arg):
            model = _encode_phase(model, args[idx], method, pre)
        model = model.start_retrieving()
        _, recalls = simulate_free_recall(model, args[n_item_args], rng)
        return recalls

    return trial, n_item_args


# ── Layer 3: Batching ─────────────────────────────────────────────


def batch_trial(
    trial_fn: Callable,
    n_args: int | None = None,
    n_static: int = 0,
) -> Callable:
    """Wrap a trial function for subjects x replications execution.

    Parameters
    ----------
    trial_fn : Callable
        Trial function with signature
        ``(model, rng, *item_args, *static_args) -> recalls``.
    n_args : int, optional
        Total number of positional arguments.  When ``None``
        (default), inferred via ``inspect.signature``.
    n_static : int
        Number of trailing arguments to mark as JIT-static
        (e.g. ``max_recall``).

    Returns
    -------
    Callable

    """
    if n_args is None:
        import inspect
        n_args = len(inspect.signature(trial_fn).parameters)
    n_extra = n_args - 2
    inner = (None, 0) + (None,) * n_extra
    outer = (0, 0) + (None,) * n_extra
    static = tuple(range(n_args - n_static, n_args)) if n_static else ()
    return jit(
        vmap(vmap(trial_fn, in_axes=inner), in_axes=outer),
        static_argnums=static,
    )


# ── Layer 4: Sweep helpers ────────────────────────────────────────


def sweep_rngs(
    rng: PRNGKeyArray,
    n_subjects: int,
    experiment_count: int,
) -> tuple[PRNGKeyArray, PRNGKeyArray]:
    """Split RNG into per-subject, per-replication keys.

    Parameters
    ----------
    rng : PRNGKeyArray
        Parent PRNG key.
    n_subjects : int
        Number of subjects.
    experiment_count : int
        Replications per subject.

    Returns
    -------
    tuple[PRNGKeyArray, PRNGKeyArray]
        ``(rngs_2d, next_rng)`` where ``rngs_2d`` has shape
        ``(n_subjects, experiment_count, 2)``.

    """
    rng, sub_rng = random.split(rng)
    rngs = random.split(sub_rng, n_subjects * experiment_count)
    return rngs.reshape(n_subjects, experiment_count, -1), rng


def film_recalled_stats(
    recalls: Integer[Array, " n_trials max_recall"],
    paradigm: Paradigm,
    n_subjects: int,
) -> tuple[float, float, float]:
    """Compute mean and 95% CI of film items recalled.

    Parameters
    ----------
    recalls : Integer[Array, " n_trials max_recall"]
        Recall sequences.
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


# ── Layer 5: High-level sweep API ─────────────────────────────────


class PreparedSweep(NamedTuple):
    """Reusable cache for parameter sweeps."""

    models: PhasedMemorySearch
    batched: Callable
    item_args: tuple
    n_subjects: int
    experiment_count: int
    item_slots: dict[str, int]


def prepare_sweep(
    params: Mapping[str, Float[Array, " n_subjects"]],
    paradigm: Paradigm,
    model_factory: PhasedMemorySearchCreateFn,
    cache_after: str = "reminder",
    tier: str = "standard",
    **scales: float,
) -> PreparedSweep:
    """Create models, configure pre-cache rates, encode through cache.

    Parameters
    ----------
    params : Mapping[str, Float[Array, " n_subjects"]]
        Per-subject fitted parameters.
    paradigm : Paradigm
        Paradigm geometry.
    model_factory : PhasedMemorySearchCreateFn
        Creator function for PhasedMemorySearch models.
    cache_after : str
        Phase boundary for caching: ``"creation"``, ``"film"``,
        ``"break"``, ``"reminder"``, ``"interference"``,
        ``"filler"``.
    tier : str
        Array-width tier: ``"standard"`` (default),
        ``"break_extended"``, ``"interference_extended"``,
        or ``"filler_extended"``.
    **scales : float
        Fixed scale factors applied before caching.

    Returns
    -------
    PreparedSweep

    """
    if cache_after not in _CACHE_POINTS:
        raise ValueError(f"Unknown cache_after: {cache_after!r}")
    if tier not in _TIER_LIST_LENGTH:
        raise ValueError(f"Unknown tier: {tier!r}")

    overrides = _TIER_ATTR_OVERRIDES[tier]
    list_length = getattr(paradigm, _TIER_LIST_LENGTH[tier])

    # 1. Create one model per subject
    n_subjects = next(iter(params.values())).shape[0]
    models = vmap(
        lambda p: model_factory(list_length, p, None),
    )(params)

    # 2. Apply fixed pre-cache scales
    configured = configure_rates(models, **scales)

    # 3. Encode phases up to the cache boundary
    idx = _CACHE_POINTS[cache_after]
    pre, post = _PHASES[:idx], _PHASES[idx:]
    if pre:
        cached = vmap(
            lambda m: _encode_prefix(m, pre, paradigm, overrides),
        )(configured)
    else:
        cached = configured

    # 4. Generate trial function and collect deduplicated item args
    trial_fn, n_item_args = _make_trial_fn(post)
    slots: dict[str, int] = {}
    item_args: list = []
    for _, _, attr, _ in post:
        if attr not in slots:
            slots[attr] = len(slots)
            actual_attr = overrides.get(attr, attr)
            item_args.append(getattr(paradigm, actual_attr))

    # 5–6. Wrap with batch_trial and return
    n_total = 2 + n_item_args + 1  # model + rng + items + max_recall
    return PreparedSweep(
        models=cached,
        batched=batch_trial(trial_fn, n_args=n_total, n_static=1),
        item_args=(*item_args, paradigm.max_recall),
        n_subjects=n_subjects,
        experiment_count=paradigm.experiment_count,
        item_slots=slots,
    )


def run_sweep(
    prepared: PreparedSweep,
    rng: PRNGKeyArray,
    **scales: float | list[float],
) -> tuple[Integer[Array, "n_values n_subjects n_reps max_recall"], PRNGKeyArray]:
    """Run a parameter sweep on prepared models.

    Parameters
    ----------
    prepared : PreparedSweep
        From ``prepare_sweep``.
    rng : PRNGKeyArray
        Parent PRNG key (consumed and returned).
    **scales : float
        Scale factors.  Exactly one must be array-valued (swept);
        scalar values are held fixed across iterations.

    Returns
    -------
    tuple[Integer[Array, "n_values n_subjects n_reps max_recall"], PRNGKeyArray]

    """
    swept_name = None
    swept_values = None
    fixed = {}
    for name, val in scales.items():
        if hasattr(val, "__len__"):
            swept_name, swept_values = name, val
        else:
            fixed[name] = val

    if swept_name is None or swept_values is None:
        raise ValueError("Exactly one scale must be array-valued (swept)")

    results = []
    for v in swept_values:
        ready = configure_rates(
            prepared.models,
            **{swept_name: float(v), **fixed},
        )
        rngs, rng = sweep_rngs(
            rng, prepared.n_subjects, prepared.experiment_count,
        )
        results.append(prepared.batched(ready, rngs, *prepared.item_args))

    return jnp.stack(results), rng


# ── Layer 6: Item-count sweeps ────────────────────────────────────


_COUNT_PHASE: dict[str, tuple[str, Callable]] = {
    "n_break": ("break_items", make_extended_break),
    "n_interference": ("interference_items", make_extended_interference),
    "n_filler": ("filler_items", make_extended_filler),
}


def run_count_sweep(
    prepared: PreparedSweep,
    rng: PRNGKeyArray,
    paradigm: Paradigm,
    phase: str,
    count_values: Sequence[int],
) -> tuple[Integer[Array, "n_values n_subjects n_reps max_recall"], PRNGKeyArray]:
    """Run an item-count sweep on prepared models.

    Parameters
    ----------
    prepared : PreparedSweep
        From ``prepare_sweep``.
    rng : PRNGKeyArray
        Parent PRNG key (consumed and returned).
    paradigm : Paradigm
        Paradigm geometry (supplies ``make_extended_*``).
    phase : str
        Phase whose item count varies: ``"n_break"``,
        ``"n_interference"``, or ``"n_filler"``.
    count_values : Sequence[int]
        Item counts to sweep over.

    Returns
    -------
    tuple[Integer[Array, "n_values n_subjects n_reps max_recall"], PRNGKeyArray]

    """
    if phase not in _COUNT_PHASE:
        raise ValueError(
            f"Unknown phase {phase!r}; expected one of {list(_COUNT_PHASE)}"
        )
    attr, make_fn = _COUNT_PHASE[phase]
    slot = prepared.item_slots[attr]

    results = []
    for n in count_values:
        items = make_fn(paradigm, int(n))
        args = list(prepared.item_args)
        args[slot] = items
        rngs, rng = sweep_rngs(
            rng, prepared.n_subjects, prepared.experiment_count,
        )
        results.append(prepared.batched(prepared.models, rngs, *args))

    return jnp.stack(results), rng
