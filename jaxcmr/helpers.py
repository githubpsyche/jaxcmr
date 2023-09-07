from copy import copy

import jaxtyping
import numpy as np
from jax import numpy as jnp, jit, lax, random
from jax.tree_util import tree_map
from plum import dispatch
from functools import partial

lb = jnp.finfo(jnp.float32).eps

ScalarInteger = jaxtyping.Integer[jaxtyping.Array, ""] | int | np.int32 | np.int64
ScalarFloat = jaxtyping.Float[jaxtyping.Array, ""] | float | np.float32 | np.float64
ScalarBool = jaxtyping.Bool[jaxtyping.Array, ""] | bool | np.bool_
PRNGKeyArray = jaxtyping.PRNGKeyArray

study_events = "The number of (possible) study events in the simulated trial"
recall_outcomes = "The number of (possible) retrieval outcomes in the simulated trial"
context_feature_units = "Number of units in the context representation"
input_features = "Number of units in the input representation"
item_features = "Number of units in item representations"
features = "Number of units in the representation"
item_count = "Number of unique items in the trial"
output_features = "Number of units in the output representation"
instances = "Number of instances in the memory"
recall_events = "Number of recall events in the simulated trial"
trial_count = "Number of trials in the experiment"

Integer = jaxtyping.Integer
Float = jaxtyping.Float
Array = jaxtyping.Array
Bool = jaxtyping.Bool


def replace(instance, **kwargs):
    """Return a copy of instance with the specified attributes replaced."""
    new_instance = copy(instance)

    for attr, value in kwargs.items():
        setattr(new_instance, attr, value)

    return new_instance


@jit
@dispatch
def normalize_to_unit_length(
    vector: Float[Array, "features"]
) -> Float[Array, "features"]:
    """Enforce magnitude of vector to 1."""
    return vector / jnp.sqrt(jnp.sum(jnp.square(vector)) + lb)


@jit
@dispatch
def normalize_to_sum_one(vector: Float[Array, "features"]) -> Float[Array, "features"]:
    """Enforce sum of vector to 1."""
    return vector / jnp.sum(vector)


@jit
@dispatch
def power_scale(
    vector: Float[Array, "features"], scale: ScalarFloat
) -> Float[Array, "features"]:
    """Scale activation vector by exponent factor using the logsumexp trick to avoid underflow."""
    log_activation = jnp.log(vector)
    return lax.cond(
        jnp.logical_and(jnp.any(vector != 0), scale != 1),
        lambda _: jnp.exp(scale * (log_activation - jnp.max(log_activation))),
        lambda _: vector,
        None,
    )


def tree_transpose(list_of_trees):
    """Convert a list of trees of identical structure into a single tree of arrays."""
    return tree_map(lambda *xs: jnp.array(xs), *list_of_trees)


@jit
@dispatch
def get_list_length(presentation: Integer[Array, "study_events"]) -> ScalarInteger:
    """Return the number of study events in a trial"""
    return jnp.sum(presentation != 0)


@jit
@dispatch
def get_item_count(presentation: Integer[Array, "study_events"]) -> ScalarInteger:
    """Return the number of unique items in a trial"""
    return jnp.max(presentation)


@jit
@dispatch
def recall_by_item_index(
    item_index_by_study_position: Integer[Array, "study_events"],
    study_position_by_recall_position: Integer[Array, "recall_events"],
) -> Integer[Array, "recall_events"]:
    """Trial recall events in terms of item index rather than study position"""
    return lax.map(
        lambda r: lax.select(r == 0, 0, item_index_by_study_position[r - 1]),
        study_position_by_recall_position,
    )


@jit
@dispatch
def recall_by_study_position(
    item_index_by_study_position: Integer[Array, "study_events"],
    item_index_by_recall_position: Integer[Array, "recall_events"],
):
    """Trial recall events in terms of (first) study position rather than item index"""
    return lax.map(
        lambda r: lax.cond(
            r == 0,
            lambda _: 0,
            lambda _: jnp.argmax(item_index_by_study_position == r) + 1,
            None,
        ),
        item_index_by_recall_position,
    )


@jit
@dispatch
def log_likelihood(likelihoods: Float[Array, "..."]) -> ScalarFloat:
    """Return the log-likelihood over a set of likelihoods"""
    return -jnp.sum(jnp.log(likelihoods))


def select_parameters_by_subject(
    subject_indices: Integer[Array, "trial_count"], parameters: list[dict]
) -> list[dict]:
    """Select parameters based on provided subject indices."""
    return [
        next(
            param["fixed"] for param in parameters if param["subject"] == subject_index
        )
        for subject_index in subject_indices
    ]


@partial(jit, static_argnums=(1, 2))
def latin_hypercube_sampling(rng_key, dim, num_samples):
    """Generate Latin Hypercube Samples (LHS) in a JIT-compatible manner."""
    # Step 1: Generate Base Points
    base_points = jnp.linspace(0, 1, num_samples)

    # Step 2: Shuffle Points in each dimension
    rng_keys = random.split(rng_key, dim)
    shuffled_points = lax.map(lambda key: random.permutation(key, base_points), rng_keys)

    return shuffled_points.T

# @jit
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

def scale_to_bounds(lhs_samples, lower_bounds, upper_bounds):
    return lower_bounds + (upper_bounds - lower_bounds) * lhs_samples