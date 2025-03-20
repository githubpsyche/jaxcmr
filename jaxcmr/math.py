import jax.numpy as jnp
from jax import lax

from jaxcmr.typing import Float, Float_, Int_


def power_scale(value: Float_, scale: Float_) -> Float:
    """Returns value scaled by the exponent factor using logsumexp trick."""
    log_activation = jnp.log(value)
    return lax.cond(
        jnp.logical_and(jnp.any(value != 0), scale != 1),
        lambda _: jnp.exp(scale * (log_activation - jnp.max(log_activation))),
        lambda _: value,
        None,
    )


def exponential_primacy_decay(
    study_index: Int_, primacy_scale: Float_, primacy_decay: Float_
):
    """Returns the exponential primacy weighting for the specified study event.

    Args:
        study_index: the index of the study event.
        primacy_scale: the scale factor for primacy effect.
        primacy_decay: the decay factor for primacy effect.
    """
    return primacy_scale * jnp.exp(-primacy_decay * study_index) + 1


def exponential_stop_probability(
    stop_probability_scale: Float_, stop_probability_growth: Float_, recall_total: Int_
):
    """Returns the exponential stop probability for the specified recall event.

    Args:
        stop_probability_scale: the scale factor for stop probability.
        stop_probability_growth: the growth factor for stop probability.
        recall_total: the total number of items recalled.
    """
    return stop_probability_scale * jnp.exp(recall_total * stop_probability_growth)
