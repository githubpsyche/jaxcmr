from jaxtyping import Integer, Float, Array, Bool
from jaxcmr.memorysearch.CMR import CMR
from jaxcmr.memorysearch.MemorySearch import exponential_primacy_weighting
from jaxcmr.context import TemporalContext
from jax import numpy as jnp
from plum import dispatch
from jaxcmr.memory import InstanceMcf, LinearAssociativeMfc
from simple_pytree import static_field

__all__ = ["InstanceCMR"]


class InstanceCMR(CMR, mutable=True):

    item_count = static_field()
    
    def __init__(
        self,
        item_count: int | Integer[Array, ""],
        presentation_count: int | Integer[Array, ""],
        encoding_drift_rate: float | Float[Array, ""],
        delay_drift_rate: float | Float[Array, ""],
        start_drift_rate: float | Float[Array, ""],
        recall_drift_rate: float | Float[Array, ""],
        shared_support: float | Float[Array, ""],
        item_support: float | Float[Array, ""],
        learning_rate: float | Float[Array, ""],
        primacy_scale: float | Float[Array, ""],
        primacy_decay: float | Float[Array, ""],
        stop_probability_scale: float | Float[Array, ""],
        stop_probability_growth: float | Float[Array, ""],
        choice_sensitivity: float | Float[Array, ""],
        trace_sensitivity: float | Float[Array, ""]
    ):
        self.mfc = LinearAssociativeMfc.create(item_count, learning_rate)
        self.mcf = InstanceMcf.create(
            item_count, presentation_count, shared_support, item_support, choice_sensitivity, trace_sensitivity
        )
        self.context = TemporalContext.create(item_count)
        self.encoding_drift_rate = encoding_drift_rate
        self.delay_drift_rate = delay_drift_rate
        self.start_drift_rate = start_drift_rate
        self.recall_drift_rate = recall_drift_rate
        self.mfc_learning_rate = learning_rate
        self.stop_probability_scale = stop_probability_scale
        self.stop_probability_growth = stop_probability_growth

        self.is_active = True
        self.item_count = item_count
        self.items = jnp.eye(item_count, item_count)
        self.encoding_index = 0
        self.recall_total = 0
        self.recall_sequence = jnp.zeros(item_count, jnp.int32)
        self.recall_mask = jnp.zeros(item_count, jnp.bool_)
        self._mcf_learning_rate = exponential_primacy_weighting(
            presentation_count, primacy_scale, primacy_decay
        )

    @classmethod
    @dispatch
    def create(
        cls,
        item_count: int | Integer[Array, ""],
        presentation_count: int | Integer[Array, ""],
        encoding_drift_rate: float | Float[Array, ""],
        delay_drift_rate: float | Float[Array, ""],
        start_drift_rate: float | Float[Array, ""],
        recall_drift_rate: float | Float[Array, ""],
        shared_support: float | Float[Array, ""],
        item_support: float | Float[Array, ""],
        learning_rate: float | Float[Array, ""],
        primacy_scale: float | Float[Array, ""],
        primacy_decay: float | Float[Array, ""],
        stop_probability_scale: float | Float[Array, ""],
        stop_probability_growth: float | Float[Array, ""],
        choice_sensitivity: float | Float[Array, ""],
        trace_sensitivity: float | Float[Array, ""]
    ):
        return cls(
            item_count,
            presentation_count,
            encoding_drift_rate,
            delay_drift_rate,
            start_drift_rate,
            recall_drift_rate,
            shared_support,
            item_support,
            learning_rate,
            primacy_scale,
            primacy_decay,
            stop_probability_scale,
            stop_probability_growth,
            choice_sensitivity,
            trace_sensitivity
        )

    @classmethod
    @dispatch
    def create(
        cls,
        item_count: int | Integer[Array, ""],
        presentation_count: int | Integer[Array, ""],
        parameters: dict,
    ):
        return cls(
            item_count,
            presentation_count,
            parameters["encoding_drift_rate"],
            parameters["delay_drift_rate"],
            parameters["start_drift_rate"],
            parameters["recall_drift_rate"],
            parameters["shared_support"],
            parameters["item_support"],
            parameters["learning_rate"],
            parameters["primacy_scale"],
            parameters["primacy_decay"],
            parameters["stop_probability_scale"],
            parameters["stop_probability_growth"],
            parameters["choice_sensitivity"],
            parameters["mcf_trace_sensitivity"]
        )
    
    @classmethod
    @dispatch
    def create(
        cls,
        item_count: int | Integer[Array, ""],
        parameters: dict,
    ):
        return cls(
            item_count,
            item_count,
            parameters["encoding_drift_rate"],
            parameters["delay_drift_rate"],
            parameters["start_drift_rate"],
            parameters["recall_drift_rate"],
            parameters["shared_support"],
            parameters["item_support"],
            parameters["learning_rate"],
            parameters["primacy_scale"],
            parameters["primacy_decay"],
            parameters["stop_probability_scale"],
            parameters["stop_probability_growth"],
            parameters["choice_sensitivity"],
            parameters["mcf_trace_sensitivity"]
        )

    @property
    def mcf_learning_rate(self) -> Float[Array, ""]:
        return self._mcf_learning_rate[self.encoding_index]