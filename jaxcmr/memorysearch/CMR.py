from plum import dispatch
from flax.struct import PyTreeNode
from jaxtyping import Integer, Float, Array
from jax import jit, numpy as jnp
from functools import partial
from jaxcmr.memory import OneWayMemory
from jaxcmr.memory.LinearAssociativeMemory import LinearAssociativeMemory as lam
from jaxcmr.context import Context, TemporalContext

#%% Type

class MemorySearch(PyTreeNode):
    pass

class CmrState(MemorySearch):
    mfc: OneWayMemory
    mcf: OneWayMemory
    context: Context
    encoding_index: int | Integer[Array, ""]

#%% Initialization

@partial(jit, static_argnums=(0,))
@dispatch
def init_base_cmr(
    item_count: int | Integer[Array, ""],
    learning_rate: float | Float[Array, ""],
    shared_support: float | Float[Array, ""],
    item_support: float | Float[Array, ""],
):
    return CmrState(
        mfc=lam.init_linear_mfc(item_count, learning_rate),
        mcf=lam.init_linear_mcf(item_count, shared_support, item_support),
        context=TemporalContext.initialize_temporal_context(item_count),
        encoding_index=0
    )

#%% Encoding
        
# def experience(
#     model: CmrState, 
#     item: Float[Array, "input_features"],
# ) -> CmrState:
#     "Encode an item into the model"

#     context_input = lam.activations(model.mfc, item)
#     mcf_learning_rate = model["primacy_weighting"][model["encoding_index"]]

#     model["context"] = TemporalContext.integrate(model['context'], context_input, model['encoding_drift_rate'])
#     model["mfc"] = lam.associate(model["mfc"], model["learning_rate"], item, model["context"])
#     model["mcf"] = lam.associate(model["mcf"], mcf_learning_rate, model["context"], encoded_item)
#     model["encoding_index"] = model["encoding_index"] + 1

#     return model