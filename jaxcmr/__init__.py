from jaxcmr.fitting import EvosaxDE, ScipyDE
from jaxcmr.helpers import load_data, make_dataset
from jaxcmr.simulation import simulate_free_recall
from jaxcmr.typing import (
    Array,
    Bool,
    Float,
    Float_,
    Int_,
    Integer,
    MemorySearch,
    MemorySearchModelFactory,
    PRNGKeyArray,
    RecallDataset,
)

__all__ = [
    "Array",
    "Bool",
    "EvosaxDE",
    "Float",
    "Float_",
    "Int_",
    "Integer",
    "MemorySearch",
    "MemorySearchModelFactory",
    "PRNGKeyArray",
    "RecallDataset",
    "ScipyDE",
    "__version__",
    "load_data",
    "make_dataset",
    "simulate_free_recall",
]

__version__ = "0.2.0rc1"
