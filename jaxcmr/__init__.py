from jaxcmr.cross_validation import cross_validate
from jaxcmr.fitting import ScipyDE
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
    "__version__",
    "Array",
    "Bool",
    "Float",
    "Float_",
    "Int_",
    "Integer",
    "MemorySearch",
    "MemorySearchModelFactory",
    "PRNGKeyArray",
    "RecallDataset",
    "ScipyDE",
    "cross_validate",
    "load_data",
    "make_dataset",
    "simulate_free_recall",
]

__version__ = "0.1.1"
