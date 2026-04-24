"""Model fitting utilities."""

from jaxcmr.fitting.evosax import (
    BatchEvosaxDE,
    EvosaxDE,
    MapEvosaxDE,
    ScanEvosaxDE,
)
from jaxcmr.fitting.scipy import ScipyDE, make_scipy_loss_fn


__all__ = [
    "BatchEvosaxDE",
    "EvosaxDE",
    "MapEvosaxDE",
    "ScanEvosaxDE",
    "ScipyDE",
    "make_scipy_loss_fn",
]
