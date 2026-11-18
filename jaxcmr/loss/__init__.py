"""Score model predictions against recall observations.

sequence_likelihood scores observed events while preserving their conditioning
history. mse compares dataset analyses of observed and simulated recalls.
Loss constructors can be classes or configured callables; fitters evaluate
candidate columns in physical parameter units. Unordered-recall likelihood
and explicitly prepared observations live under jaxcmr.experimental.
"""
