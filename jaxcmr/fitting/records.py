"""Participant alignment for saved fitting results.

These operations work on in-memory mappings. Loading files, choosing a model,
and deciding whether to rerun optimization remain the caller's responsibility.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np

from jaxcmr.typing import Array, Bool, Integer, Real

__all__ = ["combine_participant_parameters", "parameters_for_subjects"]


def combine_participant_parameters(
    parameter_sets: Sequence[
        Mapping[
            str,
            Sequence[float | int | bool]
            | Real[Array | np.ndarray, " participants"]
            | Bool[Array | np.ndarray, " participants"],
        ]
    ],
) -> dict[str, list[float | int | bool]]:
    """Combine participant parameter columns from separate fits, sorted by ID.

    Args:
        parameter_sets: Nonempty sequence of parameter-column mappings. Each
            mapping supplies the same keys, including ``subject``, and nonempty,
            aligned one-dimensional columns. Subject IDs must be integers,
            distinct across all mappings; the pooled sentinel -1 is excluded.
            These are caller preconditions, not runtime checks.

    Returns:
        New lists aligned to ascending subject IDs. Boolean parameter values
        remain boolean. Input mappings and their columns are left unchanged.

    The caller establishes that the fits describe compatible models, parameter
    definitions and observations. This combines parameter rows only; it does
    not merge optimizer diagnostics or produce a complete ``FitResult``.

    Example::

        combined = combine_participant_parameters([
            {"subject": [2], "rate": [0.4], "enabled": [True]},
            {"subject": [1], "rate": [0.2], "enabled": [False]},
        ])
        assert combined["subject"] == [1, 2]
        assert combined["enabled"] == [False, True]
    """
    combined: dict[str, list[float | int | bool]] = {
        name: [] for name in parameter_sets[0]
    }
    for parameters in parameter_sets:
        for name, values in combined.items():
            values.extend(np.asarray(parameters[name]).tolist())
    order = sorted(range(len(combined["subject"])), key=combined["subject"].__getitem__)
    return {name: [values[index] for index in order] for name, values in combined.items()}


def parameters_for_subjects(
    record: Mapping[str, Any],
    subject_ids: Sequence[int] | Integer[Array | np.ndarray, " participants"],
) -> dict[str, Real[Array, " participants"] | Bool[Array, " participants"]]:
    """Assign fitted parameter rows to requested participants, preserving order.

    A pooled record (``pooled=True``, or a sole subject ID of -1 in older records)
    supplies one parameter set to every requested participant. Other records
    are matched by subject identity; unused fitted rows are allowed. This
    prepares inputs to ``simulate_dataset`` without loading data, constructing
    models, or checking the scientific compatibility of a fit and experiment.

    Args:
        record: A fit record with nonempty, aligned one-dimensional ``fits``
            columns, including integer ``subject`` IDs. Participant rows must
            have distinct IDs covering the requested participants. A pooled
            declaration must describe exactly one row with ID -1.
            Optional ``fit_configuration.parameters.fixed`` takes precedence
            over ``fixed`` when restoring Boolean parameter types; each Boolean
            parameter named there must have a corresponding ``fits`` column.
        subject_ids: Nonempty sequence or vector of distinct integer IDs.
            Floating-point IDs, Boolean IDs and the pooled sentinel -1 are
            excluded. These and the record requirements are caller preconditions,
            not runtime checks.

    Returns:
        A new mapping of one-dimensional JAX arrays in the requested order,
        including ``subject``. Inputs remain unchanged. Runs outside JIT.

    Example::

        parameters = parameters_for_subjects(saved_fit, [12, 5])
        # Every parameter column now corresponds to participant 12, then 5.
    """
    columns = {name: np.asarray(values) for name, values in record["fits"].items()}
    fitted_subject_ids = columns["subject"]
    requested_subject_ids = np.asarray(subject_ids)
    if record.get("pooled", fitted_subject_ids.tolist() == [-1]):
        parameter_indices = np.zeros(len(requested_subject_ids), dtype=int)
    else:
        row_by_subject = {subject: index for index, subject in enumerate(fitted_subject_ids)}
        parameter_indices = np.asarray(
            [row_by_subject[subject] for subject in requested_subject_ids], dtype=int
        )
    parameters = {
        name: jnp.asarray(values)[parameter_indices]
        for name, values in columns.items() if name != "subject"
    }
    parameters["subject"] = jnp.asarray(requested_subject_ids)
    fixed = record.get("fit_configuration", {}).get("parameters", {}).get(
        "fixed", record.get("fixed", {})
    )
    for name, value in fixed.items():
        if isinstance(value, (bool, np.bool_)):
            parameters[name] = parameters[name].astype(bool)
    return parameters
