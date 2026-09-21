"""Explicit parameter coordinates shared by bounded fitting algorithms.

Construction validates bounds on the host. Conversion uses JAX operations, so
the same coordinates can be used for population search and differentiation.
"""

from collections.abc import Mapping, Sequence
from typing import Literal, SupportsFloat

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from jaxcmr.typing import Array

__all__ = ["ParameterSpace", "ParameterTransform"]

type ParameterTransform = Literal["linear", "log", "log1p", "exp_neg"]


class ParameterSpace:
    """Map bounded model parameters to unit-interval optimization coordinates.

    ``bounds`` and all model-facing values use physical parameter units.
    Selected parameters may use log, log1p, or exp(-decay) coordinates before
    their transformed interval is scaled to [0, 1]. This changes the geometry
    of optimization and uniform sampling, not the model family.

    Conversion methods use the last array axis for parameters, in ``names``
    order, and preserve all leading dimensions. They support JIT, vmap and
    differentiation using the caller's JAX precision. Construction rejects
    bounds whose intervals or transforms collapse at that precision; decay
    bounds up to 100, for example, require x64 for retention coordinates.

    Attributes:
        names: Parameter names in the insertion order of the bounds mapping.
        bounds: Original physical bounds, shaped (parameters, 2).
        transforms: Explicit transform choices; omitted names use linear scaling.
        lower: Representable physical lower bounds, rounded inward if necessary.
        upper: Representable physical upper bounds, rounded inward if necessary.
    """

    def __init__(
        self,
        bounds: Mapping[str, ArrayLike | Sequence[SupportsFloat]],
        transforms: Mapping[str, ParameterTransform] | None = None,
    ) -> None:
        """Validate physical bounds and prepare their transformed intervals.

        Args:
            bounds: A nonempty mapping from names to finite [lower, upper]
                pairs, with lower strictly below upper. Its order defines the
                parameter axis for every conversion.
            transforms: Selected names mapped to ``linear``, ``log`` (positive
                bounds), ``log1p`` (bounds above -1), or ``exp_neg`` (retention).
                The log1p offset is one in the parameter's physical units.
                Retention reverses orientation: unit zero gives maximum decay.

        Raises:
            ValueError: Bounds or transforms are invalid, or the configured
                JAX precision cannot represent the transformed interval.
        """
        self.names = tuple(bounds)
        self.transforms: dict[str, ParameterTransform] = dict(transforms or {})
        _validate_transforms(self.names, self.transforms)
        self.bounds = _validated_bounds(bounds)
        rounded = _representable_bounds(self.bounds)
        self.lower = jnp.asarray(rounded[:, 0])
        self.upper = jnp.asarray(rounded[:, 1])
        self._kinds: tuple[ParameterTransform, ...] = tuple(
            self.transforms.get(name, "linear") for name in self.names
        )
        self._set_coordinates(_transformed_bounds(self.names, self._kinds, rounded))

    def to_physical(self, unit_parameters: Array) -> Array:
        """Convert optimization coordinates to model parameter values.

        Args:
            unit_parameters: Array shaped (..., parameters), with its final
                axis in ``names`` order. Values normally lie in [0, 1].

        Returns:
            Physical parameter values with the same shape. Results are clipped
            to representable physical bounds, including small rounding errors.
            Clipping does not make invalid nonlinear inputs valid.

        Raises:
            ValueError: The input has no parameter axis or its width is wrong.
        """
        self._validate_shape(unit_parameters)
        coordinates = self._coordinate_lower + unit_parameters * self._coordinate_span
        return jnp.clip(self._inverse(coordinates), self.lower, self.upper)

    def to_unit(self, physical_parameters: Array) -> Array:
        """Convert model parameter values to optimization coordinates.

        Args:
            physical_parameters: Array shaped (..., parameters), ordered by
                ``names``. Values must be in the selected transforms' domains.

        Returns:
            Unit coordinates with the same shape, clipped to [0, 1]. These
            conversions do not perform runtime value validation inside JIT.

        Raises:
            ValueError: The input has no parameter axis or its width is wrong.
        """
        self._validate_shape(physical_parameters)
        coordinates = self._forward(physical_parameters)
        unit = (coordinates - self._coordinate_lower) / self._coordinate_span
        return jnp.clip(unit, 0.0, 1.0)

    def _validate_shape(self, parameters: Array) -> None:
        """Check static shape metadata, including while tracing a JAX function."""
        if parameters.ndim < 1 or parameters.shape[-1] != len(self.names):
            raise ValueError(
                f"Expected a final axis of {len(self.names)} parameters in "
                f"{self.names} order; got shape {parameters.shape}."
            )

    def _forward(self, parameters: Array) -> Array:
        """Apply each parameter's transform before unit-interval scaling."""
        values = []
        for index, kind in enumerate(self._kinds):
            value = parameters[..., index]
            if kind == "log":
                value = jnp.log(value)
            elif kind == "log1p":
                value = jnp.log1p(value)
            elif kind == "exp_neg":
                value = jnp.exp(-value)
            values.append(value)
        return jnp.stack(values, axis=-1)

    def _inverse(self, coordinates: Array) -> Array:
        """Undo the nonlinear transforms, retaining the final parameter axis."""
        values = []
        for index, kind in enumerate(self._kinds):
            value = coordinates[..., index]
            if kind == "log":
                value = jnp.exp(value)
            elif kind == "log1p":
                value = jnp.expm1(value)
            elif kind == "exp_neg":
                value = -jnp.log(value)
            values.append(value)
        return jnp.stack(values, axis=-1)

    def _set_coordinates(self, coordinates: np.ndarray) -> None:
        """Retain finite intervals whose inverse also survives JAX arithmetic."""
        self._coordinate_lower = jnp.asarray(coordinates[:, 0])
        self._coordinate_upper = jnp.asarray(coordinates[:, 1])
        self._coordinate_span = self._coordinate_upper - self._coordinate_lower
        if (
            not np.all(np.isfinite(coordinates))
            or not np.all(np.isfinite(self._coordinate_span))
            or np.any(np.asarray(self._coordinate_span) <= 0)
        ):
            raise ValueError(
                "Transformed bounds are not representable; enable x64 or change the transform."
            )
        # XLA can flush subnormal retention values to zero. Check the inverse
        # before clipping, so this cannot quietly narrow the physical domain.
        endpoints = jnp.stack([self._coordinate_lower, self._coordinate_upper])
        if not np.all(np.isfinite(np.asarray(self._inverse(endpoints)))):
            raise ValueError(
                "Transformed bounds are not representable; enable x64 or change the transform."
            )


def _validate_transforms(
    names: tuple[str, ...], transforms: Mapping[str, ParameterTransform]
) -> None:
    """Reject unknown names or transform kinds before constructing intervals."""
    if transforms.keys() - set(names):
        raise ValueError("transforms may name only free parameters present in bounds.")
    if any(
        kind not in ("linear", "log", "log1p", "exp_neg")
        for kind in transforms.values()
    ):
        raise ValueError(
            "Supported transforms are 'linear', 'log', 'log1p', and 'exp_neg'."
        )


def _validated_bounds(
    bounds: Mapping[str, ArrayLike | Sequence[SupportsFloat]],
) -> np.ndarray:
    """Copy one finite, strictly increasing physical interval per parameter."""
    values = np.asarray(list(bounds.values()), dtype=float)
    if not bounds or values.shape != (len(bounds), 2):
        raise ValueError(
            "bounds must provide one [lower, upper] pair per free parameter."
        )
    if not np.all(np.isfinite(values)) or np.any(values[:, 0] >= values[:, 1]):
        raise ValueError("bounds must be finite and strictly ordered.")
    return values


def _representable_bounds(bounds: np.ndarray) -> np.ndarray:
    """Round endpoints inward at the configured JAX floating-point precision.

    Ordinary nearest rounding can expand the requested domain, for example by
    moving a drift rate just below one onto a singular endpoint at exactly one.
    Inward rounding preserves the bounds and rejects collapsed intervals.
    """
    dtype = np.asarray(jnp.asarray(0.0)).dtype.type
    with np.errstate(over="ignore", under="ignore"):
        rounded = bounds.astype(dtype)
    rounded[:, 0] = np.where(
        rounded[:, 0].astype(float) < bounds[:, 0],
        np.nextafter(rounded[:, 0], dtype(np.inf)),
        rounded[:, 0],
    )
    rounded[:, 1] = np.where(
        rounded[:, 1].astype(float) > bounds[:, 1],
        np.nextafter(rounded[:, 1], dtype(-np.inf)),
        rounded[:, 1],
    )
    if not np.all(np.isfinite(rounded)) or np.any(rounded[:, 0] >= rounded[:, 1]):
        raise ValueError(
            "Bounds collapse in the configured JAX precision; enable x64 or widen them."
        )
    return rounded


def _transformed_bounds(
    names: tuple[str, ...], kinds: tuple[ParameterTransform, ...], bounds: np.ndarray
) -> np.ndarray:
    """Transform host-side intervals, reversing decreasing retention bounds."""
    intervals = []
    for name, kind, pair in zip(names, kinds, bounds, strict=True):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            if kind == "log":
                if pair[0] <= 0:
                    raise ValueError(
                        f"The log transform requires positive bounds for {name}."
                    )
                interval = np.log(pair.astype(float))
            elif kind == "log1p":
                if pair[0] <= -1:
                    raise ValueError(
                        f"The log1p transform requires bounds above -1 for {name}."
                    )
                interval = np.log1p(pair.astype(float))
            elif kind == "exp_neg":
                interval = np.exp(-pair[::-1].astype(float))
            else:
                interval = pair
            intervals.append(np.asarray(interval, dtype=bounds.dtype))
    return np.asarray(intervals)
