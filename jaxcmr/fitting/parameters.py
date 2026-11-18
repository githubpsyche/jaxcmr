"""Explicit parameter coordinates shared by bounded fitting algorithms.

Construction prepares bounds on the host. Conversion uses JAX operations, so
the same coordinates can be used for population search and differentiation.
"""

from collections.abc import Mapping, Sequence
from typing import Literal, SupportsFloat

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map
from numpy.typing import ArrayLike

from jaxcmr.typing import Array, Float

__all__ = ["ParameterSpace", "ParameterTransform", "bound_params"]

type ParameterTransform = Literal["linear", "log", "log1p", "exp_neg"]


def bound_params(
    params: Mapping[str, Float[Array, " population"]], bounds: Mapping[str, list[float]]
) -> dict[str, Float[Array, " population"]]:
    """Return a new parameter mapping clipped to the supplied lower/upper bounds."""
    return tree_map(
        lambda param, bound: jnp.minimum(jnp.maximum(bound[0], param), bound[1]),
        params,
        bounds,
    )


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
        """Prepare physical bounds and their transformed intervals.

        Parameters
        ----------
        bounds : Mapping[str, ArrayLike]
            Nonempty mapping to finite, strictly ordered lower/upper pairs. Insertion
            order defines the final parameter axis in every conversion.
        transforms : Mapping[str, ParameterTransform] or None, optional
            Selected names mapped to linear, log (positive bounds), log1p (bounds
            above -1), or exp_neg (retention). Omitted names use linear coordinates.
            Retention reverses orientation: unit zero gives maximum decay.

        Raises
        ------
        ValueError
            The configured JAX precision cannot represent the physical or transformed
            interval. Increasing precision or changing the bounds may be necessary.
        """
        self.names = tuple(bounds)
        self.transforms: dict[str, ParameterTransform] = dict(transforms or {})
        self.bounds = np.asarray(list(bounds.values()), dtype=float)
        rounded = _representable_bounds(self.bounds)
        self.lower = jnp.asarray(rounded[:, 0])
        self.upper = jnp.asarray(rounded[:, 1])
        self._kinds: tuple[ParameterTransform, ...] = tuple(
            self.transforms.get(name, "linear") for name in self.names
        )
        self._set_coordinates(_transformed_bounds(self.names, self._kinds, rounded))

    def to_physical(
        self,
        unit_parameters: Float[Array, "*batch parameters"],
    ) -> Float[Array, "*batch parameters"]:
        """Convert optimization coordinates to model parameters.

        Parameters
        ----------
        unit_parameters : Float[Array, "*batch parameters"]
            Final axis follows names; all leading batch axes are preserved.
            Values normally lie in [0, 1]. Retention coordinates reverse orientation.

        Returns
        -------
        Float[Array, "*batch parameters"]
            Values are clipped to representable physical bounds. Exact unit endpoints
            return the corresponding bound without changing transform derivatives.
        """
        coordinates = self._coordinate_lower + unit_parameters * self._coordinate_span
        physical = jnp.clip(self._inverse(coordinates), self.lower, self.upper)
        # Inverse transforms may round just inside an endpoint. Preserve exact
        # declared endpoints without changing the transform's existing derivative.
        decreasing = jnp.asarray([kind == "exp_neg" for kind in self._kinds])
        start = jnp.where(decreasing, self.upper, self.lower)
        end = jnp.where(decreasing, self.lower, self.upper)
        endpoints = jnp.where(
            unit_parameters == 0,
            start,
            jnp.where(unit_parameters == 1, end, physical),
        )
        return physical + jax.lax.stop_gradient(endpoints - physical)

    def to_unit(
        self,
        physical_parameters: Float[Array, "*batch parameters"],
    ) -> Float[Array, "*batch parameters"]:
        """Convert physical parameters to optimization coordinates.

        Parameters
        ----------
        physical_parameters : Float[Array, "*batch parameters"]
            Final axis follows names; all leading batch axes are preserved.
            Values must lie in the chosen transforms' mathematical domains.

        Returns
        -------
        Float[Array, "*batch parameters"]
            Unit coordinates clipped to [0, 1]. No runtime validation is performed
            inside the compiled numerical calculation.
        """
        coordinates = self._forward(physical_parameters)
        unit = (coordinates - self._coordinate_lower) / self._coordinate_span
        return jnp.clip(unit, 0.0, 1.0)

    def _forward(
        self,
        parameters: Float[Array, "*batch parameters"],
    ) -> Float[Array, "*batch parameters"]:
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

    def _inverse(
        self,
        coordinates: Float[Array, "*batch parameters"],
    ) -> Float[Array, "*batch parameters"]:
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

    def _set_coordinates(self, coordinates: Float[np.ndarray, " parameters 2"]) -> None:
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


def _representable_bounds(
    bounds: Float[np.ndarray, " parameters 2"],
) -> Float[np.ndarray, " parameters 2"]:
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
    names: tuple[str, ...],
    kinds: tuple[ParameterTransform, ...],
    bounds: Float[np.ndarray, " parameters 2"],
) -> Float[np.ndarray, " parameters 2"]:
    """Transform host-side intervals, reversing decreasing retention bounds."""
    intervals = []
    for name, kind, pair in zip(names, kinds, bounds, strict=True):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            if kind == "log":
                interval = np.log(pair.astype(float))
            elif kind == "log1p":
                interval = np.log1p(pair.astype(float))
            elif kind == "exp_neg":
                interval = np.exp(-pair[::-1].astype(float))
            else:
                interval = pair
            intervals.append(np.asarray(interval, dtype=bounds.dtype))
    return np.asarray(intervals)
