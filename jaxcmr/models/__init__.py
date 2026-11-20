"""Each model module supplies a `make_factory` function. It binds the component
creators and returns a factory class. Construct that class with the dataset
and any vocabulary features or similarities. Its `create_trial_model`
method takes a trial index and a parameter mapping.

The semantic factories prepare a connection matrix for each trial. The
emotional-source factories prepare a category for each local item. Both use
the selected trial's item codes when constructing its model. Each configurable
factory takes a response policy that assigns probabilities to STOP and each
item. The positional and blended models account for repeated presentations
before returning a probability for the recorded item.
"""

from collections.abc import Mapping
from inspect import signature

from jaxcmr.helpers import import_from_string
from jaxcmr.typing import MemorySearchModelFactory

# Historical CMR-family records named only the stopping part of the response.
_CMR_RESPONSE_PATHS = {
    "jaxcmr.components.termination.NoStopTermination": "jaxcmr.components.response.RecallUntilExhausted",
    "jaxcmr.components.termination.PositionalTermination": "jaxcmr.components.response.PositionalStop",
    "jaxcmr.components.termination.SupportRatioTermination": "jaxcmr.components.response.RelativePowerSupportRatio",
}


def recorded_policy_path(components: Mapping[str, str]) -> str:
    """Identify the response or stopping rule named in a saved model configuration.

    The three historical CMR stopping rules use their corresponding response
    policy names for comparison. Models can still calculate item evidence differently.
    """
    if "response_policy_create_fn" in components:
        return components["response_policy_create_fn"]
    stopping = components["termination_policy_create_fn"]
    return _CMR_RESPONSE_PATHS.get(stopping, stopping)


def model_factory_from_paths(
    model_factory: str, components: Mapping[str, str]
) -> type[MemorySearchModelFactory]:
    """Construct a configured model factory from the import paths in a fit record."""
    paths = dict(components)
    create = import_from_string(model_factory)
    if (
        "termination_policy_create_fn" in paths
        and "response_policy_create_fn" in signature(create).parameters
    ):
        old_rule = paths.pop("termination_policy_create_fn")
        if "response_policy_create_fn" not in paths:
            if old_rule not in _CMR_RESPONSE_PATHS:
                raise ValueError(
                    f"Supply a response_policy_create_fn for the saved CMR rule {old_rule!r}."
                )
            paths["response_policy_create_fn"] = _CMR_RESPONSE_PATHS[old_rule]
    return create(**{name: import_from_string(path) for name, path in paths.items()})
