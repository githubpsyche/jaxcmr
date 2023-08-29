# | export

from numba.core import types as NumbaTypes
from numba.typed import Dict as NumbaDict
from typing import Dict, Iterable
import json
import copy


class Parameters(object):
    """
    Class to manage model parameters.

    Attributes
    ----------
    fixed : dict of (str: float)
        Values of fixed parameters.

    free : dict of (str: tuple)
        Bounds of each free parameter.
    """

    def __init__(self, json_file: str = None, par_dict: dict = None) -> None:

        self.fixed = NumbaDict.empty(key_type=NumbaTypes.unicode_type, value_type=NumbaTypes.float64)
        self.free = NumbaDict.empty(
            key_type=NumbaTypes.unicode_type, value_type=NumbaTypes.Tuple([NumbaTypes.float64, NumbaTypes.float64]))
        self._fields: list[str] = ['fixed', 'free']

        if json_file is not None:
            with open(json_file, 'r') as f:
                par_dict = json.load(f)

        if par_dict is not None:
            # enforce each free parameter as a tuple
            for key, value in par_dict['free'].items():
                if not isinstance(value, tuple):
                    par_dict['free'][key] = tuple(value)
            self.set_free(par_dict['free'])
            self.set_fixed(par_dict['fixed'])

    def __repr__(self) -> str:
        parts = {}
        for name in self._fields:
            obj = getattr(self, name)
            fields = [f'{key}: {value}' for key, value in obj.items()]
            parts[name] = '\n'.join(fields)
        s = '\n\n'.join([f'{name}:\n{f}' for name, f in parts.items()])
        return s

    def copy(self):
        """Copy the parameters definition."""
        return copy.deepcopy(self)

    def to_json(self, json_file: str) -> None:
        """Write parameter definitions to a JSON file."""

        data = {
            'fixed': {k: v for k, v in self.fixed.items()},
            'free': {k: v for k, v in self.free.items()},
        }
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)

    def set_fixed(self, *args: Dict[str, float], **kwargs: float):
        """
        Set fixed parameter values.

        Examples
        --------
        >>> from cymr import parameters
        >>> param_def = parameters.Parameters()
        >>> param_def.set_fixed(a=1, b=2)
        >>> param_def.set_fixed({'c': 3, 'd': 4})
        """
        self.fixed.update(*args, **kwargs)

    def set_free(
            self, *args: Dict[str, Iterable[float]], **kwargs: Iterable[float]
    ) -> None:
        """
        Set free parameter ranges.

        Examples
        --------
        >>> from cymr import parameters
        >>> param_def = parameters.Parameters()
        >>> param_def.set_free(a=[0, 1], b=[1, 10])
        >>> param_def.set_free({'c': [3, 4], 'd': [0, 10]})
        """
        self.free.update(*args, **kwargs)


def parameter_list(jsonl_file=None, par_list=None) -> list:
    """
    Return a list of parameter names.

    Parameters
    ----------
    jsonl_file : str
        Path to a JSON file containing parameter definitions.
    par_list : dict
        Dictionary containing parameter definitions.

    Returns
    -------
    list of str
        List of parameter names.
    """

    if par_list is None:
        par_list = []
    if jsonl_file is not None:
        with open(jsonl_file, 'r') as f:
            for line in f:
                par_list.append(json.loads(line))

    return [Parameters(par_dict=par_dict).fixed for par_dict in par_list]
