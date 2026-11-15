"""Loading optimization settings without changing their values or layout."""

import json

from jaxcmr.helpers.io import load_opt_params


def test_load_opt_params_preserves_fixed_values_and_list_bounds(tmp_path):
    configuration = {"fixed": {"rate": .3, "count": 2, "enabled": True},
                     "free": {"drift": [.1, .9]}}
    path = tmp_path / "parameters.json"
    path.write_text(json.dumps(configuration), encoding="utf-8")
    loaded = load_opt_params(str(path))
    assert loaded == {"base": configuration["fixed"], "bounds": configuration["free"]}
    assert loaded["base"]["enabled"] is True
    assert isinstance(loaded["bounds"]["drift"], list)
