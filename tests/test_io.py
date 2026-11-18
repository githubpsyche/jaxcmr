"""The generic HDF5 representation does not require recall-specific fields."""

import h5py
import jax
import numpy as np
import pytest

from jaxcmr.helpers import load_data, load_hdf5_arrays, save_dict_to_hdf5


def test_hdf5_array_round_trip_preserves_fields_and_transposed_layout(tmp_path):
    data = {
        "subject": np.array([[8], [3]], dtype=np.int32),
        "rating": np.array([[0.25, 0.75, 0.5], [0.5, 0.25, 0.75]], dtype=np.float32),
        "available": np.array([[True, False, True], [False, True, True]]),
    }
    original = {name: values.copy() for name, values in data.items()}
    path = tmp_path / "judgments.h5"
    with h5py.File(path, "w") as stored:
        stored.create_group("obsolete")
    save_dict_to_hdf5(data, path)
    with h5py.File(path) as stored:
        assert set(stored) == {"data"}
        for name, values in data.items():
            np.testing.assert_array_equal(stored[f"data/{name}"][()], values.T)
    loaded = load_hdf5_arrays(path)
    assert loaded.keys() == data.keys()
    for name, values in original.items():
        assert isinstance(loaded[name], jax.Array)
        np.testing.assert_array_equal(loaded[name], values)
        np.testing.assert_array_equal(data[name], values)
    # The same file is useful generic data but is not a recall dataset.
    with pytest.raises(KeyError, match="listLength"):
        load_data(path)
