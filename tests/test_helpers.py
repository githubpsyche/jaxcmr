import jax.numpy as jnp
import pytest
from jaxcmr.helpers import apply_by_subject, make_dataset


def test_apply_by_subject_preserves_trial_fields_and_rejects_misaligned_data():
    data = make_dataset(recalls=jnp.array([[1, 0], [2, 0]]), subject=jnp.array([[1], [2]]))
    data["condition"] = jnp.array([[10], [20]])
    selected = apply_by_subject(data, jnp.array([True, True]), lambda subject: subject["condition"])
    assert [int(values[0, 0]) for values in selected] == [10, 20]
    data["word_pool"] = jnp.array([1, 2, 3])
    with pytest.raises(ValueError, match="word_pool.*2 trial rows"):
        apply_by_subject(data, jnp.array([True, True]), lambda subject: subject["recalls"])
