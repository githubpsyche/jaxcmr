import matplotlib

matplotlib.use("Agg", force=True)
import jax.numpy as jnp
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from jaxcmr.analyses import first_item_recall
from jaxcmr.typing import RecallDataset


def _make_dataset(recalls: jnp.ndarray, presentations: jnp.ndarray) -> RecallDataset:
    recalls = jnp.asarray(recalls, dtype=jnp.int32)
    presentations = jnp.asarray(presentations, dtype=jnp.int32)
    n_trials = recalls.shape[0]
    list_length = presentations.shape[1]
    return {
        "subject": jnp.ones((n_trials, 1), dtype=jnp.int32),
        "listLength": jnp.full((n_trials, 1), list_length, dtype=jnp.int32),
        "pres_itemnos": presentations,
        "recalls": recalls,
    }


def test_returns_probability_by_recall_position():
    dataset = _make_dataset(
        recalls=jnp.array([[1, 2, 3], [2, 1, 3]]),
        presentations=jnp.array([[1, 2, 3], [1, 2, 3]]),
    )

    curve = first_item_recall.first_item_recall_curve(dataset)

    expected = jnp.array([0.5, 0.5, 0.0])
    assert jnp.allclose(curve, expected).item()


def test_plot_returns_axes():
    dataset: RecallDataset = {
        "subject": jnp.array([[1], [1]], dtype=jnp.int32),
        "listLength": jnp.array([[3], [3]], dtype=jnp.int32),
        "pres_itemnos": jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32),
        "recalls": jnp.array([[1, 2, 0], [2, 1, 0]], dtype=jnp.int32),
    }
    trial_mask = jnp.array([True, True], dtype=bool)

    axis = first_item_recall.plot_first_item_recall_curve(dataset, trial_mask)

    assert isinstance(axis, Axes)
    fig = axis.figure
    assert isinstance(fig, Figure)
