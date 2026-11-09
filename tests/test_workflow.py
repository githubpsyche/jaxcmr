"""A user workflow with persisted data, a bounded fit, and reusable results."""

import json
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxcmr.analyses.spc import spc, plot_spc
from jaxcmr.components import context, linear_memory
from jaxcmr.components.termination import NoStopTermination
from jaxcmr.experimental.cross_validation import generate_fold_masks
from jaxcmr.fitting.gradient import ScipyLBFGSB
from jaxcmr.helpers import load_data, save_dict_to_hdf5
from jaxcmr.loss.transform_sequence_likelihood import ExcludeTerminationLikelihoodLoss
from jaxcmr.models.cmr import make_factory
from jaxcmr.simulation import simulate_h5_from_h5
from support import recall_fixture

PARAMS = dict(
    encoding_drift_rate=0.5,
    start_drift_rate=0.4,
    recall_drift_rate=0.6,
    shared_support=1.0,
    item_support=1.5,
    learning_rate=0.3,
    primacy_scale=1.0,
    primacy_decay=0.1,
    choice_sensitivity=2.0,
    learn_after_context_update=False,
    allow_repeated_recalls=False,
)


def test_complete_modeling_workflow(tmp_path):
    original = {key: value[:2] for key, value in recall_fixture().items()}
    data_path = tmp_path / "observations.h5"
    save_dict_to_hdf5(original, str(data_path))
    data = load_data(str(data_path))
    for key in original:
        np.testing.assert_array_equal(data[key], original[key])
    np.testing.assert_allclose(spc(data), [0.5, 0.5, 1, 1, 1, 1, 0, 0])
    factory = make_factory(
        linear_memory.init_mfc, linear_memory.init_mcf, context.init, NoStopTermination
    )
    loss = ExcludeTerminationLikelihoodLoss(factory, data, None)
    fitter = ScipyLBFGSB(
        data,
        None,
        PARAMS,
        factory,
        ExcludeTerminationLikelihoodLoss,
        {
            "bounds": {"choice_sensitivity": [0.1, 5.0]},
            "num_steps": 15,
            "best_of": 1,
            "screening_size": 0,
            "initial_params": {"choice_sensitivity": 2.0},
            "seed": 17,
        },
    )
    mask = jnp.ones(2, dtype=bool)
    result = fitter.fit_subjects(mask)
    assert result["fits"]["subject"] == [1, 2]
    for index, sensitivity in enumerate(result["fits"]["choice_sensitivity"]):
        assert 0.1 <= sensitivity <= 5.0
        indices = jnp.array([index])
        recalculated = loss(indices, PARAMS, ("choice_sensitivity",), jnp.array([[sensitivity]]))[0]
        initial = loss(indices, PARAMS, ("choice_sensitivity",), jnp.array([[2.0]]))[0]
        np.testing.assert_allclose(result["fitness"][index], recalculated, rtol=1e-6)
        assert recalculated <= initial + 1e-5
    result_path = tmp_path / "fit.json"
    result_path.write_text(json.dumps(result))
    restored = json.loads(result_path.read_text())
    assert restored == result
    simulated = simulate_h5_from_h5(
        factory,
        data,
        None,
        {name: jnp.asarray(values) for name, values in restored["fits"].items()},
        mask,
        experiment_count=2,
        rng=jax.random.PRNGKey(19),
    )
    np.testing.assert_array_equal(simulated["subject"].ravel(), [1, 1, 2, 2])
    assert np.all((simulated["recalls"] >= 0) & (simulated["recalls"] <= 8))
    axis = plot_spc(simulated, jnp.ones(4, dtype=bool), labels=["simulation"])
    line = next(line for line in axis.lines if line.get_marker() == "o")
    np.testing.assert_allclose(np.asarray(line.get_ydata(), dtype=float), spc(simulated))
    assert axis.get_ylabel() == "Recall Rate"
    axis.figure.savefig(tmp_path / "simulation.png")
    plt.close(axis.figure)


def test_cross_validation_membership_exclusions_and_partition():
    data = {"fold": np.array([[1], [1], [2], [2], [3], [3]])}
    base = jnp.array([False, True, True, False, True, True])
    train, held_out = generate_fold_masks(data, "fold", 2, base)
    np.testing.assert_array_equal(train, [False, True, False, False, True, True])
    np.testing.assert_array_equal(held_out, [False, False, True, False, False, False])
    assert not np.any(train & held_out)
    np.testing.assert_array_equal(train | held_out, base)
