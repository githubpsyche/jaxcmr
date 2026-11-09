"""Seven temporary faults must be rejected by the ordinary regression checks."""

from pathlib import Path
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
import pytest


@pytest.mark.parametrize(
    "fault,target,patch",
    [
        (
            "constant-likelihood",
            "test_loss_answers.py::test_likelihood_known_event_products",
            """
from jaxcmr.loss.transform_sequence_likelihood import MemorySearchLikelihoodLoss
MemorySearchLikelihoodLoss.present_and_predict_trials_loss = lambda self,*a,**k: jnp.asarray(1.)
""",
        ),
        (
            "empty-plot",
            "test_plots.py::test_recall_plots_draw_selected_values",
            """
from jaxcmr.analyses import spc
import matplotlib.pyplot as plt
spc.plot_spc = lambda *a,**k: plt.subplots()[1]
""",
        ),
        (
            "wrong-denominator",
            "test_transitions.py::test_source_target_counts_items_and_updates_excluded_events",
            """
from jaxcmr.analyses import source_target_crp
original = source_target_crp.tabulate_trial
def wrong(*args, **kwargs):
    actual, available = original(*args, **kwargs)
    return actual, 2*available
source_target_crp.tabulate_trial = wrong
""",
        ),
        (
            "padding-as-item",
            "test_repeated_transitions.py::test_neighbor_padding_and_real_last_position",
            """
from jaxcmr.analyses import repneighborcrp
original = repneighborcrp.tabulate_trial
def wrong(recalls, present, **kwargs):
    return original(jnp.where(recalls==0, len(present), recalls), present, **kwargs)
repneighborcrp.tabulate_trial = wrong
""",
        ),
        (
            "swapped-parameters",
            "test_loss_answers.py::test_scipy_adapter_preserves_scalar_population_order_and_transform",
            """
from jaxcmr.fitting.parameters import ParameterSpace
original = ParameterSpace.to_physical
ParameterSpace.to_physical = lambda self, values: original(self, values)[...,::-1]
""",
        ),
        (
            "stale-selection",
            "test_prepared_data.py::test_planned_cmr_preserves_values_gradients_multiplicity_and_empty_data",
            """
from jaxcmr.loss.cmr_likelihood import FastCMRLikelihoodLoss
original = FastCMRLikelihoodLoss.prepare_selections
def wrong(self, selections):
    plans = original(self, selections)
    return (plans[0],)*len(plans)
FastCMRLikelihoodLoss.prepare_selections = wrong
""",
        ),
        (
            "wrong-subject",
            "test_simulation.py::test_simulate_h5_from_h5_maps_nonconsecutive_subject_ids",
            """
import jaxcmr.simulation as simulation
simulation._parameter_indices_for_subjects = lambda subjects, params: jnp.zeros_like(subjects)
""",
        ),
    ],
    ids=[
        "constant-likelihood",
        "empty-plot",
        "wrong-denominator",
        "padding-as-item",
        "swapped-parameters",
        "stale-selection",
        "wrong-subject",
    ],
)
def test_deliberate_fault_is_detected(fault, target, patch, tmp_path):
    report = tmp_path / (fault + ".xml")
    target = str(Path(__file__).parent / target)
    arguments = [target, "-q", "-p", "no:cacheprovider", "--junitxml=" + str(report)]
    script = "import jax.numpy as jnp\nimport pytest\n" + textwrap.dedent(patch)
    script += "\nraise SystemExit(pytest.main(" + repr(arguments) + "))\n"
    run = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    (tmp_path / (fault + ".log")).write_text(run.stdout + run.stderr)
    assert run.returncode == 1, run.stdout + run.stderr
    failures = ET.parse(report).findall(".//testcase/failure")
    assert failures, run.stdout + run.stderr
    assert any(
        "AssertionError" in (failure.text or "")
        or "ValueError: not enough values to unpack" in failure.get("message", "")
        for failure in failures
    ), run.stdout + run.stderr
