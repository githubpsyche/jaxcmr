import json
import os
import tempfile
from pathlib import Path

import numpy as np

from jaxcmr.fitting import make_subject_trial_masks
from jaxcmr.runner import aggregate_subject_results, render_slurm_array_script


def test_masks_intersect_subject_when_base_mask_applied():
    """Behavior: Intersects subject and base trial masks.

    Given:
      - subject vector with three unique subjects
    When:
      - creating per-subject masks with a base mask that excludes one trial
    Then:
      - each subject mask contains only that subject's selected trials
    Why this matters:
      - ensures per-subject fitting respects trial filters
    """
    # Arrange / Given
    subjects = np.array([1, 1, 2, 2, 3, 3]).reshape(-1, 1)
    base_mask = np.array([True, False, True, True, False, True])

    # Act / When
    masks, uniq = make_subject_trial_masks(base_mask, subjects.flatten())

    # Assert / Then
    assert list(uniq) == [1, 2, 3]
    assert np.array_equal(masks[0], np.array([True, False, False, False, False, False]))
    assert np.array_equal(masks[1], np.array([False, False, True, True, False, False]))
    assert np.array_equal(masks[2], np.array([False, False, False, False, False, True]))


def test_aggregates_results_when_subject_jsons_ok():
    """Behavior: Merges per-subject FitResult payloads.

    Given:
      - two per-subject JSON results with consistent parameter sets
    When:
      - aggregating paths
    Then:
      - merged fitness and fits contain both subjects in order
    Why this matters:
      - ensures downstream code can consume a single results file
    """
    # Arrange / Given
    d = tempfile.mkdtemp(prefix="jaxcmr_test_")
    p1 = Path(d) / "fit_subject_1.json"
    p2 = Path(d) / "fit_subject_2.json"
    base_payload = {
        "status": "ok",
        "subject": 0,
        "fit": {
            "fixed": {"a": 1.0},
            "free": {"b": [0.0, 1.0]},
            "fitness": [10.0],
            "fits": {"a": [1.0], "b": [0.2], "subject": [1]},
            "hyperparameters": {"bounds": {"b": [0.0, 1.0]}},
            "fit_time": 1.0,
        },
    }
    with open(p1, "w") as f:
        json.dump(base_payload, f)
    base_payload["fit"]["fitness"] = [5.0]
    base_payload["fit"]["fits"]["b"] = [0.8]
    base_payload["fit"]["fits"]["subject"] = [2]
    with open(p2, "w") as f:
        json.dump(base_payload, f)

    # Act / When
    merged = aggregate_subject_results([str(p1), str(p2)])

    # Assert / Then
    assert merged["fitness"] == [10.0, 5.0]
    assert merged["fits"]["subject"] == [1, 2]
    assert merged["fits"]["b"] == [0.2, 0.8]


def test_renders_slurm_script_when_given_subject_list():
    """Behavior: Emits a valid SLURM array script.

    Given:
      - a list of subject ids and basic cluster parameters
    When:
      - rendering the SLURM array job script
    Then:
      - the script references array size and subject mapping
    Why this matters:
      - supports clean submission to HPC clusters via arrays
    """
    # Arrange / Given
    subjects = [101, 203, 305]

    # Act / When
    script = render_slurm_array_script(
        subjects=subjects,
        conda_env="jaxcmr",
        python="/path/to/python",
        workdir="/work/jaxcmr",
        out_dir="/work/out",
        cfg_path="/work/cfg.json",
        job_name="fitjob",
        partition=None,
        time_limit="01:00:00",
        mem="4G",
        cpus_per_task=1,
    )

    # Assert / Then
    assert "#SBATCH --array=0-2" in script
    assert "SUBJECTS=(101 203 305)" in script
    assert "--subject-id $SUBJECT_ID" in script
    assert "-m jaxcmr.runner run-subject" in script

