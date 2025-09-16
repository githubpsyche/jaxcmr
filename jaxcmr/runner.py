"""CLI utilities to fit models locally or via SLURM.

Provides a per-subject runner, an optional local multi-subject orchestrator,
SLURM array-script generation, and aggregation of per-subject results. Defaults
to easy local usage while enabling clean SLURM submission when desired.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from .fitting import ScipyDE, make_subject_trial_masks
from .helpers import generate_trial_mask, load_data
from .typing import Float_, RecallDataset


# -----------------------------
# Data classes and config utils
# -----------------------------


@dataclass(frozen=True)
class FitConfig:
    """Returns a normalized fit configuration.

    Args:
      data_path: Path to an HDF5 dataset file.
      model_factory_path: Dotted import path to a MemorySearchModelFactory implementation.
      free: Parameter bounds keyed by parameter name.
      fixed: Fixed parameter values keyed by name.
      hyperparams: Optional overrides for the fitting algorithm.
      trial_query: Optional trial selection expression evaluated against `data`.
      connections_path: Optional path to a connectivity matrix (HDF5 or npy).
    """

    data_path: str
    model_factory_path: str
    free: dict[str, list[float]]
    fixed: Mapping[str, Float_]
    hyperparams: dict[str, Any] | None = None
    trial_query: str | None = None
    connections_path: str | None = None


def _import_from_string(import_string: str) -> Any:
    """Returns attribute resolved from a dotted import path.

    Args:
      import_string: Module path and attribute like "pkg.mod.ClassName".
    """

    module_name, attr_name = import_string.rsplit(".", 1)
    module = __import__(module_name, fromlist=[attr_name])
    return getattr(module, attr_name)


def _load_connections(path: Optional[str]) -> Optional[np.ndarray]:
    """Returns connectivity matrix loaded from file when provided.

    Supports `.npy`, `.npz`, or `h5` under group "data" with dataset "connections".

    Args:
      path: Optional file path to connectivity data.
    """

    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"connections file not found: {path}")
    if p.suffix == ".npy":
        return np.load(p)
    if p.suffix == ".npz":
        return np.load(p)["connections"]
    if p.suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(p, "r") as f:
            return f["/data"]["connections"][()].T  # type: ignore
    raise ValueError(f"Unsupported connections format: {p.suffix}")


# ------------------
# Per-subject runner
# ------------------


def run_subject_fit(
    cfg: FitConfig,
    subject_id: int,
    out_path: str,
) -> dict[str, Any]:
    """Returns the per-subject fit result and writes JSON to `out_path`.

    Applies the base `trial_query` and intersects with the subject-specific mask.
    When the subject has no selected trials, writes a small JSON payload and returns it.

    Args:
      cfg: Fit configuration with paths, model factory, and parameters.
      subject_id: Subject identifier to fit.
      out_path: Output file path for the JSON result.
    """

    # Load dataset and connections
    data: RecallDataset = load_data(cfg.data_path)
    connections = _load_connections(cfg.connections_path)

    # Build trial mask for this subject
    base_mask = generate_trial_mask(data, cfg.trial_query)
    subject_vector = data["subject"].flatten()
    subject_mask = (subject_vector == subject_id) & base_mask.astype(bool)
    selected = int(np.sum(np.array(subject_mask)))

    result: dict[str, Any]
    if selected == 0:
        result = {
            "status": "skipped",
            "reason": "no trials for subject after masking",
            "subject": subject_id,
        }
        return _write_json(out_path, result)
    # Instantiate algorithm and fit
    ModelFactory = _import_from_string(cfg.model_factory_path)
    algo = ScipyDE(
        dataset=data,
        connections=connections,  # type: ignore[arg-type]
        base_params=dict(cfg.fixed),
        model_factory=ModelFactory,
        loss_fn_generator=_import_from_string(
            "jaxcmr.likelihood.MemorySearchLikelihoodFnGenerator"
        ),
        hyperparams={"bounds": cfg.free, **(cfg.hyperparams or {})},
    )
    fit = algo.fit(trial_mask=subject_mask, fit_to_subjects=False)
    payload: dict[str, Any] = {
        "status": "ok",
        "subject": subject_id,
        "fit": fit,
        "config": {
            "data_path": cfg.data_path,
            "model_factory_path": cfg.model_factory_path,
            "trial_query": cfg.trial_query,
        },
    }

    return _write_json(out_path, payload)


def _write_json(out_path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Returns payload after writing JSON to `out_path`.

    Args:
      out_path: Output file path for the JSON payload.
      payload: Serializable mapping to write.
    """

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f)
    return payload


# -------------
# Aggregation
# -------------


def aggregate_subject_results(paths: Iterable[str]) -> dict[str, Any]:
    """Returns a merged FitResult across subjects from JSON payloads.

    Args:
      paths: Iterable of file paths to per-subject JSON results produced by `run_subject_fit`.
    """

    merged: dict[str, Any] | None = None
    for p in paths:
        with open(p, "r") as f:
            rec = json.load(f)
        if rec.get("status") != "ok":
            continue
        fit = rec["fit"]
        if merged is None:
            merged = {
                "fixed": fit["fixed"],
                "free": fit["free"],
                "fitness": list(fit.get("fitness", [])),
                "fits": {k: list(v) for k, v in fit["fits"].items()},
                "hyperparameters": fit.get("hyperparameters", {}),
                "fit_time": float(fit.get("fit_time", 0.0)),
            }
            continue

        # Consistency: ensure same parameter sets
        if set(merged["fixed"]) != set(fit["fixed"]) or set(merged["free"]) != set(
            fit["free"]
        ):
            raise ValueError("Inconsistent parameter sets across subject results")

        merged["fitness"].extend(fit.get("fitness", []))
        for k, v in fit["fits"].items():
            merged["fits"][k].extend(v)
        merged["fit_time"] += float(fit.get("fit_time", 0.0))

    return merged or {
        "fixed": {},
        "free": {},
        "fitness": [],
        "fits": {"subject": []},
        "hyperparameters": {},
        "fit_time": 0.0,
    }


# ---------------------
# SLURM array script IO
# ---------------------


def render_slurm_array_script(
    subjects: list[int],
    conda_env: str,
    python: str,
    workdir: str,
    out_dir: str,
    cfg_path: str,
    job_name: str = "jaxcmr-fit",
    partition: str | None = None,
    time_limit: str = "2:00:00",
    mem: str = "8G",
    cpus_per_task: int = 2,
) -> str:
    """Returns a SLURM array-job script.

    Uses a bash array to map `SLURM_ARRAY_TASK_ID` to subject ids and runs the
    per-subject runner for each array index.

    Args:
      subjects: Subject identifiers to fit.
      conda_env: Name of the conda environment to activate.
      python: Absolute path to the Python executable in that environment.
      workdir: Absolute path to the repository working directory.
      out_dir: Directory where per-subject results are written.
      cfg_path: Path to a JSON config consumable by `run-subject`.
      job_name: SLURM job name.
      partition: Optional SLURM partition or queue name.
      time_limit: Walltime in HH:MM:SS.
      mem: Memory per task.
      cpus_per_task: Logical CPUs per task.
    """

    subs = " ".join(str(int(s)) for s in subjects)
    part_line = f"#SBATCH --partition={partition}\n" if partition else ""
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --array=0-{len(subjects)-1}
{part_line}#SBATCH --time={time_limit}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --output={out_dir}/logs/%x_%A_%a.out
#SBATCH --error={out_dir}/logs/%x_%A_%a.err

set -euo pipefail

mkdir -p {out_dir}/logs
cd {workdir}

export CONDA_DEFAULT_ENV={conda_env}
source "$HOME/miniconda3/etc/profile.d/conda.sh" || true
conda activate {conda_env} || true

SUBJECTS=({subs})
SUBJECT_ID=${{SUBJECTS[$SLURM_ARRAY_TASK_ID]}}

{python} -m jaxcmr.runner run-subject \
  --config {cfg_path} \
  --subject-id $SUBJECT_ID \
  --out {out_dir}/fit_subject_${{SUBJECT_ID}}.json
"""


# ---------
# CLI
# ---------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="jaxcmr-runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # run-subject
    p_sub = sub.add_parser("run-subject", help="Fit a single subject and write JSON")
    p_sub.add_argument("--config", required=True, help="Path to JSON fit config")
    p_sub.add_argument("--subject-id", required=True, type=int)
    p_sub.add_argument("--out", required=True, help="Output JSON path")

    # aggregate
    p_agg = sub.add_parser("aggregate", help="Aggregate per-subject JSON results")
    p_agg.add_argument("--glob", required=True, help="Glob for subject JSONs")
    p_agg.add_argument("--out", required=True, help="Output JSON path")

    # run-dataset (local sequential/parallel)
    p_ds = sub.add_parser("run-dataset", help="Run all subjects locally")
    p_ds.add_argument("--config", required=True)
    p_ds.add_argument("--out-dir", required=True)
    p_ds.add_argument("--parallel", action="store_true")
    p_ds.add_argument("--workers", type=int, default=0, help="0=auto when parallel")

    # slurm-script
    p_slurm = sub.add_parser("slurm-script", help="Render a SLURM array script")
    p_slurm.add_argument("--subjects", required=True, help="Comma-separated subject ids")
    p_slurm.add_argument("--conda-env", required=True)
    p_slurm.add_argument("--python", required=True)
    p_slurm.add_argument("--workdir", required=True)
    p_slurm.add_argument("--out-dir", required=True)
    p_slurm.add_argument("--config", required=True)
    p_slurm.add_argument("--job-name", default="jaxcmr-fit")
    p_slurm.add_argument("--partition", default=None)
    p_slurm.add_argument("--time", default="2:00:00")
    p_slurm.add_argument("--mem", default="8G")
    p_slurm.add_argument("--cpus", default=2, type=int)
    p_slurm.add_argument("--out", required=False, help="Write script to this path")

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Returns exit code after running requested subcommand.

    Args:
      argv: Optional override for CLI args (used in tests).
    """

    ns = _parse_args(list(argv or sys.argv[1:]))

    if ns.cmd == "run-subject":
        with open(ns.config, "r") as f:
            cfg_dict = json.load(f)
        cfg = FitConfig(
            data_path=cfg_dict["data_path"],
            model_factory_path=cfg_dict["model_factory_path"],
            free=cfg_dict["parameters"]["free"],
            fixed=cfg_dict["parameters"]["fixed"],
            hyperparams=cfg_dict.get("hyperparams"),
            trial_query=cfg_dict.get("trial_query"),
            connections_path=cfg_dict.get("connections_path"),
        )
        run_subject_fit(cfg, ns.subject_id, ns.out)
        return 0

    if ns.cmd == "aggregate":
        import glob

        paths = sorted(glob.glob(ns.glob))
        merged = aggregate_subject_results(paths)
        Path(ns.out).parent.mkdir(parents=True, exist_ok=True)
        with open(ns.out, "w") as f:
            json.dump(merged, f)
        return 0

    if ns.cmd == "slurm-script":
        subjects = [int(s) for s in ns.subjects.split(",") if s]
        script = render_slurm_array_script(
            subjects=subjects,
            conda_env=ns.conda_env,
            python=ns.python,
            workdir=ns.workdir,
            out_dir=ns.out_dir,
            cfg_path=ns.config,
            job_name=ns.job_name,
            partition=ns.partition,
            time_limit=ns.time,
            mem=ns.mem,
            cpus_per_task=ns.cpus,
        )
        if ns.out:
            Path(ns.out).parent.mkdir(parents=True, exist_ok=True)
            with open(ns.out, "w") as f:
                f.write(script)
        else:
            sys.stdout.write(script)
        return 0

    if ns.cmd == "run-dataset":
        with open(ns.config, "r") as f:
            cfg_dict = json.load(f)
        cfg = FitConfig(
            data_path=cfg_dict["data_path"],
            model_factory_path=cfg_dict["model_factory_path"],
            free=cfg_dict["parameters"]["free"],
            fixed=cfg_dict["parameters"]["fixed"],
            hyperparams=cfg_dict.get("hyperparams"),
            trial_query=cfg_dict.get("trial_query"),
            connections_path=cfg_dict.get("connections_path"),
        )
        data: RecallDataset = load_data(cfg.data_path)
        base_mask = generate_trial_mask(data, cfg.trial_query)
        _, uniq = make_subject_trial_masks(base_mask, data["subject"].flatten())
        subjects = [int(s) for s in uniq]
        Path(ns.out_dir).mkdir(parents=True, exist_ok=True)

        def _one(sid: int) -> None:
            out_file = Path(ns.out_dir) / f"fit_subject_{sid}.json"
            run_subject_fit(cfg, sid, str(out_file))

        if ns.parallel:
            from concurrent.futures import ProcessPoolExecutor

            workers = None if ns.workers == 0 else int(ns.workers)
            with ProcessPoolExecutor(max_workers=workers) as ex:
                list(ex.map(_one, subjects))
        else:
            for sid in subjects:
                _one(sid)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
