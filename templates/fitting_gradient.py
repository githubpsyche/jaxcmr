"""Fit regular CMR to Lohnas-Kahana 2014 participants with JAX gradients.

Run from a jaxcmr development checkout with its dependencies installed:

    python templates/fitting_gradient.py --subject 2 --seed 700 --seconds 4

For multiple participants with shared likelihood compilation:

    python templates/fitting_gradient.py --subjects 31 36 --seed 700 --seconds 4

For a minimal example with actual repeated presentations, add --trial-index 82.
Use --output PATH to retain the full FitResult, timings and data provenance.
This example fits recall order conditional on observed recall length. The
NoStopTermination policy and loss do not estimate stopping behavior.
The defaults use the validated screened trust-constr/BFGS recipe and optimized
regular-CMR model. Use --optimizer lbfgsb or --model reference to
compare alternatives through the same public interfaces.

The model uses primacy_scale and primacy_decay. The fitter internally searches
over exp(-primacy_decay); reported parameters and bounds remain physical.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxcmr.components import context, linear_memory
from jaxcmr.components.termination import NoStopTermination
from jaxcmr.fitting import ScipyLBFGSB, ScipyTrustConstr
from jaxcmr.helpers import load_data
from jaxcmr.loss.transform_sequence_likelihood import ExcludeTerminationLikelihoodLoss
from jaxcmr.models.cmr import make_factory
from jaxcmr.models.optimized_cmr import ModelFactory as OptimizedCMRFactory
from jaxcmr.typing import RecallDataset


def fit_cohort(
    fitter,
    dataset,
    reference_factory,
    base_params,
    bounds,
    rows,
    args,
    construction_seconds,
):
    """Use the public preparation/warmup/fit methods for each selected subject."""
    subjects = np.asarray(dataset["subject"]).reshape(-1)
    masks = [jnp.asarray(subjects == subject) for subject in args.subjects]
    preparation_seconds = fitter.prepare_selections(masks)
    canonical = ExcludeTerminationLikelihoodLoss(reference_factory, dataset, None)
    results = []
    for subject, mask in zip(args.subjects, masks, strict=True):
        warmup = fitter.warmup(mask)
        result = fitter.fit(mask, subject_id=subject)
        before = time.perf_counter()
        indices = jnp.asarray(np.flatnonzero(np.asarray(mask)), dtype=jnp.int32)
        parameters = jnp.asarray([result["fits"][name][0] for name in bounds])
        checked = float(
            jax.jit(
                lambda p, selected: canonical(
                    selected, base_params, tuple(bounds), p[:, None]
                )[0]
            )(parameters, indices)
        )
        np.testing.assert_allclose(result["fitness"][0], checked, rtol=1e-10, atol=1e-8)
        results.append(
            {
                "subject": subject,
                "fit": result,
                "warmup_seconds": warmup,
                "validation_seconds": time.perf_counter() - before,
                "verified_nll": checked,
                "row_indices": rows[np.asarray(mask)].tolist(),
                "recall_count": int(
                    np.count_nonzero(np.asarray(dataset["recalls"])[np.asarray(mask)])
                ),
            }
        )
    return {
        "subjects": args.subjects,
        "results": results,
        "construction_seconds": construction_seconds,
        "preparation_seconds": preparation_seconds,
        "setup_plus_fit_seconds": construction_seconds
        + preparation_seconds
        + sum(row["warmup_seconds"] + row["fit"]["fit_time"] for row in results),
        "data_path": str(args.data.resolve()),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "row_indices": rows.tolist(),
        "model": args.model,
        "jax_x64": True,
        "optimizer": type(fitter).__name__,
        "loss": type(fitter.loss_fn).__name__,
        "canonical_verification_loss": "ExcludeTerminationLikelihoodLoss",
        "termination": "NoStopTermination",
        "timing_note": "Setup and fitting exclude data loading, interpreter startup and independent validation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/LohnasKahana2014.h5",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--subject", type=int, default=2)
    selection.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        help="Fit each participant with one shared preparation plan.",
    )
    parser.add_argument("--seed", type=int, default=700)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--local-seconds", type=float, default=1.0)
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument(
        "--optimizer", choices=("trust-constr", "lbfgsb"), default="trust-constr"
    )
    parser.add_argument(
        "--model",
        choices=("optimized", "reference"),
        default="optimized",
        help="Choose the CMR implementation; both use the same generic likelihood.",
    )
    parser.add_argument(
        "--trial-index",
        type=int,
        help="Optional original file row belonging to this participant.",
    )
    parser.add_argument(
        "--output", type=Path, help="New JSON file for the fit and provenance."
    )
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("The output file already exists; choose a new path.")

    # Set precision before constructing data or model arrays. Retention values
    # corresponding to decay near 100 are not representable reliably in float32.
    jax.config.update("jax_enable_x64", True)
    example_started = time.perf_counter()
    raw = load_data(str(args.data))
    subject_ids = args.subjects if args.subjects is not None else [args.subject]
    available = np.asarray(raw["subject"]).reshape(-1)
    if len(set(subject_ids)) != len(subject_ids) or any(
        subject not in available for subject in subject_ids
    ):
        parser.error("Select distinct participants present in the data file.")
    rows = np.flatnonzero(np.isin(available, subject_ids))
    if args.trial_index is not None:
        if args.subjects is not None:
            parser.error("trial-index is supported with --subject, not --subjects.")
        if args.trial_index not in rows:
            parser.error("trial-index must belong to the selected participant.")
        rows = np.array([args.trial_index])
    dataset: RecallDataset = {
        "subject": jnp.asarray(raw["subject"][rows], dtype=jnp.int32),
        "listLength": jnp.asarray(raw["listLength"][rows], dtype=jnp.int32),
        "pres_itemnos": jnp.asarray(raw["pres_itemnos"][rows], dtype=jnp.int32),
        "recalls": jnp.asarray(raw["recalls"][rows], dtype=jnp.int32),
    }
    # Keep all repeated study events. Recall responses are position codes; the
    # selected model's prediction path maps them to presented item labels.
    for present, recall, length in zip(
        np.asarray(dataset["pres_itemnos"]),
        np.asarray(dataset["recalls"]),
        np.asarray(dataset["listLength"]).reshape(-1),
        strict=True,
    ):
        occupied = np.flatnonzero(recall != 0)
        responses = recall[occupied]
        if length != len(present) or np.any(present <= 0):
            parser.error("This example requires complete, unpadded study lists.")
        if np.any((responses < 1) | (responses > length)):
            parser.error(
                "Intrusions need an explicit likelihood convention before fitting."
            )
        if not np.array_equal(occupied, np.arange(len(occupied))):
            parser.error(
                "This example expects recall padding only after the last response."
            )
        if len(np.unique(present[responses - 1])) != len(responses):
            parser.error(
                "This no-repeat-recall configuration cannot score repeated responses."
            )

    eps = float(np.finfo(float).eps)
    bounds = {
        "encoding_drift_rate": [eps, 1.0 - eps],
        "start_drift_rate": [eps, 1.0 - eps],
        "recall_drift_rate": [eps, 1.0 - eps],
        "shared_support": [eps, 100.0],
        "item_support": [eps, 100.0],
        "learning_rate": [eps, 1.0 - eps],
        "primacy_scale": [eps, 100.0],
        "primacy_decay": [eps, 100.0],
        "choice_sensitivity": [eps, 100.0],
    }
    base_params = {"learn_after_context_update": False, "allow_repeated_recalls": False}
    reference_factory = make_factory(
        linear_memory.init_mfc, linear_memory.init_mcf, context.init, NoStopTermination
    )
    factory = OptimizedCMRFactory if args.model == "optimized" else reference_factory
    fitter_class = ScipyTrustConstr if args.optimizer == "trust-constr" else ScipyLBFGSB
    loss_class = ExcludeTerminationLikelihoodLoss
    settings = {
        "bounds": bounds,
        "transforms": {"primacy_decay": "exp_neg"},
        "sampling_transforms": {
            "primacy_decay": "exp_neg",
            **{
                name: "log1p"
                for name in (
                    "shared_support",
                    "item_support",
                    "primacy_scale",
                    "choice_sensitivity",
                )
            },
        },
        "screening_size": 64,
        "screening_batch_size": 16,
        "best_of": args.starts,
        "min_start_distance": 0.1,
        "num_steps": 1000,
        "gradient_tolerance": 1e-7,
        "seed": args.seed,
        "time_limit_seconds": args.seconds,
        "local_time_limit_seconds": args.local_seconds,
    }
    if args.optimizer == "lbfgsb":
        settings.update(
            max_fun=15000, relative_tolerance=1e-12, max_line_search_steps=50
        )
    construction_started = time.perf_counter()
    fitter = fitter_class(dataset, None, base_params, factory, loss_class, settings)
    construction_seconds = time.perf_counter() - construction_started
    if args.subjects is not None:
        payload = fit_cohort(
            fitter,
            dataset,
            reference_factory,
            base_params,
            bounds,
            rows,
            args,
            construction_seconds,
        )
        payload["example_seconds"] = time.perf_counter() - example_started
        if args.output is not None:
            with args.output.open("x") as file:
                json.dump(payload, file, indent=2, allow_nan=False)
                file.write("\n")
        print(
            json.dumps(
                {
                    "subjects": args.subjects,
                    "setup_plus_fit_seconds": payload["setup_plus_fit_seconds"],
                    "fits": [
                        {
                            "subject": row["subject"],
                            "nll": row["verified_nll"],
                            "warmup_seconds": row["warmup_seconds"],
                            "fit_seconds": row["fit"]["fit_time"],
                        }
                        for row in payload["results"]
                    ],
                },
                indent=2,
            )
        )
        return
    mask = jnp.ones(len(rows), dtype=bool)
    warmup_seconds = fitter.warmup(mask)
    result = fitter.fit(mask, subject_id=args.subject)

    # Independently evaluate the reported physical values with the original
    # likelihood interface, without the optimizer's parameter transform.
    validation_started = time.perf_counter()
    parameters = jnp.array([result["fits"][name][0] for name in bounds])
    checked = float(
        ExcludeTerminationLikelihoodLoss(reference_factory, dataset, None)(
            jnp.arange(len(rows), dtype=jnp.int32),
            base_params,
            tuple(bounds),
            parameters[:, None],
        )[0]
    )
    np.testing.assert_allclose(checked, result["fitness"][0], rtol=1e-10, atol=1e-8)
    validation_seconds = time.perf_counter() - validation_started
    payload = {
        "fit": result,
        "warmup_seconds": warmup_seconds,
        "construction_seconds": construction_seconds,
        "warmup_plus_fit_seconds": warmup_seconds + result["fit_time"],
        "construction_plus_warmup_plus_fit_seconds": construction_seconds
        + warmup_seconds
        + result["fit_time"],
        "validation_seconds": validation_seconds,
        "example_seconds": time.perf_counter() - example_started,
        "verified_nll": checked,
        "data_path": str(args.data.resolve()),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "row_indices": rows.tolist(),
        "list_count": len(rows),
        "recall_count": int(np.count_nonzero(np.asarray(dataset["recalls"]))),
        "model": args.model,
        "jax_x64": True,
        "optimizer": fitter_class.__name__,
        "loss": loss_class.__name__,
        "canonical_verification_loss": "ExcludeTerminationLikelihoodLoss",
        "termination": "NoStopTermination",
    }
    if args.output is not None:
        with args.output.open("x") as file:
            json.dump(payload, file, indent=2, allow_nan=False)
            file.write("\n")
    print(
        json.dumps(
            {
                "subject": args.subject,
                "lists": len(rows),
                "recalls": payload["recall_count"],
                "nll": checked,
                "construction_seconds": construction_seconds,
                "warmup_seconds": warmup_seconds,
                "fit_seconds": result["fit_time"],
                "setup_plus_fit_seconds": payload[
                    "construction_plus_warmup_plus_fit_seconds"
                ],
                "optimizer": fitter_class.__name__,
                "likelihood": loss_class.__name__,
                "parameters": result["fits"],
                "converged": result["converged"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
