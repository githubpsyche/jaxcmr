"""Workflow helpers for fitting, simulation, and visualization pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import jax.numpy as jnp
import numpy as np
from jax import random
from matplotlib import pyplot as plt
from matplotlib import rcParams

from jaxcmr import repetition
from jaxcmr.fitting import ScipyDE
from jaxcmr.helpers import (
    find_project_root,
    generate_trial_mask,
    import_from_string,
    load_data,
    save_dict_to_hdf5,
)
from jaxcmr.math import cosine_similarity_matrix
from jaxcmr.simulation import simulate_h5_from_h5
from jaxcmr.typing import FitResult, RecallDataset


CallableLike = Callable[..., Any]


@dataclass(slots=True)
class PipelineConfig:
    """Configuration bundle for a fit→simulate→plot workflow."""

    base_run_tag: str
    best_of: int
    max_subjects: int
    target_directory: str
    data_tag: str
    data_path: str
    embedding_path: str
    trial_query: str | None
    model_name: str
    model_factory: str | CallableLike
    loss_fn_generator: str | CallableLike
    simulate_trial_fn: str | CallableLike
    fitting_method: str | type[ScipyDE] = ScipyDE
    parameters: Mapping[str, Mapping[str, Any]] | None = None
    hyperparameters: Mapping[str, Any] | None = None
    experiment_count: int = 1
    seed: int = 0
    allow_repeated_recalls: bool = False
    filter_repeated_recalls: bool = False
    redo_fits: bool = False
    redo_sims: bool = False
    redo_figures: bool = False
    figure_labels: tuple[str, str] = ("Model", "Data")
    contrast_name: str = "Source"
    comparison_analyses: Sequence[str | CallableLike] = ()


@dataclass(slots=True)
class PipelineArtifacts:
    """Artifacts produced by a completed pipeline run."""

    fit_result: FitResult
    simulation: RecallDataset
    figure_paths: list[Path]


def derive_run_tag(base_run_tag: str, best_of: int, max_subjects: int) -> str:
    """Returns the run tag derived from the base tag and configuration.

    Args:
      base_run_tag: Identifier shared across related runs.
      best_of: Count of restarts used when fitting per subject.
      max_subjects: Subject cap; zero indicates that all subjects are used.
    """
    run_tag = f"{base_run_tag}_best_of_{best_of}"
    if max_subjects:
        run_tag += f"_nsubs_{max_subjects}"
    return run_tag


def ensure_product_directories(
    target_directory: str | Path, products: Sequence[str]
) -> dict[str, Path]:
    """Returns product directories, creating them when missing.

    Args:
      target_directory: Base directory that houses all pipeline artifacts.
      products: Subdirectory names to provision under the base directory.
    """
    base_path = Path(target_directory)
    base_path.mkdir(parents=True, exist_ok=True)
    directories: dict[str, Path] = {}
    for product in products:
        product_path = base_path / product
        product_path.mkdir(parents=True, exist_ok=True)
        directories[product] = product_path
    return directories


def resolve_callable(target: str | CallableLike) -> CallableLike:
    """Returns the callable referenced by ``target``.

    Args:
      target: Dotted import path or callable to use directly.
    """
    if isinstance(target, str):
        return import_from_string(target)
    return target


def compute_semantic_matrices(embedding_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns connectivity and distance matrices derived from embeddings.

    Args:
      embedding_path: Location of the NumPy embedding file.
    """
    embeddings = np.load(str(embedding_path)).astype(np.float32)
    connections = cosine_similarity_matrix(embeddings)
    distances = 1.0 - connections
    return connections, distances


def run_fit(
    dataset: RecallDataset,
    connections: np.ndarray | None,
    trial_mask: jnp.ndarray,
    fit_path: Path,
    parameters: Mapping[str, Mapping[str, Any]],
    fitting_cls: type[ScipyDE],
    model_factory: CallableLike,
    loss_fn_generator: CallableLike,
    hyperparameters: Mapping[str, Any] | None,
    metadata: Mapping[str, Any],
    redo: bool = False,
) -> FitResult:
    """Returns fit results, loading from disk when available.

    Args:
      dataset: Trial-indexed dataset to fit the model against.
      connections: Optional semantic similarity matrix aligned to the dataset.
      trial_mask: Boolean mask selecting the trials included in the fit.
      fit_path: Location where fit results are stored as JSON.
      parameters: Dictionary with ``fixed`` and ``free`` parameter definitions.
      fitting_cls: Fitting algorithm class exposing a ``fit`` method.
      model_factory: Factory used to instantiate the memory model.
      loss_fn_generator: Generator that produces loss functions per trial mask.
      hyperparameters: Optional overrides forwarded to the fitting class.
      metadata: Additional key-value pairs persisted alongside the fit.
      redo: Forces a fresh fit when set to ``True``.
    """
    if fit_path.exists() and not redo:
        with fit_path.open() as handle:
            results: FitResult = json.load(handle)
        if "subject" not in results["fits"]:
            results["fits"]["subject"] = results.get("subject", [])
        results.update(metadata)
        return results

    fixed_params = dict(parameters.get("fixed", {}))
    free_params = dict(parameters.get("free", {}))

    hyperparams = dict(hyperparameters or {})
    hyperparams["bounds"] = free_params

    fitter = fitting_cls(
        dataset,
        connections,
        fixed_params,
        model_factory,
        loss_fn_generator,
        hyperparams=hyperparams,
    )

    results = dict(fitter.fit(trial_mask))
    results.update(metadata)

    with fit_path.open("w") as handle:
        json.dump(results, handle, indent=2)

    return results


def run_simulation(
    dataset: RecallDataset,
    results: FitResult,
    connections: np.ndarray,
    trial_query: str | None,
    experiment_count: int,
    seed: int,
    simulate_trial_fn: CallableLike,
    model_factory: CallableLike,
    sim_path: Path,
    redo: bool = False,
    filter_repeated_recalls: bool = False,
) -> RecallDataset:
    """Returns simulated trials corresponding to the fit configuration.

    Args:
      dataset: Empirical dataset used for fitting.
      results: Fit results whose parameters seed the simulations.
      connections: Semantic connectivity matrix aligned to the word pool.
      trial_query: Optional query string reused to mask trials.
      experiment_count: Number of synthetic datasets to simulate per subject.
      seed: Seed used to initialize the RNG for simulations.
      simulate_trial_fn: Callable that simulates a single trial.
      model_factory: Factory that produces model instances for simulation.
      sim_path: Location where simulations are cached in HDF5 format.
      redo: Forces a fresh simulation when ``True``.
      filter_repeated_recalls: Applies repetition filtering after simulation.
    """
    if sim_path.exists() and not redo:
        sim = load_data(str(sim_path))
        if filter_repeated_recalls:
            sim["recalls"] = repetition.filter_repeated_recalls(sim["recalls"])
        return sim

    rng = random.PRNGKey(seed)
    rng, rng_iter = random.split(rng)

    trial_mask = generate_trial_mask(dataset, trial_query)
    params = {
        key: jnp.array(value)
        for key, value in results["fits"].items()
        if key != "subject"
    }
    params["subject"] = jnp.array(results["fits"]["subject"])

    sim = simulate_h5_from_h5(
        model_factory=model_factory,
        dataset=dataset,
        connections=connections,
        parameters=params,
        trial_mask=trial_mask,
        experiment_count=experiment_count,
        rng=rng_iter,
        simulate_trial_fn=simulate_trial_fn,
    )

    save_dict_to_hdf5(sim, str(sim_path))
    if filter_repeated_recalls:
        sim["recalls"] = repetition.filter_repeated_recalls(sim["recalls"])
    return sim  # type: ignore[return-value]


def render_figures(
    analyses: Sequence[CallableLike],
    sim: RecallDataset,
    data: RecallDataset,
    figure_dir: Path,
    figure_prefix: str,
    trial_query: str | None,
    distances: np.ndarray,
    redo: bool = False,
    labels: Sequence[str] = ("Model", "Data"),
    contrast_name: str = "Source",
    display_callback: Callable[[Path], None] | None = None,
) -> list[Path]:
    """Returns paths to generated comparison figures.

    Args:
      analyses: Plotting callables that accept ``datasets`` and ``trial_masks``.
      sim: Simulated dataset to compare against empirical observations.
      data: Empirical dataset used for fitting.
      figure_dir: Directory for rendered figures.
      figure_prefix: Filename stem applied to each figure.
      trial_query: Optional query reused to mask trials for both datasets.
      distances: Semantic distance matrix derived from embeddings.
      redo: Forces regeneration of existing figures when ``True``.
      labels: Labels applied to the legend order ``[sim, data]``.
      contrast_name: Legend title used by downstream analyses.
      display_callback: Optional hook invoked with the saved figure path.
    """
    figure_paths: list[Path] = []
    data_mask = generate_trial_mask(data, trial_query)
    sim_mask = generate_trial_mask(sim, trial_query)
    color_cycle = [style["color"] for style in rcParams["axes.prop_cycle"]]

    for analysis in analyses:
        suffix = analysis.__name__.removeprefix("plot_")
        figure_name = f"{figure_prefix}_{suffix}.png"
        figure_path = figure_dir / figure_name

        if figure_path.exists() and not redo:
            if display_callback is not None:
                display_callback(figure_path)
            figure_paths.append(figure_path)
            continue

        axis = analysis(
            datasets=[sim, data],
            trial_masks=[np.array(sim_mask), np.array(data_mask)],
            color_cycle=color_cycle,
            labels=list(labels),
            contrast_name=contrast_name,
            axis=None,
            distances=distances,
        )

        figure = axis.figure if axis is not None else plt.gcf()
        figure.savefig(figure_path, bbox_inches="tight", dpi=600)
        if display_callback is not None:
            display_callback(figure_path)
        plt.close(figure)
        figure_paths.append(figure_path)

    return figure_paths


def run_pipeline(
    config: PipelineConfig, display_callback: Callable[[Path], None] | None = None
) -> PipelineArtifacts:
    """Run the entire fit→simulate→plot workflow.

    Args:
      config: Aggregated configuration for the workflow.
      display_callback: Optional hook invoked with each rendered figure path.
    """
    project_root = Path(find_project_root())
    data_path = project_root / config.data_path
    embedding_path = project_root / config.embedding_path

    data = load_data(str(data_path), config.max_subjects)
    connections, distances = compute_semantic_matrices(embedding_path)

    run_tag = derive_run_tag(config.base_run_tag, config.best_of, config.max_subjects)

    product_dirs = ensure_product_directories(
        config.target_directory, ["fits", "simulations", "figures"]
    )

    figure_prefix = f"{config.data_tag}_{config.model_name}_{run_tag}"
    fit_path = product_dirs["fits"] / f"{figure_prefix}.json"
    sim_path = product_dirs["simulations"] / f"{figure_prefix}.h5"

    model_factory = resolve_callable(config.model_factory)
    loss_fn_generator = resolve_callable(config.loss_fn_generator)
    simulate_trial_fn = resolve_callable(config.simulate_trial_fn)

    fitting_cls = resolve_callable(config.fitting_method)
    if not isinstance(fitting_cls, type):
        raise TypeError("fitting_method must resolve to a class.")

    parameter_block = config.parameters or {}
    fixed_parameters = dict(parameter_block.get("fixed", {}))
    free_parameters = dict(parameter_block.get("free", {}))
    fixed_parameters["allow_repeated_recalls"] = config.allow_repeated_recalls
    parameters = {"fixed": fixed_parameters, "free": free_parameters}

    trial_mask = generate_trial_mask(data, config.trial_query)

    metadata = {
        "data_query": config.trial_query,
        "model": config.model_name,
        "name": figure_prefix,
    }

    hyperparameters = dict(config.hyperparameters or {})
    hyperparameters.setdefault("best_of", config.best_of)

    fit_result = run_fit(
        dataset=data,
        connections=connections,
        trial_mask=trial_mask,
        fit_path=fit_path,
        parameters=parameters,
        fitting_cls=fitting_cls,
        model_factory=model_factory,
        loss_fn_generator=loss_fn_generator,
        hyperparameters=hyperparameters,
        metadata=metadata,
        redo=config.redo_fits,
    )

    simulation = run_simulation(
        dataset=data,
        results=fit_result,
        connections=connections,
        trial_query=config.trial_query,
        experiment_count=config.experiment_count,
        seed=config.seed,
        simulate_trial_fn=simulate_trial_fn,
        model_factory=model_factory,
        sim_path=sim_path,
        redo=config.redo_sims,
        filter_repeated_recalls=config.filter_repeated_recalls,
    )

    analyses = [resolve_callable(item) for item in config.comparison_analyses]

    figure_paths = render_figures(
        analyses=analyses,
        sim=simulation,
        data=data,
        figure_dir=product_dirs["figures"],
        figure_prefix=figure_prefix,
        trial_query=config.trial_query,
        distances=distances,
        redo=config.redo_figures,
        labels=config.figure_labels,
        contrast_name=config.contrast_name,
        display_callback=display_callback,
    )

    return PipelineArtifacts(
        fit_result=fit_result,
        simulation=simulation,
        figure_paths=figure_paths,
    )
