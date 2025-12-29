# RepFR Implementation Plan (Starter Scope)

## Goals
- Rebuild the Chapter 3 pipeline using updated `jaxcmr.models` and TalmiEEG-style render notebooks.
- Start minimal: one dataset, two models, and a small analysis set that matches the fitting template pattern.

## Scope (v0)
- Dataset: `data/LohnasKahana2014.h5`
- Models:
  - `jaxcmr.models.cmr.make_factory`
  - `jaxcmr.models.no_reinstate_cmr.make_factory`
- Analyses:
  - comparison_analyses: `spc`, `crp`, `pnr` (PFR via default PNR behavior)
  - single_analyses: repetition lag-CRP via `repcrp`
- Render notebooks: `render_analyses.ipynb`, `render_fitting.ipynb`
- Templates: use root `templates/` notebooks (no repfr-specific copies yet).
- Parameter bounds: pull from TalmiEEG reference notebooks for the most equivalent models.
- Fitting/simulation settings: match TalmiEEG baseline (NoStop termination + exclude-stop likelihood + forced-stop simulation).

## Notebook spine (TalmiEEG pattern)
- `projects/repfr/notebooks/render_analyses.ipynb`
  - Loops over analysis configs and calls root `templates/` via papermill.
  - Writes rendered notebooks to `projects/repfr/notebooks/rendered/`.
  - Writes figures to `projects/repfr/results/figures/analyses/`.
- `projects/repfr/notebooks/render_fitting.ipynb`
  - Loops over model configs and calls root `templates/fitting.ipynb` via papermill.
  - Controls `redo_fits`, `redo_sims`, and `redo_figures`.
  - Writes rendered notebooks to `projects/repfr/notebooks/rendered/`.
  - Writes results to `projects/repfr/results/{fits,simulations,figures}`.

## Parameters and queries (initial defaults)
- `trial_query`: start with thesis default `data['list_type'] > 0`.
- `control_trial_query`: use thesis default `data['list_type'] == 1` for control-based analyses.
- `allow_repeated_recalls`: `False` (match TalmiEEG baseline).
- `filter_repeated_recalls`: `True` (match TalmiEEG baseline and standard transition gating).

## Repeated recalls handling (decision + rationale)
- Default: `allow_repeated_recalls=False` and `filter_repeated_recalls=True`.
  - Matches TalmiEEG baseline and the analysis gate in `jaxcmr.repetition.filter_repeated_recalls` (counts only first mention per trial).
  - Keeps comparisons clean for `spc`, `crp`, `pnr`, and `repcrp`, which assume a "not yet recalled" gate.
- Optional sensitivity run (later): add a second run tag with `allow_repeated_recalls=True` and `filter_repeated_recalls=False` if we want to study repeated-recall behavior explicitly.

## Reference bounds (TalmiEEG baseline CMR)
Source: `projects/TalmiEEG/notebooks/render_model_fitting.ipynb` model `WeirdCMRNoStop` (factory `jaxcmr.models.cmr.make_factory`).

- Fixed parameters:
  - `allow_repeated_recalls`: `False`
  - `learn_after_context_update`: `False`
- Free parameter bounds (use for both CMR and NoReinstateCMR initially):
  - `encoding_drift_rate`: `[2.220446049250313e-16, 0.9999999999999998]`
  - `start_drift_rate`: `[2.220446049250313e-16, 0.9999999999999998]`
  - `recall_drift_rate`: `[2.220446049250313e-16, 0.9999999999999998]`
  - `shared_support`: `[2.220446049250313e-16, 100.0]`
  - `item_support`: `[2.220446049250313e-16, 100.0]`
  - `learning_rate`: `[2.220446049250313e-16, 0.9999999999999998]`
  - `primacy_scale`: `[2.220446049250313e-16, 100.0]`
  - `primacy_decay`: `[2.220446049250313e-16, 100.0]`
  - `choice_sensitivity`: `[2.220446049250313e-16, 100.0]`
- Termination: TalmiEEG baseline uses `NoStopTermination` (see `templates/fitting.ipynb` component paths).

## Reference fitting/simulation settings (TalmiEEG baseline)
Source: `templates/fitting.ipynb`.

- Component paths:
  - `mfc_create_fn`: `jaxcmr.components.linear_memory.init_mfc`
  - `mcf_create_fn`: `jaxcmr.components.linear_memory.init_mcf`
  - `context_create_fn`: `jaxcmr.components.context.init`
  - `termination_policy_create_fn`: `jaxcmr.components.termination.NoStopTermination`
- Algorithms:
  - `sim_alg_path`: `jaxcmr.simulation.simulate_study_free_recall_and_forced_stop`
  - `loss_fn_path`: `jaxcmr.loss.transform_sequence_likelihood.ExcludeTerminationLikelihoodFnGenerator`
  - `fit_alg_path`: `jaxcmr.fitting.ScipyDE`
- Run defaults:
  - `base_run_tag`: `fixed_term`
  - `experiment_count`: `200`
  - `max_subjects`: `0`
  - `filter_repeated_recalls`: `True`
  - `handle_elis`: `False`
  - DE hyperparams: `relative_tolerance=0.001`, `popsize=15`, `num_steps=1000`, `cross_rate=0.9`, `diff_w=0.85`, `best_of=3`
## Outputs
- Rendered notebooks: `projects/repfr/notebooks/rendered/`
- Figures: `projects/repfr/results/figures/analyses/`
- Fits: `projects/repfr/results/fits/`
- Simulations: `projects/repfr/results/simulations/`

## First implementation steps
1. Add `projects/repfr/notebooks/render_analyses.ipynb` (papermill loop, root templates).
2. Add `projects/repfr/notebooks/render_fitting.ipynb` (papermill loop, root fitting template, bounds above).
3. Dry-run with LohnasKahana2014 + CMR, then enable NoReinstateCMR.
