# Model-comparison correction and caller migration

This change corrects the existing comparison APIs. It does not publish a package,
refit models, regenerate historical research outputs, or change model behavior.
Callers and saved outputs from earlier versions must not be assumed to have been
migrated merely because the library tests pass.

## Score and participant conventions

- Information criteria and BMS require **summed negative log likelihood (NLL)**
  per fitted parameter set. `FitResult.fitness` remains a general optimizer
  objective; MSE and averaged/scaled NLL are not interchangeable with summed NLL.
  The sign or objective cannot be inferred from the score's numerical value.
- `calculate_aic` now uses `sum_rows(2*NLL + 2*k)`, agreeing with AIC weights.
  A pooled fit contributes one row/penalty. Independently fitted participants
  each contribute their own parameter penalty. Remove prior negation workarounds.
  If a caller holds actual log likelihoods, explicitly convert them to NLL first;
  per-observation contributions must be summed into their fitted parameter row.
- `calculate_bic_scores(results, n_obs)` requires observation counts and returns
  ascending BIC: `sum_rows(2*NLL + k*log(n_obs))`. The count vector is attached to
  the **first input record's original row order**, before ID alignment. A scalar
  is accepted only for one fitted row. AICc and BMS's optional `n_obs` follow the
  same count convention. Choose the scientific observation unit explicitly;
  neither list count nor recall-event count is universally prescribed here.
- Subject IDs are read from `fits['subject']`, legacy top-level `subject`, or CV
  `subjects`. Multiple supplied fields must agree in order. IDs must be unique
  integers or strings, with identical sets across models. All participant-wise
  comparisons, and aggregate comparisons of multiple fitted rows, require IDs.
  An unidentified single pooled row is accepted for aggregate criteria only.
- Scores must be nonempty finite 1D arrays. Counts must be positive integers,
  match the fit rows, and exceed `k+1` for AICc. Duplicate model names, conflicting
  IDs, nonfinite scores, and mismatched samples raise `ValueError`. There is no
  automatic deletion of participants. Select a justified common sample upstream.
- Paired functions align to a stable ID order. Strict winner proportions retain
  ties as non-wins for both models. A pooled `subject=-1` row is rejected by
  participant-wise functions. Undefined t intervals for fewer than two
  participants remain missing; that is distinct from accepting missing inputs.

## Changed returns and reporting

```python
mean, halfwidth = pairwise_aic_differences(results)
mean, halfwidth = pairwise_aicc_differences(results, n_obs)
median, lower, upper = pairwise_median_aic_differences(results)
median, lower, upper = pairwise_median_aicc_differences(results, n_obs)
```

The unused equivalence Boolean and `equivalence_margin` argument are removed.
Mean intervals remain `mean ± halfwidth`. Display median intervals as
`median [lower, upper]`: percentile endpoints need not be centered on the median.
Direction remains row model minus column model; negative favors the row model.
The bootstrap retains confidence 0.95, 10,000 resamples and seed 42 by default.

```python
comparison_records, adjustments = floor_nested_fitness(raw_results, nesting_pairs)
adjusted_aic = calculate_aic(comparison_records)
```

The nested-model helper no longer mutates input fits or returns a count. Use
`len(adjustments)` for the number of substitutions. `adjustments` contains
`subject`, `parent`, `child`, `original_nll`, and `adjusted_nll`. An unidentified
pooled row has `subject=None`. Pairs are processed in supplied order; chained
substitutions retain the original raw child score in each table entry.

The comparison records contain names, free parameter names, copied NLL scores,
IDs where available, and `comparison_only=True`. They deliberately omit fitted
parameter estimates. Keep raw records for parameter summaries, simulations and
CV. Explicitly label any table using the adjusted records and save its adjustment
table beside it. Passing an empty pair list returns unchanged *comparison*
records and an empty table, not original fit objects.

This preserves the intended reporting adjustment for incomplete optimization.
It trusts declared nesting and does not verify objective equivalence, feasible
parameter embeddings or parameter bounds. An adjusted score is not a newly
achieved fit. Tuple/list/stringified-pair notebook inputs remain supported.

## BMS correction

BMS retains the AIC/AICc log-evidence approximation, symmetric Dirichlet prior,
Monte Carlo seed 0, sample count, and output columns. The protected-probability
calculation now includes the Dirichlet prior/posterior contribution previously
missing from the alternative-model free energy. The March 2026 correction to
the null-model evidence is retained. Nonconvergence now raises `RuntimeError`.

The numerical checks use independently calculated posterior and omnibus-risk
values for two- and three-model examples, including a non-unit prior, and check
invariance to participant-specific evidence offsets. The free-energy reference
is [SPM's implementation of Rigoux et al.](https://github.com/spm/spm/blob/main/spm_BMS_bor.m),
particularly equations A.17 and A.20. No reference solver or extra test framework
is maintained. AIC-derived evidence remains an approximation, not an integrated
model likelihood; changing that approximation is separate work.

## Caller inventory (11 November 2026)

This inventory examines **source cells**, not imports captured in saved notebook
outputs. Generated staging copies are rebuilt from authoritative sources.

| Consumer | Status and required work |
|---|---|
| JAXCMR `guide/overview/evaluation.ipynb` | Migrated: pass saved NLL records directly, remove negation workaround, explain corrected parameter counting and interval returns. |
| JAXCMR `templates/model_comparison.ipynb` | Migrated: use library AIC on the already recalculated pooled objectives. Keep parameter cells, output paths, row order, and pooled-only inference unchanged. |
| RepFR `scripts/evaluate_first_occurrence.py` | External migration required: change `mean, ci, _` to `mean, ci`. Its `_subjectwise_aic` import is retained, with original row order; the script already checks identical IDs before its additional positional calculations. Do not rerun or overwrite historical tables in this change. |
| RepFR `analyses/render_model_comparison.ipynb` and `repfr/comparison_config.py` | Pre-existing mismatch: the launcher supplies `data_names`, model-name/tag lists, `nesting_pairs`, etc.; the current template uses explicit `fit_paths` and `simulation_paths`. This requires a separate launcher/template migration. Old comparison imports shown in launcher outputs are generated snapshots, not live source calls. Preserve prepare-only operation and output layout during that migration. |
| ECMR3 `analyses/Dupertuys2026_model_comparison_pooled.ipynb` and the four `...ecmr_semantic*_pooled.ipynb` variants | External migration required for retained participant-wise branches: mean unpacking, median endpoints, nonmutating nested adjustments. Their default empty nesting list and pooled branch bypass those functions and compute AIC directly. Default pooled scores are not changed by these library fixes; dormant branches still need migration. |
| ECMR3 `archive/compare_eCMR_variants.ipynb` | Historical source uses the old interfaces. Migrate only if intentionally reviving it; preserve archived outputs. |
| RepFR `archive/proposal_repfr/figures.ipynb` | Historical AIC-weight caller; verify IDs and objective conventions before rerunning. Preserve archived outputs. |
| LPP-eCMR `vendor/jaxcmr/jaxcmr/summarize.py` | Independent vendored implementation, untouched. Updating the shared checkout does not fix that copy or its historical results. |

The older CRU comparison notebooks recorded in JAXCMR commit `09db8acd` passed
saved NLL records directly to the former subtracting AIC/BIC helpers. Those
historical tables need an explicit scientific-results audit before reuse.
The March research log also records a deliberate correction to participant-wise
parameter penalties; this change preserves that correction.

Potential historical numerical changes: AIC/BIC convention or penalty fixes,
asymmetric median intervals, protected exceedance probabilities, and mispaired
participants. Existing well-formed AIC weights and pooled AIC calculations should
agree. These observations identify affected calculations, not revised manuscript
conclusions. No external research file or historical output was rewritten.
