# Transcript: Parameter Correlations Discussion

## Conversation

User:
check fit jsons at paths specified by projects/repfr/notebooks/rendered/fitting_LohnasKahana2014_McfReinfPositionalCMRNoStop_rerun_best_of_1.ipynb. Any apparently correlations between parameter values?

Assistant:
Pulled the fit path from the notebook variables and checked projects/repfr/results/fits/LohnasKahana2014_McfReinfPositionalCMRNoStop_rerun_best_of_1.json. This file has 35 subject fits.

Strongest parameter correlations across subjects (Pearson):
- shared_support vs item_support: r = 0.776
- primacy_decay vs choice_sensitivity: r = 0.713
- primacy_scale vs primacy_decay: r = -0.638
- primacy_scale vs mcf_first_pres_reinforcement: r = 0.638

Additional monotonic relationships (Spearman rho >= 0.6):
- shared_support vs item_support: rho = 0.800
- primacy_scale vs mcf_first_pres_reinforcement: rho = 0.732
- encoding_drift_rate vs choice_sensitivity: rho = -0.688
- primacy_decay vs mcf_first_pres_reinforcement: rho = -0.657
- primacy_scale vs primacy_decay: rho = -0.652
- shared_support vs choice_sensitivity: rho = 0.603

Fixed (no variance) in this fit, so no correlations:
- allow_repeated_recalls, learn_after_context_update, mfc_sensitivity, mfc_first_pres_reinforcement

If you want, I can run the same correlation scan for the other positional variants or plot scatter matrices for the top pairs.

User:
check fullreinpositionalcmrnostop

Assistant:
Checked projects/repfr/results/fits/LohnasKahana2014_FullReinfPositionalCMRNoStop_rerun_best_of_1.json (35 subjects). Key correlations:

Fixed (no variance):
- allow_repeated_recalls, learn_after_context_update, mfc_sensitivity

Strong Pearson (|r| >= 0.6):
- shared_support vs item_support: r = 0.709
- primacy_scale vs primacy_decay: r = -0.705
- primacy_scale vs mcf_first_pres_reinforcement: r = 0.647

Strong Spearman (|rho| >= 0.6):
- primacy_scale vs primacy_decay: rho = -0.802
- shared_support vs item_support: rho = 0.791
- learning_rate vs mfc_first_pres_reinforcement: rho = 0.712
- shared_support vs choice_sensitivity: rho = 0.703
- primacy_scale vs mcf_first_pres_reinforcement: rho = 0.665
- encoding_drift_rate vs shared_support: rho = -0.607
- primacy_decay vs mcf_first_pres_reinforcement: rho = -0.605

If you want the same scan for any other model, tell me the fit name and I will run it.

User:
any relationship between mcf_first_pres_reinforcement and mfc_first_pres_reinforcement

Assistant:
For projects/repfr/results/fits/LohnasKahana2014_FullReinfPositionalCMRNoStop_rerun_best_of_1.json:
- Pearson r = -0.173
- Spearman rho = 0.035

So there is essentially no monotonic relationship between mcf_first_pres_reinforcement and mfc_first_pres_reinforcement in this fit.

User:
exploring jaxcmr/models/cmr, what might explain the correlation between shared_support and item_support fits? if correlation recurs across datasets, does this imply that we could/should collapse into a single parameter in some way?

Assistant:
Short answer: the correlation is expected given how shared_support and item_support enter MCF, and yes, repeated correlations across datasets are a strong hint of practical non-identifiability. That suggests you should consider reparameterizing or fixing one, but only after a quick identifiability check.

Why the correlation is plausible:
- In jaxcmr/components/linear_memory.py, init_mcf fills the MCF matrix with shared_support and then overwrites the diagonal with item_support. That means retrieval depends mostly on the contrast between diagonal and off-diagonal, not their absolute scale.
- In jaxcmr/models/cmr.py, item probabilities normalize by the total activation sum. So any global scaling of MCF can be partially canceled by normalization and by choice_sensitivity, making multiple (shared_support, item_support) pairs fit similarly.

What this implies if the correlation recurs:
- It is a sign of practical non-identifiability: the data constrain only a low-dimensional combination (e.g., ratio or difference) rather than the two parameters independently.
- That does not prove they should be collapsed, but it is a good reason to test a simpler parameterization.

Options to collapse or stabilize:
1. Reparameterize into base + contrast. Example: base = shared_support, delta = item_support - shared_support (or enforce delta >= 0).
2. Use a ratio parameter. Example: item_support = shared_support * (1 + ratio), fit shared_support and ratio.
3. Fix one parameter. Example: set shared_support to a constant (or 1.0) and let item_support vary, or vice versa.

How to decide:
- Compare fit quality vs. complexity (AIC/BIC or cross-validated likelihood) between the 2-parameter and collapsed models.
- Run a recoverability check: simulate with known parameters, refit, and see if both are recovered or if only their combination is.

User:
base + contrast would be same # of parameters. same for shared_support and ratio. we are observing that the base and contrast params are correlated during fitting

Assistant:
You are right: reparameterizing as base + contrast or base + ratio keeps the same degrees of freedom, so it will not fix non-identifiability by itself. If those new parameters are still correlated in fits, that is a clear sign the likelihood has a ridge: the data mainly constrain some combination (e.g., diagonal/off-diagonal contrast), not both dimensions independently.

Why this happens in CMR:
- In jaxcmr/components/linear_memory.py, MCF is initialized with a uniform shared_support, then the diagonal is overwritten with item_support. That means retrieval depends primarily on relative diagonal vs off-diagonal contrast, not absolute scale.
- In jaxcmr/models/cmr.py, item probabilities are normalized, so any global scaling of MCF is largely absorbed (and further confounded with choice_sensitivity). This makes the absolute level of shared_support weakly identified.

What it implies:
- If base+contrast (or base+ratio) are still correlated across datasets, the model effectively only sees one dimension (contrast or ratio). That is practical non-identifiability, not just bad parameterization.

What to do if you want to reduce the ridge:
1. Collapse to one parameter: fix a base and fit a contrast or ratio only.
2. Regularize instead of collapsing: put a tight prior/penalty on the base.
3. Diagnose the ridge explicitly: profile likelihood by fixing shared_support on a grid and refitting item_support.

Bottom line: repeated correlation across datasets means the model is not distinguishing the two supports. Reparameterization can make optimization nicer, but if the correlation persists, the only way to remove it is to fix or constrain one dimension.

User:
id be okay fixing shared_support but what would be a principled way to do that? i know using 0 or eps creates problems, in part because it can imply zero or nil retrieval support for low-support items

Assistant:
You are right that fixing to 0/eps is a bad idea. The most principled fix is to treat shared_support as a scale/gauge parameter and pin it to a reasonable positive reference level, because the model mostly cares about relative support (diagonal vs off-diagonal) and probabilities are normalized in jaxcmr/models/cmr.py. That makes absolute scale weakly identified.

Concrete, defensible ways to fix shared_support:
- Unit-scale gauge: set shared_support = 1.0 and fit item_support (and/or a contrast delta = item_support - 1.0). This preserves nonzero off-diagonal support and fixes the scale.
- Empirical reference: fix shared_support to a robust central value from prior fits (e.g., median across subjects/datasets). This keeps the scale close to what the model already tends to use.
- Normalized MCF: set shared_support = 1.0 and interpret item_support as the relative boost above baseline; optionally constrain item_support >= shared_support so the diagonal is never weaker than off-diagonal.

If the correlation persists across datasets, the only way to remove the ridge is to fix one dimension (or impose a tight prior). Reparameterization alone will not do it.

If you want, tell me which reference value you prefer (e.g., 1.0 vs median from fits), and I can help you update the fitting config or compute the median across datasets.
