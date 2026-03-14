# Model comparison log

After correcting the AIC weights formula and re-running the full model comparison, the Emotion-only model (k=10) dominates all LPP-informed variants across AIC weights, median ΔAIC, and BMS. This contradicts the manuscript's GLMM analysis (Early LPP predicts recall for negative items, β=.14, p<.001; no effect for neutral items, β=.01, p=.488) and the cat_lpp_by_recall diagnostic, which the Emotion-only model cannot reproduce.

## Issues identified

### 1: AIC weights penalty was counted once instead of per-subject

`calculate_aic_weights` in `jaxcmr/summarize.py` computed `aic = 2k + 2·Σ NLL`, counting the parameter penalty once across all subjects. Because each subject is fit independently with their own k free parameters, the correct formula is `aic = 2kN + 2·Σ NLL`, where N is the number of subjects. The single-penalty version inflated the apparent advantage of more complex models by undercharging them. With 38 subjects and up to 14 free parameters, the discrepancy is substantial: the interaction model's penalty should be 2×14×38 = 1064, not 2×14 = 28.

Fixed by multiplying k by `n_subjects = len(model["fitness"])` in `calculate_aic_weights`. After correction, AIC weights shift decisively toward simpler models: the Emotion-only model receives AICw ≈ 1.0, and all LPP variants are penalised below 10⁻⁹.

### 2: LPP is channeled through a single, narrow pathway

In eCMR (`eeg_full_ecmr.py:212–244`), φ_emot modulates only `emotion_mcf` (emotional context → item associations). It does not affect `mcf` (temporal context → item, which is purely primacy-driven), `mfc` or `emotion_mfc` (item → context, which use a fixed learning rate), or item representation strength. The GLMM, by contrast, captures LPP's total effect on recall through whatever mechanisms exist. If LPP's real effect operates through multiple encoding pathways (e.g., also through temporal encoding strength or attentional modulation of context drift), the model's narrow channeling artificially limits LPP's impact and makes the additional parameters appear less worthwhile.

Addressed in entry 10. Added `phi_emot_modulates_temporal` flag to `eeg_full_ecmr.py`. Broadening to temporal MCF does not reliably improve fit by median ΔAIC. A subset of subjects benefits, driving aggregate AICw improvement (~70×), but the median subject is indifferent.

### 3: lpp_main applies to neutral items where the data shows no effect

The `lpp_main` term in the φ_emot construction (`eeg_full_ecmr.py:128`) applies to all items equally:

```python
lpp_main = lpp_main_scale * (lpp_centered - lpp_main_threshold)
```

The GLMM shows LPP has zero predictive power for neutral recall (β=.01, p=.488). Yet the Emo+LPP model spends 2 parameters (scale + threshold) fitting this non-existent effect. Worse, for neutral items in eCMR, a nonzero lpp_main strengthens their associations with the neutral pole of emotional context, potentially hurting fit by over-predicting neutral recall through the emotional pathway. The optimizer must compromise: large lpp_main_scale captures the emotional LPP signal but pollutes neutral items; small lpp_main_scale protects neutrals but underweights the emotional effect.

Addressed by the parsimonious model (entry 8), which eliminates the main-effect term entirely and restricts LPP to emotional items only.

### 4: Threshold parameters are partially redundant

`lpp_main_threshold` and `lpp_inter_threshold` shift the zero-point of already trial-mean-centered LPP values. After centering, the mean LPP within each trial is 0, so a threshold acts as an additional intercept. Combined with the `max(0, φ_emot)` clipping (`eeg_full_ecmr.py:216`), the threshold shifts where the clipping activates. This is a valid degree of freedom, but the parameter is partially degenerate with `emotion_scale` (both shift the effective baseline for emotional items). Each threshold costs 2 AIC units per subject (76 total across 38 subjects) for a nuance that may not substantially improve fit.

Addressed by the parsimonious model (entry 8), which fixes both thresholds to 0.

### 5: Over-parameterization for a small signal

The interaction model uses 4 LPP parameters (lpp_main_scale, lpp_main_threshold, lpp_inter_scale, lpp_inter_threshold). With corrected AIC, each parameter costs 2 per subject, totalling 76 per parameter across 38 subjects. Total LPP penalty: 4×76 = 304 AIC units.

The average per-subject NLL improvement from Emotion→Emo+LPP+Int is approximately 2.09 nats. Total NLL improvement: 2.09×38 = 79.4, yielding a net AIC change of 2×79.4 − 304 = −145.2 (i.e., 145 AIC units worse than Emotion-only). The LPP signal improves fit, but 4 parameters are far too many to express it efficiently.

Addressed by the parsimonious model (entry 8), which uses 1 LPP parameter. AIC penalty drops from 304 to 76. The model only needs to average >1.0 nats of NLL improvement per subject to clear the AIC bar, compared to >4.0 for the 4-parameter model.

### 6: Additive primacy + φ_emot coupling biases LPP influence toward late positions

All current eCMR fits use `modulate_emotion_by_primacy=False` (additive mode). The emotional MCF learning rate is `primacy + max(−primacy, φ_emot)`. Since φ_emot ≥ 0 after clipping, this simplifies to `primacy + φ_emot`. For early serial positions where primacy is large, φ_emot is a small additive increment (proportionally). For late positions where primacy ≈ 0, φ_emot dominates. This means LPP has proportionally more influence on late items and less on early items.

Not addressed in this pass. This is a model design choice inherited from TLD19. Whether it matches the data's actual LPP × serial position profile is an empirical question that could inform future model revisions.

### 7: Set likelihood is insensitive to within-category LPP discrimination

The set likelihood (50 Monte Carlo permutation samples per trial, `set_permutation_likelihood.py:76`) predicts which items are recalled as an unordered set. With 16-item lists and moderate recall rates, the category-level signal (emotional > neutral) dominates set prediction. The within-category LPP discrimination—which specific emotional items are recalled—provides marginal improvement in per-trial set prediction even when it is clearly present in the data (as shown by the GLMM and cat_lpp_by_recall diagnostic). This is less a specification error than a fundamental tension between the optimization target (set NLL) and the summary statistic diagnostic (LPP-by-recall conditional distributions).

Not addressed in this pass. This is an inherent property of the evaluation metric. It could be mitigated by increasing permutation count (reducing Monte Carlo noise that masks small improvements), switching to ordered recall likelihood (which would give LPP more leverage through recall-order effects), or adding the cat_lpp_by_recall pattern as an auxiliary fitting objective.

### 8: Parsimonious fix — eCMR Emo×LPP model

Created a new model variant (`eCMREmotionTimesLPP` in `render_model_fitting_ecmr.ipynb`) that directly maps the GLMM finding: LPP predicts recall specifically for emotional items, with no effect on neutral items. The φ_emot construction reduces to:

```python
phi_emot = emotion_scale * is_emotional + lpp_inter_scale * lpp_centered * is_emotional
```

Configuration: `lpp_main_scale=0` (fixed), `lpp_main_threshold=0` (fixed), `lpp_inter_threshold=0` (fixed), `lpp_inter_scale` (free, bounds [ε, 100]). This addresses issues 3 (neutral contamination eliminated), 4 (thresholds fixed to 0), and 5 (1 parameter instead of 4).

The model has 11 free parameters (k=11): the 10 base parameters shared with Emotion-only plus lpp_inter_scale. AIC penalty for the single LPP parameter: 76. The model clears the AIC bar if mean per-subject NLL improvement exceeds 1.0 nat relative to Emotion-only.

Added to both the main and full model comparisons in `model_comparison.ipynb`.

**Results (2026-03-10).** Fit completed (best of 3). AICw: Emotion 0.9999, Emo×LPP 5.6×10⁻⁶. Median ΔAIC (Emotion vs Emo×LPP): −1.16 [−2.58, 0.26] — CI includes 0, models statistically indistinguishable. BMS: Emotion expected frequency 0.405, xp 0.750; Emo×LPP expected frequency 0.319, xp 0.248; protected XP uniform (all 0.167). Emo×LPP is clearly the best LPP model in the full comparison (expected frequency 0.198 vs < 0.05 for all other LPP variants) but does not beat Emotion-only.

### 9: New comparison metrics added

Two metrics were added to `jaxcmr/summarize.py` and wired into the model comparison notebook alongside the corrected AIC weights:

**Median ΔAIC with bootstrap CIs.** `pairwise_median_aic_differences` uses the same per-subject AIC differences as `pairwise_aic_differences` but reports the median instead of the mean. CIs use the percentile bootstrap (n=10,000, seed=42) rather than t-based intervals, since the median's sampling distribution is not assumed normal. Robust to outlier subjects who may disproportionately favor one model.

**Bayesian Model Selection.** `bayesian_model_selection` implements random-effects BMS (Stephan et al., 2009) with protected exceedance probability (Rigoux et al., 2014). Uses per-subject log model evidence approximated as −½ AIC_s. Variational Bayes iteration on Dirichlet parameters; exceedance probabilities via Monte Carlo sampling (100,000 draws); Bayesian omnibus risk for protected XP. Returns expected frequency, exceedance probability, and protected XP per model.

Both functions share a `_subjectwise_aic` helper with the existing `pairwise_aic_differences` and `winner_comparison_matrix`, refactored during this pass to avoid code duplication.

### 10: Broad φ_emot models — 2×2 pathway breadth × LPP

Added `phi_emot_modulates_temporal` flag to `eeg_full_ecmr.py`. When True, the temporal MCF learning rate becomes `primacy + max(−primacy, φ_emot)` instead of just `primacy`, allowing φ_emot to boost encoding through both the emotional and temporal context–item pathways. Backward-compatible: default is False, preserving all existing model behavior.

This creates a 2×2 design crossing pathway breadth (narrow vs broad) with LPP (absent vs present):

| Model              | Pathway | LPP | k  |
|:-------------------|:--------|:----|:---|
| eCMR Emotion       | narrow  | no  | 10 |
| eCMR Emo×LPP       | narrow  | yes | 11 |
| eCMR Emotion Broad | broad   | no  | 10 |
| eCMR Emo×LPP Broad | broad   | yes | 11 |

Within each LPP condition the parameter count is identical, so AIC differences are pure NLL.

**Results (2026-03-10).**

*Broadening alone* (eCMR Emotion vs eCMR Emotion Broad): median ΔAIC 0.02 [−0.18, 0.23] — no effect. BMS expected frequency worse (0.066 vs 0.112 in full comparison). Broadening without LPP does not help and slightly hurts by BMS.

*Broadening with LPP* (eCMR Emo×LPP vs eCMR Emo×LPP Broad): median ΔAIC −0.06 [−0.40, 0.28] — no effect at the median. BMS expected frequency better (0.159 vs 0.087).

*LPP within broad architecture* (eCMR Emotion Broad vs eCMR Emo×LPP Broad): median ΔAIC −0.83 [−2.05, 0.38] — CI includes 0, same pattern as narrow.

*Aggregate metrics.* AICw: eCMR Emo×LPP Broad 3.9×10⁻⁴ (2nd in full comparison), eCMR Emotion Broad 2.3×10⁻⁴ (3rd). Both ~70× better than their narrow counterparts by AICw. BMS main comparison: eCMR Emo×LPP Broad expected frequency 0.187 (2nd behind Emotion's 0.310), eCMR Emotion Broad 0.118. Protected XP uniform for all models.

*Interpretation.* The median subject sees no benefit from broadening. A tail of subjects benefits substantially, driving aggregate metrics (AICw, BMS) while leaving the median unchanged. Broadening alone hurts by BMS; it only helps when LPP is present. This suggests φ_emot's temporal pathway contribution is useful specifically when it carries item-level LPP signal, not when it merely amplifies the categorical emotion effect.

### 11: CMR+LPP — can LPP replace category labels?

Tested whether LPP operating through the temporal pathway alone (single-context architecture, no `is_emotional` anywhere) can approach eCMR's performance. Model: `EEGLPPParsimonious` in `render_model_fitting.ipynb`, using `eeg_cmr.make_factory` with `lpp_main_scale` free and `lpp_main_threshold` fixed to 0 (k=10, matching eCMR).

**Motivation.** If the LPP is truly an ideal index of emotional processing, category labels should be redundant — LPP amplitude alone should carry enough information to reproduce the emotional recall advantage. Testing this requires removing category labels entirely (including the emotional context layer, which uses `is_emotional` for source features and pole assignment). The single-context architecture is the cleanest test.

**Results (2026-03-11).**

- CMR+LPP vs CMR: Mean ΔAIC 0.33 [−0.77, 1.44] — CI includes 0. LPP adds nothing.
- CMR+LPP vs eCMR (k=10): Mean ΔAIC 3.73 [1.23, 6.22] — CI excludes 0. Median ΔAIC 1.52 [0.18, 2.86] — also excludes 0. eCMR substantially better.
- AIC weights: LPP-eCMR 0.997, eCMR 0.003, CMR+LPP ≈ 0, CMR ≈ 0.
- BMS: CMR+LPP EF=0.089, XP=0.0003, protected XP=0.25 (chance). LPP-eCMR EF=0.407, XP=0.721.
- cat_spc: CMR+LPP produces faint negative-above-neutral separation (LPP correlates with category), far weaker than eCMR.
- cat_lpp_by_recall: CMR+LPP reproduces recalled/unrecalled LPP separation for negative items (shared with LPP-eCMR).

**Interpretation.** The LPP is far from replacing category labels. The emotional context architecture — which routes emotional items through a shared context pole and gives them an encoding boost via `is_emotional` — captures structure in recall that a continuous neural signal through the temporal pathway cannot recover. The LPP's value is as a supplement to the categorical architecture (LPP-eCMR), not a substitute.

## Manuscript narrative

The GLMM establishes that LPP predicts recall for emotional but not neutral items (β=.14, p<.001 and β=.01, p=.488 respectively). These are fixed-effect estimates — they test whether the population-average association is non-zero. The modelling question is whether this group-level statistical association translates into individual-level prediction improvement when embedded as a mechanistic encoding-strength parameter in eCMR.

The answer is: at the aggregate level, yes (AICw=.997, BMS XP=.721); at the individual level, the advantage is not uniform (median ΔAIC=−1.21, protected XP at chance). A subset of participants benefits substantially from the LPP term, driving the aggregate advantage, while for most participants the improvement is too small to offset the AIC penalty. This divergence is both informative — it characterizes heterogeneity in the LPP-recall association — and a practical constraint of per-subject fitting with limited data (9 lists per participant).

The qualitative diagnostic (@fig-lpp-recall) provides complementary evidence: LPP-eCMR reproduces the recalled/unrecalled LPP separation for emotional items, while eCMR cannot. This is genuine generalization to an untrained summary statistic and demonstrates what the LPP term adds to the model's explanatory scope independently of whether it uniformly improves numerical fit.

The paper is scoped as a methodological contribution — demonstrating a workflow for incorporating single-trial neural measures into a generative memory model, evaluating prediction at both aggregate and individual levels, and assessing qualitative adequacy against untrained diagnostics — not as a claim about which specific encoding mechanism LPP reflects. The TalmiEEG dataset is the worked example.

## Core results needed

**1. Emo×LPP clears model comparison metrics relative to Emotion-only.** AIC, median ΔAIC, and/or BMS should favor the LPP-informed model. This is necessary but not the most interesting result.

**2. Show where improved fits come from.** The stronger argument is demonstrating which summary statistics the Emotion-only model fails on and Emo×LPP captures. The cat_lpp_by_recall diagnostic is the key: the Emotion-only model cannot reproduce the recalled/unrecalled LPP separation for emotional items. The Emo×LPP model should. This is what distinguishes the generative model contribution from the GLMM — the GLMM says "LPP predicts recall," the generative model says "here's the mechanism, and it produces this behavioral signature as a consequence." Critically, the model is fit to set likelihood, not to cat_lpp_by_recall, so reproducing that diagnostic is genuine generalization to an untrained summary statistic.

## Strengthening analyses

**Cross-validation.** AIC is an asymptotic approximation of leave-one-out CV. With only 9 trials per subject, the approximation may be poor. Direct CV would more convincingly demonstrate that LPP improves out-of-sample prediction. CV infrastructure already exists (see `*_cv.json` files in the fits directory). Expensive: each model requires ~3 hours × 38 folds for leave-one-subject-out.

**Neutral LPP test.** A matched k=11 model with LPP on all items and no threshold (`eCMRGeneralLPP`) would cleanly test whether extending LPP to neutral items helps. This directly translates the GLMM null result (β=.01, p=.488 for neutrals) into the generative framework. If the general model doesn't beat the emotion-specific model despite equal parameter counts, the emotion-specificity of the LPP effect is not just a statistical pattern but a property of the encoding mechanism. Requires one new fit (~3 hours).

## Contingency

The Issue 2 contingency (broadening φ_emot to temporal MCF) has been executed (entry 10). Broadening does not reliably help at the median-subject level. The remaining candidate explanations for why LPP doesn't clear the AIC bar are issue 6 (additive primacy × φ_emot coupling biases LPP to late positions) and issue 7 (set likelihood insensitivity to within-category LPP discrimination).

## Circularity defense

Emo×LPP is designed to match the GLMM finding (LPP predicts emotional but not neutral recall), so "you built the model to match, so of course it matches" is a fair objection. The defense is that the model is fit to set likelihood — a completely different objective — and reproducing the cat_lpp_by_recall pattern is genuine generalization to an untrained summary statistic. The manuscript should be explicit about this distinction.
