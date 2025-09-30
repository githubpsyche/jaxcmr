## Model comparison

We compared four variants under a common fitting and scoring pipeline (Methods): **Standard CMR** (`WeirdCMR`), **ICMR‑NoSPR** (`WeirdCMRDistinctContexts`), **ICMR‑OS** (`WeirdPositionalCMR`), and **ICMR‑Reinf** (`WeirdStudyReinfPositionalCMR`). ICMR‑Reinf introduces one additional reinforcement parameter; all other pairwise comparisons are parameter‑matched. Fit quality is reported as mean negative log‑likelihood (−log L) across participants with bootstrapped **95% CIs over subjects**. Where informative, we also report AIC model weights (AICw), which penalize parameter count.

Across datasets, three patterns emerge. First, **ICMR‑OS** substantially improves fit relative to Standard CMR in **all** datasets, with especially large gains in the serial‑recall tasks (@logan2021serial; @kahana2000interresponse). Second, adding study‑phase reinforcement without contextual knitting (**ICMR‑Reinf**) yields a further, reliable improvement in **free recall** (@lohnas2014retrieved; @broitman2024neural) despite its extra parameter, but brings no advantage over OS in **serial recall**. Third, **ICMR‑NoSPR** improves over Standard CMR in free recall but is not sufficient on its own—consistent with the results sections above.

### Fit quality (−log L; lower is better)

| Dataset                                    |     Standard CMR |       ICMR‑NoSPR |              ICMR‑OS |           ICMR‑Reinf |
| ------------------------------------------ | ---------------: | ---------------: | -------------------: | -------------------: |
| **Lohnas & Kahana (2014)** (free recall)   | 1668.19 ± 146.93 | 1661.98 ± 146.77 |     1661.01 ± 147.02 | **1658.99 ± 146.97** |
| **Broitman & Kahana (2024)** (free recall) | 1268.71 ± 279.66 | 1263.46 ± 277.99 |     1260.15 ± 278.41 | **1259.17 ± 278.37** |
| **Logan et al. (2021)** (serial recall)    |   770.66 ± 59.75 |                — |   **729.19 ± 60.18** |       730.08 ± 60.47 |
| **Kahana & Jacobs (2000)** (serial recall) | 3852.12 ± 530.13 |                — | **3759.75 ± 516.93** |     3761.08 ± 516.74 |

: **Summary of fit quality across datasets.** Values are participant means ± bootstrapped 95% CIs (subjects). ICMR‑NoSPR was not evaluated for the serial‑recall datasets. {#tbl-fit}

As compact deltas relative to Standard CMR: Lohnas −log L improves by **−6.21** (NoSPR), **−7.18** (OS), **−9.20** (Reinf); Broitman by **−5.25**, **−8.56**, **−9.54**; Logan (serial) by **−41.47** (OS) and **−40.58** (Reinf); Kahana & Jacobs (serial) by **−92.37** (OS) and **−91.04** (Reinf).

### Penalized comparisons (AIC model weights)

To separate parameter‑count effects from fit, we report AICw in two steps. First, a **pairwise** comparison between **ICMR‑OS** and **Standard CMR** (parameter‑matched) across all datasets; second, a **three‑way** comparison in free recall that **includes ICMR‑Reinf** (adds one parameter).

| Dataset                      | AICw (Standard) | AICw (ICMR‑OS) |
| ---------------------------- | --------------: | -------------: |
| **Lohnas & Kahana (2014)**   |   3.25 × 10⁻¹¹⁰ |       **1.00** |
| **Broitman & Kahana (2024)** |   8.24 × 10⁻¹²⁴ |       **1.00** |
| **Logan et al. (2021)**      |               0 |       **1.00** |
| **Kahana & Jacobs (2000)**   |               0 |       **1.00** |

: **Pairwise AIC weights (ICMR‑OS vs Standard CMR).** OS is preferred in every dataset. {#tbl-aic-os-vs-cmr}

| Free‑recall dataset          | AICw (Standard) | AICw (ICMR‑OS) | AICw (ICMR‑Reinf) |
| ---------------------------- | --------------: | -------------: | ----------------: |
| **Lohnas & Kahana (2014)**   |   1.91 × 10⁻¹⁴⁰ |   5.87 × 10⁻³¹ |          **1.00** |
| **Broitman & Kahana (2024)** |   1.95 × 10⁻¹³⁷ |   2.37 × 10⁻¹⁴ |          **1.00** |

: **Three‑way AIC weights in free recall.** After penalization, ICMR‑Reinf is preferred; OS is second; Standard CMR is not competitive. {#tbl-aic-fr}

### Participant‑level win rates

As a complementary, model‑agnostic summary, we report the fraction of participants whose individual fit favored one model over another (lower −log L). These “win rates” mirror the AIC results: **ICMR‑OS** beats Standard CMR for most or all participants in every dataset (Lohnas **80%**, Broitman **79%**, Logan **100%**, Kahana & Jacobs **100%**). In **free recall**, **ICMR‑Reinf** beats ICMR‑OS for a majority of participants (Lohnas **63%**, Broitman **64%**), whereas in **serial recall** the two are roughly tied (Logan **46%** for Reinf; Kahana & Jacobs **53%** for OS).

### Interpretation

Taken together with the diagnostic analyses, the model‑comparison results support a simple division of labor. **Occurrence‑specific reinstatement at test** (ICMR‑OS) is necessary to eliminate the cross‑occurrence interference predicted by Standard CMR and dominates in serial recall. **Study‑phase reinforcement without contextual knitting** (ICMR‑Reinf) adds a small but reliable advantage in free recall by capturing the boosted access to first‑occurrence context, while preserving the non‑interference profile. **ICMR‑NoSPR** establishes that CMR’s original study‑phase reinstatement is not required for overall spacing or benchmark fits, but by itself it does not resolve the interference signature. Across datasets, **Standard CMR** is consistently least favored under penalized fit.
