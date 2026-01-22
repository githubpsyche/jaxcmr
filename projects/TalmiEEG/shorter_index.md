---
bibliography: references.bib
---

Building on the behavioral and ERP results reported above, we test mechanistic accounts of how study-phase neural activity can modulate encoding within retrieved-context models.
We embed Early LPP as an item-level modulator of learning strength in eCMR and compare three variants that differ only in how LPP enters the encoding rule.

## eCMR Specification

We use a connectionist CMR architecture with a temporal context state and bidirectional item-context associations.
We implement emotion and Early LPP as modulators of learning strength, and we do not include a separate emotional context layer.
This keeps the retrieved-context dynamics fixed across variants while localizing emotion/LPP effects to a single encoding pathway.

We denote temporal context as $c^T$ and item features as $f_i$.
$M^{FC}$ maps item features to temporal context, and $M^{CF}$ maps temporal context to item features.

| Symbol                     | Meaning                           |
|----------------------------|-----------------------------------|
| $c^T$                      | temporal context                  |
| $f_i$                      | item feature vector (one-hot)     |
| $M^{FC}$                   | feature-to-context associations   |
| $M^{CF}$                   | context-to-feature associations   |
| $\beta_{enc}$              | encoding drift rate               |
| $\beta_{start}$            | start-list drift rate             |
| $\beta_{rec}$              | recall drift rate                 |
| $\alpha$                   | shared support in $M^{CF}$        |
| $\delta$                   | self-support in $M^{CF}$          |
| $\gamma$                   | learning rate for $\Delta M^{FC}$ |
| $\phi_s$                   | primacy scale                     |
| $\phi_d$                   | primacy decay                     |
| $\tau_c$                   | choice sensitivity                |
| $g_i$                      | learning strength                 |
| $L_i$                      | list-centered Early LPP           |
| $\kappa_L,\lambda_L$       | LPP main-effect slope, offset     |
| $\kappa_{EL},\lambda_{EL}$ | LPP-by-emotion slope, offset      |

: Parameters and structures specifying eCMR. {#tbl-ecmr-parameters}

Pre-experimental memory is initialized as

$$
M^{FC}_{pre(ij)} =
\begin{cases}
1 - \gamma & \text{if } i=j \\
0 & \text{if } i \neq j
\end{cases}
$$

$$
M^{CF}_{pre(ij)} =
\begin{cases}
\delta & \text{if } i=j \\
\alpha & \text{if } i \neq j
\end{cases}
$$

During encoding, temporal contextual input is retrieved as $c^{IN}_i = M^{FC} f_i$ and normalized to unit length.
Temporal context updates according to

$$
c^T_i = \rho_i c^T_{i-1} + \beta_{enc} c^{IN}_i
$$

$$
\rho_i = \sqrt{1 + \beta_{enc}^2\left[\left(c^T_{i-1} \cdot c^{IN}_i\right)^2 - 1\right]} - \beta_{enc}\left(c^T_{i-1} \cdot c^{IN}_i\right)
$$

Feature-to-context learning proceeds via Hebbian updates as:

$$
\Delta M^{FC}_{ij} = \gamma f_i c^T_j
$$

The primacy-graded attention term is defined as:

$$
\phi_i = \phi_s e^{-\phi_d(i-1)} + 1
$$

The complete learning-strength term is defined differently in each model variant (see below).
In the base case, it is:

$$
g_i = \phi_{emot,i} + \phi_i
$$

Altogether, context-to-feature associations are updated as:

$$
\Delta M^{CF}_{ij} = g_i c^T_j f_i
$$

At recall onset, temporal context is shifted toward the pre-list state as

$$
c^T_{start} = \rho_{N+1} c^T_N + \beta_{start} c^T_0
$$

At each recall step, temporal context cues item activations according to:

$$
A = M^{CF} c^T
$$

Given this cue, at each recall step, the probability of recalling item $i$ is

$$
P(i) = \frac{A_i^{\tau_c}}{\sum_k A_k^{\tau_c}}
$$

After recall, the retrieved item's temporal context is reinstated via $M^{FC}$ and integrated with drift rate $\beta_{rec}$, and the process iterates.
We omit an explicit termination mechanism; instead, we match each trial's observed output length during simulation (see below).

## Model Variants

We compare three variants that differ only in how emotion and Early LPP enter the learning-strength term $g_i$ in $\Delta M^{CF}_{ij}$.
Let $e_i$ be an indicator that equals 1 for emotional items and 0 for neutral items.

We define the emotion term in the learning strength as $\phi_{emot,i} = \phi_{emot} e_i$.
In the emotion-only model, LPP effects are removed by setting $\kappa_L = 0$ and $\kappa_{EL} = 0$, yielding

$$
g_i = \phi_{emot,i} + \phi_i
$$

In the main-effect model, the interaction term is removed by setting $\kappa_{EL} = 0$, yielding

$$
g_i = \phi_{emot,i} + \phi_i + \kappa_L (L_i - \lambda_L)
$$

In the separate-slope model, both LPP terms are allowed, yielding

$$
g_i = \phi_{emot,i} + \phi_i + \kappa_L (L_i - \lambda_L) + \kappa_{EL} (L_i - \lambda_{EL}) e_i
$$

In implementation, negative values of $g_i$ are clamped to 0 in all variants.

## Evaluation Approach

We evaluated each model by how well it predicts the unordered set of recalled items on each list (not recall order), because our data is missing recall order information and because our primary behavioral outcomes and benchmarks target item selection rather than order-dependent dynamics.
Let $S_t$ denote the set of items recalled on trial $t$, and let $r$ range over recall sequences that contain exactly the items in $S_t$ (in any order).
We define the *set likelihood* as

$$
P(S_t \mid \theta) = \sum_{r:\, r\ \text{yields}\ S_t} P(r \mid \theta),
$$

where $\theta$ are the model parameters.

Evaluating this sum exactly is infeasible because the number of sequences that yield a given set grows rapidly with $|S_t|$.
We therefore approximate $P(S_t\mid\theta)$ by Monte Carlo sampling: for each trial we draw $K$ random permutations $r_1,\dots,r_K$ of $S_t$, compute $P(r_k\mid\theta)$, and estimate the sum as $|S_t|!\,\frac{1}{K}\sum_{k=1}^K P(r_k\mid\theta)$.
Because the factorial factor does not depend on $\theta$, we maximize the (unscaled) mean permutation likelihood $\frac{1}{K}\sum_k P(r_k\mid\theta)$, which is proportional to the set likelihood and yields identical fits and $\Delta$AIC comparisons.

We maximize the objective with differential evolution [@storn1997differential].
Because the optimizer is stochastic, we run three independent fits per participant and model variant and retain the best-performing fit.
We compare variants using AIC computed from the approximate set log-likelihood.
We then assess qualitative adequacy by simulating each fitted model on the study lists while matching each trial's observed output length and plotting the benchmark summaries.
This set-likelihood approach evaluates models on their ability to reproduce the full recalled sets, providing a stringent test of mechanistic hypotheses.

## Comparison Results

The Emotion + LPP (interaction) model provides the best fit by all comparison metrics.
@tbl-talmi-delta-aic reports pairwise delta AIC values used in the comparisons above.
Parameter estimates were similar across variants, so inference is driven primarily by relative fit and benchmark behavior.

|   | Emotion + LPP (main effects) | Emotion + LPP (interaction) | Emotion-only |
|------------------|------------------|------------------|------------------|
| Emotion + LPP (main effects) | NA | 1.12 \[0.13, 2.11\] | -2.86 \[-4.14, -1.57\] |
| Emotion + LPP (interaction) | -1.12 \[-2.11, -0.13\] | NA | -3.98 \[-5.71, -2.25\] |
| Emotion-only | 2.86 \[1.57, 4.14\] | 3.98 \[2.25, 5.71\] | NA |

: Pairwise delta AIC (row minus column; mean \[95% t-based CI\]). {#tbl-talmi-delta-aic tbl-colwidths="\[34,22,22,22\]"}

@fig-eeg_benchmark_fits summarizes empirical benchmarks and their  simulations for all fitted variants.
In the Emotion-only simulations, the model captures much of the emotional enhancement of memory in the Category-SPC but shows little separation between recalled and unrecalled negative items in Early LPP. This pattern indicates that emotion labels alone can reproduce the overall category advantage without explaining LPP-linked recall differences.
In the Emotion + LPP (main effects) simulations, the model fits both summary statistics but assigns a larger recalled–unrecalled difference for neutral items and a smaller difference for negative items.
This pattern suggests that a single LPP slope captures the overall association while misallocating the emotion-specific separation.
In the Emotion + LPP (interaction) simulations, the Category-SPC preserves emotional enhancement of memory and the Early LPP panel shows a clearer emotion-dependent separation between recalled and unrecalled items.
Together these benchmarks indicate that combining emotion and LPP with an interaction provides the most faithful account of both the category-level advantage and the emotion-specific LPP–recall pattern.

::: {#fig-eeg_benchmark_fits layout-ncol="2"}
![](results/figures/fitting/TalmiEEG_EEGEmotionOnly_50_set_likelihood_fixed_term_best_of_3_cat_spc.png)

![](results/figures/fitting/TalmiEEG_EEGEmotionOnly_50_set_likelihood_fixed_term_best_of_3_cat_lpp_by_recall.png)

![](results/figures/fitting/TalmiEEG_EEGMainEffects_50_set_likelihood_fixed_term_best_of_3_cat_spc.png)

![](results/figures/fitting/TalmiEEG_EEGMainEffects_50_set_likelihood_fixed_term_best_of_3_cat_lpp_by_recall.png)

![](results/figures/fitting/TalmiEEG_EEGMainEffectsPlusInteraction_50_set_likelihood_fixed_term_best_of_3_cat_spc.png)

![](results/figures/fitting/TalmiEEG_EEGMainEffectsPlusInteraction_50_set_likelihood_fixed_term_best_of_3_cat_lpp_by_recall.png)

Benchmark simulations for all model variants compared to empirical data.
The top row corresponds to the empirical benchmarks.
Proceeding rows correspond to Emotion-only, Emotion + LPP (main effects), and Emotion + LPP (interaction), respectively.
Columns show Category-SPC (left) and the LPP-by-recall diagnostic (right).
:::
