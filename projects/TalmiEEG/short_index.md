# Formal Modeling

Building on the behavioral and ERP results reported above (negative > neutral recall and an emotion-dependent Early LPP subsequent-memory effect), we test mechanistic accounts of how study-phase neural activity can modulate encoding within retrieved-context models.
We embed Early LPP as an item-level modulator of learning strength in eCMR and compare three variants that differ only in how LPP enters the encoding rule.

## Modeling Methods

### Inputs and data constraints

- Lists are modeled as 20-item sequences because the first two study positions were excluded during preprocessing.
- The behavioral data indicate which items were recalled but not recall order, so model evaluation targets recalled *sets* per list.
- We use the Early LPP measure defined in the ERP analyses above. For modeling, $L_i$ denotes list-centered Early LPP for item $i$ (item value minus the list mean).
- Study events lacking an LPP measurement are imputed using the participant’s within-list mean for items of the same valence (negative vs. neutral).

### eCMR specification

We use a connectionist CMR architecture with a temporal context and implement valence and Early LPP as modulators of learning strength rather than a separate emotional context state.
We denote temporal context as $c^T$, with item features $f_i$.
The matrix $M^{FC}$ binds item features to temporal context, and $M^{CF}$ binds temporal context to item features.

| Symbol | Name | Description |
|------------------------|------------------------|------------------------|
| $c^T$ | temporal context | Recency-weighted summary of prior items. |
| $f_i$ | item features | One-hot representation of item $i$. |
| $M^{FC}$ | item-to-temporal-context memory | Feature-to-context associations. |
| $M^{CF}$ | context-to-item memory | Temporal context cueing item features. |
| $\beta_{enc}$ | encoding drift rate | Integration rate of temporal context during encoding. |
| $\beta_{start}$ | start drift rate | Start-list context integration at recall onset. |
| $\beta_{rec}$ | recall drift rate | Integration rate of temporal context during recall. |
| $\alpha$ | shared support | Uniform pre-experimental support in $M^{CF}$. |
| $\delta$ | item support | Self-support in $M^{CF}$. |
| $\gamma$ | learning rate | Feature-to-context learning rate in $M^{FC}$. |
| $\phi_s$ | primacy scale | Initial boost to $M^{CF}$ learning. |
| $\phi_d$ | primacy decay | Decay rate of the primacy boost. |
| $\tau_c$ | choice sensitivity | Exponent for Luce-style competition. |
| $\phi_{emot}$ | negative-item learning boost | Value is $\phi_{emot}$ for negative items and 0 for neutral items. |
| $L_i$ | centered Early LPP | List-centered Early LPP for item $i$. |
| $\kappa_L$ | LPP main scale | Slope for LPP main effect. |
| $\lambda_L$ | LPP main threshold | Centering offset for the LPP main effect. |
| $\kappa_{EL}$ | LPP interaction scale | Slope for LPP-by-valence interaction. |
| $\lambda_{EL}$ | LPP interaction threshold | Centering offset for the interaction term. |

: Parameters and structures specifying eCMR. {#tbl-ecmr-parameters}

CMR initializes $M^{FC}$ and $M^{CF}$ with pre-experimental associations.

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

Here, $\gamma$ scales the ratio of new feature-to-context associations formed during the experiment to pre-experimental item–context links.
The parameters $\delta$ and $\alpha$ set pre-experimental self-support and shared support in context-to-feature memory, respectively.
During encoding, temporal contextual input is retrieved as $c^{IN}_i = M^{FC} f_i$ and normalized to unit length.
Temporal context updates according to:

$$
c^T_i = \rho_i c^T_{i-1} + \beta_{enc} c^{IN}_i
$$

$$
\rho_i = \sqrt{1 + \beta_{enc}^2\left[\left(c^T_{i-1} \cdot c^{IN}_i\right)^2 - 1\right]} - \beta_{enc}\left(c^T_{i-1} \cdot c^{IN}_i\right)
$$

Here, $\beta_{enc}$ sets the drift rate of temporal context during encoding and $\rho_i$ normalizes the context vector.
Feature-to-context learning uses a Hebbian update:

$$
\Delta M^{FC}_{ij} = \gamma f_i c^T_j
$$

This update uses $\gamma$ as the feature-to-context learning rate.
Primacy is implemented by scaling the temporal context-to-feature learning rate:

$$
\phi_i = \phi_s e^{-\phi_d(i-1)} + 1
$$

Here, $\phi_s$ sets the initial primacy boost and $\phi_d$ controls its decay across serial positions.
Learning strength is defined as:

$$
g_i = \phi_{emot,i} + \phi_i
$$

Here, $\phi_{emot,i}$ equals $\phi_{emot}$ for negative items and 0 for neutral items.
The primacy term $\phi_i$ enters additively to capture an independent primacy contribution to learning strength.
Context-to-feature learning uses the learning strength $g_i$:

$$
\Delta M^{CF}_{ij} = g_i c^T_j f_i
$$

This update uses $g_i$ to bind items in $M^{CF}$.
At the start of recall, temporal context is shifted toward the pre-list state:

$$
c^T_{start} = \rho_{N+1} c^T_N + \beta_{start} c^T_0
$$

Here, $\beta_{start}$ controls how much start-list context is reinstated at retrieval onset.
At each recall step, temporal context cues item activations:

$$
A = M^{CF} c^T
$$

At each recall step, the probability of recalling item $i$ is:

$$
P(i) = \frac{A_i^{\tau_c}}{\sum_k A_k^{\tau_c}}
$$

Here, $\tau_c$ controls the sharpness of the Luce-style competition over item activations.
We omit an explicit termination mechanism here, consistent with our focus on recalled items rather than stopping decisions.
When an item is recalled, its temporal context is reinstated via $M^{FC}$ and integrated with $\beta_{rec}$, and the process iterates.
Here, $\beta_{rec}$ sets the drift rate of temporal context during recall.

### Model variants

Motivated by the mixed-effects results reported above (main effects of valence and Early LPP plus an interaction), we compare three closely related eCMR variants that differ only in how $L_i$ enters learning.
These comparisons test whether Early LPP contributes incremental information about recalled sets beyond the valence label, and whether the observed interaction requires an explicit valence-dependent LPP term.

We implement the three variants by constraining the learning strength $g_i$ used in $\Delta M^{CF}_{ij}$.
Let $e_i$ be an indicator that equals 1 for negative items and 0 for neutral items.
In the emotion-only model, LPP effects are removed by setting $\kappa_L = 0$ and $\kappa_{EL} = 0$, yielding:

$$
g_i = \phi_{emot,i} + \phi_i
$$

In the main-effect model, the interaction term is removed by setting $\kappa_{EL} = 0$, yielding:

$$
g_i = \phi_{emot,i} + \phi_i + \kappa_L (L_i - \lambda_L)
$$

Here, $\kappa_L$ and $\lambda_L$ define a single LPP slope and centering applied uniformly across items.
In the separate-slope model, both LPP terms are allowed, yielding:

$$
g_i = \phi_{emot,i} + \phi_i + \kappa_L (L_i - \lambda_L) + \kappa_{EL} (L_i - \lambda_{EL}) e_i
$$

Here, $\kappa_{EL}$ and $\lambda_{EL}$ allow an additional LPP slope for negative items beyond the shared LPP term.
In implementation, negative values of $g_i$ are clamped to 0 in all variants.

### Fitting objective: set likelihood

Our modeling goal is to explain which specific items are recalled on each trial, given their position, valence, and LPP, rather than only the average recall rates summarized in serial-position curves.
Accordingly, we fit models by maximizing the likelihood of the observed recall data under each parameter setting.
Because recall order is unavailable, we score models on how well they predict the *set* of recalled items on each list.
We treat each list as a 20-item sequence (with the first two study positions excluded) and do not score termination events; stopping policy is therefore out of scope for fitting.

In a standard likelihood analysis with full recall order, we would compute, for each trial, the probability that the model generates the exact observed recall sequence, and then maximize the product of those probabilities across trials.
Here we replace the sequence with the unordered set of recalled items.
Let $S_t$ denote the set of items recalled on trial $t$, and let $r$ range over recall sequences that produce the same set $S_t$ (possibly with different orders).
We define the *set likelihood* as

$$
P(S_t \mid \theta) = \sum_{r:\, r\ \text{yields}\ S_t} P(r \mid \theta),
$$

where $\theta$ are the model parameters.
For each list and parameter setting, this quantity assigns higher probability to sets that are frequently produced by the model’s cue-dependent, competitive retrieval dynamics, regardless of the order in which items are recalled.

Evaluating this sum exactly is infeasible for lists with many recalled items, because the number of sequences that yield a given set grows rapidly with set size.
We therefore approximate $P(S_t \mid \theta)$ with a Monte Carlo estimate based on permutations of the observed set.
For each trial, we sample a fixed number of random permutations of $S_t$, treat each permutation as a possible recall sequence, and compute its probability under the model.
Averaging these probabilities over permutations yields an unbiased estimate of the set likelihood for that trial.
We then sum log likelihoods across trials for each participant and choose parameter values that maximize this approximate set-based log likelihood.

We maximize the objective with differential evolution [@storn1997differential].
Because differential evolution is stochastic, we run three independent optimizations per participant and model variant and retain the best-performing fit.
For model comparison, we compute per-subject AIC values and summarize mean $\Delta$AIC with t-based 95% confidence intervals, alongside AIC weights and winner ratios.
As a qualitative check on plausibility, we also simulate each fitted model variant on the same list structure participants experienced and compare simulated benchmarks against the empirical patterns.
In simulation, we generate recall sequences only up to the observed number of recalls for each trial, so benchmark comparisons focus on item selection rather than stopping policy.

## Modeling Results

### Model comparison

The Emotion + LPP (interaction) model provides the best fit by all comparison metrics.
Relative to Emotion + LPP (main effects), its mean $\Delta$AIC is -1.12 with a 95% t-based confidence interval of \[-2.11, -0.13\].
Relative to Emotion-only, its mean $\Delta$AIC is -3.98 with a 95% t-based confidence interval of \[-5.71, -2.25\].
Emotion + LPP (main effects) also improves over Emotion-only, with mean $\Delta$AIC of -2.86 \[ -4.14, -1.57 \].
AIC weights place essentially all mass on the interaction model, with the other two near zero.
Winner ratios tell the same story, favoring the interaction model for roughly 0.66 of subjects against main effects and 0.82 against Emotion-only, and favoring main effects for roughly 0.74 against Emotion-only.
Together these results indicate that adding an interaction provides a modest but consistent improvement over main effects, and both LPP models outperform Emotion-only.
Estimated parameters are broadly similar across variants, so the key evidence comes from relative fit metrics and benchmark patterns rather than large shifts in fitted values.

### Parameter estimates (brief summary)

In the interaction model, mean estimates (± 95% t-based CI) support a baseline negative-item learning advantage and additional Early LPP modulation: $\phi_{emot} = 4.91 \pm 0.81$, $\kappa_L = 39.53 \pm 12.91$, and $\kappa_{EL} = 53.95 \pm 11.17$.

### Benchmark reproduction (qualitative)

Across fitted simulations, the interaction model preserves the negative > neutral recall advantage while producing a stronger Early LPP separation for subsequently recalled negative items than for neutral items.
