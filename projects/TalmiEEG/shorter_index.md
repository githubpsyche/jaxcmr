---
bibliography: references.bib
---

Building on the behavioral and ERP results reported above, we test mechanistic accounts of how study-phase neural activity can modulate encoding within retrieved-context models.
We embed Early LPP as an item-level modulator of learning strength in eCMR and compare three variants that differ only in how LPP enters the encoding rule.

## eCMR Specification

We use a connectionist CMR architecture with a temporal context state.
We implement emotion and Early LPP as modulators of learning strength, rather than adding a separate emotional context layer.
This choice keeps the model aligned with standard retrieved-context dynamics while ensuring that emotion-only, LPP main-effects, and LPP interaction variants differ only in a single encoding pathway.

We denote temporal context as $c^T$ and item features as $f_i$.
$M^{FC}$ maps item features to temporal context, and $M^{CF}$ maps temporal context to item features.

| Symbol | Name | Description |
|------------------------|------------------------|------------------------|
| $c^T$ | temporal context | Recency-weighted summary of prior items |
| $f_i$ | item features | One-hot representation of item $i$ |
| $M^{FC}$ | item-to-temporal-context memory | Feature-to-context associations |
| $M^{CF}$ | context-to-item memory | Temporal context cueing item features |
| $\beta_{enc}$ | encoding drift rate | Integration rate of temporal context during encoding |
| $\beta_{start}$ | start drift rate | Start-list context integration at recall onset |
| $\beta_{rec}$ | recall drift rate | Integration rate of temporal context during recall |
| $\alpha$ | shared support | Uniform pre-experimental support in $M^{CF}$ |
| $\delta$ | item support | Self-support in $M^{CF}$ |
| $\gamma$ | learning rate | Feature-to-context learning rate in $M^{FC}$ |
| $\phi_s$ | primacy scale | Initial boost to $M^{CF}$ learning |
| $\phi_d$ | primacy decay | Decay rate of the primacy boost |
| $\tau_c$ | choice sensitivity | Exponent for Luce-style competition |
| $\phi_{emot}$ | emotional learning boost | Value is $\phi_{emot}$ for emotional items and 0 for neutral items |
| $L_i$ | centered Early LPP | List-centered Early LPP for item $i$ |
| $\kappa_L$ | LPP main scale | Slope for LPP main effect |
| $\lambda_L$ | LPP main threshold | Centering offset for the LPP main effect |
| $\kappa_{EL}$ | LPP interaction scale | Slope for LPP-by-emotion interaction |
| $\lambda_{EL}$ | LPP interaction threshold | Centering offset for the interaction term |

: Parameters and structures specifying eCMR. {#tbl-ecmr-parameters}

At the start of each list, $M^{FC}$ and $M^{CF}$ are initialized to pre-experimental values.

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

$\gamma$ scales the ratio of new feature-to-context associations formed during the experiment to pre-experimental item-context links.
$\delta$ and $\alpha$ set pre-experimental self-support and shared support in context-to-feature memory, respectively.

During encoding, temporal contextual input is retrieved as $c^{IN}_i = M^{FC} f_i$ and normalized to unit length.
Temporal context updates according to

$$
c^T_i = \rho_i c^T_{i-1} + \beta_{enc} c^{IN}_i
$$

$$
\rho_i = \sqrt{1 + \beta_{enc}^2\left[\left(c^T_{i-1} \cdot c^{IN}_i\right)^2 - 1\right]} - \beta_{enc}\left(c^T_{i-1} \cdot c^{IN}_i\right)
$$

Feature-to-context learning uses a Hebbian update.

$$
\Delta M^{FC}_{ij} = \gamma f_i c^T_j
$$

Primacy is implemented by scaling the temporal context-to-feature learning rate.

$$
\phi_i = \phi_s e^{-\phi_d(i-1)} + 1
$$

Learning strength is defined as

$$
g_i = \phi_{emot,i} + \phi_i
$$

$\phi_{emot,i}$ equals $\phi_{emot}$ for emotional items and 0 for neutral items.
The primacy term $\phi_i$ enters additively to capture an independent contribution to learning strength.

Context-to-feature learning uses the learning strength $g_i$.

$$
\Delta M^{CF}_{ij} = g_i c^T_j f_i
$$

At the start of recall, temporal context is shifted toward the pre-list state.

$$
c^T_{start} = \rho_{N+1} c^T_N + \beta_{start} c^T_0
$$

At each recall step, temporal context cues item activations.

$$
A = M^{CF} c^T
$$

At each recall step, the probability of recalling item $i$ is

$$
P(i) = \frac{A_i^{\tau_c}}{\sum_k A_k^{\tau_c}}
$$

We omit an explicit termination mechanism, consistent with our focus on recalled items rather than stopping decisions.
After each recall, the recalled item's temporal context is reinstated via $M^{FC}$ and integrated with drift rate $\beta_{rec}$, and the process iterates.

## Model Variants

We compare three variants that differ only in how Early LPP enters the learning-strength term $g_i$ in $\Delta M^{CF}_{ij}$.
This isolates whether LPP adds explanatory value beyond emotion labels and whether that value differs by item type, while keeping the retrieved-context dynamics fixed.

Let $e_i$ be an indicator that equals 1 for emotional items and 0 for neutral items.
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

We evaluated each model by how well it predicts the unordered set of recalled items on each list (not recall order), because our data is missing recall order information and because our primary behavioral outcomes and benchmarks target item selection rather than order-dependent dynamics. Let $S_t$ denote the set of items recalled on trial $t$, and let $r$ range over recall sequences that contain exactly the items in $S_t$ (in any order). We define the *set likelihood* as

$$
P(S_t \mid \theta) = \sum_{r:\, r\ \text{yields}\ S_t} P(r \mid \theta),
$$

where $\theta$ are the model parameters.

Evaluating this sum exactly is infeasible because the number of sequences that yield a given set grows rapidly with $|S_t|$. We therefore approximate $P(S_t\mid\theta)$ by Monte Carlo sampling: for each trial we draw $K$ random permutations $r_1,\dots,r_K$ of $S_t$, compute $P(r_k\mid\theta)$, and estimate the sum as $|S_t|!\,\frac{1}{K}\sum_{k=1}^K P(r_k\mid\theta)$. Because the factorial factor does not depend on $\theta$, we maximize the (unscaled) mean permutation likelihood $\frac{1}{K}\sum_k P(r_k\mid\theta)$, which is proportional to the set likelihood and yields identical fits and $\Delta$AIC comparisons.

We maximize the objective with differential evolution [@storn1997differential]. Because the optimizer is stochastic, we run three independent fits per participant and model variant and retain the best-performing fit. We compare variants using AIC computed from the approximate set log-likelihood. We then assess qualitative adequacy by simulating each fitted model on the study lists while matching each trial's observed output length and plotting the benchmark summaries.
This set-likelihood approach evaluates models on their ability to reproduce the full recalled sets, providing a stringent test of mechanistic hypotheses.
