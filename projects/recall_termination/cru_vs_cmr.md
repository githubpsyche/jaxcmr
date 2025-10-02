Different implementations of CMR treat recall termination differently.
In several implementations of CMR, a leaky, competitive accumulation process determines which item wins each recall competition [e.g., @polyn2009context; @lohnas2015expanding; @healey2016four].
In these implementations, each simulated recall takes a certain amount of time, and recall terminates when the elapsed recall time exceeds the duration of the recall period.
It is computationally expensive to estimate the likelihood of any given recall event using the leaky, competitive accumulation process, leading to the development of a simplified recall termination process amenable to likelihood-based model evaluation [@kragel2015neural; @morton2016predictive; @hong2024modulation].
With this simplified framework, CMR treats recall termination as a separate process from item retrieval that does not depend on the content of the current context or on which items have been recalled so far.
For CMR, the probability of stopping the recall sequence without retrieving any more items, depends on the output position $j$ and is given by:

$$
P(\text{stop}, j) = \theta_\text{s} e^{j\theta_\text{r}}
$$

Here, $\theta_\text{s}$ and $\theta_\text{r}$ govern the initial stopping probability and the rate at which this probability increases, respectively, which is modeled as an exponential function.

---

5.  **Recall Termination**.
    CRU encodes an "end-of-list" state that competes with items to terminate recall, whereas CMR separately calculates a stopping probability using $\theta_\text{s}$ and $\theta_\text{r}$.
    In our design, we can adopt either the CRU-style item-based mechanism or the CMR-style exponential stop rule.
    This factor addresses how each model handles the decision to stop retrieval -- especially in free recall, where participants spontaneously cease recall rather than exhaustively listing items in order.
    Here we show that CRU's context-based termination mechanism causes the model to collapse when addressing free recall datasets with strong recency effects, and that incorporating a CMR-style stopping probability can help CRU better capture these patterns.
    However, by contrast, we show that CRU's end-of-list mechanism is more effective than CMR's stopping probability in capturing the strict order of serial recall.
    It's possible that in serial recall participants explicitly encode and retrieve an end-of-list marker as a signal that typing is done [@logan2018automatic], but that in free recall participants instead attempt to continue recalling items until it is too difficult to do so or time runs out.


---

::: {#tbl-fits}
| Model Variant | -LL (±95% CI) |
|:--------------------------------------------------|:--------------------|
| CMR (Free $\gamma$, $\alpha$, $\delta$, $\phi_\text{s}$, $\phi_\text{d}$, $\beta_{\text{start}}$ and Position-Based Termination) | 587.13 ± 16.84 |
| CRU with Free $\gamma$, $\phi_\text{s}$, $\phi_\text{d}$, $\beta_{\text{start}}$ | 606.05 ± 16.56 |
| CRU with Free $\alpha$, $\delta$, $\phi_\text{s}$, $\phi_\text{d}$, $\beta_{\text{start}}$ | 608.00 ± 17.00 |
| CMR with CRU's Context-Based Termination | 627.86 ± 17.35 |
| CRU with Free $\phi_\text{s}$, $\phi_\text{d}$, $\beta_{\text{start}}$ | 645.04 ± 17.49 |
| CRU with Free $\beta_\text{start}$ | 651.35 ± 17.31 |
| CRU | 724.01 ± 17.76 |

Negative log-likelihood (±95% CI) averaged across participants for selected model variants fit to PEERS free recall data.
$\gamma$: item-to-context learning rate; $\alpha$: shared support; $\delta$: self-support; $\phi_\text{s}$: primacy scale; $\phi_\text{d}$: primacy decay; $\beta_\text{start}$: start context integration rate.
:::

{{< include figcmr.md >}}

---

## Position- vs Context-Based Mechanisms for Recall Termination

{{< include figfreetermination.md >}}

CRU and CMR's mechanisms for recall termination are fundamentally different.
This difference cannot be captured by toggling a parameter from a fixed to a freely adjustable value, but rather, these mechanisms can only be swapped between variants.
CMR uses exponentially increasing stopping probabilities $\theta_\text{s}$ and $\theta_\text{r}$ to model recall termination; the probability of termination scales only with the number of recalls made so far.
By contrast, CRU treats the end of a study sequence as a special item associated in memory with the final state of the study context.
This item competes with other items for retrieval at each new recall event, and its activation can terminate recall.
In this specification, the probability of termination depends on the state of context at each recall event, and can be influenced by the same mechanisms that influence the probability of recalling other items.

Performance differences between CMR's position-based recall termination mechanism and CRU's context-based recall termination mechanism are substantial.
While @fig-cmr shows baseline CMR's performance on these benchmarks, @fig-freetermination shows the performance of CMR with CRU's context-based recall termination mechanism.
While patterns in response initiation are well-captured, using context-based recall termination mechanism leads CMR to predict overly high recall rates for all study list positions and to fail to capture the sharpness of the lag-contiguity effect.
The success of CRU's context-based recall termination mechanism depends on how consistently participants terminate recall after recalling the final items from the study list.
In most serial recall datasets, participants tend to perform this way, and the mechanism correspondingly predicts that the probability of terminating recall scales with the number of recalls made so far, as context drifts from its start-of-list state to its end-of-list state.
By contrast, in free recall datasets where participants exhibit a strong recency effect in recall initiation, this mechanism can predict early termination of recall upon or even before retrieving the last item in the study list.

---

::: {#tbl-fits-serial}
| Model Variant | −LL (±95% CI) |
|:--------------------------------------------------|:--------------------|
| CRU with Free $\alpha$, $\delta$, $\phi_\text{s}$, $\phi_\text{d}$ | 1428.98 ± 222.09 |
| CMR with CRU's Context-Based Termination | 1431.39 ± 223.74 |
| CRU with Free $\gamma$, $\alpha$, $\delta$ | 1436.94 ± 222.11 |
| CRU with Free $\phi_\text{s}$, $\phi_\text{d}$, $\beta_{\text{start}}$ | 1443.14 ± 221.79 |
| CRU | 1482.94 ± 230.86 |
| CMR with Own Position-Based Termination Rule | 1508.45 ± 208.88 |

Negative log-likelihood (±95% CI) averaged across participants for selected model variants fit to the @logan2021serial dataset.
$\gamma$: item-to-context learning rate; $\alpha$: shared support; $\delta$: self-support; $\phi_\text{s}$: primacy scale; $\phi_\text{d}$: primacy decay; $\beta_\text{start}$: start context integration rate.
All CRU variants in this table use the context-based end-of-list termination mechanism unless otherwise noted.
:::

---


The overall goodness of fit for each model variant is presented in @tbl-fits-serial.
The models are ordered by overall goodness of fit. 
These log-likelihood differences are highly reliable at the group level, based on wAIC comparisons (not reported). 
The baseline version of CRU is near the bottom of the list, demonstrating the utility of adding some mechanisms from CMR to improve model fits. 
Standard CMR is at the bottom of the list, demonstrating that some of CRU's mechanisms are critical to fitting serial recall data. 
When CRU's recall termination rule is incorporated into CMR, this improves the model's performance, but it still falls far short of the best CRU model.
The most successful model is a version of CRU that uses a few CMR mechanisms: adjustable context-to-feature shared support ($\alpha$) and self-support ($\delta$) associations, and the associative primacy gradient ($\phi_\text{s}$, and $\phi_\text{d}$). 
This model provides the best fit at the group level as well as for 100% of the individual participants in comparison to baseline CRU and CMR with both CRU's context-based recall termination mechanism and item identification confusability mechanism.

---

## An Advantage for Context-Based Recall Termination in Serial Recall

<!-- A parallel contrast emerges when comparing how CMR and CRU determine when to stop recall in strict serial recall.
CMR typically uses a position-based rule with two parameters $\theta_\text{s}$, $\theta_\text{r}$, making the probability of stopping grow exponentially with each additional output.
CRU, by contrast, treats the end-of-list event as any other stored episode in memory, associated with the final state of the study context and using no special parameters to configure its stopping probability.
This draws on the same mechanisms that drive recall, and as the context drifts toward the end-of-list state, the probability of recalling that state increases. -->

In our free recall evaluation, we found that CMR's position-based termination mechanism outperformed CRU's context-based termination mechanism.
This is because the latter can lead to premature termination when recall initiates with a strong recency effect, as is common in free recall.
However, in our serial recall evaluation, we find that embedding CRU's context-based termination in CMR improves overall model fits, despite dropping two free parameters [@tbl-fits-serial].
The standard CMR recall termination mechanism underestimates recall rates for later list items, as shown in [@fig-serial-srac, Bottom Row].

In serial recall, recall termination is not a function of the number of recalls made, but rather a function of the current state of the context, as predicted by CRU's unique specification.
By tying the end-of-list state to the evolving context, CRU effectively models this behavior with zero additional parameters.
In contrast, CMR's exponential growth in stop probability often forces the model either to under- or overshoot the actual stopping point.
Resolving this discrepancy may require a more complex model of recall termination that either incorporates insights from both the context-based and position-based mechanisms or that allows for strategic control over recall termination depending on the task at hand.

---

The present analysis sidesteps comparison of CRU and CMR's distinct recall competition mechanisms based on the justification that neither model is committed to a specific mechanism for recall competition [@polyn2009context; @morton2016predictive; @logan2021serial] and simulation using the probabilistic choice rule is more computationally efficient.
Nonetheless, with response time distributions providing important constraints for accounts of recall initiation [e.g., @osth2019using] and termination [e.g., @dougherty2007motivated], the racing diffusion model of recall competition favored in demonstrations of CRU [@logan2018automatic; @logan2021serial] potentially offers a tractable framework for addressing response time distributions as a theoretical constraint within the likelihood-based fitting approach used here [@tillman2020sequential].
To benefit from this approach, future work should clarify assumptions about when recall competitions begin and end across responses in free and serial recall tasks, and how these assumptions can be tested against response time data [@logan2021serial].

---

<!-- TODO: LOGAN: I think it would be worth considering the idea that CMR and CRU are different configurations of the same architecture.  You talk about the formal similarity making them part of the same family.  But it is also worth considering it from the perspective of a subject endowed with capabilities to represent percepts, memories, and responses in a single memory system that spans a great variety of items, contexts, situations etc.  The subject is also endowed with processes that can be applied to those representations to encode items into memory and retrieve them when asked to.  I think of the subject as a homunculus that chooses among the processes to choose a set that, when applied to perception or memory or both, produces the behavior required for the task.  CRU and CMR may be different strategies for applying the same set of memory abilities to different tasks.  I think of a strategy as a choice of parameters that make a computational model run in a certain way, and I think that subjects choose strategies and implement them by programming their CMRs and CRUs.  If you're interested in this idea, my 2001 Logan and Gordon paper describes how it lets Bundesen's TVA model account for dual task performance and proposes the idea that task sets are configurations of parameters (and that reconfiguration means changing parameters) more generally.  -->

<!-- TODO: Related: My second suggestion would be to delve more deeply into model comparisons at the level of individuals. Just as different tasks may entail different strategies, different participants may engage in different strategies even within the same task. I am thinking in particular of Qiong Zhang's work showing that, within CMR, the “optimal” strategy for performing a free recall task is to turn it into a serial recall task of the kind modeled by CRU (Zhang, Griffiths, & Norman, 2022, “Optimal Policies for Free Recall”). So even within the PEERS dataset, is it the case that there are individual participants who are better accounted for by CRU, suggesting that they adopt this “optimal” strategy? That is just one example of a question that might be asked here, but I think the authors' model comparison approach puts them in a great position to address these kinds of questions. Questions about strategies also support the development of more unified theories by showing how a shared framework of retrieved context models can account for and explain individual differences. -->
