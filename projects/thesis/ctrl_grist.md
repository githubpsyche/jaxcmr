### Section X. “What is a Repetition Effect?”

*(to be inserted immediately before the Methods/Control‑analysis subsection)*

---

#### 1 Why the definition matters

Experiments that sprinkle repeated items into a study list often report “repetition effects” whenever any measure—accuracy, latency, transition rate—differs between *mixed* and *control* lists. But a mixed list differs from its control in many ways (overall list length, neighbour identities, rehearsal time). Unless the analysis controls for those confounds, the effect that is attributed to “repetition” may simply reflect ordinary serial‑position dynamics or item similarity. A rigorous definition is therefore a **contrast conditioned on serial position and list composition**.

#### 2 Formal statement

Let $S_i$ be a study position. Denote by

$$
P\bigl(R_{S_i}=1\mid\textsc{rep}\bigr)\quad\text{and}\quad 
P\bigl(R_{S_i}=1\mid\textsc{nov}\bigr)
$$

the probabilities that the item at $S_i$ is recalled when, respectively, it **repeats** a token shown earlier in the same list or is **novel**. We define the **repetition effect** at that position as

$$
\boxed{\;\Delta_{\text{rep}}(S_i)=P\!\bigl(R_{S_i}=1\mid\textsc{rep}\bigr)\;-\;
                     P\!\bigl(R_{S_i}=1\mid\textsc{nov}\bigr)\;}
\tag{1}
$$

The goal of any analysis is to estimate $\Delta_{\text{rep}}$ without bias from factors that influence *both* probabilities equally.

#### 3 Common analytical pitfalls

| Pitfall                                                                                                     | Why it biases $\Delta_{\text{rep}}$                                                                    |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **List‑level comparisons** (e.g., mean recall in mixed vs control lists)                                    | Confounds repetition with global list properties (e.g., two fewer unique words, extra rehearsal time). |
| **Unmatched serial positions** (control lists keep the same words but shift them to new lags)               | Alters primacy/recency exposure, inflating or reversing the contrast.                                  |
| **Single‑position matching for double tokens** (treating only one of the two occurrences as the “repeater”) | Halves the denominator in the control condition, guaranteeing a positive $\Delta_{\text{rep}}$.        |
| **Counting second recalls of the same logical item**                                                        | Over‑weights repeats in mixed lists and under‑weights them in controls, again biasing the estimate.    |

Several influential studies mix at least one of these pitfalls, leading to contradictory claims about whether repetition impairs or facilitates associative transitions.

#### 4 Our principled control method

To obtain an unbiased estimator of (1) we enforce three design rules:

1. **Position‑matched shuffles.** For every mixed list we generate 100 random shuffles *within the same word set* until all intra‑list repeats vanish. The item at $S_i$ remains at $S_i$; only its identity (repeat vs novel) changes. These shuffles supply the empirical distribution of $P(R_{S_i}=1\mid\textsc{nov})$.
2. **Dual‑position matching.** If the mixed list contains tokens of the same word at $S_a$ and $S_b$, a recall at *either* $S_a$ or $S_b$ in the control counts as a “match.” This honours the null hypothesis that the memory system could have treated the occurrences interchangeably; omitting one position would artificially depress the baseline.
3. **Duplicate‑recall filtering.** After the first recall of a logical item, subsequent occurrences are excluded from transition tallies in *both* mixed and control lists. This keeps the at‑risk sets identical and prevents the same response from inflating one side of the contrast.

Figure X.1 (next page) schematises the workflow; Algorithm 1 in Appendix B gives pseudocode.

#### 5 Interpretation

Because the two conditions differ **only** in whether the token at $S_i$ was previously studied, any surviving discrepancy in recall probability, transition rate, or neighbour bias can be *uniquely* attributed to repetition. In the analyses that follow we compute (1) for (i) recall probability, (ii) forward‑ and backward‑neighbour transition rates in free recall, and (iii) chaining errors in serial recall. These estimates allow us to test the specific associative‑interference predictions of standard CMR against the trace‑selective variant of Instance‑CMR introduced later in the chapter.

---

*This section establishes the conceptual and methodological ground rules before the reader encounters the algorithmic details of the permutation control analysis.*



---

Below is a streamlined way to frame the section, followed by a fully worked‑out *model paragraph* you can drop into the manuscript.  I concentrate on three things:

* **Conceptual precision** – nailing down what qualifies as a “repetition effect.”
* **Operational definition** – mapping the concept onto an analysis plan.
* **Implementation choices** – making your control analysis and its assumptions transparent.

---

## 1 Clarify the concept before the statistics

**Pitfall to avoid.**  Many papers call *any* difference observed in a list that happens to contain repeated items a “repetition effect.”  That confuses ordinary practice effects (better memory because you saw *something*) with the distinctive impact of seeing *the same* item twice.

**Tighter definition.**  A *repetition effect* is the incremental change in recall probability (or in whatever dependent measure you choose) that is uniquely attributable to the fact that the current study event *duplicates* an earlier one:

$$
\Delta_{\text{rep}} = P(\text{Recall}\mid \text{current item is repeat}) - P(\text{Recall}\mid \text{current item is novel})
$$

This conditional framing matters: we are not comparing repeated *lists* with non‑repeated lists; we are comparing repeated *events* with novel events, within matched list structures as closely as possible.

---

## 2 Map the concept onto an analysis pipeline

1. **Define the two event sets.**
   *Target* events (repeats) vs. *control* events (novels) matched on serial position, global list context, and any other nuisance variables you can hold constant.

2. **Specify the outcome.**
   Decide whether you care about (a) *item recall* or (b) *transition probability* from serial position $S_i$ to $S_j$.  Make this explicit up front; the rest of your pipeline hangs on it.

3. **State the null.**
   Under $H_0$, a repeated study event behaves like an independent token; its earlier occurrence is irrelevant.  Operationally, that means shuffling the mapping between study items and recall attempts while preserving overall list structure.

4. **Construct the baseline.**
   Generate $N$ (e.g., 100) shuffled control trials per participant.  For each shuffle, compute the same statistic you will compute for the mixed lists.  Store the distribution; your observed statistic is then compared to that empirical null.

5. **Handle multi‑position items.**
   Decide whether “recall of item *A* in control position 7” should count as “matching” a repeat that appeared in positions 3 and 7 of the mixed list.  You argue *yes* (both positions count).  Spell out why: excluding one position biases the control downward and guarantees a positive “effect” even if participants treated the two occurrences independently.

6. **Filter duplicate recalls consistently.**
   Whether you are in the main or control analysis, toss out second and later recalls of the *same* logical item before computing transition rates.  This makes the numerator and denominator of your transition metric comparable across conditions.

---

## 3 Anticipate the likely objections

* **“Double‑counting” criticism.**  By counting both positions of a repeater, are you inflating transition opportunities?  Yes—but *symmetrically,* because the same rule holds in the null.  The relevant question is whether the *difference* between observed and null changes.
* **Shuffling destroys higher‑order structure.**  True.  But if your phenomenon of interest is tied to first‑order serial transitions, shuffling within lists (not across lists) preserves most relevant structure while severing the specific contingency that defines a repetition.
* **Model‑based alternatives.**  A reviewer may ask for an item‑response‑theory or REM‑style model of repeated exposures.  Be explicit that your goal here is an *agnostic* baseline, not a process model.


I think we need something more convincing that situates the approach against alternative (mistaken) approaches. Let's draft an entire section titled "What is a Repetition Effect?"This should be structured like academic writing -- without using tables, lists, or a subheading per paragraph. Reasonably long as not to be dense. In particular, we should take time to walk the reader through the biases in the literature this approach avoids and the consequences that the biases otherwise inflict. I think we still need to be clear at the beginning to identify the definition of repetition effects that we are critiquing. We should similarly explain why failure to filter duplicate recalls would "inflate the repeat condition".  


 



2/2


---

## 4 Rewrite suggestion (one cohesive paragraph)

**Repetition effects** are often described loosely as “performance changes caused by presenting an item more than once,” but that phrasing misses the point.  Practice improves memory for *any* item; what we care about is the *incremental* change that occurs *because* the current study event is a *repeat* of a prior one.  Formally, we define the repetition effect for an event at serial position $S_i$ as

$$
\Delta_{\text{rep}} = P(\text{recall}_{S_i}\mid \text{repeat}) - P(\text{recall}_{S_i}\mid \text{novel}),
$$

where both probabilities are computed within lists that are identical except for whether the item at $S_i$ reoccurs elsewhere.  To estimate $\Delta_{\text{rep}}$ without conflating it with generic list‑level differences, we pair each *mixed* list (containing intra‑serial repeats) with a *control* list in which the same items are randomly shuffled until no repeats remain.  We generate 100 such shuffles per participant to build an empirical null distribution of recall and transition statistics.  Crucially, if an item appears twice in the mixed list—say at $S_3$ and $S_7$—a recall at either position in the control list is counted as a “match”; otherwise the null would be artificially deflated, guaranteeing a spurious positive effect.  Duplicate recalls of the same logical item are removed before any transition is scored, in both main and control analyses, so that the event sets entering the comparison are perfectly aligned.  What survives this filtering is, by construction, the component of performance that can only be attributed to encountering an *old*, not a *new*, item.

Feel free to lift and adapt: the paragraph already encapsulates your stance, the conditional definition, the control logic, and the key methodological decision about multi‑position items, all in \~180 words.

---

### Final checks before you move on

* Does every technical choice in the pipeline trace back to your conceptual definition?
* Can you point to at least one competing paper and show exactly where its control analysis violates your definition?
* Have you run a quick simulation confirming that your pipeline returns *no* effect when the generative process treats repeats as independent tokens?

If you can answer “yes” to all three, the section will be hard to dismiss—even by scholars who favor different control schemes.
