---
title: "Retrieved-Context Across Tasks: Opening + Ch.2 + Ch.3"
subtitle: "Unifying CRU/CMR & Rethinking Repetition"
author: "TBD"
date: today
format:
  pptx:
    slide-level: 2
    # reference-doc: theme.pptx   # <- Optional: point to your PowerPoint theme/template
bibliography: references.bib      # <- Optional: provide your .bib file to enable citations
toc: false
---

## Roadmap

- **Opening:** What we test (not architecture) & why retrieved-context theory (RCT)
- **Chapter 1 (setup):** Instance–CMR equivalence → compare *strategies*, not architectures
- **Chapter 2:** CRU↔CMR factorial comparison in free & serial recall
- **Chapter 3:** Repetition without blending—diagnostics and new variants
- **Take‑home & next steps**

---

## Opening: Problem & Promise

- Goal: A **single retrieved‑context account** spanning free & serial recall and repeated experience
- Key lever in RCT: **context reinstatement** to cue what comes next
- Two big questions that drive Chapters 2–3:`
  1. Do CRU vs CMR differences matter *once architecture is equated*?
  2. When items repeat, does memory **blend** episodes or **keep traces separate**?

---

## Opening: Retrieved‑Context Theory (RCT) in one slide

- Context **drifts** across study; items bind to that context
- Recall **reinstates** features tied to the last item → cues temporal neighbors
- Captures: **recency**, **lag‑contiguity**, **primacy/recency trade‑offs**, prior‑list intrusions
- Implementations: **TCM**, **CMR**, **CRU**

> We treat these as **configurations** of the same RCT toolkit (Ch. 1).

---

## Chapter 1 (Setup): Instance–CMR equivalence

- Instance‑CMR shows **trace-based** and **associative-matrix** storage are functionally equivalent for our tasks
- Consequence: CRU and CMR can be compared as **parameterized strategies** within a shared architecture
- We’ll focus on **algorithmic toggles** (what gets reinstated; how recall starts/stops; what pre-experimental structure is present)

**Take‑away:** In Ch. 2–3, differences are **not architectural**—they are **settings** in one family.

---

## Chapter 2: Bridging CRU & CMR (Goal)

- Question: Can CRU’s **streamlined** approach extend to free recall? Does CMR’s **added machinery** help serial recall?
- Approach: **Factorial toggles** that turn CRU into CMR‑like variants (and vice‑versa) while holding decision rule constant
- Fit **free recall (PEERS subset)** and **serial recall (Logan 2021)** using the same likelihood pipeline

---

## Ch.2: Factorial toggles (mechanism levers)

- **Feature→Context learning** \(γ\) (dynamic vs inert): reinstates study context for backward transitions
- **Pre‑experimental Context→Item** \(δ,\;α\): self‑support & shared support scaffolds
- **Primacy gradient** \(ϕ_s,\;ϕ_d\) vs CRU’s position‑scaled \(β,g\)
- **Recall initiation** \(β_\text{start}\): blend end‑ and start‑of‑list context (CMR) vs start‑only (CRU)
- **Termination**: CRU end‑of‑list **context‑based** vs CMR **position‑based** \(θ_s, θ_r\)
- **Confusable item identification** \(g, g_\text{dec}\) (serial‑recall only)

> Decision rule held constant (Luce) to isolate mechanism effects.

---

## Ch.2: Datasets & fitting

- **Free recall:** PEERS words (16‑item lists; strong recency at initiation)
- **Serial recall:** Logan 2021 letters (5–7 letters; confusability relevant)
- Likelihood fits per participant; **same scoring** pipeline across variants
- Simulate fitted models to check **signature curves** (SPC, PFR, lag‑CRP)

---

## Ch.2 (Free recall): What matters

- **CMR best overall** (more degrees, designed for free recall)
- CRU **fails** with fixed start‑of‑list **initiation**; enabling \(β_\text{start}\) **rescues recency**
- Adding CMR’s **primacy gradient** improves early‑list accuracy beyond CRU’s \(β\) scaling
- **Dynamic** \(γ\) + **pre‑experimental** \(δ,α\) needed to match **bidirectional lag‑CRP** (backward transitions)
- CRU’s **context‑based termination** collapses on lists with strong recency → prefer **CMR \(θ_s, θ_r\)** here

**Minimal CRU→CMR upgrades for FR:** \(β_\text{start}\) + \(ϕ_s,ϕ_d\) + \(γ\) (+ \(δ,α\) for full lag‑CRP).

- *Add figure:* PFR showing recency; lag‑CRP asymmetry

---

## Ch.2 (Serial recall): What matters

- **CRU’s context‑based termination** works **better** than CMR’s \(θ_s, θ_r\)
- **Item identification \(g, g_\text{dec}\)** required to model intrusions/errors
- Adding **\(α,δ\)** and **\(ϕ_s,ϕ_d\)** to CRU **helps** log‑likelihood, but small effect on accuracy curves
- Tuning \(γ\) or \(β_\text{start}\) **not critical** for strict serial order

**Bottom line (SR):** Keep CRU’s **end‑of‑list termination** + **confusability**; CMR extras yield modest gains.

- *Add figure:* Serial accuracy by position; error breakdown

---

## Ch.2: Summary & synthesis

- Treat CRU/CMR as **strategies** in RCT
- **Free recall** wants: flexible **initiation**, **primacy** scaling, **backward‑friendly reinstatement**, and **position‑based stopping**
- **Serial recall** wants: **context‑based stopping**, **confusability**, and standard primacy
- Factorial toggles map where each task sits on the shared **parameter manifold**

---

## Chapter 3: Same item, different traces

- Core question: When an item repeats, does recall cue a **composite** of occurrences or **select one episode**?
- Stakes: Blending predicts **interference** (knit neighborhoods, balanced access); occurrence‑specific access predicts **non‑interference**
- Challenge: Cleanly estimate **repetition effects** → need a **matched, symmetric baseline**

---

## Ch.3: Baseline that makes the test fair

- For each mixed list with a repeater at positions \(i, j\), pair many **position‑matched control lists**
- In each control list, treat \(i\) and \(j\) as **one pseudo‑item** (symmetric scoring)
- This removes **list‑composition** and **scoring asymmetries** that inflate apparent effects

> We analyze **mixed–minus–baseline** contrasts throughout.

---

## Ch.3 Diagnostics (1): Cross‑occurrence neighbor knitting

- Trigger: recall a **neighbor** of one occurrence; center lags on the **other** occurrence
- **Composite** predicts above‑baseline peaks at the other occurrence’s **±1/±2** neighbors
- **Data:** curves sit at **baseline** (no knitting)
- **Standard CMR:** predicts **peaks** → wrong direction

- *Add figure:* Centered neighbor lag‑CRP (data ≈ flat; CMR ↑ at neighbors)

---

## Ch.3 Diagnostics (2): Outgoing repetition lag‑CRP

- After recalling the **repeater**, where next?
- **Data:** **first‑occurrence neighborhood > second**, **beyond baseline**
- **Standard CMR:** **balanced access** (blending) → wrong direction

- *Add figure:* Two centered curves (first vs second); separation larger than baseline

---

## Ch.3 Diagnostics (3): Serial forward chaining

- After correct report through position \(i\): compare \(i\!\to\!i{+}1\) vs **cross‑occurrence** \(i\!\to\!j{+}1\)
- **Data:** no elevation of \(i\!\to\!j{+}1\) relative to baseline → forward chain **intact**
- **Standard CMR:** elevates \(i\!\to\!j{+}1\) errors → interference that isn’t there

- *Add figure:* Cross‑occurrence forward‑error panel (data vs CMR)


---

## Ch.3: Variants we test (holding temporal‑context dynamics fixed)

- **Standard CMR:** study‑phase reinstatement + composite reinstatement at test
- **CMR‑NoSPR:** remove study‑phase reinstatement (distinct contexts at study)
- **ICMR‑OS (Occurrence‑Specific):** distinct at study **and** **select a single occurrence** at test (no blending)
- **ICMR‑Reinf:** OS **+ reinforce first occurrence** at study (no knitting), 1 extra parameter

---

## CMR‑NoSPR: What changes & what it buys

- **Removes** study‑phase reinstatement → repetitions encode to **distinct** contexts
- **Keeps** composite reinstatement at test
- **Results:**
  - Benchmarks & spacing **preserved**
  - **Eliminates neighbor knitting**
  - Still shows **balanced access** after a repeater (issue remains)

- *Add figure:* Neighbor CRP flat; outgoing rep‑CRP still balanced

---

## ICMR‑OS: Select one occurrence at test

- At test, occurrence traces **compete**; reinstatement routes through **one** episode (no new df; reuse trace‑sensitivity)
- **Results:**
  - Benchmarks & spacing **preserved**
  - **No knitting**; **no balanced access** (separation matches baseline)
  - Serial recall: **no cross‑occurrence forward error**

- *Add figure:* Outgoing rep‑CRP shows baseline‑level first>second; serial error panel flat

---

## ICMR‑Reinf: Reinforcement without blending

- At second presentation, **strengthen first occurrence’s trace**; **do not** reinstate its context
- Test remains **occurrence‑specific** (as in OS); adds **1 parameter** (bounded)
- **Results:**
  - Benchmarks & spacing **preserved**
  - **No knitting**; **no balanced access**
  - **Boosted first‑over‑second separation** (matches data)
  - Selective **lag = 0 bump** in first→second neighbor analysis (matches data)
  - Serial recall: forward chain **intact**

- *Add figure:* Outgoing rep‑CRP with boosted separation; neighbor CRP with lag=0 bump

---

## Ch.3: Fit summary (across datasets)

- Parameter‑matched: **ICMR‑OS** beats Standard CMR (free & serial)
- Free recall: **ICMR‑Reinf** > OS (small but reliable gain; 1 extra parameter)
- Serial recall: **OS ≈ Reinf**; both >> Standard CMR
- Interpretation: **Occurrence‑specific test-phase reinstatement is necessary**; **reinforcement** explains first‑occurrence boost

- *Add table or bullet metrics:* −log L or AICw summary

---

## Big picture (Ch.2 + Ch.3)

- **One RCT family**; **CRU/CMR** are **task‑tuned settings**
- **Free recall:** flexible **initiation** + **primacy** + **backward‑friendly reinstatement** + **position‑based stopping**
- **Serial recall:** **context‑based stopping** + **confusability**
- **Repetition:** keep occurrences **separate** at study; **select** one at test; if needed, **reinforce** earlier trace—not blend

---

## Practical guidance (for modelers)

- Decide **what gets reinstated** (composite vs occurrence‑specific) before adding capacity
- Use **position‑matched symmetric baselines** for repetition effects
- Test **termination** rules by task (context‑ vs position‑based)
- Treat CRU/CMR assumptions as **toggles**; avoid confounding with decision rules

---

## Limitations & next steps

- Mixed‑list spacing slope (steeper than baseline) may need **rehearsal/strategy** components
- Direct neural markers of **occurrence‑specific reinstatement**
- Extend to richer materials with graded **semantic overlap**
- Response‑time constraints: consider **racing diffusion** for recall competitions

---

## Thank you / Questions

- Backups: additional diagnostics, parameter map, equations, and dataset details

---

## Appendix: Parameter map (quick)

- \(β_\text{enc}, β_\text{rec}, β_\text{start}\): context integration (encode/recall/start)
- \(γ\): feature→context learning (supports backward transitions)
- \(δ, α\): pre‑experimental context→item support (self/shared)
- \(ϕ_s, ϕ_d\): primacy scaling of \(M^{CF}\) learning
- \(θ_s, θ_r\): position‑based stopping
- \(g, g_\text{dec}\): item identification sensitivity (serial) 

---

## Appendix: Termination rules (contrast)

- **CRU (context‑based):** end‑of‑list episode competes with items
- **CMR (position‑based):** \(P(\text{stop}, j) = θ_s e^{j θ_r}\)
- **Empirical:** FR prefers position‑based; SR prefers context‑based

---

## Appendix: Baseline scoring (repetition)

- Symmetric pseudo‑item in controls (positions \(i, j\) count as **one**)
- Mixed–minus–baseline contrasts → isolate repetition effect (not list composition)
- Avoid **inflation** from scoring the two control positions independently

---

## References

- Add `references.bib` to resolve citations (e.g., @howard2002distributed; @polyn2009context; @logan2021serial; @lohnas2014retrieved).
