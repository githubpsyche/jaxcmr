Below is a tight, NIH‑style **Specific Aims** page aligned to your three‑aim structure. It’s written to fit on one page when formatted in standard NIH margins/fonts.

---

**Project Title:** Retrieval Intentionality in Episodic Memory: Control, Task Policy, and Neurocognitive Constraints in Retrieved‑Context Models

**Overall Objective.** Build and test a mechanistic account of the distinction between **voluntary** and **involuntary** recall within an emotion/reward‑sensitive retrieved‑context framework. We will adjudicate competing computational accounts, formalize “strategic” recall as a task‑dependent policy that adapts with practice, and link EEG signals to model parameters governing control and stopping.

**Central Hypothesis.** Recall events arise from a shared cue‑driven generator; the voluntary/involuntary distinction primarily reflects **control (gating/monitoring)** and **stopping policies** that are conditioned on goals and experience. An item‑independent temporal context component and state‑dependent failures of control together explain “spontaneous” intrusions. EEG indices track these control and policy states.

**Significance.** The field lacks a decisive, mechanistic account of retrieval intentionality. Clarifying whether involuntary recall requires a distinct route (vs. failures of control/policy) bears directly on models of intrusive memory and on how reward/emotion reshape memory performance. This project provides falsifiable tests, unifies explanations across free, reward‑weighted, and suppression instructions, and grounds interpretation in neurocognitive signals.

**Innovation.** (i) Formal, head‑to‑head comparison of **three** mechanistic families for intentionality; (ii) a **normative policy** layer (ordering and stopping) that predicts strategic recall and practice effects across tasks; (iii) a **neurocognitive** model in which EEG dynamically modulates control/stop parameters **within** the same retrieved‑context pipeline—rather than as a separate decoder.

---

### **Specific Aim 1 — Formalize and adjudicate competing accounts of the voluntary/involuntary distinction.**

**Working models.**

* **A. No‑distinction:** All recalls arise from the same cue‑driven generator; intrusions are by‑products during intentional search.
* **B. Shared generator + control:** Same generator plus an instruction‑dependent **gate/monitor** and **stopping policy** that can fail, yielding involuntary recalls.
* **C. Separate route:** A secondary, context‑decoupled generator produces a subset of involuntary recalls.

**Approach.** Implement A/B/C as extensions of eCMR/CMR‑style models; incorporate **response latency and termination** explicitly. Fit hierarchical models to existing free‑recall datasets with emotion and reward manipulations, including instructions to **suppress** recall. Evaluate with out‑of‑sample likelihood, posterior predictive checks (intrusion rates/types, contiguity, latency, stop‑time distributions), and model‑recovery tests.

**Primary tests.** Does adding an online **gate/stop** (B) outperform A across tasks? Is a **separate route** (C) required to capture context‑insensitive intrusions?

**Milestones/criteria.** Clear winner among A/B/C on held‑out trials; identifiable gate vs. stop parameters (validated by recovery); open code/specification.

**Risk & alternative.** If A matches B, conclude minimality (no extra route/control needed) and refine claims accordingly; if C is required, localize boundary conditions (task, emotion, practice) where it is needed.

---

### **Specific Aim 2 — Model strategic recall as a task policy that adapts with instructions and practice.**

**Rationale.** “Strategic” recall depends on task goals (free, reward‑weighted, suppression) and experience. A **normative policy** (expected‑value ordering + adaptive stop rule) predicts when recalls should occur; **deviations** index involuntary leakage.

**Approach.** Layer a **utility‑based policy** and **dynamic stopping** on the best generator from Aim 1. Allow policy parameters (gate, stop, ordering weight) to **learn across lists/sessions** (multi‑trial learning; practice). Fit to reward‑manipulated and suppression‑instruction datasets; quantify a **policy gap** (optimal vs. produced recalls) and test whether it explains intrusions beyond the generator alone. Optionally include an **item‑independent temporal drift** component to capture cross‑list intrusions.

**Predictions.** (i) Reward and suppression instructions reshape ordering and termination; (ii) practice reduces policy gaps and latencies; (iii) emotion/reward‑matched intrusions emerge when gate/stop parameters are weak or when temporal drift brings past contexts near.

**Milestones/criteria.** Policy layer improves prediction of **recall order** and **stop times**; practice effects captured by parameter change; cross‑task generalization (train on one instruction, predict another) with minimal refitting.

**Risk & alternative.** If a fixed policy suffices, report as constraint on control learning; focus interpretation on generator dynamics and temporal drift.

---

### **Specific Aim 3 — Build a neurocognitive model linking EEG to voluntary/involuntary recall.**

**Rationale.** If control and stopping govern intentionality, **trial‑wise** neural signals should modulate those parameters and improve prediction of intrusions and termination.

**Approach.** In existing EEG free‑recall datasets (emotion/reward), compute retrieval‑locked and pre‑response features (e.g., control/effort indices, reinstatement measures). Use them as **priors/modulators** on **gate strength**, **stop threshold**, and (optionally) **temporal drift**, evaluated on **held‑out trials**. Include **shuffled/phase‑randomized** controls to test specificity and guard against circularity.

**Milestones/criteria.** EEG‑informed models show reliable gains in predictive likelihood for **intrusion occurrence** and **stop times**; individual differences in EEG parameters track instruction effects and practice‑related changes.

**Risk & alternative.** If EEG adds no predictive value, report as a principled null and refine feature selection or timescale in follow‑up work.

---

**Expected Outcomes.** A decisive comparison of intentionality mechanisms; a policy‑aware model that explains instruction and practice effects; and a neurocognitive account tying EEG to the control and stopping parameters that distinguish voluntary from involuntary recall.

**Impact.** This program clarifies whether involuntary recall requires a distinct route or emerges from state‑dependent control failures, provides normative predictions for reward/suppression contexts, and delivers an open, extensible neurocognitive modeling toolkit for intrusive memory and related phenomena.

**Scope & Deliverables (12 months, no new data).** Three analysis papers (Aims 1–3), open code and preregistered comparison plans, and a reusable pipeline integrating EEG with retrieved‑context modeling.
