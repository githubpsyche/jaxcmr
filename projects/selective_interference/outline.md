# Intrusive and Voluntary Memory in a Single-System Framework: A Retrieved-Context Account of Selective Interference

This document outlines a proposed Psych Review–style theoretical paper and accompanying simulation program. In the proposed paper, we would:

- Treat the selective interference effect (post-film Tetris reduces intrusions while sparing voluntary memory) as the central explanatory target.
- Show how this pattern can be derived from a single-system retrieved-context model (CMR3) without invoking separate traces or specialised visuospatial consolidation mechanisms.
- Use simulations to generate mechanistic predictions and boundary conditions that can explain both positive and null findings in the trauma-film/Tetris literature.
- Situate this account within broader work on contextual binding, intrusive memories, and emotional disorders.

We would advance three main claims:

1. **Single-system sufficiency.**
   A single, context-binding episodic memory system (as implemented in CMR3) is sufficient to reproduce the selective interference effect: intrusions decrease after certain post-film tasks, while recognition and intentional free recall are relatively spared. 
   This does not require separate "intrusive" and "voluntary" memory traces or a dedicated visuospatial consolidation mechanism.

1. **Retrieval mode and control as key determinants.** 
   The critical dissociations arise from how retrieval is driven and controlled.
   Unguided intrusions are primarily context→item retrievals as everyday context drifts near trauma-related states. 
   Intentional free recall is also context→item but starts from a deliberately biased context and uses output gating.
   Recognition and similar probe-based tasks rely heavily on item→context retrieval and probe-based decision rules. 
   Differences in retrieval mode (context→item vs item→context) and in the strength of control (starting-context bias and gating) explain why some tasks are more susceptible to post-film competition than others.

2. **Reinterpretation of Tetris/visuospatial interventions.**
   Tetris and related tasks can be interpreted as interventions that generate dense competitor episodes that (a) are bound into contexts near the trauma film, (b) suppress rehearsal of trauma items during the interval, and (c) may share arousal/engagement properties with the film's most salient moments.
   This reframes the role of visuospatial properties: they matter insofar as they drive segmentation, engagement, and context overlap, not because they uniquely disrupt sensory trace consolidation.

The remainder of the document expands these claims into a section-by-section manuscript plan and a concrete CMR3 simulation program.

## Introduction: The selective interference puzzle

**Goal:** Define the empirical target and the theoretical puzzle succinctly.

- Introduce the trauma-film + Tetris paradigm and the canonical finding: reminder + visuospatial task reduces intrusive memories but leaves voluntary memory largely intact (e.g., Holmes et al., Lau‑Zhu et al., James et al., Iyadurai et al., Kessler et al.).
- Define "selective interference" precisely:
  - Outcome 1: intrusive images/flashbacks (lab tasks, diaries).
  - Outcome 2: voluntary free recall/recognition.
  - Manipulation: post-film Tetris or related tasks vs control.

- Emphasise why this pattern has been interpreted as evidence for:
  - separate-trace / dual-representation frameworks, and/or
  - visuospatial working-memory interference with consolidation.

- Introduce the gap. These explanations are not embedded in a formal episodic memory model and often remain silent on:
  - why voluntary memory is spared,
  - how associative cueing manipulations behave,
  - why some replications/meta-analyses show attenuated or absent effects.

- State the aim. We will derive the selective interference pattern from a single-system retrieved-context model (CMR3) and contrast its predictions with dominant alternatives.

**Possible Figure 1:** Schematic of the phenomenon (intrusions vs voluntary memory; Tetris vs control).

## Empirical landscape

**Goal:** Compactly organise the main empirical constraints the model must accommodate.

### Core lab studies:

- Early Holmes et al. work showing Tetris (vs control) reduces subsequent intrusions while sparing voluntary free recall.
- Lau‑Zhu et al. (2019, 2021) showing intrusion reduction with spared recognition/free recall, including manipulations of cueing (trauma vs foil cues) and vigilance–intrusion tasks vs recognition tasks.
- Other lab variants manipulating:
  - timing (during vs post-film),
  - task type (visuospatial vs verbal),
  - cueing (trauma cues vs non-trauma cues).

These provide the basic selective interference pattern and show that it generalises across some task variations.

### Extensions and translation

- Reminder+Tetris designs at delayed intervals (e.g., 24 h, 3 days).
- Emergency department and real-world trauma-like implementations.
- Imagery rescripting and other non-Tetris interventions that sometimes produce similar intrusion reductions.

These studies extend the paradigm beyond immediate post-film interventions and into more ecologically valid settings.

### Meta-analytic and replication evidence

- Recent meta-analyses on visuospatial interventions and intrusive memories (mixed effect sizes; stronger for some designs, weaker or absent for others).
- Multisite/preregistered replication attempts reporting:
  - robust immediate or lab-based effects in some conditions, and
  - weaker or absent effects on long-term diary intrusions in others.

We will use this to frame the empirical target as **heterogeneous**: the effect exists under some conditions, is weaker or fails under others. This motivates a mechanistic model that can explain when and why selective interference occurs instead of assuming a simple law.

**Possible Table 1:** Summary of a representative set of studies and key variables we will address (rows: study; columns: timing, task type, intrusion measure, voluntary measure, main effect).

## Existing theoretical accounts

**Goal:** Provide a fair but critical review of dominant accounts.

### Visuospatial working-memory interference / consolidation accounts

**Representative paper:**  
Holmes, E. A., James, E. L., Coode-Bate, T., & Deeprose, C. (2009). *Can playing the computer game "Tetris" reduce the build-up of flashbacks for trauma?* PLoS ONE.

**Core idea:**  
Post-film visuospatial tasks load visuospatial working memory during a critical window, disrupting consolidation of sensory-perceptual aspects of the trauma memory that support later intrusive imagery. More abstract/narrative aspects rely less on visuospatial resources, so voluntary recall and recognition are relatively spared.

**Possible strengths:**

- Provides a straightforward cognitive mechanism grounded in dual-task interference and WM resource theories.
- Naturally explains why visuospatial tasks often outperform purely verbal tasks at reducing intrusive imagery.

**Possible shortcomings:**

- The route from transient WM load to long-term episodic consolidation is underspecified and not formalised in a retrieval model, limiting task-level predictions.
- It does not, by itself, explain spared recognition/free recall or heterogeneous findings (e.g., mixed diary effects, some non-visuospatial interventions) without additional assumptions.

### Dual-representation / separate trace theories

**Representative paper:**  
Brewin, C. R., Gregory, J. D., Lipton, M., & Burgess, N. (2010). Intrusive images in psychological disorders: Characteristics, neural mechanisms, and treatment implications. Psychological Review.

**Core idea:**  
Trauma creates two partly separable representations: sensory-bound, poorly contextualised traces that drive flashback-type intrusions, and contextualised, verbally accessible traces that support narrative memory. Selective interference is attributed to interventions preferentially weakening the sensory-bound trace while leaving contextualised/narrative memory relatively intact.

**Possible strengths:**

- Fits clinical phenomenology of vivid, sensory intrusions coexisting with more coherent narratives, and offers a clear conceptual dissociation between intrusive and voluntary memory.
- Has been influential in framing imagery-focused treatments and motivating hypotheses about selective effects on "flashback" memory.

**Possible shortcomings:**

- The two-trace architecture is rarely embedded in a detailed retrieval model; it is unclear how the traces interact during particular tasks (free recall, recognition, cue-driven intrusion tasks).
- Explanations can become post hoc: it is often unclear which phenomena are attributed to which trace and how an intervention like Tetris acts at the representational level.

### Reconsolidation-based accounts

**Representative paper:**  
James, E. L., Bonsall, M. B., Hoppitt, L., Tunbridge, E. M., Geddes, J. R., Milton, A. L., & Holmes, E. A. (2015). *Computer game play reduces intrusive memories of experimental trauma via reconsolidation-update mechanisms.* Psychological Science.

**Core idea:**  
A reminder cue reactivates the trauma memory into a labile reconsolidation state; engaging in Tetris during this window introduces competing perceptual information that disrupts restabilisation of sensory elements that underpin intrusions. Contextual/narrative aspects are assumed to be less affected, preserving voluntary access.

**Possible strengths:**

- Ties behavioural effects to a broader neurobiological literature on reconsolidation and updating, providing a principled role for reminders and precise timing.
- Offers a way to explain effects of delayed reminder+Tetris protocols that occur after initial consolidation.

**Possible shortcomings:**

- Evidence that these human paradigms genuinely tap reconsolidation (rather than standard re-encoding in a new context) is indirect, and the computational story remains largely verbal.
- Reconsolidation windows and boundary conditions are debated, and several patterns (including mixed replication success) can plausibly be captured by simpler context-binding and competition mechanisms.

### Retrieved-context theory / CMR3 as a single-system alternative

**Representative paper:**  
Cohen, R. T., & Kahana, M. J. (2022). A memory-based theory of emotional disorders.Psychological Review.

**Core idea:**  
Episodic memory is supported by a single system in which items are bound to a drifting context that includes temporal, semantic, and emotional features. Intrusions occur when current context overlaps with trauma-related states and trauma items win a context→item competition. Voluntary recall and recognition depend on how context is seeded and controlled (for free recall) or on item→context retrieval (for recognition); post-film tasks like Tetris add competitor episodes in nearby contexts.

**Possible strengths:**

- Provides a fully specified computational model that already accounts for wide-ranging memory phenomena and incorporates valence/arousal, allowing explicit simulations rather than purely verbal theorising.
- Explains intrusive vs voluntary dissociations via retrieval mode and control within a single system, without needing separate traces or a dedicated visuospatial module.

**Possible shortcomings:**

- The model is abstract and parameter-rich; without clear constraints there is a risk of flexibility, and mapping to neural implementation is non-trivial.
- CMR3 has not yet been applied directly to unintentional recall or trauma-film/Tetris paradigms; this project would be the first serious test of whether its mechanisms can capture the involuntary/voluntary distinction, selective interference findings, and their boundary conditions.

**Transition sentence:**  
We next outline the CMR3 framework at a conceptual level, focusing on the aspects most relevant to selective interference (context-binding, retrieval modes, and control), before mapping the trauma-film/Tetris paradigm into this framework and presenting simulations.

## The retrieved-context / CMR3 framework

**Goal:** Introduce the single-system framework we will use at a conceptual level and make explicit how it instantiates the ingredients in Claims 1–2. Detailed equations and parameterisation will be deferred to a Box/Appendix.

### Overview and link to the main claims

- Retrieved-context models (CMR, eCMR, CMR3) treat episodic memory as a system in which:
  - Items are bound to a slowly drifting context representation.
  - Context includes temporal, semantic, and emotional (valence, arousal) features.
  - Retrieval is governed by learned context–item and item–context associations.

- This single architecture will underlie all intrusive and voluntary phenomena we model, directly instantiating **Claim 1** (single-system sufficiency).

### Architecture and emotional context (conceptual)

We will describe the core ingredients informally:

- **Context state:** a vector that drifts over time, integrating recent experience and emotional features.
- **Item representations:** feature vectors (including emotional features) that are associated with the context at encoding.
- **Associative matrices:** context→item and item→context links learned during encoding.

CMR3 extends earlier models by:

- Treating **valence** and **arousal** as explicit components of context and item features.
- Encoding high-arousal items with stronger context–item links and allowing them to pull context into similar emotional regions.

A schematic figure here (context vector, item vectors, bidirectional links, emotional subspace highlighted) should suffice. The full equations and parameter table will be deferred to Box/Appendix 1.

This subsection anchors the idea that "trauma context" and "post-film context" correspond to particular regions in a shared context space.

### Retrieval modes: context→item and item→context

We then define two ways the model can be "read out":

- **Context→item mode:**  
  A context cue activates items via context→item links; multiple items receive activation and compete for retrieval. This is the default mode for free recall and spontaneous intrusions.

- **Item→context mode:**  
  A probe item retrieves its associated context via item→context links. The retrieved context is then compared to a target context or used to drive a decision (e.g., in recognition).

We map these modes to task classes at a coarse level:

- Unguided intrusions and free recall are primarily context→item.
- Recognition and similar probe-based judgements are primarily item→context.

We stress that these are not separate systems; they are different uses of a single associative architecture.
This operationalises the retrieval-mode component of Claim 2. 
Later sections will refer back to "context→item" and "item→context" retrieval as defined here, rather than reintroducing these concepts.

### Retrieval control: starting context and output gating

We introduce the model's two main control levers:

1. **Starting-context bias**  
   - Retrieval can be initialised from a context state biased toward particular episodes or time periods (e.g., the film session vs a later lab context).  
   - This implements the notion of steering mental context at the start of intentional recall.

2. **Output evaluation / gating**  
   - When an item wins the context→item competition, its retrieved context can be compared to the current goal.  
   - Items with poor match can be rejected and have their context integrated only weakly, limiting drift away from the target context.

We link these to behaviour:

- **Unguided intrusions:** minimal starting-context control; little or no output gating.
- **Intentional free recall:** strong starting-context bias toward the film session; strong gating against off-target items.
- **Recognition:** primarily item→context plus a decision rule; relatively little reliance on context-driven competition among many items.

This operationalises the control component of Claim 2. Simulation 3 will manipulate these control parameters (high vs low control) rather than introducing new mechanisms.

### Where the full model specification lives

To keep the main text accessible while reassuring technical readers:

- The main text will remain at the schematic/conceptual level described above.
- Box / Appendix 1 will provide:
  - The full CMR3 equations (context update, learning rules, retrieval dynamics).
  - A parameter table.
  - Any modifications relative to published implementations (e.g., specific emotional-context settings used here).

All simulations later in the paper will instantiate this single specification (with limited parameter tweaks), rather than using different ad hoc models.



## Mapping the trauma-film/Tetris paradigm into CMR3

**Goal:** Specify how we represent the trauma-film experiments within CMR3 (items, contexts, tasks) and derive a verbal version of the selective interference account before turning to simulations.

**Opening sentence:**  
Having outlined the core CMR3 architecture, retrieval modes, and control levers, we now specify how the trauma-film/Tetris paradigm maps into this framework.

### Representational mapping

We first map components of the paradigm onto model constructs:

- **Trauma film:**  
  Modelled as a sequence of high-arousal negative items, encoded with strong item–context bindings as context drifts through a characteristic temporal/emotional trajectory.

- **Post-film tasks:**
  - Tetris-like condition: many neutral items, high engagement and segmentation, encoded in contexts adjacent in time (and, depending on assumptions, overlapping in arousal/engagement) with late-film contexts.
  - Control conditions: fewer/weaker items and more opportunity for spontaneous trauma-film rehearsal during the same interval.

- **Intrusion measures:**
  - Lab tasks (e.g., vigilance–intrusion tasks, cue-provocation tasks): context constrained by instructions and occasional trauma/foil cues.
  - Diary measures: context follows more variable, naturalistic trajectories.

This is a mapping step; we do not yet invoke competition or interference here.

### The context-binding account of selective interference

We then state the core mechanism:

- Post-film items are bound in context states adjacent to trauma-film states, so they become additional candidates whenever the system revisits that region in context→item mode (as defined in Section 4.3).
- As those contexts are later revisited, trauma-film items must compete with a denser set of strongly encoded post-film competitors; they therefore win less often, lowering the probability of an intrusion.
- The underlying item representations and their associations remain intact. When:
  - a film item is used as a probe (recognition; item→context mode), or
  - retrieval is strongly steered toward the film (intentional free recall with strong starting-context bias and gating as in Section 4.4),

  trauma items can still be accessed, explaining why voluntary measures are spared.

This subsection cashes out Claim 1 ("single-system sufficiency") in verbal CMR3 terms.

### Why post-film task properties matter

Finally, we unpack which properties of the post-film interval matter, in a way that foreshadows the simulations and implements Claim 3:

- **Competitor density and strength:**  
  Tasks that generate many, strongly encoded episodes in the trauma-adjacent context region (high segmentation and engagement) produce more competition and hence greater intrusion reduction.

- **Rehearsal suppression:**  
  Highly engaging tasks reduce spontaneous trauma-film reactivation during the post-film interval, preventing additional strengthening of trauma–context bindings.

- **Context overlap:**  
  Reminder cues and shared situational/emotional features help ensure that post-film items are encoded into the trauma-adjacent context region. The degree of overlap modulates the size and selectivity of the interference effect (e.g., effects on emotional vs neutral content).

This sets up the link to the simulation programme, which will instantiate these properties as manipulable parameters (event count/strength, rehearsal vs no rehearsal, degree of context overlap).

## Simulations

**Goal:** Implement a minimal but diagnostic set of simulations that instantiate the above ideas and generate testable predictions.

### Simulation 1. Benchmark: post-list interference in standard recall vs recognition

**Aim:**  
Show that even in a neutral list-learning setting, CMR3 predicts stronger interference on free recall than on recognition when a post-list distractor list is added.

**Design sketch:**

- Encode List A (target items).
- Two conditions:
  - A-only.
  - A followed by List B (post-list distractors).
- Test:
  - Free recall of List A from an end-of-list context cue (context→item mode).
  - Recognition of List A vs new foils using item→context retrieval.

**Metrics:**

- Proportion of A items recalled.
- Recognition accuracy (hits – false alarms) for A items.

**Expected pattern:**

- Adding List B reduces recall of A (context competition).
- Recognition of A is minimally affected.

This benchmark demonstrates that mode-dependent interference is intrinsic to the model, independent of trauma/Tetris specifics.

### Simulation 2. Selective interference in a trauma-film analogue

**Aim:**  
Capture the canonical pattern: a post-film Tetris-like condition reduces intrusive memories while leaving recognition largely intact, in a trauma-film–inspired setup.

**Design sketch:**

- **Phase 1: "Film" encoding**  
  Sequence of high-arousal negative items; strong context–item binding.

- **Phase 2: Post-film manipulation**
  - Tetris condition: reminder cue followed by many strongly encoded neutral items (high segmentation, high engagement) in trauma-adjacent context.
  - Control condition: reminder cue but few/weak neutral items, or none (more opportunity for rehearsal).
  - Optional verbal condition: intermediate competitor density and engagement.

- **Phase 3: Testing**
  - Intrusions:
    - Simulate context drift over a series of "days" or "probes."
    - Intrusions defined as trauma-film items retrieved in context→item mode with minimal control (as in Section 4.4), under:
      - lab-like conditions (fixed context with occasional trauma/foil cues),
      - diary-like conditions (more variable context trajectory).
  - Recognition:
    - Item-cued recognition of film items vs foils (item→context mode).

**Metrics:**

- Intrusion frequency over simulated time (per condition).
- Recognition accuracy (per condition).
- Optional: voluntary free recall performance.

**Expected pattern:**

- Tetris condition shows lower intrusion rates than Control, especially in contexts overlapping with post-film context.
- Recognition differs little between conditions relative to intrusions.
- Depending on parameterisation, voluntary free recall shows limited interference, anticipating Simulation 3.

### Simulation 3. Intentional vs unintentional context→item retrieval (control mechanisms)

**Aim:**  
Explain how intrusions can be reduced while voluntary free recall is spared, using control over starting context and output gating (Section 4.4).

**Design sketch:**

- Use the same film + post-film conditions as in Simulation 2.
- Define two retrieval regimes:

  1. **Unguided intrusions**
     - Context drifts autonomously.
     - Minimal starting-context control and gating.
     - Intrusions = trauma-film items crossing threshold in context→item mode.

  2. **Intentional free recall**
     - Starting context strongly biased toward the film session (e.g., early-film context).
     - Strong output gating: if a retrieved item's context mismatches the film goal, it is rejected and its context integrated weakly.

- Additional manipulation: control strength as a parameter.
  - High control: strong bias + strong gating.
  - Low control: weaker bias + weaker gating (mimicking divided attention, time pressure, or stress).

**Metrics:**

- Intrusion rates (unguided regime).
- Recall probability for film items (intentional regime) across control levels.
- Degree to which post-film items intrude into intentional recall.

**Expected pattern:**

- With high control:
  - Post-film Tetris reduces intrusions.
  - Voluntary free recall of film items is largely spared across conditions.
- With reduced control:
  - Voluntary free recall begins to show interference effects more similar to intrusions (lower recall in Tetris vs Control), generating concrete experimental predictions.

### Simulation 4. Emotional vs neutral material and the arousal-subspace claim

**Aim:**  
Explore how emotional features (valence/arousal) modulate interference in CMR3, and to what extent post-film task arousal/emotionality is needed to produce selective effects on emotional material.

**Design sketch:**

- Encode mixed episodes containing:
  - High-arousal negative items (trauma-like).
  - Neutral items.
- Post-film conditions:
  - High-arousal competitors (e.g., neutral content encoded under high arousal/engagement or reinstated trauma context).
  - Low-arousal competitors (low engagement; minimal arousal change; weaker emotional overlap).
- Test intrusions, voluntary recall, and recognition separately for high-arousal vs neutral film items.

**Metrics:**

- Intrusion reduction for high-arousal vs neutral film items across conditions.
- Voluntary recall/recognition for each item type.

**Expected pattern:**

- When competitors are high-arousal and temporally close, interference effects on intrusions are stronger for high-arousal items than for neutral ones.
- When competitors are low-arousal, interference is less selectively emotional.
- This simulation converts the "post-film traces are in a high-arousal region" idea into specific, testable constraints.

### Simulation 5 (optional). Reminder + delayed Tetris / reconsolidation-style designs

**Aim:**  
Show that reminder+Tetris effects at delays (e.g., 24 h, 3 days) can arise via context reinstatement and new competitor encoding, without invoking special reconsolidation mechanisms.

**Design sketch:**

- Phase 1: Film encoding.
- Phase 2: Delay (context drifts away from trauma region).
- Phase 3: Reminder cue reinstating trauma context, followed by:
  - Tetris-like competitor encoding vs reminder-only control.
- Phase 4: Intrusion measures over subsequent time.

**Metrics:**

- Intrusion frequency as a function of post-reminder condition.
- Any effects on voluntary measures.

**Expected pattern:**

- When the reminder successfully reinstates trauma context and is followed by competitor encoding, additional intrusion reduction emerges relative to reminder-only control.
- This offers a single-system interpretation of some "reconsolidation-like" findings.

## Comparing accounts: who predicts what?

In this section we will systematically contrast:

- Visuospatial WM/consolidation accounts.
- Dual-representation / separate-trace theories.
- Reconsolidation-based accounts.
- The CMR3 retrieved-context account.

We will organise the comparison by phenomena, not by theory:

- Intrusions vs recognition/free recall.
- Effects of associative cueing.
- Timing of intervention (immediate vs delayed).
- Effects on emotional vs neutral content.
- Immediate vs long-term diary intrusions.
- Dependence on event segmentation / competitor density.

A summary table will indicate, for each phenomenon, whether each theory:

- clearly predicts it,
- clearly does not predict it,
- or can accommodate it only with additional assumptions.

This section will highlight where the CMR3 account adds explanatory precision and where it remains underdetermined.

## Implications for intervention design

Using the retrieved-context account, we will derive mechanistic design principles:

- Tetris is one instance of a broader class of tasks that:
  - generate many well-segmented, strongly encoded competitor episodes in contexts overlapping with trauma,
  - suppress trauma rehearsal during critical periods.
- Other tasks (imagery rescripting, alternative visuospatial activities, possibly certain semantic tasks with high segmentation and context overlap) may satisfy the same mechanistic criteria and thus be effective.

We can reinterpret mixed empirical results and meta-analytic heterogeneity in terms of:

- insufficient competitor density,
- poor context overlap,
- ineffective rehearsal suppression,
- or weak retrieval control at test.

This feeds into concrete design principles for post-trauma interventions and lab paradigms.

## Limitations and open questions

We will explicitly acknowledge:

- **Model abstraction:** CMR3 is highly abstract; parameters and context dimensions are not trivially tied to specific neural substrates.
- **Scope of fitting:** We will be fitting qualitative patterns rather than individual datasets. A full data-fitting exercise would be ideal but is beyond the scope of a single paper.
- **Empirical robustness:** The empirical base for Tetris/selective interference is heterogeneous; large-sample replications and meta-analyses show mixed outcomes, which we treat as constraints rather than nuisances.
- **Unresolved predictions:** Some key predictions (e.g., about arousal-mediated effects, the role of control, and competitor density) are currently untested and will require targeted experiments.



## 10. Immediate next steps (project level)

From the modelling side, the immediate next steps are:

- Finalise simulation specifications (parameters, metrics) for Simulations 1–3.
- Implement and sanity-check CMR3 simulations using an existing codebase or a faithful reimplementation.
- Generate preliminary figures illustrating the key qualitative patterns (recall vs recognition interference; intrusion trajectories; control vs no-control retrieval).
- Circulate these preliminary results to the team for feedback, then refine the manuscript plan and decide which additional simulations (e.g., Simulations 4–5) to prioritise.
