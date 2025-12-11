# Intrusive and Voluntary Memory in a Single-System Framework: A Retrieved-Context Account of Selective Interference

This document outlines a proposed Psych Review–style theoretical paper and accompanying simulation program. In the proposed paper, we would:

- Treat the selective interference effect (post-film Tetris reduces intrusions while sparing voluntary memory) as the central explanatory target.
- Show how this pattern can be derived from a single-system retrieved-context model (CMR3) without invoking separate traces or specialised visuospatial consolidation mechanisms.
- Use simulations to generate mechanistic predictions and boundary conditions that can explain both positive and null findings in the trauma-film/Tetris literature.
- Situate this account within broader work on contextual binding, intrusive memories, and emotional disorders.

We would advance three main claims:

1. **Single-System Sufficiency**. A single, context-binding episodic memory system (as implemented in CMR3) is sufficient to reproduce the key selective interference pattern: intrusions decrease after certain post-film tasks, while recognition and intentional free recall are relatively spared. This does not require separate "intrusive" and "voluntary" memory traces or a dedicated visuospatial consolidation mechanism.
2. **Retrieval-mode + control as key determinants**. The critical dissociations arise from how retrieval is driven and controlled. Intrusions are largely unguided context→item retrievals as everyday context drifts near trauma-related states. When post-film items are bound into that region, they compete with trauma-film items and make intrusions less likely. Recognition and similar probe-based tasks are largely item→context. The probe retrieves its own associated context, so additional post-film competitors do not get a chance to "win" against the probe in the same way. Intentional/guided free recall is also context→item but benefits from control mechanisms (starting-context steering and output gating) that can shield voluntary recall from competition effects that reduce intrusions.
3. **Re-interpretation of Tetris/visuospatial interventions**. Tetris and related tasks can be interpreted as interventions that generate dense competitor episodes that a) are bound into contexts near the trauma film,  b) suppress rehearsal of trauma items during the interval, and c) may share arousal/engagement properties with the film's most salient moments. This reframes the role of visuospatial properties: they matter insofar as they drive segmentation and engagement and help populate the relevant context region, not because they uniquely disrupt sensory trace consolidation.

The remainder of the document expands these claims into a section-by-section manuscript plan and a concrete CMR3 simulation program.

## Introduction: The selective interference puzzle

Goal: Define the empirical target and the theoretical puzzle succinctly.

Introduce the trauma-film + Tetris paradigm and the canonical finding: reminder + visuospatial task reduces intrusive memories but leaves voluntary memory largely intact (including Holmes et al., Lau-Zhu et al., James et al., Iyadurai et al., etc.).

Define "selective interference" precisely:

– Outcome 1: intrusive images/flashbacks (lab tasks, diaries)
– Outcome 2: voluntary free recall/recognition
– Manipulation: post-film Tetris or related tasks vs control.

Emphasise why this pattern has been interpreted as evidence for: separate-trace / dual-representation frameworks, and/or visuospatial working-memory interference with consolidation.

Introduce the gap. These explanations are not embedded in a formal episodic memory model and often remain silent on: 

- Why voluntary memory is spared
- How associative cueing manipulations behave
- Why some replications/meta-analyses show attenuated or absent effects.

State the aim. We'll derive the selective interference pattern from a single-system retrieved-context model (CMR3) and to contrast its predictions with dominant alternatives.

Possible Figure 1: Schematic of the phenomenon (intrusions vs voluntary memory; Tetris vs control).

## Empirical landscape

Goal: Compactly organize the main empirical constraints the model must accommodate.

Core lab studies:

- Early Holmes et al. work showing Tetris (vs control) reduces subsequent intrusions while sparing voluntary free recall.
- Lau‑Zhu et al. (2019, 2021) showing intrusion reduction with spared recognition/free recall, including manipulations of cueing (trauma vs foil cues) and vigilance–intrusion tasks vs recognition tasks.
- Other lab variants manipulating timing (during vs post-film), task type (visuospatial vs verbal), and cueing.
- Maybe: Holmes et al. (2009, 2010); James et al. (2015); Iyadurai et al. (2018); Kessler et al. (2020); Lau‑Zhu et al. (2019, 2021)?

Extensions and translation:

- Reminder+Tetris designs at delayed intervals (e.g., 24h, 3 days).
- Emergency department and real-world trauma-like implementations.
- Imagery rescripting and other non-Tetris interventions.

Meta-analytic and replication evidence:

- Recent meta-analyses on visuospatial interventions and intrusive memories (mixed effect sizes; stronger for certain designs).
- Multisite/preregistered replication attempts that report either (A) robust immediate or lab-based effects in some conditions, or (B) weaker or absent effects on long-term diary intrusions in others.

We'll use this to frame the empirical target as heterogeneous: the effect exists under some conditions, is weaker or fails under others. 
This correspondingly motivates a mechanistic model that can explain when and why selective interference occurs instead of assuming a simple law.

Possibly include a table summarizing a representative set of studies and key variables we'll try to address (rows: study; columns: timing, task type, intrusion measure, voluntary measure, main effect).

## Existing theoretical accounts

Goal: provide a fair but critical review of dominant accounts. 

### Visuospatial working-memory interference / consolidation accounts

Representative paper:
Holmes, E. A., James, E. L., Coode-Bate, T., & Deeprose, C. (2009). Can playing the computer game "Tetris" reduce the build-up of flashbacks for trauma? PLoS ONE.

Core idea: 
Post-film visuospatial tasks load visuospatial working memory during a critical window, disrupting consolidation of sensory-perceptual aspects of the trauma memory that support later intrusive imagery. More abstract/narrative aspects rely less on visuospatial resources, so voluntary recall and recognition are relatively spared.

Possible strengths:

- Provides a straightforward cognitive mechanism grounded in dual-task interference and WM resource theories.
- Naturally explains why visuospatial tasks often outperform purely verbal tasks at reducing intrusive imagery.

Possible shortcomings (?):

- The route from transient WM load to long-term episodic consolidation is underspecified and not formalised in a retrieval model, making task-specific predictions limited.
- It does not, by itself, explain spared recognition/free recall or heterogeneous findings (e.g., mixed diary effects, some non-visuospatial interventions) without additional assumptions.
 
### Dual-representation / separate trace theories

Representative paper:
Brewin, C. R., Gregory, J. D., Lipton, M., & Burgess, N. (2010). Intrusive images in psychological disorders: Characteristics, neural mechanisms, and treatment implications. Psychological Review.

Core idea:
Trauma creates two partly separable representations: sensory-bound, poorly contextualised traces that drive flashback-type intrusions, and contextualised, verbally accessible traces that support narrative memory. Selective interference is attributed to interventions preferentially weakening the sensory-bound trace while leaving contextualised/narrative memory relatively intact.

Possible strengths:

- Fits clinical phenomenology of vivid, sensory intrusions coexisting with more coherent narratives, and offers a clear conceptual dissociation between intrusive and voluntary memory.
- Has been influential in framing imagery-focused treatments and in motivating hypotheses about selective effects on "flashback" memory.

Possible shortcomings:

- The two-trace architecture is rarely embedded in a detailed retrieval model: it is unclear how the traces interact during particular tasks (free recall, recognition, cue-driven intrusion tasks).
- It is often post hoc which phenomena are attributed to which trace, and what exactly an intervention like Tetris does at the representational level (e.g., weakens sensory traces is rarely operationalised).

### Reconsolidation-based accounts

Representative paper:
James, E. L., Bonsall, M. B., Hoppitt, L., Tunbridge, E. M., Geddes, J. R., Milton, A. L., & Holmes, E. A. (2015). Computer game play reduces intrusive memories of experimental trauma via reconsolidation-update mechanisms. Psychological Science.

Core idea:
A reminder cue reactivates the trauma memory into a labile reconsolidation state; engaging in Tetris during this window introduces competing perceptual information that disrupts restabilisation of sensory elements that underpin intrusions. Contextual/narrative aspects are assumed to be less affected, preserving voluntary access.

Possible strengths:

- Ties behavioural effects to a broader neurobiological literature on reconsolidation and updating, providing a principled role for reminders and precise timing.
- Offers a way to explain effects of delayed reminder+Tetris protocols that occur after initial consolidation.

Possible shortcomings:

- Evidence that these human paradigms genuinely tap reconsolidation (rather than standard re-encoding in a new context) is indirect, and the computational story remains largely verbal.
- Reconsolidation windows and boundary conditions are debated, and several patterns (including mixed replication success) can plausibly be captured by simpler context-binding and competition mechanisms.

### Retrieved-context theory / CMR3 (single-system alternative)

Representative paper:
Cohen, R. T., & Kahana, M. J. (2022). A memory-based theory of emotional disorders. Psychological Review.

Core idea:
Episodic memory is supported by a single system in which items are bound to a drifting context that includes temporal, semantic, and emotional features. Intrusions occur when current context overlaps with trauma-related states and trauma items win a context→item competition. Voluntary recall and recognition depend on how context is seeded and controlled (for free recall) or on item→context retrieval (for recognition), and post-film tasks like Tetris add competitor episodes in nearby contexts.

Possible strengths:

- Provides a fully specified computational model that already accounts for wide-ranging memory phenomena and incorporates valence/arousal, allowing explicit simulations rather than purely verbal theorising.
- Explains intrusive vs voluntary dissociations via retrieval mode and control within a single system, without needing separate traces or a dedicated visuospatial module.

Possible shortcomings:

- The model is abstract and parameter-rich; without clear constraints there is a risk of flexibility, and mapping to neural implementation is non-trivial.
- CMR3 has not yet been applied directly to address unintentional recall or trauma-film/Tetris paradigms, so this project would be the first serious test of whether its mechanisms can capture the involuntary/voluntary distinction, selective interference findings, and applicable boundary conditions.

## The retrieved-context / CMR3 framework

Goal: provide a more detailed presentation of the CMR3 model (including specification details) and how it can be applied to the selective interference phenomenon.

### Overview and Basic Architecture

Briefly introduce the retrieved-context family (CMR / eCMR / CMR3) as models in which:

- Slowly drifting context vectors with temporal, semantic, and emotional components.
- Encoding of context→item and item→context associations.
- Retrieval dynamics: context→item competition; recalled items retrieving associated context and updating the current state.

Emphasise that CMR3 extends earlier CMR/eCMR models by treating valence and arousal as context dimensions and by strengthening high-arousal encoding.

Possible Figure 2: Schematic of context state, item vectors, and bidirectional associations, with emotional features highlighted.

### Retrieval modes: context→item vs item→context

Define two generic retrieval modes in the model:

- Context→item: given a context cue, items compete based on their learned context→item strengths.
- Item→context: given a probe item, its associated context is retrieved via item→context links and compared to a target context (or used to drive decisions).

Map these abstract modes to classes of tasks:

- Unguided intrusions and free recall are primarily context→item.
- Recognition and similar probe-based judgements rely heavily on item→context, using the probe as the entry point.

Make clear that these are not separate systems; they are different ways of using the same associative structure.

### Control over retrieval: starting context and output evaluation

Interprets intrusions in VIT and diary paradigms as context→item retrievals driven by drifting context rather than top-down control. Intrusions occur when the naturally drifting context enters regions strongly associated with trauma-film items and those items cross threshold. 

Distinguishes unguided context→item retrieval from voluntary free recall. Free recall is also context→item but includes:

- **Starting-context bias**. the initial context at retrieval can be seeded toward start-of-film states via instructions/cues, increasing the likelihood of retrieving film items.
- **Output evaluation / gating**. retrieved items can be evaluated against current goals (e.g., how well their retrieved context matches a target) and either accepted or rejected; rejected items have weaker impact on the evolving context.

Unguided intrusions correspond to minimal use of these control levers (spontaneous context drift, little gating). Intentional free recall corresponds to stronger starting-context control and stronger gating. Recognition uses item→context plus a decision rule, with relatively little reliance on context-driven competition among many items.

We will later instantiate these ideas in specific parameter choices (e.g., strength of starting-context bias; thresholds for gating), but the conceptual distinction belongs here.

### Emotional context and intrusive memory

Explain how CMR3’s emotional context dimensions are relevant:

- High-arousal events (e.g., trauma-film hotspots) are encoded more strongly and pull context into characteristic regions of "emotional space."
- Because context includes emotional features, later states that share arousal/valence components will preferentially cue those trauma items.

On this view, intrusions are simply context→item retrievals when everyday context wanders into these trauma-associated regions. Post-film tasks can create competitor items in this same region if they are encoded in overlapping contexts (temporal, situational, and possibly emotional).

## Mapping the trauma-film/Tetris paradigm into CMR3

### Representational mapping

Trauma film. Modelled as a sequence of high-arousal negative items; strong context–item bindings; progression through a specific temporal/emotional context trajectory.

Post-film tasks:

- Tetris (or similar fast-paced, segmented task) modelled as a sequence of neutral items encoded in contexts adjacent in time (and, depending on assumptions, partially overlapping in arousal/engagement) to the film context.
- Verbal or low-engagement controls represented as far fewer or weaker items; more opportunity for trauma rehearsal during that interval.

Intrusion measures:

- Lab: contexts constrained by task instructions (e.g., vigilance-intrusion tasks with trauma vs foil cues).
- Diary: more variable naturalistic context trajectories.

### The context-binding account of selective interference

Post-film items, encoded in context states adjacent to trauma-film states, become additional possible responses to any context cue in that neighbourhood.

As the system revisits that region (in everyday life or in lab tasks), trauma-film items must now compete with a larger set of strong competitors.

As a result, the probability that a trauma item crosses threshold for context-driven retrieval (an intrusion) is reduced.

However, the item representations and their associations are intact, so item-cued recognition and well-controlled free recall can still access trauma items when specifically probed.

### Why post-film task properties matter

Competitor density and strength. High event segmentation and engagement (e.g., many Tetris moves/rounds) lead to many strongly encoded post-film items; more competition.

Rehearsal suppression. Engaging tasks reduce spontaneous reactivation of the trauma film during the intervention window.

Emotional/context overlap. If we assume that a reminder reinstates a high-arousal trauma context into which Tetris items are then bound, this increases emotional-context overlap; alternatively, we can treat temporal/situational overlap as the primary driver and arousal as a modulator.


## Simulations

Goal: minimal but diagnostic simulations that instantiate the above ideas and generate testable predictions.

### Simulation 1. Benchmark: post-list interference in standard recall vs recognition

Aim: Show that even in a neutral list-learning setting, CMR3 predicts stronger interference on free recall than on recognition when a post-list distractor list is added.

Design sketch

Encode List A (target items).

Two conditions:

- A-only.
- A followed by List B (post-list distractors).

Test:

- Free recall of List A from an end-of-list context cue.
- Recognition of List A vs new foils, using item→context retrieval.

Metrics:

- Proportion of A items recalled in free recall.
- Recognition accuracy (hits – false alarms) for A items.

Expected pattern:

- Adding List B reduces recall of A (context competition).
- Recognition of A is minimally affected.

This benchmark demonstrates that mode-dependent interference is intrinsic to the model, independent of trauma/Tetris specifics.

### Simulation 2. Selective interference in a trauma-film analogue

Aim. Capture the canonical pattern: post-film Tetris-like condition reduces intrusive memories while leaving recognition largely intact, in a trauma-film–inspired setup.


Phase 1: "Film" encoding. Sequence of high-arousal negative items; strong context–item binding.

Phase 2: Post-film manipulation.

- Condition Tetris: reminder cue followed by many strongly encoded neutral items (high segmentation, high engagement).
- Condition Control: reminder cue but few/weak neutral items, or none.
- Optionally: Condition Verbal: intermediate competitor density and engagement.

Phase 3: Testing

- Intrusions. Simulate context drift over a series of "days" or "probes". Count spontaneous retrievals of film items crossing threshold under: (A) lab-like conditions (e.g., fixed context with occasional trauma/foil cues), (B) diary-like conditions (more variable context trajectory).

- Recognition: Item-cued recognition of film items vs foils.

Metrics:

- Intrusion frequency over simulated time (per condition).
- Recognition accuracy (per condition).
- Optional: voluntary free recall performance.

Expected pattern:

- Condition Tetris shows lower intrusion rates than Control, especially in contexts overlapping with post-film context.
- Recognition differs little between conditions relative to intrusions.
- Depending on parameterisation, voluntary free recall should also show limited interference, anticipating Simulation 3.

---

### Simulation 3. Intentional vs unintentional context→item retrieval (control mechanisms)

Aim: Explain how intrusions can be reduced while voluntary free recall is spared, using control over starting context and output gating.

Use the same film + post-film conditions from Simulation 2.

Define two retrieval regimes:

Unguided intrusions:

- Context drifts autonomously.
- No strong control over starting state or gating.
- Intrusions defined as trauma-film items exceeding threshold.

Intentional free recall:

- Starting context biased toward the film session (e.g., toward early-film context).
- Strong output gating: if a retrieved item’s associated context mismatches the target (film), it is rejected and its context is integrated weakly.

Additional manipulation:

- Control strength as a parameter
- High control: strong bias + strong gating.
- Low control: weaker bias + weaker gating (e.g., mimicking divided attention or time pressure).

Metrics

- Intrusion rates (unguided regime).
- Recall probability for film items (intentional regime) across control levels.
- Degree to which post-film items intrude into intentional recall.

Expected pattern

With high control: Post-film Tetris reduces intrusions and spares voluntary free recall.

With reduced control: voluntary free recall begins to show interference effects similar to intrusions (lower recall in Tetris vs Control), generating concrete experimental predictions.

### Simulation 4. Emotional vs neutral material, and the arousal-subspace claim

Aim: Explore how emotional features (valence/arousal) modulate interference in CMR3, and to what extent post-film task arousal/emotionality is needed to produce selective effects on emotional material.

Encode mixed episodes containing high-arousal negative items (trauma-like) and neutral items.

Post-film conditions:

- High-arousal competitors (e.g., neutral content but encoded under high arousal / engagement / reinstated trauma context).
- Low-arousal competitors (low engagement; minimal arousal change).

Test intrusions, voluntary recall, and recognition separately for high-arousal vs neutral items.

Metrics

- Intrusion reduction for high-arousal vs neutral film items across conditions.
- Voluntary recall/recognition for each item type.

Expected pattern:

- When competitors are high-arousal and temporally close, interference effects on intrusions are stronger for high-arousal items than for neutral ones.
- When competitors are low-arousal, interference is less selectively emotional.
- This simulation will allow us to turn the "post-film traces are in a high-arousal region" idea into specific, testable constraints.

### Simulation 5 (optional). Reminder + delayed Tetris / reconsolidation-style designs

Aim: Show that reminder+Tetris effects at delays (e.g., 24h, 3 days) can arise via context reinstatement and new competitor encoding, without invoking special reconsolidation mechanisms.

- Phase 1: Film encoding.
- Phase 2: Delay (context drifts away).
- Phase 3: Reminder cue reinstating trauma context, followed by Tetris-like competitor encoding vs control.
- Phase 4: Intrusion measures over subsequent time.

Metrics:

- Intrusion frequency as a function of post-reminder condition.
- Any effects on voluntary measures.

Expected pattern

- When reminder successfully reinstates trauma context and is followed by competitor encoding, additional intrusion reduction emerges relative to reminder-only control.
- This provides a single-system interpretation of some "reconsolidation-like" findings.

## Synthesis, implications, limitations

### Comparing accounts: who predicts what?

Here systematically contrast:

- Visuospatial WM/consolidation,
- dual representation / separate trace,
- reconsolidation accounts,
- CMR3.

Organize around phenomena:

- Intrusions vs recognition/free recall,
- effect of associative cueing,
- timing of intervention (immediate vs delayed),
- effects on emotional vs neutral content,
- immediate vs long-term diary intrusions,
- dependence on event segmentation / competitor density.

A clear table with "predicted / not predicted / flexible" entries will be useful.

### Implications for intervention design

If the account is right, Tetris is one way to create dense competitor traces; not uniquely privileged. Imagery rescripting, other visuospatial tasks, or even semantic tasks with high segmentation could also work, with different trade-offs.

We can re-interpret mixed results and meta-analytic heterogeneity in terms of:
  - insufficient competitor density,
  - poor context overlap,
  - or ineffective rehearsal suppression.

This feeds into design principles: how to engineer post-trauma experiences that safely increase competition in the trauma neighbourhood.

### Section 9. Limitations and open questions

Here we'll be frank. Some possible observations:

CMR3 is still highly abstract; parameters and context dimensions are not trivially tied to specific neural substrates.

We are fitting patterns, not individual datasets; a full data-fitting exercise would be ideal but is beyond the scope of a single paper.

The empirical base for Tetris/selective interference is not as robust as once thought; large-sample replications are mixed.

Some key predictions (e.g., about arousal-mediating effects, control manipulations) are currently untested.