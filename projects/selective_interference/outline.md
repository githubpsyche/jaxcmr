# Intrusive and Voluntary Memory in a Single-System Framework: A Retrieved-Context Account of Selective Interference

This document outlines a proposed Psych Review–style theoretical paper and accompanying simulation program. In the proposed paper, we would:

- Treat the selective interference effect (post-film Tetris reduces intrusions while sparing voluntary memory) as the central explanatory target.
- Show how this pattern can be derived from a single-system retrieved-context model (CMR3) without invoking separate traces or specialised visuospatial consolidation mechanisms.
- Use simulations to generate mechanistic predictions and boundary conditions that can explain both positive and null findings in the trauma-film/Tetris literature.
- Situate this account within broader work on contextual binding, intrusive memories, and emotional disorders.

We would advance three main claims:

1. Single-System Sufficiency. A single, context-binding episodic memory system (as implemented in CMR3) is sufficient to reproduce the key selective interference pattern: intrusions decrease after certain post-film tasks, while recognition and intentional free recall are relatively spared. This does not require separate "intrusive" and "voluntary" memory traces or a dedicated visuospatial consolidation mechanism.
2. Retrieval-mode + control as key determinants. The critical dissociations arise from how retrieval is driven and controlled. Intrusions are largely unguided context→item retrievals as everyday context drifts near trauma-related states. When post-film items are bound into that region, they compete with trauma-film items and make intrusions less likely. Recognition and similar probe-based tasks are largely item→context. The probe retrieves its own associated context, so additional post-film competitors do not get a chance to "win" against the probe in the same way. Intentional/guided free recall is also context→item but benefits from control mechanisms (starting-context steering and output gating) that can shield voluntary recall from competition effects that reduce intrusions.
3. Re-interpretation of Tetris/visuospatial interventions. Tetris and related tasks can be interpreted as interventions that generate dense competitor episodes that a) are bound into contexts near the trauma film,  b) suppress rehearsal of trauma items during the interval, and c) may share arousal/engagement properties with the film's most salient moments. This reframes the role of visuospatial properties: they matter insofar as they drive segmentation and engagement and help populate the relevant context region, not because they uniquely disrupt sensory trace consolidation.

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

Goal: introduce the single-system framework we will use, at a conceptual level, and prepare the ground for both the mapping to the trauma-film paradigm and the later formal model specification.

### Core architecture (conceptual)

We will briefly situate CMR3 within the retrieved-context family (CMR / eCMR / CMR3). The key ingredients are:

- A slowly drifting context state that integrates temporal, semantic, and emotional features over time.
- Item–context bindings learned in two directions: context→item (context cues items) and item→context (items can retrieve their associated context).
- Retrieval as competition under a context cue, with recalled items feeding back to update context.

We will emphasise that CMR3 specifically extends earlier models by treating valence and arousal as explicit context dimensions and by allowing high-arousal items to be encoded more strongly and to pull context into similar "emotional regions."

A schematic figure here (context vector, item vectors, bidirectional links, emotional subspace highlighted) should be enough; the full equations and parameter table will be deferred to a Box/Appx.

### Retrieval modes: context→item and item→context

We then define the two ways the model can be "read out":

- In context→item mode, a context cue activates items according to their context→item strengths and they compete for retrieval.
- In item→context mode, a probe item retrieves its associated context via item→context links, and that retrieved context is compared to a target context or used to drive a decision.

At this point we map these modes to broad task classes:

- Unguided intrusions and free recall are primarily context→item.
- Recognition and similar probe-based judgements rely mainly on item→context.

We stress that these are not separate systems; they are different uses of a single associative architecture. This section is where we unpack the "retrieval-mode" part of Claim 2 once and for all.

### Retrieval control: starting context and output evaluation

Next we introduce the two control "levers" the model allows, without repeating all trauma-film specifics:

- Starting-context bias: retrieval can be initialised from a context state biased toward a particular episode or time period (e.g., the film session vs a later lab session).
- Output evaluation / gating: when an item wins the competition, its retrieved context can be compared to the current goal; items with poor match can be censored or have their context only weakly integrated, limiting drift away from the target context.

We link these in abstract terms to different kinds of behaviour:

- Unguided intrusions: minimal control; context drifts with ongoing experience, little or no gating.
- Intentional free recall: strong starting-context bias toward the film and strong gating against off-target items.
- Recognition: primarily item→context plus a decision rule; less dependence on context-driven competition among many items.

We will later instantiate these control settings in the simulations (e.g., high vs low control parameters) rather than here. This subsection completes the "control" part of Claim 2.

### Emotional context and intrusive memory

Finally in this section we note how CMR3’s emotional context features are relevant for intrusive memories:

- High-arousal negative items (e.g., trauma-film hotspots) are encoded more strongly and cluster in a characteristic region of context space.
- Later states that share these emotional/contextual features are more likely to cue those items in context→item retrieval, providing a natural route to intrusions.

On this view, intrusions are not a separate system: they are context→item retrievals when everyday context wanders into trauma-associated regions. Post-film tasks can create competitor items in these regions if they are encoded under overlapping temporal/situational (and possibly emotional) contexts.

We end the section by signalling that:

- Box/Appx 1 will provide the full mathematical specification of CMR3 as used here (state vectors, learning rules, parameters), including any deviations from published implementations.
- The next section will apply this conceptual framework to the trauma-film/Tetris paradigm.

## Mapping the trauma-film/Tetris paradigm into CMR3

Goal: specify how we represent the trauma-film experiments within CMR3 (items, contexts, tasks) and derive a verbal version of the selective interference account before turning to simulations.

### Representational mapping

We first map components of the paradigm onto model constructs:

- Trauma film: modelled as a sequence of high-arousal negative items, encoded with strong item–context bindings as context drifts through a characteristic temporal/emotional trajectory.
- Post-film tasks:

  - Tetris-like condition: many neutral items, high engagement/segmentation, encoded in contexts adjacent in time (and, depending on assumptions, overlapping in arousal/engagement) with late-film contexts.
  - Control conditions: fewer/weaker items and more opportunity for spontaneous trauma-film rehearsal during the same interval.
- Intrusion measures:

  - Lab tasks (e.g., vigilance–intrusion tasks, cue-provocation tasks): context constrained by instructions and occasional trauma/foil cues.
  - Diary measures: context follows more variable, naturalistic trajectories.

This is purely a mapping step; we do not yet discuss competition or interference here.

### The context-binding account of selective interference

We then state the core mechanism compactly:

- Post-film items are bound in context states adjacent to trauma-film states, so they become additional candidates whenever the system revisits that region in context→item mode.
- As those contexts are later revisited, trauma-film items must compete with a denser set of strongly encoded post-film competitors; they therefore win less often, reducing intrusion probability.
- The underlying item representations and their associations remain intact; when a film item is used as a probe (recognition) or when retrieval is strongly steered toward the film (intentional free recall with control), trauma items can still be accessed.

This subsection is where we cash out Claim 1 ("single-system sufficiency") in verbal CMR3 terms.

### Why post-film task properties matter

We close the mapping section by unpacking which properties of the post-film interval matter, in a way that directly foreshadows the simulations:

- Competitor density and strength: tasks that generate many, strongly encoded episodes in the relevant context region (high segmentation and engagement) produce more competition.
- Rehearsal suppression: strongly engaging tasks reduce spontaneous trauma-film reactivation during the post-film interval, preventing further strengthening of trauma–context bindings.
- Context overlap: reminder cues and shared situational/emotional features can ensure that post-film items are indeed encoded into the trauma-adjacent context region, which modulates the size and selectivity of the interference effect.

This sets up the logical link to the simulation programme, which will instantiate these properties as manipulable parameters (event count/strength, rehearsal vs no rehearsal, degree of context overlap).


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