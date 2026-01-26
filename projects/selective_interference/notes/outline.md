# Intrusive and Voluntary Memory in a Single-System Framework: A Retrieved-Context Account of Selective Interference

This document outlines a proposed Psych Review–style theoretical paper and accompanying simulation program. In the proposed paper, we would:

- Treat the selective interference effect (post-film Tetris reduces intrusions while sparing voluntary memory) as the central explanatory target.
- Show how this pattern can be derived from a single-system retrieved-context model (CMR3) without invoking separate traces or specialised visuospatial consolidation mechanisms.
- Use simulations to generate mechanistic predictions and boundary conditions that can explain both positive and null findings in the trauma-film/Tetris literature.
- Situate this account within broader work on contextual binding, intrusive memories, and emotional disorders.

We would advance three main claims:

1. **Single-system sufficiency.**  
   A single, context-binding episodic memory system (as implemented in CMR3) is sufficient to reproduce the selective interference effect: intrusions decrease after certain post-film tasks, while recognition and intentional free recall are relatively spared. This does not require separate "intrusive" and "voluntary" memory traces or a dedicated visuospatial consolidation mechanism.

2. **Retrieval mode and control as key determinants.**  
   The critical dissociations arise from how retrieval is driven and controlled. Unguided intrusions are primarily context→item retrievals as everyday context drifts near trauma-related states. Intentional free recall is also context→item but starts from a deliberately biased context and uses output gating. Recognition and similar probe-based tasks rely heavily on item→context retrieval and probe-based decision rules. Differences in retrieval mode (context→item vs item→context) and in the strength of control (starting-context bias and gating) explain why some tasks are more susceptible to post-film competition than others.

3. **Reinterpretation of Tetris/visuospatial interventions.**  
   Tetris and related tasks can be interpreted as interventions that generate dense competitor episodes that (a) are bound into contexts near the trauma film, (b) suppress rehearsal of trauma items during the interval, and (c) may share arousal/engagement properties with the film's most salient moments. This reframes the role of visuospatial properties: they matter insofar as they drive segmentation, engagement, and context overlap, not because they uniquely disrupt sensory trace consolidation.

The remainder of the document expands these claims into a section-by-section manuscript plan and a concrete CMR3 simulation program.

## Introduction: The selective interference puzzle

Goal: define the empirical target and the theoretical puzzle succinctly.

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

Possible Figure 1: schematic of the phenomenon (intrusions vs voluntary memory; Tetris vs control).

## Empirical landscape

Goal: compactly organise the main empirical constraints the model must accommodate.

### Core lab studies

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

We will use this to frame the empirical target as heterogeneous: the effect exists under some conditions, is weaker or fails under others. This motivates a mechanistic model that can explain when and why selective interference occurs instead of assuming a simple law.

Possible Table 1: summary of a representative set of studies and key variables we will address (rows: study; columns: timing, task type, intrusion measure, voluntary measure, main effect).

## Existing theoretical accounts

Goal: provide a fair but critical review of dominant accounts.

### Visuospatial working-memory interference / consolidation accounts

Representative paper:  
Holmes, E. A., James, E. L., Coode-Bate, T., & Deeprose, C. (2009). Can playing the computer game "Tetris" reduce the build-up of flashbacks for trauma? PLoS ONE.

Core idea:  
Post-film visuospatial tasks load visuospatial working memory during a critical window, disrupting consolidation of sensory-perceptual aspects of the trauma memory that support later intrusive imagery. More abstract/narrative aspects rely less on visuospatial resources, so voluntary recall and recognition are relatively spared.

Possible strengths:

- Straightforward cognitive mechanism grounded in dual-task interference and WM resource theories.
- Naturally explains why visuospatial tasks often outperform purely verbal tasks at reducing intrusive imagery.

Possible shortcomings:

- The route from transient WM load to long-term episodic consolidation is underspecified and not formalised in a retrieval model, limiting task-level predictions.
- Does not, by itself, explain spared recognition/free recall or heterogeneous findings (e.g., mixed diary effects, some non-visuospatial interventions) without additional assumptions.

### Dual-representation / separate trace theories

Representative paper:  
Brewin, C. R., Gregory, J. D., Lipton, M., & Burgess, N. (2010). Intrusive images in psychological disorders: Characteristics, neural mechanisms, and treatment implications. Psychological Review.

Core idea:  
Trauma creates two partly separable representations: sensory-bound, poorly contextualised traces that drive flashback-type intrusions, and contextualised, verbally accessible traces that support narrative memory. Selective interference is attributed to interventions preferentially weakening the sensory-bound trace while leaving contextualised/narrative memory relatively intact.

Possible strengths:

- Fits clinical phenomenology of vivid, sensory intrusions coexisting with more coherent narratives, and offers a clear conceptual dissociation between intrusive and voluntary memory.
- Influential in framing imagery-focused treatments and hypotheses about selective effects on "flashback" memory.

Possible shortcomings:

- The two-trace architecture is rarely embedded in a detailed retrieval model; it is unclear how the traces interact during tasks such as free recall, recognition, and cue-driven intrusion paradigms.
- Explanations can become post hoc: often unclear which phenomena are attributed to which trace and how an intervention like Tetris acts at the representational level.

### Reconsolidation-based accounts

Representative paper:  
James, E. L., Bonsall, M. B., Hoppitt, L., Tunbridge, E. M., Geddes, J. R., Milton, A. L., & Holmes, E. A. (2015). Computer game play reduces intrusive memories of experimental trauma via reconsolidation-update mechanisms. Psychological Science.

Core idea:  
A reminder cue reactivates the trauma memory into a labile reconsolidation state; engaging in Tetris during this window introduces competing perceptual information that disrupts restabilisation of sensory elements that underpin intrusions. Contextual/narrative aspects are assumed to be less affected, preserving voluntary access.

Possible strengths:

- Ties behavioural effects to a broader neurobiological literature on reconsolidation and updating, providing a principled role for reminders and timing.
- Offers a way to explain effects of delayed reminder+Tetris protocols that occur after initial consolidation.

Possible shortcomings:

- Evidence that these human paradigms genuinely tap reconsolidation (rather than standard re-encoding in a new context) is indirect, and the computational story remains largely verbal.
- Reconsolidation windows and boundary conditions are debated, and several patterns (including mixed replication success) can plausibly be captured by simpler context-binding and competition mechanisms.

### Retrieved-context theory / CMR3 as a single-system alternative

Representative paper:  
Cohen, R. T., & Kahana, M. J. (2022). A memory-based theory of emotional disorders. Psychological Review.

Core idea:  
Episodic memory is supported by a single system in which items are bound to a drifting context that includes temporal, semantic, and emotional features. Intrusions occur when current context overlaps with trauma-related states and trauma items win a context→item competition. Voluntary recall and recognition depend on how context is seeded and controlled (for free recall) or on item→context retrieval (for recognition); post-film tasks like Tetris add competitor episodes in nearby contexts.

Possible strengths:

- Fully specified computational model that already accounts for wide-ranging memory phenomena and incorporates valence/arousal, allowing explicit simulations rather than purely verbal theorising.
- Explains intrusive vs voluntary dissociations via retrieval mode and control within a single system, without separate traces or a dedicated visuospatial module.

Possible shortcomings:

- Abstract and parameter-rich; without clear constraints there is a risk of flexibility, and mapping to neural implementation is non-trivial.
- CMR3 has not yet been applied directly to unintentional recall or trauma-film/Tetris paradigms; this project would be the first serious test of whether its mechanisms can capture the involuntary/voluntary distinction, selective interference findings, and their boundary conditions.

Transition: we next outline the CMR3 framework at a conceptual and formal level, focusing on the aspects most relevant to selective interference (context-binding, retrieval modes, and control), before mapping the trauma-film/Tetris paradigm into this framework and presenting simulations.

## The retrieved-context / CMR3 framework

Goal: introduce the single-system framework we will use, first conceptually and then in formal detail, and make explicit how it instantiates the ingredients in Claims 1–2.

### Conceptual overview and link to the main claims

- Retrieved-context models (CMR, eCMR, CMR3) treat episodic memory as a system in which:
  - Items are bound to a slowly drifting context representation.
  - Context includes temporal, semantic, and emotional (valence, arousal) features.
  - Retrieval is governed by learned context–item and item–context associations.
- This single architecture will underlie all intrusive and voluntary phenomena we model, directly instantiating Claim 1 (single-system sufficiency).

### Formal model specification: state representations and encoding

Plan: present the core CMR3 machinery in the main text, but focus on the pieces that matter for selective interference.

- Define notation:
  - Item feature vectors (e.g., $f$): include semantic, source, and emotional features.
  - Context vectors (e.g., $c$): high-dimensional vectors that drift over time.
  - Associative matrices:
    - $M_{CF}$ (context→item or context→features).
    - $M_{FC}$ (item→context).
- Encoding dynamics:
  - Specify the context update rule (e.g., $c_t$ as a recency-weighted blend of previous context and current item features, with separate scaling for emotional vs non-emotional components).
  - Specify how $M_{CF}$ and $M_{FC}$ are updated at each study event (Hebbian-style learning with learning rates; stronger learning for high-arousal items).
- Highlight which parameters will later matter:
  - Context drift rate.
  - Learning rates for neutral vs high-arousal items.
  - Relative weighting of temporal vs emotional features in context.

The goal is to give enough formal detail that readers can see exactly how trauma-film events and post-film tasks are encoded into the same context space.

### Formal model specification: retrieval dynamics and control

Plan: present the retrieval equations and explicit control parameters.

- Retrieval modes:
  - Context→item:
    - Given a context state $c$, activation of each item is a function of $M_{CF}$ and $c$ (e.g., $a = M_{CF} \cdot c$).
    - Items compete for recall via a standard choice rule (e.g., Luce or softmax).
    - Retrieved item feeds back via $M_{FC}$ to update context (e.g., $c' = $ blend of old context and Item→context(cue)).
  - Item→context:
    - Given a probe item, retrieve its associated context via $M_{FC}$.
    - Compare retrieved context to a target context (e.g., via dot product or similarity measure) to drive recognition decisions.
- Control parameters:
  - Starting-context bias:
    - A parameter controlling how strongly initial $c$ at retrieval is biased toward a specified context (e.g., the film session vs later lab context).
  - Output gating:
    - A similarity threshold or weighting that determines how strongly retrieved items update context, and whether they are accepted as outputs, as a function of match between retrieved context and a goal context.
  - Stopping and threshold rules:
    - Parameters controlling when retrieval stops (e.g., global threshold on activation or probability of recall) will be specified, but kept simple across simulations.

Here we can include a small set of core equations, with prose guiding readers through what each component does. The aim is not to reproduce the entire CMR3 paper, but to make the key operations visible and to show which parameters we will manipulate or hold fixed in the simulations.

### Retrieval modes in task terms

Once the formal pieces are defined, we explicitly map them to behavioural regimes:

- Unguided intrusions:
  - Context→item retrieval.
  - Starting context follows ongoing experience or a generic lab context.
  - Control parameters: low starting-context bias; minimal gating.
- Intentional free recall:
  - Context→item retrieval.
  - Starting context strongly biased toward the film session.
  - Control parameters: strong gating based on match to film context.
- Recognition:
  - Item→context retrieval plus similarity-based decision rule.
  - Minimal use of context-driven competition among many items; control parameters matter less.

This section locks in the retrieval-mode and control vocabulary that later sections will reference without redefinition.

## Mapping the trauma-film/Tetris paradigm into CMR3

Goal: specify how we represent the trauma-film experiments within CMR3 (items, contexts, tasks), both verbally and in terms of the formal machinery defined above, and derive a verbal version of the selective interference account before turning to simulations.

Opening: having outlined the CMR3 architecture and equations, we now specify how the trauma-film/Tetris paradigm maps into this framework.

### Representational mapping (verbal)

- Trauma film:
  - Modelled as a sequence of high-arousal negative items $f_1, f_2, ..., f_N$, each with strong emotional components.
  - Context $c_t$ drifts along a distinctive trajectory during film viewing, with emotional components elevated.
- Post-film tasks:
  - Tetris-like condition:
    - Many neutral items $g_1, g_2, ..., g_M$ (moves/trials), encoded in contexts $c_{N+1}...c_{N+M}$ that are temporally adjacent to late-film contexts and may share arousal/engagement features (depending on assumptions).
  - Control conditions:
    - Fewer/weaker items and more opportunity for unstructured trauma rehearsal during the same interval.
- Intrusion measures:
  - Lab tasks: retrieval contexts $c_{test}$ are constrained by instructions and occasional trauma/foil cues.
  - Diary measures: $c_{test}$ follows more variable, naturalistic trajectories.

### Representational mapping (formal sketch)

Plan: tie the above to specific model objects.

- Define study phases:
  - Film phase: sequence of updates to $c$ and $M_{CF}, M_{FC}$ for each $f_i$ with high-arousal learning parameters.
  - Post-film phase:
    - Tetris condition: sequence of updates for $g_j$ with neutral item features but with context states $c_t$ that inherit from late film and/or reinstated trauma context.
    - Control condition: fewer such updates or different context states (e.g., more drift away).
- Define test phases:
  - Intrusion context trajectories:
    - For lab-based intrusions (e.g., vigilance–intrusion task), define $c_{test}$ as a random walk around a lab context, occasionally perturbed by trauma or foil cues via $M_{FC}$.
    - For diaries, define $c_{test}$ trajectories that mix lab-like and everyday-like contexts.

This subsection specifies, at a high level, which equations are used in which phase, and how film vs post-film items are distinguished only by their features and context states, not by separate stores.

### The context-binding account of selective interference

Mechanism, now explicitly grounded in the formalism:

- During film encoding, $f_i$ items acquire strong context→item links in states $c_i$ with high emotional components.
- In the Tetris condition, $g_j$ items are also encoded into $c_t$ states that are near $c_N$ in context space, so $M_{CF}$ now contains many strong links from trauma-adjacent contexts to $g_j$ as well as $f_i$.
- During later context→item retrieval (intrusions), when $c_{test}$ drifts into the trauma-adjacent region:
  - Activation $a = M_{CF} \cdot c_{test}$ includes both $f_i$ and $g_j$.
  - With more strong competitors $g_j$, trauma items $f_i$ are less likely to win the competition and cross threshold, reducing intrusions.
- Item representations and $M_{FC}$ remain intact:
  - In item→context recognition, probe items $f_i$ retrieve their own associated contexts via $M_{FC}$; $g_j$ do not compete at the same stage in the process.
  - In intentional free recall, starting-context bias and gating keep $c_{test}$ closer to film contexts and reduce the impact of off-target $g_j$ on context drift.

This is where we explicitly cash out Claim 1 in terms of the model's equations and operations.

### Why post-film task properties matter (formal hooks for simulations)

We connect task-level properties to model parameters that will be manipulated in simulations:

- Competitor density and strength:
  - Number of $g_j$ items and their learning rate parameters determine how many strong post-film competitors are linked to trauma-adjacent context states.
- Rehearsal suppression:
  - Probability of re-presenting or internally "replaying" $f_i$ during the post-film interval (which would further strengthen their $M_{CF}$ and $M_{FC}$ links) is reduced in the Tetris condition and higher in some control conditions.
- Context overlap:
  - Degree to which post-film context states $c_{N+1...}$ remain close to $c_N$ in context space depends on:
    - whether a reminder reinstates trauma context before Tetris,
    - the drift rate parameters,
    - how strongly emotional and situational features are carried over.
  - These factors will be implemented as explicit parameter differences between conditions.

These links give the simulation section clear levers to pull, without needing to re-explain the theory.

## Simulations

Aim: specify a small set of simulations that directly test the three main claims, using the formal CMR3 specification laid out earlier. Each simulation should be simple, map onto well-defined empirical contrasts, and reuse as many parameters as possible.

### Simulation 1: Benchmark list A/B interference

Aim. Show that, in a neutral list-learning setting, adding a post-list distractor list reduces free recall of List A much more than recognition of List A, purely as a function of context-based competition. This establishes that mode-dependent interference is an intrinsic property of the model.

Design sketch.

- Study phase:
  - List A: N items, neutral, encoded with standard (neutral) learning rates.
  - Condition A-only: no further encoding.
  - Condition A+B: immediately study List B (N neutral items) after A.
- Test phase:
  - Free recall of List A:
    - Initial context c_test seeded to the end-of-List-A context (in A-only) or end-of-List-B context (in A+B), per the standard CMR3 implementation.
    - Retrieval in context→item mode using the equations already specified.
  - Recognition of List A:
    - Mixed list of A-items and new foils.
    - Recognition decisions based on item→context retrieval and similarity to the List-A context.

Key parameters.

- Same drift, learning, and decision parameters as the published CMR3 benchmark fits, except for any minimal changes we commit to globally.
- No emotional features used here; emotional components of f and c are set to zero.

Outcome measures.

- Proportion of A items recalled in free recall (A-only vs A+B).
- Recognition accuracy for A items (hits minus false alarms) across conditions.

Target pattern.

- Robust reduction in free recall of A when List B is added, driven by context competition.
- Much smaller effect (or no effect) on recognition of A, because item→context retrieval is relatively insensitive to extra context→item competitors.

Role in the paper.

- Demonstrates that the free recall vs recognition asymmetry is a generic property of context-based models, not a special feature of trauma-film paradigms.
- Provides a neutral benchmark and a sanity check on our CMR3 implementation.

### Simulation 2: Trauma-film analogue and selective interference

Aim. Show that, when a "film" phase is followed by a high-density post-film encoding phase (Tetris-like) vs a low-density control phase, the model produces a selective reduction in intrusions with relatively spared recognition.

Design sketch.

- Film phase:
  - Encode a sequence of N "film" items $f_1,...,f_N$ with high-arousal negative features.
  - Use the emotional-learning parameters so that these items have stronger context–item associations and pull the context into a high-arousal region.

- Post-film manipulation:
  - Condition Tetris:
    - Present M neutral items $g_1,...,g_M$ in quick succession.
    - Context $c_t$ at the start of this phase is set to the late-film context (possibly after a reminder cue), so that $g_j$ are encoded into trauma-adjacent contexts.

  - Condition Control:
    - Either no new items, or a very small number of weakly encoded neutral items, with more time for possible rehearsal of film items.
  - Optional condition Verbal:
    - Fewer post-film items with semantic but not visuospatial features, allowing us to manipulate competitor density vs modality.

- Test phase:
  - Intrusions in a lab-like context:
    - Define a test context $c_{test}$ that is a mixture of lab context and occasional cue-driven reinstatement of film context (via $M_{FC}$ and a film- or foil-cue manipulation).
    - Run context→item retrieval without strong control (low starting-context bias, minimal gating).
    - Count the number of $f_i$ intrusions over many simulated "trials".

  - Intrusions in diary-like contexts:
    - Define a sequence of $c_{test}$ states that drift over a more heterogeneous trajectory.
    - Again run context→item retrieval under low control and count $f_i$ intrusions over simulated "days".

  - Recognition:
    - Present film items and foils as probes.
    - Use item→context retrieval and a simple similarity-based decision rule (as in the formal spec).

Key parameters.

- $M$ (number of post-film items), and their learning strength in the Tetris condition vs Control.
- Degree of context overlap between film and post-film phases (controlled by the reminder and drift parameters).
- Threshold and stopping rules for intrusions (kept fixed across conditions).

Outcome measures.

- Intrusion frequency (lab-like and diary-like) in Tetris vs Control vs Verbal.
- Recognition accuracy for film items in each condition.
- Optional: free recall of film items in a high-control regime (see Simulation 3).

Target pattern.

- Tetris condition shows lower intrusion rates than Control, with Verbal intermediate if competitor density is lower.
- Recognition differences across conditions are small relative to intrusion differences.
- If we include free recall here, any Tetris effect should be smaller than for intrusions.

Role in the paper.

- Directly implements the selective interference effect in an idealised trauma-film setting.
- Provides a model-based explanation of why intrusions are more affected than recognition.

### Simulation 3: Intentional vs unintentional context→item retrieval

Aim. Show that the same post-film competitor encoding that reduces intrusions can leave intentional free recall relatively spared when retrieval control is high, and that lowering control makes free recall more intrusion-like.

Design sketch.

- Use the same Film and Post-film phases as in Simulation 2 (Tetris vs Control).
- Define two retrieval regimes, using the formal control parameters:

  - Unguided intrusions:
    - Retrieval in context→item mode.
    - Initial $c_{test}$ drawn from a distribution of lab-like or everyday contexts, with no explicit bias toward film context.
    - Control parameters: low starting-context bias; minimal gating.
    - Count spontaneous $f_i$ recalls across many iterations (as in Simulation 2).

  - Intentional free recall:
    - Retrieval in context→item mode.
    - Initial $c_{test}$ is explicitly biased toward the film-session context (for example a linear combination of early-film context states).
    - Control parameters: strong gating (items whose retrieved context is dissimilar to the film-context template have reduced influence on $c$ and may be rejected from output).
    - Count correctly recalled $f_i$ and incorrectly recalled $g_j$.

- Manipulate control strength:

  - High-control condition:
    - Strong starting-context bias.
    - High gating threshold.
  - Low-control condition:
    - Weaker bias; lower gating threshold (mimicking divided attention, time pressure, or stress).

Outcome measures.

- Intrusion rates (unguided regime) across post-film conditions (reproducing Simulation 2).
- Free recall of film items (intentional regime) across post-film conditions and control levels.
- Rate at which $g_j$ items intrude into intentional recall.

Target pattern.

- Unguided intrusions: Tetris vs Control difference similar to Simulation 2.
- Intentional free recall:
  - With high control, little or no Tetris vs Control difference, and relatively few $g_j$ intrusions.
  - With low control, Tetris vs Control differences start to appear, and $g_j$ items intrude more into recall.

Role in the paper.

- Explicitly instantiates the "control" part of Claim 2 in the formal model.
- Generates concrete predictions that can be tested by manipulating retrieval load or instruction strictness in trauma-film experiments.

### Simulation 4: Emotional vs neutral material

Aim. Test how emotional features and arousal-based learning parameters affect selective interference, and determine when post-film tasks differentially affect emotional vs neutral items.

Design sketch.

- Film phase:
  - Encode a mixed list of items: some "emotional" (high-arousal negative) and some "neutral" (low arousal).
  - Use CMR3's emotional encoding parameters so that emotional items have stronger context–item bindings and cluster in emotional-context space.

- Post-film manipulation:
  - High-arousal competitor condition:
    - Tetris-like or other arousing tasks encoded under contexts with elevated arousal (through a reminder or through task properties), so $g_j$ land in or near the emotional region of context.

  - Low-arousal competitor condition:
    - Post-film tasks that are low engagement/low arousal; $g_j$ encoded in more neutral contexts.

  - Control condition:
    - No additional post-film encoding.

- Test phase:
  - Intrusions in context→item mode under low control, as in Simulation 2.
  - Optional free recall and recognition for emotional vs neutral items under high control (for recall) and item→context (for recognition).

Outcome measures.

- Intrusion reduction for emotional vs neutral items in each post-film condition.
- Voluntary recall and recognition of each item type.

Target pattern.

- When competitors are encoded in high-arousal, trauma-adjacent contexts, intrusions from emotional items are reduced more than from neutral items.
- When competitors are low-arousal and contextually distant, interference is less emotion-selective and more uniformly tied to temporal proximity.

Role in the paper.

- Makes the "high-arousal region" premise empirically precise.
- Clarifies whether emotional selectivity is an optional add-on or an integral part of the interference mechanism.

### Simulation 5 (optional): Delayed reminder + Tetris

Aim. Show that reminder+Tetris effects at delays (24 h, 3 days, etc.) can emerge from context reinstatement and competitor encoding, without additional reconsolidation machinery.

Design sketch.

- Film phase as in Simulation 2.

- Delay:
  - Allow context to drift away from the trauma region via the standard drift dynamics.
- Reminder phase:
  - Present a reminder cue that retrieves the film context via M_FC and reinstates a trauma-like $c_{remind}$ (by mixing retrieved context into the current context).

- Post-reminder manipulation:
  - Reminder+Tetris condition:
    - Encode many $g_j$ items starting from $c_{remind}$.
  - Reminder-only control:
    - Present the reminder but no subsequent $g_j$ encoding, or only a few weak competitors.

- Test phase:
  - Intrusions under context→item retrieval over subsequent "days" as in Simulation 2.
  - Optional recognition and free recall.

Outcome measures.

- Intrusion rates in Reminder+Tetris vs Reminder-only across simulated days.
- Any changes in voluntary measures.

Target pattern.

- Additional reduction in intrusions in the Reminder+Tetris condition relative to Reminder-only, proportional to the degree of context reinstatement and competitor density.
- No need to posit a special reconsolidation module; the effect emerges from the same CMR3 operations.

Role in the paper.

- Provides a single-system interpretation of delayed reminder+Tetris findings and clarifies what is gained (and not gained) by invoking reconsolidation terminology.

## Comparing accounts

Here we contrast the main theories across a common set of phenomena, using the simulations as concrete instantiations for CMR3.

Organisation.

- For each phenomenon or constraint, summarise:
  - What the data look like (briefly).
  - What each theory predicts or can accommodate.
  - What the CMR3 simulations show.

Key phenomena:

- Selective interference: intrusions reduced, recognition/free recall spared (Simulations 1–3).
- Effects of associative cueing: trauma vs foil cues in lab-based intrusions and recognition tasks (built into the way c_test is perturbed in Simulations 2–3).
- Timing and context overlap:
  - During-film vs post-film interventions; delayed reminder+Tetris (Simulation 5).
- Emotional specificity:
  - Emotional vs neutral items and the arousal-subspace claim (Simulation 4).
- Heterogeneity and boundary conditions:
  - Immediate vs diary measures; tasks with low competitor density or poor context overlap.

For each, a short paragraph can lay out, for example:

- Visuospatial WM accounts: straightforward narrative for modality and timing, less clear on retrieval-mode specificity and diary heterogeneity.
- Dual-representation accounts: good fit to intrusive vs voluntary dissociations, less precise about task dynamics and context manipulations.
- Reconsolidation accounts: emphasise reminders and timing, but lack explicit retrieval predictions and can be mimicked by context reinstatement.
- CMR3 account: explains selective interference, retrieval-mode differences, and several boundary conditions with one set of operations, but relies on abstract context representations and requires careful parameter constraints.

A single comparative table can make these correspondences explicit (rows = phenomena, columns = theories, entries = predicted / flexible / not predicted).

## Implications for intervention design

This section draws out practical and conceptual implications of the CMR3 account for designing and interpreting intrusive-memory interventions.

Points to develop:

- Mechanistic design principles:
  - Post-film tasks are effective when they:
    - generate many strongly encoded competitor episodes in context near the trauma memory,
    - suppress trauma rehearsal during that window,
    - maintain sufficient context overlap (temporal, situational, emotional) with the trauma context.
- Tetris as one instance:
  - Tetris is effective not because it uniquely taps visuospatial WM, but because it happens to punch all three levers (segmentation, engagement, context overlap) in a convenient way.
- Alternative interventions:
  - Imagery rescripting, other visuospatial tasks, or even some verbal/semantic tasks could work if they can be engineered to produce a similar pattern of competitor encoding and rehearsal suppression.
- Interpretation of heterogeneity:
  - Mixed results across studies and meta-analyses can be reinterpreted in terms of:
    - insufficient competitor density,
    - poor context overlap (e.g., no reminder; different setting),
    - inadequate engagement,
    - or weak control at test (e.g., noisy intrusion measures).
- Experiment design directions:
  - Manipulate event segmentation and competitor count at fixed modality.
  - Manipulate arousal/engagement while holding segmentation constant.
  - Manipulate retrieval control (divided attention, instructions) to test Simulation 3 predictions.

The aim is to show that the model gives concrete guidance about how to tweak interventions, not just a post hoc story about Tetris.

## Limitations and open questions

Finally, we should be explicit about what this approach does not solve and where it is most fragile.

Model-related limitations:

- Abstraction and parameterisation:
  - CMR3 treats context and emotion at a high level of abstraction; mapping these directly to neural circuits or clinical variables will require additional assumptions.
  - The model has many parameters; our simulations must use a small, principled subset of manipulations to avoid overfitting.
- Scope of fits:
  - In this paper we will target qualitative patterns, not full quantitative fits to each dataset.
  - A more ambitious programme would fit the same parameter set to trauma-film and standard list-learning data simultaneously.

Empirical and conceptual limitations:

- Empirical robustness:
  - The trauma-film/Tetris effect is not uniformly robust across labs, samples, and measures; some key studies have failed to replicate certain aspects.
  - Our account treats this heterogeneity as a feature to be explained (via competitor density, context overlap, and control), but we will need new experiments to test that.
- Unresolved predictions:
  - The arousal-subspace hypothesis and its proposed tests (Simulation 4) require physiological or self-report measures of arousal to be collected alongside trauma-film paradigms.
  - The role of retrieval control in real-world settings (stress, sleep deprivation, comorbidity) may differ from the neat high/low control manipulations we simulate.
- Integration with biological theories:
  - Although we do not invoke a dedicated reconsolidation mechanism, our account is not incompatible with biological consolidation and reconsolidation processes.
  - Future work will need to articulate how context-binding and competition mechanisms relate to synaptic and systems-level changes over time.

The discussion can close by emphasising that a single-system, retrieved-context account does not aim to replace all existing ideas, but to provide a mechanistic core that clarifies which aspects of the selective interference effect require additional constructs (e.g., reconsolidation, dual representations) and which do not.
