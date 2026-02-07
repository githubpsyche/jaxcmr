# Intrusive and Voluntary Memory in a Single-System Framework: A Retrieved-Context Account of Selective Interference

This is a concise outline of a proposed Psych Review–style paper and simulation program to share with collaborators.

## Big-picture aim

Develop and test a single-system retrieved-context account (CMR3) of the selective interference effect in trauma-film paradigms, where post-film Tetris reduces intrusions while sparing voluntary memory. The paper will:

- Derive this pattern from a context-binding model without invoking separate traces or dedicated visuospatial consolidation mechanisms.
- Use simulations to explain both successful and failed Tetris effects across paradigms.
- Clarify how retrieval mode and control processes shape intrusions vs voluntary memory.
- Generate design principles for intrusive-memory interventions.

## Core claims

1. Single-system sufficiency  
   A single context-binding system (CMR3) can produce reduced intrusions alongside relatively spared recognition/intentional recall. No "intrusive vs voluntary" trace separation or special visuospatial consolidation module is required.

2. Retrieval mode and control  
   Dissociations arise from how retrieval is driven and controlled:
   - Intrusions: largely uncontrolled context→item retrieval as context drifts near trauma states.
   - Voluntary free recall: context→item retrieval with strong control (starting-context bias, output gating).
   - Recognition: item→context retrieval and similarity-based decisions.
   Retrieval mode and control level determine which measures are vulnerable to post-film competition.

3. Reinterpretation of Tetris interventions  
   Tetris works by creating dense, strongly encoded competitor episodes in trauma-adjacent context, while suppressing rehearsal, and sometimes sharing arousal/engagement properties with film hotspots. Visuospatial demands matter insofar as they drive segmentation, engagement, and context overlap, not because they uniquely disrupt a sensory trace.

---

## Proposed paper structure

### 1. Introduction: the selective interference puzzle

- Define selective interference in trauma-film + Tetris paradigms:
  - Intrusions reduced (lab tasks, diaries).
  - Voluntary free recall / recognition largely spared.
- Note prevailing interpretations:
  - Dual-representation / separate traces.
  - Visuospatial WM/consolidation and reconsolidation accounts.
- Identify gaps:
  - Lack of a formal episodic retrieval model.
  - Limited treatment of retrieval mode, cueing, and failures to replicate.
- State contribution:
  - Single-system CMR3 account, with explicit simulations and testable predictions.

### 2. Empirical landscape (constraints for the model)

- Core lab findings:
  - Tetris vs control: intrusions down, voluntary memory spared (Holmes et al.; Lau‑Zhu et al.).
  - Variations in timing (during vs after film), task modality, and cueing.
- Extensions/translation:
  - Delayed reminder+Tetris protocols.
  - Emergency department / real-world implementations.
  - Non-Tetris interventions (imagery rescripting, alternative tasks).
- Meta-analyses and replications:
  - Mixed but non-zero effects of visuospatial interventions.
  - Stronger effects in some lab settings than in long-term diary follow-ups.
- Take-home: the empirical pattern is heterogeneous and constraining, not a single clean effect.

### 3. Existing theoretical accounts

For each, briefly state core idea, strengths, and gaps relative to selective interference.

- Visuospatial WM/consolidation:
  - WM load during a consolidation window disrupts sensory-perceptual traces underlying intrusions.
  - Good on modality and timing; weak on retrieval-mode specificity and mixed diary results.
- Dual-representation / separate traces:
  - Sensory-bound vs contextualised traces; interventions weaken sensory-bound ones.
  - Good on phenomenology (flashbacks vs narratives); underspecified retrieval dynamics and task predictions.
- Reconsolidation-based accounts:
  - Reminder puts memory in labile state; Tetris updates/disrupts re-stabilisation.
  - Good on reminders and timing; limited computational detail and sometimes mimicked by simpler context-reinstatement mechanisms.
- Retrieved-context / CMR3:
  - Single system with drifting context (temporal, semantic, emotional) and bidirectional context–item associations.
  - Not yet applied to trauma-film/Tetris; this paper develops that application.

### 4. Retrieved-context / CMR3 framework

Goal: introduce the model components needed for the rest of the paper, conceptually and formally.

- State representations:
  - Item feature vectors (semantic, source, emotional).
  - Context vectors that drift over time and include emotional dimensions.
  - Associative matrices M_CF (context→item) and M_FC (item→context).
- Encoding:
  - Context update rule (recency-weighted blend of previous context and current item).
  - Hebbian updates to M_CF and M_FC, with stronger learning for high-arousal items.
- Retrieval dynamics:
  - Context→item: activation a = M_CF · c; competition and retrieval; retrieved item feeds back via M_FC to update context.
  - Item→context: probe uses M_FC to retrieve context; similarity to target context drives recognition decisions.
- Control parameters:
  - Starting-context bias (how strongly retrieval starts in film vs other contexts).
  - Output gating (how strongly retrieved items update context and whether they are accepted).
  - Simple stopping/threshold rules.

Map to tasks:

- Intrusions: context→item; low bias; minimal gating.
- Intentional free recall: context→item; strong bias to film; strong gating.
- Recognition: item→context; similarity decision; weak dependence on context-based competition.

### 5. Mapping the trauma-film/Tetris paradigm into CMR3

Goal: specify how film, post-film tasks, and outcome measures correspond to model objects.

- Film:
  - Sequence of high-arousal negative items f_i; strong context–item bindings in an emotional region of context space.
- Post-film interventions:
  - Tetris: many neutral items g_j encoded in trauma-adjacent context states (via temporal proximity and/or reminder-driven reinstatement); high segmentation and engagement.
  - Control tasks: fewer/weaker g_j; more room for f_i rehearsal; possibly more drift away from trauma context.
- Tests:
  - Lab intrusions: context→item retrieval from lab-like c_test with occasional trauma/foil cues.
  - Diary intrusions: context→item retrieval along more heterogeneous c_test trajectories.
  - Recognition: item→context retrieval for f_i vs foils.

Mechanism (selective interference):

- Tetris condition:
  - Trauma-adjacent contexts become associated with many g_j as well as f_i.
  - During later context→item retrieval, g_j compete with f_i; f_i cross threshold less often, reducing intrusions.
- Voluntary tasks:
  - Item representations and M_FC remain intact.
  - Recognition and controlled free recall can still access f_i when probes or starting-context/gating favor film items.

Link back to claims:

- Implements single-system sufficiency (Claim 1) and Tetris-as-competitor encoding (Claim 3) using the retrieval-mode/control machinery in Claim 2.

---

## Simulation program

Aim: a small set of simulations that instantiate the account and speak directly to key empirical contrasts.

### Simulation 1: Neutral benchmark (List A vs A+B)

- Standard list-learning setup:
  - List A alone vs List A followed by List B.
- Show:
  - Adding List B reduces free recall of A (context→item competition).
  - Recognition of A is much less affected (item→context).
- Role:
  - Demonstrates mode-dependent interference as a generic property, independent of trauma/Tetris.

### Simulation 2: Trauma-film analogue (Tetris vs control)

- Film phase:
  - High-arousal film items encoded as f_i.
- Post-film phase:
  - Tetris condition: many g_j encoded in trauma-adjacent context.
  - Control: few/weak g_j and/or more f_i rehearsal.
- Test:
  - Intrusions (lab-like and diary-like) in low-control context→item mode.
  - Recognition of f_i vs foils in item→context mode.
- Show:
  - Intrusions lower in Tetris vs Control; recognition nearly unchanged.
- Role:
  - Core selective interference pattern from a single-system model.

### Simulation 3: Retrieval control and spared free recall

- Use same film + post-film encoding as Simulation 2.
- Retrieval regimes:
  - Unguided intrusions: low bias, low gating.
  - Intentional free recall: strong bias to film context; strong gating; vary control strength.
- Show:
  - Tetris reduces intrusions in low-control regime.
  - With high control, voluntary free recall of film items is minimally affected by Tetris vs Control.
  - Lowering control makes free recall more intrusion-like (greater Tetris vs Control effect, more post-film intrusions).
- Role:
  - Formal test of the control component of Claim 2; links to manipulations of instruction, load, or stress.

### Simulation 4: Emotional vs neutral items

- Mixed film lists of emotional and neutral items.
- Post-film conditions:
  - High-arousal competitors (g_j encoded in trauma-like contexts).
  - Low-arousal competitors (g_j encoded in more neutral contexts).
  - No-competitor control.
- Test:
  - Intrusions and voluntary measures for emotional vs neutral items.
- Show:
  - Emotional items are more strongly affected when competitors are high-arousal and contextually close; less selective effects when competitors are low-arousal.
- Role:
  - Makes the "high-arousal region" premise precise; links emotional features to selective interference.

### Simulation 5 (optional): Delayed reminder + Tetris

- Film, delay, reminder, then:
  - Reminder+Tetris vs Reminder-only.
- Test:
  - Intrusions over later days.
- Show:
  - Additional intrusion reduction in Reminder+Tetris via context reinstatement and competitor encoding, without new mechanisms.
- Role:
  - Single-system interpretation of "reconsolidation-style" effects.

---

## Comparison, implications, and limitations

### Comparing accounts

- Use a table and brief narrative to compare:
  - Visuospatial WM, dual-representation, reconsolidation, and CMR3.
- Organise by phenomena:
  - Intrusions vs recognition/free recall.
  - Cueing manipulations.
  - Timing (immediate vs delayed).
  - Emotional selectivity.
  - Diary vs lab measures; heterogeneity across studies.
- Emphasise:
  - Where CMR3 matches or improves on other accounts.
  - Where it is underdetermined or compatible with them.

### Implications for intervention design

- Derive design principles from the model:
  - Maximise competitor density and strength in trauma-adjacent context.
  - Minimise trauma rehearsal during key windows.
  - Ensure sufficient temporal/situational/emotional overlap with trauma context.
- Reinterpret Tetris:
  - One instantiation of these principles, not a unique visuospatial fix.
- Suggest alternative tasks and model-driven experiment designs:
  - Vary segmentation, modality, arousal, and retrieval control to test the theory.

### Limitations and open questions

- Model abstraction and parameterisation:
  - High-level context/emotion representations; need for careful parameter constraints.
- Empirical robustness:
  - Heterogeneous Tetris effects as a target, not an inconvenience.
- Unresolved predictions:
  - Arousal-subspace hypothesis and retrieval control effects need direct testing.
- Integration:
  - How retrieved-context mechanisms fit with biological consolidation/reconsolidation frameworks.

