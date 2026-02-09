# Selective Interference Simulation Plan

## Goal
Develop the narrative in `notes/context_binding.md` into a well-formed paper (`index.qmd`) with simulations that progressively build a single-system retrieved-context account of selective interference.

## Core commitments

**Context→item vs item→context dissociation**: The architectural distinction between context-driven retrieval (free recall, intrusions) and item-driven retrieval (recognition) is the foundation of the account. Context→item retrieval is vulnerable to competitor encoding; item→context retrieval is not. This alone explains the intrusion–recognition dissociation.

**Competitor encoding**: Interference is implemented by encoding actual items (via `model.experience()`) in trauma-adjacent context, NOT by drift alone. The competitor-encoding account reframes visuospatial tasks (Tetris) as one instantiation of a broader strategy: any task that generates strongly encoded events in trauma-adjacent context will interfere with later context→item retrieval of film content.

**Recognition via item→context similarity**: `mfc.probe(item)` retrieves associated context; dot-product with current context gives the recognition signal. Context drifts between recognition tests (integrating retrieved context, same as during free recall), so current context serves as a dynamic reference rather than a frozen snapshot. Because encoding competitors only modifies competitor rows of M_FC (items are one-hot), `mfc.probe(film_item)` returns the same context vector regardless of competitor encoding — making recognition largely immune. The residual effect of competitor encoding on recognition comes only from the small difference in test-onset context state (since encoding B items drifts context further before `start_retrieving()`), not from any change to the item→context associations themselves.

**Retrieval control via choice sensitivity (tau)**: Voluntary recall is distinguished from involuntary by sharper competition (higher `choice_sensitivity`) in the Luce choice rule. Higher tau amplifies the advantage of strongly-matched items, functionally approximating CMR2's context-monitoring mechanism. Combined with `start_drift_rate`, this produces graded immunity: recognition (item→context) > intentional recall with control > unguided recall.

**Reminder-based context reinstatement**: Reminders of a previously-viewed trauma film retrieve associated context through item→context associations, driving current context back toward the trauma-film region. Subsequent experiences (interference task) are then bound to this reinstated context, creating competitors even at temporal delay. This provides an alternative to reconsolidation accounts of delayed interference.

**eCMR arousal context**: High-arousal items encode and reinstate shared arousal context features (using eCMR's `emotional_mcf` mechanism, interpreting `is_emotional` as `is_high_arousal`). High-arousal competitors preferentially interfere with recall of high-arousal study events via shared arousal context, producing arousal-selective interference.

## Simulation philosophy

The full trauma-film paradigm involves many interacting components, each involving implementation decisions with ambiguous consequences. Early simulations isolate individual mechanisms in simplified settings, using parameter-shifting explorations to show how stronger or weaker mechanism engagement affects outcomes. Each simulation resolves open questions before later simulations depend on the answers. Each varies 1-2 parameters while holding others at values established by previous simulations, preventing combinatorial explosion. Work order and presentation order may diverge — we may work on Sim 2 before Sim 1 since Sim 2 findings inform Sim 1's configuration.

---

## Narrative arc

The paper's argument: a single retrieved-context system can explain selective interference without dual traces, reconsolidation windows, or modality-specific disruption.

### Simulation progression (6 simulations)

| Sim | Conceptual question | What it clarifies for the reader | Commitment it resolves for later sims |
|-----|-------------------|--------------------------------|--------------------------------------|
| 1 | Does competitor encoding produce mode-dependent interference? | The recall/recognition dissociation is architectural — competitors hurt context→item (recall) more than item→context (recognition). No trauma-specific mechanism needed. | Validates the recognition pathway and basic competitor-encoding mechanism. |
| 2 | What intensifies interference from competing events? | Explores interference intensifiers (MCF encoding strength, context proximity, competitor density) via dose-response tracking. Identifies which factors drive interference and where ceilings emerge. | Resolves how to parameterize competitor encoding in later sims. |
| 3 | Why is intentional free recall sometimes spared? | Starting-context bias (`start_drift_rate`) and sharpened competition (`choice_sensitivity`) provide graded protection for intentional context→item recall. | Clarifies how to operationalize voluntary vs involuntary retrieval. Establishes control parameter values. |
| 4 | How do reminder cues during recall affect performance? | Film cues reinstate trauma context, partially overriding retrieval-control mechanisms. This explains heterogeneity across cued vs uncued paradigms. | Informs test-phase design for Sims 5-6 and interpretation of VIT-style paradigms. |
| 5 | How do reminders enable delayed interference? | Reminders reinstate trauma context via item→context retrieval. Competitors encoded after reinstatement land in trauma-adjacent context, producing interference even at delay. No reconsolidation window needed. | Validates reminder + competitor encoding as the delayed-interference mechanism. |
| 6 | Does arousal-matched competition produce arousal-selective interference? | High-arousal competitors preferentially reduce intrusions of high-arousal film items via shared arousal context features (eCMR). | (Final extension — no downstream dependencies.) |

### Postdictions vs. predictions

| Sim | Postdiction (reproduces known finding) | Prediction (novel or underexplored) |
|-----|---------------------------------------|-------------------------------------|
| 1 | Recognition is more resistant to retroactive interference than free recall. | The dissociation arises because competitors only affect context→item retrieval; item→context associations in M_FC are structurally unaffected by additional items encoded in shared context. |
| 2 | More engaging interference tasks produce stronger effects; simply increasing task duration shows diminishing returns (Holmes et al. 2009; James et al. 2015). | Interference is primarily driven by MCF encoding strength and context proximity of competitors, not count alone. Sequential encoding shows diminishing returns due to context drift away from the film region. |
| 3 | Intentional free recall of film content is sometimes spared despite being context→item (Lau-Zhu et al. 2019 Exp 1 vs Exp 2). | `start_drift_rate` and `choice_sensitivity` produce graded immunity across context→item tasks. Both contribute, but their individual effects differ in shape. |
| 4 | Selective interference findings are inconsistent across VIT-style paradigms that present film cues during the test phase. | Film cues reinstate trauma context, partially overriding retrieval-control mechanisms and flattening the voluntary/involuntary distinction. Cue-free paradigms are cleaner tests of the control hypothesis. Cue frequency and strength modulate the effect. |
| 5 | Delayed Tetris + reminder reduces intrusions (James et al. 2015); Tetris without reminder is less effective. | Reminder reinstates trauma context so competitor encoding effectiveness is a function of reinstatement strength, not temporal delay per se. Reminder-only (no competitors) has no effect. |
| 6 | Emotional film content produces more intrusions than neutral (Brewin 2014; Holmes & Bourne 2008). | High-arousal competitors preferentially reduce intrusions of high-arousal film items. Arousal-matched interference shows a flatter serial position profile than the recency-weighted pattern in Sims 1-5 — arousal context bridges items independently of temporal position. |

---

## Simulation designs (presentation order)

All simulations use fitted parameters from HealeyKahana2014 as a base, with parameter shifting to explore mechanism engagement. The primary visualization for Sims 2-6 is the **serial position curve (SPC)** pooling study and interference items as study positions, with formatting to distinguish film vs. interference regions (following the pattern in `projects/selective_interference/code/start_drift_rate_parameter_shifting.ipynb`). Only Sim 1 includes recognition.

A general prediction across Sims 1-5: interference should show a **recency gradient** — later-studied film items are disproportionately suppressed because they share more temporal context with competitors (which are encoded immediately after the film phase). This position-dependent interference pattern is a natural consequence of context drift and should be visible in the SPC as greater A-only vs A+B divergence at late film positions than early ones. Sim 6 then contrasts this: arousal context features operate independently of temporal position, so arousal-matched interference should partially flatten this recency gradient for high-arousal film items.

### Sim 1: Context-Based Competition in Free Recall vs. Recognition

**Purpose**: Establish that competitor encoding hurts context→item retrieval (free recall) but leaves item→context retrieval (recognition) intact. This is the fundamental architectural dissociation, demonstrated in a clean neutral setting.

**Design** (A/B list):
**Study A**: `experience(1), ..., experience(N_A)` — target items
**Study B** (interference condition only): `experience(N_A+1), ..., experience(N_A+N_B)` — competitor items
**Free recall test**: `start_retrieving()` → `simulate_free_recall()` → record full recall sequence including any B intrusions
**Recognition test**: `start_retrieving()`, then for each A item: compute recognition signal `dot(normalize(mfc.probe(item)), context.state)`, then integrate retrieved context into current context (drift, same as free recall). Raw signal strength is the DV — no decision threshold needed.

**Key comparison**: A-only vs A+B conditions, measured by both context→item (free recall) and item→context (recognition) tests.

**Primary visualization**:
*Panel A*: SPC for free recall in A-only vs A+B conditions. Shows recall probability at each study position. In the A+B condition, A items show reduced recall probability while B items appear as a new region. Number of A-item recalls (intrusions in the trauma-film analogy) is the headline DV.
*Panel B*: Recognition signal SPC — recognition signal strength plotted by study position for A items, in A-only vs A+B conditions. The two curves should nearly overlap, preserving the same serial position structure (primacy/recency shape). This shows immunity isn't just an aggregate — the full position-by-position pattern is unchanged. Any residual difference reflects only the small context-state offset from encoding B items before `start_retrieving()`, not changes to item→context associations.

**Infrastructure needed**:
Recognition pathway: `mfc.probe(item)` → normalize → dot with `context.state` → context integrates retrieved context (new, ~20 lines, in `jaxcmr/selective_interference/recognition.py`)
Multi-phase encoding: encode A items, then B items, then test (orchestration using existing `experience()`)

**Key findings**: The dissociation exists. Competitor encoding impairs context→item retrieval of target items while leaving item→context retrieval largely intact. The near-immunity of recognition is architectural: competitor encoding doesn't modify film items' M_FC associations.

---

### Sim 2: Manipulating Interference Density

**Purpose**: Explore what intensifies interference from competing events. Evaluates proposed interference intensifiers via dose-response tracking.

**Interference intensifiers to evaluate**:
**MCF encoding strength** (boosted MCF learning rate): Higher engagement → stronger M_CF associations for competitors → more competition during context→item retrieval. Implementable via a separate MCF learning rate for interference items.
**Context proximity** (interference-specific `encoding_drift_rate`): Competitors encoded with lower drift stay in film-adjacent context longer → more context overlap → stronger competition. Implemented as a separate `encoding_drift_rate` for interference events.
**Number of competitors**: More encoded events = more competition, subject to context-drift ceiling.

**Design** (film + interference):
**Film phase**: `experience(1), ..., experience(N_film)` — film hotspot items, using standard encoding parameters
**Interference phase**: `experience(N_film+1), ..., experience(N_film+M)` — competitor items, using interference-specific encoding_drift_rate and MCF learning rate
**Test**: Context→item recall (free recall)

**Parameter-shifting explorations** (~10 values each):
**Interference MCF learning rate sweep**: From baseline to boosted values, holding other parameters fixed. Does stronger encoding produce more interference?
**Interference `encoding_drift_rate` sweep**: From low (competitors stay in film context) to high (competitors drift away quickly). Shows the context-proximity effect and where the ceiling emerges.
**Competitor count sweep**: With fixed interference parameters, vary M. Shows diminishing returns due to context drift.
**Context trajectory visualization**: Plot `dot(context.state, film_reference)` after each competitor encoding step, for several interference `encoding_drift_rate` values. Directly shows how context leaves the film region.

**Primary visualization**: SPC with film/interference boundary (vertical line). Multiple lines for parameter-shifted values. Intrusion count (number of film items recalled) as a summary DV across conditions.

**Infrastructure needed**:
Interference-specific encoding parameters: separate `encoding_drift_rate` and MCF learning rate for the interference phase (implemented by modifying model state via `model.replace()` before encoding competitors)
Context trajectory tracking: record `dot(context.state, reference)` after each `experience()` call (new diagnostic, in `jaxcmr/selective_interference/context_tracking.py`)

**Key findings**: Which intensifiers work under CMR logic. The dose-response shape for each intensifier. Parameter values to carry forward into Sims 3-6.

---

### Sim 3: Intentional vs Unintentional Context→Item Retrieval

**Purpose**: Demonstrate how retrieval-control mechanisms produce graded immunity for intentional free recall. Both `start_drift_rate` and `choice_sensitivity` (tau) are explored individually and in combination.

**Two control mechanisms**:
**Starting-context bias** (`start_drift_rate`): Biases initial retrieval context toward start-of-film, away from interference region. Already in `CMR.start_retrieving()`.
**Choice sensitivity** (`choice_sensitivity` / tau): Higher tau sharpens the Luce choice rule, amplifying the advantage of items with strongest context→item support. Functionally approximates CMR2 context monitoring by suppressing weakly-matched items (including competitors encoded in partially-overlapping context).

**Design**: Film + interference encoding (using parameters from Sim 2), then free recall under varied control settings.

**Parameter-shifting explorations**:
**`start_drift_rate` sweep** (~10 values, default tau): How does starting-context bias shift the SPC? At what value does recall of film items recover toward control levels?
**`choice_sensitivity` sweep** (~10 values, default `start_drift_rate`): How does sharpened competition affect film-item vs competitor-item retrieval? At what tau is voluntary recall fully spared?
**2×2 summary** (`start_drift_rate` low/high × tau low/high): Do the mechanisms interact? Does tau add value beyond `start_drift_rate` alone?

**Primary visualization**: SPC with film/interference boundary, parameter-shifted lines. One figure per sweep (start_drift_rate sweep, tau sweep), plus the 2×2 summary. Intrusion count as a summary DV.

**Infrastructure needed**: None beyond Sims 1-2. Both `start_drift_rate` and `choice_sensitivity` are existing parameters in `cmr.py`.

**Key findings**: Graded immunity across context→item tasks. Both mechanisms contribute. Establishes control parameter values for Sims 4-6.

---

### Sim 4: How Do Reminder Cues During Recall Affect Performance?

**Purpose**: Clarify the role of film cues presented during the test phase (as in VIT paradigms). Film cues reinstate trauma context via `cue_context()`, which may override the retrieval-control mechanisms from Sim 3.

**Design**: Film + interference encoding (Sim 2 parameters), retrieval control (Sim 3 parameters), with varied cue presentation at test.

**Parameter-shifting explorations**:
**`cue_drift_rate` sweep** (~10 values): How strongly does `cue_context()` pull context back to the film region before each retrieval attempt?
**Cue probability sweep** (~10 values, e.g., 0.0 to 1.0): Not every retrieval attempt is preceded by a cue. How does cue frequency modulate the effect?
**Cue × control interaction** (select informative combinations, not full grid): At high cue frequency and strong cue drift rate, does the cue override `start_drift_rate` and tau?

**Primary visualization**: SPC with film/interference boundary, parameter-shifted lines. One figure per sweep. Intrusion count as a summary DV.

**Infrastructure needed**: Existing `cue_context()` from `jaxcmr/experimental/selective_interference.py`. Requires a cued retrieval loop: pre-generate a binary cue mask (Bernoulli draws from cue probability) for each recall step, then use `lax.cond(cue_mask[i], cue_fn, identity)` before each recall attempt within the `lax.scan` loop. This modified retrieval function goes in `jaxcmr/selective_interference/paradigm.py` alongside the paradigm orchestration.

**Key findings**: Film cues reinstate trauma context, overriding retrieval-control mechanisms. Cue frequency and strength modulate the effect. Cue-free paradigms yield cleaner tests of the control hypothesis.

---

### Sim 5: Delayed Interference with Reminder

**Purpose**: Show that context reinstatement via reminder enables delayed interference without a reconsolidation window. The reminder retrieves associated context, driving the model back to the film region so competitors can be encoded there.

**Design**:
**Film phase**: Encode film items with standard parameters
**Delay**: Drift context to out-of-list state with `drift_rate=1.0` (locked, not varied)
**Reminder**: Walk through film items in order using `cue_context()` for each (updates context but NOT M_FC/M_CF) with a reminder-specific drift rate
**Interference phase**: Encode competitor items (using Sim 2 parameters)
**Test**: Context→item recall (free recall), using demonstrative control parameters from Sim 3. No cues during recall.

**Parameter-shifting explorations**:
**Reminder drift rate sweep** (~10 values): How strongly does the reminder reinstate film context? At what drift rate is reinstatement sufficient for competitor encoding to produce interference?

**Conditions**:
*Reminder + competitors*: Full reminder sequence → interference encoding → test
*Reminder only*: Full reminder sequence → no interference → test
*No reminder + competitors*: Skip reminder → interference encoding (in distant context) → test

**Primary visualization**: SPC with film/interference boundary, parameter-shifted lines for reminder drift rate. Intrusion count as a summary DV across the three conditions.

**Infrastructure needed**: Reminder sequence function: loop calling `cue_context()` for each film item in order with a specified drift rate. Uses existing `cue_context()` from `selective_interference.py`. Existing `drift_context()` for the delay phase.

**Key findings**: Reminder reinstatement is sufficient for delayed interference. Without reminder, competitors are ineffective. Reminder-only (no competitors) has no effect on later retrieval.

---

### Sim 6: Arousal-Specific Interference

**Purpose**: Extend the account to arousal-specific interference using eCMR. High-arousal items share arousal context features, creating a context subspace where arousal-matched competitors compete preferentially.

**Design**:
**Film phase**: Mixed list of high-arousal and neutral items. `is_emotional` flag (interpreted as `is_high_arousal`) marks which items are high-arousal.
**Interference phase**: Only high-arousal competitors (all marked `is_high_arousal`). Encoded using Sim 2 interference parameters.
**Key IV**: Ratio of high-arousal to neutral items in the film phase
**Test**: Context→item recall. Score separately for high-arousal and neutral film items.

**Parameter-shifting explorations**:
**`emotion_scale` sweep** (~10 values): How does the strength of the arousal context feature affect the degree of arousal-selective interference? At what `emotion_scale` do high-arousal competitors preferentially suppress high-arousal film items?
**Arousal ratio sweep** (e.g., 20/80, 50/50, 80/20 high-arousal/neutral in film): Does higher concentration of high-arousal film items amplify the arousal-selective interference?

**Primary visualization**: SPC with film/interference boundary. Separate lines or panels for high-arousal vs. neutral film items. Parameter-shifted by `emotion_scale`. Intrusion counts split by arousal category.

**Infrastructure needed**: Use eCMR model variant from `jaxcmr/models_eeg/eeg_ecmr.py` directly, with `is_emotional` interpreted as `is_high_arousal`. Requires constructing appropriate `is_emotional` and `lpp_centered` arrays for the synthetic film + interference item sequence.

**Key findings**: High-arousal competitors preferentially reduce intrusions of high-arousal film items via shared arousal context. The effect scales with `emotion_scale` and with the ratio of high-arousal items in the film. Critically, arousal-matched interference should show a flatter serial position profile than the recency-weighted pattern seen in Sims 1-5 — because arousal context features bridge film and competitor items independently of temporal position, high-arousal film items are suppressed more uniformly across study positions while neutral film items retain the standard recency gradient.

---

## Code organization

### Library module: `jaxcmr/selective_interference/`

Flat subpackage with project-specific infrastructure. Follows existing library conventions (cf. `jaxcmr/models/`, `jaxcmr/components/`): Pytree-based state, JAX-compatible, type-annotated.

**`__init__.py`** — Public API re-exports.
**`recognition.py`** — Item→context recognition pathway. Given a model and a sequence of item indices, computes recognition signal at each position: `dot(normalize(mfc.probe(item)), context.state)`, then integrates retrieved context (drift). Returns signal-by-position array. Used in Sim 1.
**`context_tracking.py`** — Context trajectory diagnostics. Records `dot(context.state, reference)` after each `experience()` call to track how context drifts through film and interference phases. Returns trajectory array for plotting. Used in Sims 1, 2, 5.
**`paradigm.py`** — Paradigm orchestration for synthetic (non-h5) simulations. Handles multi-phase encoding: film items with standard parameters, parameter swap (via `model.replace()`) to interference-specific `encoding_drift_rate` and MCF learning rate, competitor encoding. Also handles delay drift, reminder sequences (loop of `cue_context()` calls), and the cued retrieval loop for Sim 4. Used across all sims.
**`plotting.py`** — Reusable visualization code refactored from `projects/selective_interference/code/start_drift_rate_parameter_shifting.ipynb`. Includes: SPC with film/interference boundary (vertical line, region labels, parameter-shifted multi-line formatting), context trajectory plots, and recognition signal SPC. Shared across all 6 sim notebooks.

### Notebooks: `projects/selective_interference/code/`

Simulation notebooks live in the project directory, not the library:

`sim1_mode_dissociation.ipynb`
`sim2_interference_density.ipynb`
`sim3_retrieval_control.ipynb`
`sim4_cues_at_test.ipynb`
`sim5_delayed_interference.ipynb`
`sim6_arousal.ipynb`

### Existing code reused as-is

`jaxcmr/experimental/selective_interference.py` — `drift_context()`, `cue_context()`, `simulate_vigilance_block()`
`jaxcmr/simulation.py` — `simulate_free_recall()` (used for the recall phase after encoding; `parameter_shifted_simulate_h5_from_h5()` is NOT directly usable since our paradigms are synthetic, not h5-driven)
`jaxcmr/analyses/spc.py` — `plot_spc()` for serial position curves
`jaxcmr/models_eeg/eeg_ecmr.py` — eCMR model for Sim 6

---

## Implementation plan (work order)

**Shared infrastructure** — encoding/paradigm orchestration, plotting (refactored from existing notebook), recognition pathway, context tracking
**Sim 2** (interference density) — worked before Sim 1 since its findings about intensifiers inform Sim 1's configuration
**Sim 1** (mode dissociation) — uses infrastructure + parameter values informed by Sim 2
**Sim 3** (retrieval control) — `start_drift_rate` and `choice_sensitivity` sweeps
**Sim 4** (cues at test) — extends Sim 3 with cue manipulation
**Sim 5** (delayed interference) — reminder sequence + existing `drift_context()`/`cue_context()`
**Sim 6** (arousal-specific) — eCMR model variant; last

## Key existing files

`jaxcmr/models/cmr.py` — base CMR: `experience()`, `start_retrieving()`, `activations()`, `outcome_probabilities()`
`jaxcmr/experimental/selective_interference.py` — `drift_context()`, `cue_context()`, `simulate_vigilance_block()`
`jaxcmr/simulation.py` — `simulate_free_recall()` (line 143); `parameter_shifted_simulate_h5_from_h5()` is reference for the parameter-shifting pattern but not directly usable for synthetic paradigms
`jaxcmr/components/context.py` — `TemporalContext.integrate()`, context drift mechanics
`jaxcmr/components/termination.py` — termination policies (composable pattern)
`jaxcmr/models_eeg/eeg_ecmr.py` — eCMR: `emotional_mcf`, `is_emotional`, `emotion_scale`
`jaxcmr/analyses/spc.py` — `plot_spc()` for serial position curves
`projects/selective_interference/code/start_drift_rate_parameter_shifting.ipynb` — existing parameter-shifting + SPC visualization demo

## Verification

Each simulation should produce:
**Parameter-shifting SPC** showing how mechanism engagement shifts recall of film vs interference items
**Context trajectory visualizations** where relevant (Sims 1, 2, 5)
**Intrusion count** (number of film items recalled) as summary DV across conditions
Clear labeling of postdiction vs. prediction for each result
