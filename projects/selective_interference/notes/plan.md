# Selective Interference Simulation Plan

## Goal

Develop a paper with simulations that progressively build a single-system retrieved-context account of selective interference.

## Core commitments

1.  **Context-to-item vs item-to-context dissociation**: The architectural distinction between context-driven retrieval (free recall, intrusions) and item-driven retrieval (recognition) is the foundation of the account.
    Context-to-item retrieval is vulnerable to competitor encoding; item-to-context retrieval is not.
    This alone explains the intrusion-recognition dissociation.

2.  **Competitor encoding**: Interference is implemented by encoding actual items in trauma-adjacent context, NOT by drift alone.
    The competitor-encoding account reframes visuospatial tasks (Tetris) as one instantiation of a broader strategy: any task that generates strongly encoded events in trauma-adjacent context will interfere with later context-to-item retrieval of film content.

3.  **Recognition via item-to-context similarity**: Probing M_FC with a test item retrieves the item's associated context; the dot product with current context gives the recognition signal.
    Context drifts between recognition tests (integrating retrieved context, same as during free recall), so current context serves as a dynamic reference rather than a frozen snapshot.
    Because encoding competitors only modifies competitor rows of M_FC (items are orthogonal), probing M_FC with a film item returns the same context vector regardless of competitor encoding, making recognition largely immune.
    The residual effect of competitor encoding on recognition comes only from the small difference in test-onset context state (since encoding B items drifts context further before the start-of-retrieval context reinstatement), not from any change to the item-to-context associations themselves.

4.  **Retrieval control via choice sensitivity (tau)**: Voluntary recall is distinguished from involuntary by sharper competition (higher tau or `choice sensitivity`) in the Luce choice rule.
    Higher tau amplifies the advantage of strongly-matched items, functionally approximating CMR2's context-monitoring mechanism without the added complexity of a full retrieval-monitoring implementation.
    Combined with starting-context reinstatement, this produces graded immunity: recognition (item-to-context) \> intentional recall with control \> unguided recall.

5.  **Reminder-based context reinstatement**: Reminders of a previously-viewed trauma film retrieve associated context through item-to-context associations, driving current context back toward the trauma-film region.
    Two scale parameters give continuous control over pre-interference context state: `reminder_start_drift_scale` controls how much context drifts toward start-of-list after film encoding (simulating the passage of time), and `reminder_drift_scale` controls how strongly the reminder reinstates film context.
    Competitors encoded after reinstatement land in reinstated context, producing interference even at temporal delay.
    This provides an alternative to reconsolidation accounts of delayed interference.
    Importantly, reminders modulate interference *targeting* (which film items are most vulnerable) rather than interference *intensity* (how many film items are recalled).
    This mechanism is explored parametrically in Sim 2 and demonstrated as a focused delayed-interference result in Sim 5.

6.  **eCMR arousal context**: High-arousal items encode and reinstate shared arousal context features (using eCMR's emotional source memory mechanism, interpreting the emotion flag as high-arousal).
    High-arousal competitors preferentially interfere with recall of high-arousal study events via shared arousal context, producing arousal-selective interference.

## Simulation philosophy

The full trauma-film paradigm involves many interacting components, each involving implementation decisions with ambiguous consequences.
Early simulations isolate individual mechanisms in simplified settings, using parameter-shifting explorations to show how stronger or weaker mechanism engagement affects outcomes.
Each simulation resolves open questions before later simulations depend on the answers.
Each varies 1-2 parameters while holding others at values established by previous simulations, preventing combinatorial explosion.
Work order and presentation order may diverge; we may work on Sim 2 before Sim 1 since Sim 2 findings inform Sim 1's configuration.

------------------------------------------------------------------------

## Narrative arc

The paper's argument: a single retrieved-context system can explain selective interference without dual traces, reconsolidation windows, or modality-specific disruption.

### Simulation progression (6 simulations)

| Sim | Conceptual question | What it clarifies for the reader | Commitment it resolves for later sims |
|----------------|------------------------------------------|------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| 1 | Does competitor encoding produce mode-dependent interference? | The recall/recognition dissociation is architectural: competitors hurt context-to-item retrieval (recall) more than item-to-context retrieval (recognition). No trauma-specific mechanism needed. | Validates the recognition pathway and basic competitor-encoding mechanism. |
| 2 | What intensifies interference, and how does pre-interference context state shape impairment? | Explores interference intensifiers (M_CF encoding strength, context proximity, competitor density) and shows how delay + reminder modulate which film items are most vulnerable. | Resolves how to parameterize competitor encoding in later sims. Establishes the delay/reminder mechanism developed in Sim 5. |
| 3 | Why is intentional free recall sometimes spared? | Starting-context reinstatement and sharpened competition (tau) provide graded protection for intentional context-to-item recall. | Clarifies how to operationalize voluntary vs involuntary retrieval. Establishes control parameter values. |
| 4 | How do reminder cues during recall affect performance? | Film cues reinstate trauma context, partially overriding retrieval-control mechanisms. This explains heterogeneity across cued vs uncued paradigms. | Informs test-phase design for Sims 5-6 and interpretation of VIT-style paradigms. |
| 5 | How do reminders enable delayed interference? | Reminders reinstate trauma context via item-to-context retrieval. Competitors encoded after reinstatement land in trauma-adjacent context, producing interference even at delay. No reconsolidation window needed. | Validates reminder + competitor encoding as the delayed-interference mechanism. |
| 6 | Does arousal-matched competition produce arousal-selective interference? | High-arousal competitors preferentially reduce intrusions of high-arousal film items via shared arousal context features (eCMR). | (Final extension; no downstream dependencies.) |

### Postdictions vs. predictions

| Sim | Postdiction (reproduces known finding) | Prediction (novel or underexplored) |
|-----------------|----------------------------|---------------------------|
| 1 | Recognition is more resistant to retroactive interference than free recall. | The dissociation arises because competitors only affect context-to-item retrieval; item-to-context associations in M_FC are structurally unaffected by additional items encoded in shared context. |
| 2 | More engaging interference tasks produce stronger effects; simply increasing task duration shows diminishing returns (Holmes et al. 2009; James et al. 2015). | Interference is primarily driven by M_CF encoding strength and context proximity of competitors, not count alone. Sequential encoding shows diminishing returns due to context drift. Reminders modulate interference *targeting* (which film items are vulnerable) rather than *intensity* (how many are recalled): the reminder reinstates context overlap so competitors are encoded in film-adjacent context, but total film recall is relatively stable across reminder strength. |
| 3 | Intentional free recall of film content is sometimes spared despite being context-to-item (Lau-Zhu et al. 2019 Exp 1 vs Exp 2). | Starting-context reinstatement and tau produce graded immunity across context-to-item tasks. Both contribute, but their individual effects differ in shape. |
| 4 | Selective interference findings are inconsistent across VIT-style paradigms that present film cues during the test phase. | Film cues reinstate trauma context, partially overriding retrieval-control mechanisms and flattening the voluntary/involuntary distinction. Cue-free paradigms are cleaner tests of the control hypothesis. Cue frequency and strength modulate the effect. |
| 5 | Delayed Tetris + reminder reduces intrusions (James et al. 2015); Tetris without reminder is less effective. | Reminder reinstates trauma context so competitor encoding effectiveness is a function of reinstatement strength, not temporal delay per se. No reconsolidation window is needed. Reminder-only (no competitors) has no effect. |
| 6 | Emotional film content produces more intrusions than neutral (Brewin 2014; Holmes & Bourne 2008). | High-arousal competitors preferentially reduce intrusions of high-arousal film items. Arousal-matched interference shows a flatter serial position profile than the recency-weighted pattern in Sims 1-5, because arousal context bridges items independently of temporal position. |

------------------------------------------------------------------------

## Simulation designs (presentation order)

All simulations use fitted parameters from Healey & Kahana (2014) free recall data as a base, with parameter shifting to explore mechanism engagement.
The primary visualization for Sims 2-6 is the **serial position curve (SPC)** pooling study and interference items as study positions, with formatting to distinguish film vs. interference regions.
Only Sim 1 includes recognition.

A general prediction across Sims 1-5: interference should show a **recency gradient** in which later-studied film items are disproportionately suppressed because they share more temporal context with competitors (which are encoded immediately after the film phase).
This position-dependent interference pattern is a natural consequence of context drift and should be visible in the SPC as greater control vs interference divergence at late film positions than early ones.
Sim 6 then contrasts this: arousal context features operate independently of temporal position, so arousal-matched interference should partially flatten this recency gradient for high-arousal film items.

### Sim 1: Context-Based Competition in Free Recall vs. Recognition

**Purpose**: Establish that competitor encoding hurts context-to-item retrieval (free recall) but leaves item-to-context retrieval (recognition) intact.
This is the fundamental architectural dissociation, demonstrated in a clean neutral setting.

**Design** (A/B list): - **Study A**: Encode N_A target items using standard encoding parameters - **Study B** (interference condition only): Encode N_B competitor items immediately after Study A - **Free recall test**: Reinstate start-of-list context, then simulate free recall, recording the full recall sequence including any B intrusions - **Recognition test**: Reinstate start-of-list context, then for each A item probe M_FC to retrieve its associated context, compute the dot product with current context as the recognition signal, and integrate retrieved context (drift).
Raw signal strength is the DV with no decision threshold needed.

**Key comparison**: A-only vs A+B conditions, measured by both context-to-item (free recall) and item-to-context (recognition) tests.

**Primary visualization**: - *Panel A*: SPC for free recall in A-only vs A+B conditions.
Shows recall probability at each study position.
In the A+B condition, A items show reduced recall probability while B items appear as a new region.
Number of A-item recalls is the headline DV.
- *Panel B*: Recognition signal SPC showing recognition signal strength plotted by study position for A items, in A-only vs A+B conditions.
The two curves should nearly overlap, preserving the same serial position structure (primacy/recency shape).
This shows immunity isn't just an aggregate; the full position-by-position pattern is unchanged.
Any residual difference reflects only the small context-state offset from encoding B items before starting retrieval, not changes to item-to-context associations.

**Key findings**: The dissociation exists.
Competitor encoding impairs context-to-item retrieval of target items while leaving item-to-context retrieval largely intact.
The near-immunity of recognition is architectural: competitor encoding doesn't modify film items' M_FC associations.

------------------------------------------------------------------------

### Sim 2: Manipulating Interference Intensity and Pre-Interference Context State

**Purpose**: Explore what intensifies interference from competing events, and how pre-interference context state shapes the pattern of impairment.
The delay and reminder sweeps here establish the two-parameter context-control mechanism developed further in Sim 5.

**Trial sequence**: Film encoding (16 items, fitted parameters) → delay drift toward start-of-list (`reminder_start_drift_scale` x fitted `start_drift_rate`) → reminder (reinstate film context via M_FC without learning, at `reminder_drift_scale` x fitted `encoding_drift_rate`) → interference encoding (up to 32 items, with interference-specific drift and M_CF scales) → filler encoding (suppresses interference recency) → `start_retrieving()` → free recall.

**Interference intensifiers**: 1.
**M_CF encoding strength**: Scale factor on M_CF learning rate during interference.
Higher engagement produces stronger context-to-item associations for competitors.
2.
**Context proximity**: Scale factor on encoding drift rate during interference.
Lower drift keeps competitors in film-adjacent context.
3.
**Number of competitors**: More encoded events means more competition, subject to a context-drift ceiling.

**Pre-interference context state**: Two parameters control the context state at the moment competitors are encoded: - **`reminder_start_drift_scale`**: Controls the delay drift toward start-of-list after film encoding (simulates passage of time) - **`reminder_drift_scale`**: Controls how strongly the reminder reinstates film context

Together, these give continuous control over the pre-interference context landscape. Sim 5 develops this mechanism into a focused delayed-interference demonstration.

**Parameter-shifting explorations** (\~10 values each): 1.
**Filler count calibration**: Determines how many filler items are needed to suppress interference recency (N_FILLER_DEFAULT = 16) 2.
**M_CF learning rate sweep**: Does stronger encoding produce more interference?
(Monotonic decrease in film recall) 3.
**Encoding drift rate sweep**: Shows context-proximity dose-response with ceiling 4.
**Context trajectory visualization**: Similarity to pre-interference context after each competitor encoding step, at several drift scales 5.
**Competitor count sweep**: Shows diminishing returns due to context drift 6.
**Start-of-list drift scale sweep**: How the delay between film and interference modulates interference 7.
**Reminder drift scale sweep**: How strongly the reminder reinstates film context, with diagnostic plots showing film-item SPC and post-reminder context state

**Primary visualization**: SPC with film/interference/filler boundary lines.
Multiple lines for parameter-shifted values.
Film items recalled (mean with 95% CI) as a summary DV across conditions.

**Key findings**: M_CF encoding strength and context proximity are the primary drivers of interference intensity.
Competitor count shows diminishing returns.
Reminders modulate interference *targeting* (which film items are vulnerable) rather than *intensity* (how many are recalled): total film recall is relatively stable across the reminder sweep, but the SPC shape shifts.
The delay drifts context away from film, reducing interference from context overlap; the reminder reinstates it.
No reconsolidation window is needed.
Parameter values to carry forward into Sims 3-6.

------------------------------------------------------------------------

### Sim 3: Intentional vs Unintentional Context-to-Item Retrieval

**Purpose**: Demonstrate how retrieval-control mechanisms produce graded immunity for intentional free recall.
Both starting-context reinstatement and choice sensitivity (tau) are explored individually and in combination.

**Two control mechanisms**: 1.
**Starting-context reinstatement**: Biases initial retrieval context toward start-of-film, away from the interference region.
This is CMR's existing start-of-retrieval drift mechanism.
2.
**Choice sensitivity (tau)**: Higher tau sharpens the Luce choice rule, amplifying the advantage of items with strongest context-to-item support.
This functionally approximates CMR2's context-monitoring mechanism by suppressing weakly-matched items (including competitors encoded in partially-overlapping context) without requiring the full monitoring implementation.

**Design**: Film + interference encoding (using parameters from Sim 2), then free recall under varied control settings.

**Parameter-shifting explorations**: 1.
**Starting-context reinstatement sweep** (\~10 values, default tau): How does starting-context bias shift the SPC?
At what value does recall of film items recover toward control levels?
2.
**Tau sweep** (\~10 values, default starting-context reinstatement): How does sharpened competition affect film-item vs competitor-item retrieval?
At what tau is voluntary recall fully spared?
3.
**2x2 summary** (starting-context reinstatement low/high x tau low/high): Do the mechanisms interact?
Does tau add value beyond starting-context reinstatement alone?

**Primary visualization**: SPC with film/interference boundary, parameter-shifted lines.
One figure per sweep, plus the 2x2 summary.
Intrusion count as a summary DV.

**Key findings**: Graded immunity across context-to-item tasks.
Both mechanisms contribute.
Establishes control parameter values for Sims 4-6.

------------------------------------------------------------------------

### Sim 4: How Do Reminder Cues During Recall Affect Performance?

**Purpose**: Clarify the role of film cues presented during the test phase (as in VIT paradigms).
Film cues reinstate trauma context via item-to-context retrieval, which may override the retrieval-control mechanisms from Sim 3.

**Design**: Film + interference encoding (Sim 2 parameters), retrieval control (Sim 3 parameters), with varied cue presentation at test.

**Parameter-shifting explorations**: 1.
**Cue drift rate sweep** (\~10 values): How strongly does a film cue pull context back to the film region before each retrieval attempt?
2.
**Cue probability sweep** (\~10 values, e.g., 0.0 to 1.0): Not every retrieval attempt is preceded by a cue.
How does cue frequency modulate the effect?
3.
**Cue x control interaction** (select informative combinations, not full grid): At high cue frequency and strong cue drift rate, does the cue override starting-context reinstatement and tau?

**Primary visualization**: SPC with film/interference boundary, parameter-shifted lines.
One figure per sweep.
Intrusion count as a summary DV.

**Key findings**: Film cues reinstate trauma context, overriding retrieval-control mechanisms.
Cue frequency and strength modulate the effect.
Cue-free paradigms yield cleaner tests of the control hypothesis.
Informs test-phase design for Sims 5-6 and interpretation of VIT-style paradigms.

------------------------------------------------------------------------

### Sim 5: Delayed Interference with Reminder

**Purpose**: Show that context reinstatement via reminder enables delayed interference without a reconsolidation window.
The reminder retrieves associated context, driving the model back to the film region so competitors can be encoded there.
This simulation presents a focused demonstration of the delay/reminder mechanism explored parametrically in Sim 2.

**Design**:
- **Film phase**: Encode film items with standard parameters
- **Delay**: Drift context toward start-of-list (`reminder_start_drift_scale` x fitted `start_drift_rate`)
- **Reminder**: Walk through film items in order, probing M_FC for each and integrating the retrieved context (updates context but does NOT update M_FC or M_CF) at `reminder_drift_scale` x fitted `encoding_drift_rate`
- **Interference phase**: Encode competitor items (using Sim 2 parameters)
- **Test**: Context-to-item recall (free recall), using control parameters from Sim 3. No cues during recall.

**Parameter-shifting explorations**:
1. **Reminder drift scale sweep** (~10 values): How strongly does the reminder reinstate film context? At what drift rate is reinstatement sufficient for competitor encoding to produce interference?

**Conditions**:
- *Reminder + competitors*: Full reminder sequence, then interference encoding, then test
- *Reminder only*: Full reminder sequence, no interference, then test
- *No reminder + competitors*: Skip reminder, interference encoding (in distant context), then test

**Primary visualization**: SPC with film/interference boundary, parameter-shifted lines for reminder drift scale. Intrusion count as a summary DV across the three conditions.

**Key findings**: Reminder reinstatement is sufficient for delayed interference. Without reminder, competitors are ineffective. Reminder-only (no competitors) has no effect on later retrieval.

------------------------------------------------------------------------

### Sim 6: Arousal-Specific Interference

**Purpose**: Extend the account to arousal-specific interference using eCMR.
High-arousal items share arousal context features, creating a context subspace where arousal-matched competitors compete preferentially.

**Design**: - **Film phase**: Mixed list of high-arousal and neutral items - **Interference phase**: Only high-arousal competitors.
Encoded using Sim 2 interference parameters.
- **Key IV**: Ratio of high-arousal to neutral items in the film phase - **Test**: Context-to-item recall.
Score separately for high-arousal and neutral film items.

**Parameter-shifting explorations**: 1.
**Arousal context strength sweep** (\~10 values): How does the strength of the arousal context feature affect the degree of arousal-selective interference?
At what strength do high-arousal competitors preferentially suppress high-arousal film items?
2.
**Arousal ratio sweep** (e.g., 20/80, 50/50, 80/20 high-arousal/neutral in film): Does higher concentration of high-arousal film items amplify the arousal-selective interference?

**Primary visualization**: SPC with film/interference boundary.
Separate lines or panels for high-arousal vs. neutral film items.
Parameter-shifted by arousal context strength.
Intrusion counts split by arousal category.

**Key findings**: High-arousal competitors preferentially reduce intrusions of high-arousal film items via shared arousal context.
The effect scales with arousal context strength and with the ratio of high-arousal items in the film.
Critically, arousal-matched interference should show a flatter serial position profile than the recency-weighted pattern seen in Sims 1-5, because arousal context features bridge film and competitor items independently of temporal position while neutral film items retain the standard recency gradient.

------------------------------------------------------------------------

## Work order

1.  **Shared infrastructure** (recognition pathway, context trajectory diagnostics, paradigm orchestration, shared visualization)
2.  **Sim 2** (interference intensity + pre-interference context state) -- done; findings inform all subsequent sims
3.  **Sim 1** (mode dissociation) -- uses parameter values informed by Sim 2
4.  **Sim 3** (retrieval control) -- starting-context reinstatement and tau sweeps
5.  **Sim 4** (cues at test) -- extends Sim 3 with cue manipulation
6.  **Sim 5** (delayed interference) -- reminder sequence using Sim 2 delay/reminder mechanism
7.  **Sim 6** (arousal-specific) -- eCMR model variant; last

## Verification

Each simulation should produce: 1.
**Parameter-shifting SPC** showing how mechanism engagement shifts recall of film vs interference items 2.
**Context trajectory visualizations** where relevant (Sims 1, 2, 5) 3.
**Intrusion count** (number of film items recalled) as summary DV across conditions 4.
Clear labeling of postdiction vs. prediction for each result