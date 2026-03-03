# Selective Interference: The Argument

A single retrieved-context system explains selective interference without dual traces, reconsolidation windows, or modality-specific disruption.

## Prerequisites

These are presented in their own paper sections before any simulations.

- **Model specification**: eCMR / CMR3 is the central model, including the emotional context channel. The reader knows the full architecture — context-to-item and item-to-context retrieval, M_CF and M_FC associations, temporal and emotional context, retrieval control mechanisms (start-of-context reinstatement, choice sensitivity), and the reminder mechanism — before encountering any simulation.
- **Paradigm mapping**: The trauma-film paradigm is translated to model operations: film encoding → delay (context drift toward start-of-list) → reminder (context reinstatement via item-to-context retrieval, without new learning) → interference encoding (competitors in reinstated context) → filler encoding (suppresses interference recency) → recall.
- **Calibration**: Parameters fitted to Healey & Kahana (2014) free-recall data. Paradigm geometry tuned via break count, filler count, and reminder parameter sweeps. Presented as methodology — necessary setup, not part of the argument.

## The argument

### Core — The selective interference effect

Interference x control factorial. Film → reminder → competitor encoding → recall, under minimal control (low start-drift, low tau) vs directed control (high start-drift, high tau), crossed with interference present vs absent. The headline result: competitor encoding fully impairs unguided recall, partially impairs directed recall, and leaves recognition completely intact — a gradient that the separate-trace assumption cannot produce.

Postdiction: Intentional free recall of film content is sometimes spared despite being context-to-item (Lau-Zhu et al. 2019 Exp 1 vs Exp 2), while intrusion measures are reliably reduced by interference tasks.

Prediction: The selective effect arises from retrieval control operating within the same architecture — not from separate memory systems or different trace types. Minimal control leaves recall vulnerable to competition from items encoded in shared context; directed control steers retrieval away from the interference region and sharpens competition to favor target items.

*Existing accounts assume that intrusive and voluntary memories come from separate memory stores, so an interfering task can selectively disrupt one store while the other remains intact. Our account: there is one memory system. The dissociation arises from how retrieval is configured (unguided vs directed), not from what kind of trace is accessed.*

### Core — Context reinstatement replaces reconsolidation

Three conditions at delay: reminder + competitors, no-reminder + competitors, reminder only. Plus a reinstatement strength sweep (reminder_drift_scale). Only reminder + competitors produces interference. Without a reminder, competitors at delay land in distant context and don't overlap with film items. Reminder alone reinstates context temporarily but creates no competing traces.

Postdiction: Delayed Tetris + reminder reduces intrusions; neither alone is effective.

Prediction: Interference effectiveness depends on two continuous variables: how strongly the reminder reinstates film context, and how much subsequent experience has caused the reinstated context to drift before the interference task begins. Stronger reinstatement means more context overlap between competitors and film items; more intervening experience erodes that overlap. There is no reconsolidation window and no labile trace, but there is a temporal constraint arising from context drift rather than a discrete molecular process.

*Existing accounts assume that a reminder reactivates a consolidated memory trace, making it labile during a time-limited window. An interfering task during this window disrupts restabilization. Our account: the reminder reinstates context via item-to-context retrieval, driving the model back into the film region. Competitors encoded there produce interference through the same mechanism as immediate interference — context overlap and retrieval competition. The critical variables are reinstatement strength and context maintenance — how much intervening experience erodes the reinstated state before the interference task begins. Timing matters, but through continuous context drift rather than a discrete window.*

### Decompose control — What protects voluntary recall

Holding interference constant, vary control mechanisms individually and in combination:

- **Start-drift sweep**: Biasing initial retrieval context toward start-of-film, away from the interference region.
- **Tau sweep**: Sharpening competition in the Luce choice rule, amplifying the advantage of strongly-matched film items over weakly-matched competitors.
- **Start-drift x tau interaction**: Are they additive or interactive? Does tau add value beyond start-drift alone?

Postdiction: Intentional free recall can be spared despite being a context-to-item task (Lau-Zhu et al. 2019). The degree of sparing varies across studies and task demands.

Prediction: Start-drift and tau produce graded protection with different effect shapes. Start-drift shifts the SPC by reweighting which film items are initially cued. Tau sharpens the competition globally, concentrating retrieval probability on the strongest candidates. Combined, they approximate context monitoring without a full monitoring implementation. The graded immunity ordering — directed recall > unguided recall — falls out of the same architecture at different operating points.

*Existing accounts assume voluntary memory is spared because it draws on a separate, undisrupted memory store. This offers no mechanism for graded protection — memory is either in the intact store or the disrupted one. Our account: free recall uses the same retrieval pathway as intrusions (context-to-item) but is partially protected by control mechanisms. Protection is continuous and graded — not full immunity but reduced vulnerability. This predicts when free recall would become more vulnerable: weaken control (e.g., reduce executive resources, add secondary load at test) and free recall should look more like unguided recall.*

### Decompose interference — What makes interference stronger

Holding control constant, vary interference along four axes:

- **MCF encoding strength**: Stronger M_CF learning during interference produces stronger context-to-item associations for competitors, increasing retrieval competition.
- **Context proximity**: Lower encoding drift rate during interference keeps competitors closer to film context, increasing context overlap and competition.
- **Competitor count**: More competitors means more competition, but with diminishing returns — later competitors drift further from film context, reducing their overlap.
- **Arousal-context overlap**: Shared arousal context between interference items and film items broadens interference beyond temporal proximity. Arousal context bridges items independently of temporal position, producing a flatter serial position profile for high-arousal film items while neutral film items retain the standard recency gradient.

Postdiction: More engaging interference tasks produce stronger effects; simply increasing task duration shows diminishing returns (Holmes et al. 2009; James et al. 2015). Emotional film content produces more intrusions than neutral (Brewin 2014; Holmes & Bourne 2008).

Prediction: Interference is driven by encoding quality and context overlap, not count alone. Sequential encoding shows diminishing returns due to context drift. Arousal-matched competitors preferentially suppress recall of high-arousal film items via shared arousal context, partially flattening the temporal recency gradient that characterizes interference in the other axes.

*Existing accounts assume visuospatial tasks are specifically effective because they compete for the sensory-perceptual resources used to consolidate image-based memory traces. Verbal tasks should be ineffective or counterproductive. Our account: what matters is how strongly events are encoded and how much their encoding context overlaps with the film context. Any strongly-encoded events in film-adjacent context will interfere, regardless of modality.*

### Follow-ups — Interpreting paradigm variations

These sections are loosely connected by the theme of explaining why the selective interference effect is not always observed and how experimental design choices modulate it.

**Recognition immunity.** Recognition uses a different retrieval pathway from free recall and intrusions. It probes item-to-context associations (M_FC), retrieving the context a film item was encoded in and comparing it to current context. Competitor encoding only modifies competitor rows of M_FC (items are orthogonal), so probing with a film item returns the same context vector regardless of how many competitors were encoded. Recognition is structurally immune — not protected by control, but architecturally untouched.

This is a different kind of protection than free recall enjoys. Free recall is partially protected by retrieval control (same pathway as intrusions, different operating point — limited but nonzero interference remains). Recognition is completely spared by architecture (different pathway entirely). The separate-trace assumption attributes both to "the voluntary memory store is intact" and cannot distinguish the two mechanisms or predict the gradient. Our account predicts they should dissociate: weaken retrieval control and free recall becomes more vulnerable, approaching the level of unguided recall, while recognition remains completely immune regardless.

Postdiction: Recognition is more resistant to retroactive interference than free recall.

Prediction: The immunity is architectural, not a threshold artifact. The full position-by-position recognition signal is unchanged by competitor encoding. Any residual difference reflects only the small context-state offset from encoding competitors before the start-of-retrieval reinstatement.

**Test-phase cues.** Film cues presented during recall reinstate trauma context via item-to-context retrieval, partially overriding the retrieval-control mechanisms that protect voluntary recall. This blurs the voluntary/involuntary distinction and explains heterogeneity across cued vs uncued paradigms.

Postdiction: Selective interference findings are inconsistent across VIT-style paradigms that present film cues during the test phase.

Prediction: Cue frequency and cue strength modulate the override. At high cue intensity, the voluntary/involuntary distinction flattens. Cue-free paradigms yield cleaner tests of the selective interference effect.

**Paradigm design implications.** What the model suggests about designing experiments more sensitive to the selective interference effect — task parameters, retrieval instructions, and cue handling that maximize the contrast between voluntary and involuntary recall conditions.

## Cross-cutting predictions

- **Recency gradient**: Across the core and decomposition sections, interference should disproportionately suppress late film items because they share more temporal context with competitors. Arousal-context overlap partially flattens this gradient for high-arousal items.
- **Graded immunity**: Recognition (completely spared) > directed recall (partially spared) > unguided recall (fully vulnerable). This gradient falls out of the architecture — recognition uses a structurally immune pathway, while directed and unguided recall differ in control, not trace type.

## Work order

Work order may diverge from presentation order. Current state: calibration and interference sweeps are rendered; control sweeps partially rendered; core factorial, arousal, recognition, and cue simulations are unimplemented.
