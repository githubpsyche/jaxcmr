# Same item, different traces

## Introduction
- Frames episodic memory under repeated/overlapping experience as a tension between integrating related episodes and differentiating their episode-specific details.
- Introduces repeated-item list learning across free and serial recall (e.g., *canoe* at positions 5 and 15) and motivates cross-occurrence transitions as behavioral diagnostics of whether occurrences remain separable or blend during retrieval.
- Positions retrieved-context theory as the dominant account of temporal organization in recall and previews the central finding: under repetition, retrieval stays tied to a single occurrence context rather than blending, motivating an instance-based alternative.

### Retrieved-Context Theory and Repetition
- Contrasts passive-drift context theories with item-based context reinstatement in RCT and highlights why reinstatement is a linchpin of RCT’s successes across free recall, serial recall, and multi-list phenomena.
- Explains RCT’s account of repetition effects via contextual variability plus associative study-phase retrieval, linking these mechanisms to spacing/superadditive repetition benefits and related testing-effect accounts.
- Argues that the same item-driven mechanism predicts cross-occurrence blending at both study and retrieval, which should undermine occurrence-specific access to contextual detail.
- States the core empirical claim: across six published datasets and refined position-matched baselines, three blending-derived predictions are not supported, while standard CMR simulations robustly produce them.
- Generalizes the mismatch as an extension of the classic serial-recall noninterference constraint and argues it pressures any account in which item identity is the retrieval unit for contextual associations.

### Episode-Specific Reinstatement Within a Retrieved-Context Framework
- Proposes a three-part solution starting with an instance-based reformulation of CMR: separate traces per study event and a continuum of reinstatement from item-level composites to single-occurrence selection.
- Motivates instance-based modeling as underused in retrieved-context work due to tradition (not theory) and as a route to selective access across repeated occurrences.
- Argues that separate traces alone are insufficient and introduces the second component: trace-level competition so the winning occurrence reinstates its own context rather than a blend.
- Introduces the third component: item-independent context-evolution at study so repeated occurrences do not drift toward overlapping contextual states that would preserve interference.
- Defines Instance-CMR as the combined changes (item-independent dynamics + trace competition + trace storage) that preserve retrieved-context principles while enabling episode-specific access and reconciling integration with differentiation.

### Roadmap
- Lays out the paper sequence: validate Instance-CMR on non-repetition benchmarks and fits, test the three diagnostic predictions across six repeated-item datasets against standard CMR, ablate study-phase retrieval, and show how Instance-CMR accounts for repeated-item transitions while preserving retrieved-context phenomena before a final extension.
- Concludes the introduction by stating the theoretical upshot: item–context associations alone cannot mediate access to episode-specific detail, and temporal context must evolve item-independently within traces to mark episode identity even when item features support transfer.

---

## Modeling Framework
- **RCT overview** — Items couple to drifting context; reinstatement at study and recall; one rule accommodates two classic ideas (contextual variability + study-phase retrieval); explains benchmarks (lag-contiguity, primacy-recency, distractor-robust recency, PLIs, serial order)
- [Figure: CMR schematic] — Illustrates contextual variability and study-phase retrieval hypotheses
- **The blending problem** — Same reinstatement rules blend support across occurrences; at study (contexts become more similar) and at retrieval (composite cue contacts all occurrences); implies cues to one occurrence also cue the other
- **Three levers** — Introduces experimental strategy: hold temporal-context dynamics fixed, manipulate (1) unique vs item-specific context at study, (2) composite vs competitive reinstatement at test, (3) reinforcement without knitting
- **Position features implementation** — How to allow unique contexts: traces concatenate item + position + context features; position features are orthogonal and serve as storage key; item features only identify what's retrieved
- **Occurrence-specific reinstatement implementation** — How to approximate single-trace selection: eligible traces weighted by similarity to current cue; reuses trace-sensitivity parameter (no new DOF); converges to one trace as sharpening increases
- **Reinforcement implementation** — How to boost first occurrence without knitting: strengthen first-occurrence trace at second presentation; apply to both item→context and context→item stores; one bounded parameter
- **Variant summary** — Instance-CMR = CMR at baseline; CMR-NoSPR removes study-phase reinstatement; ICMR-OS adds competitive reinstatement; ICMR-Reinf adds reinforcement; all else unchanged

## Methodological Framework

### Defining Repetition Effects
- **What counts as a repetition effect?** — Tempting but wrong: any change after multiple presentations; correct: special consequence of same material twice (not just more material); formal definition invokes counterfactual baseline
- **Why mixed-list singletons fail** — Repetitions change rehearsal, output competition, contextual drift for all items; singletons in mixed lists are contaminated controls
- **Position-matched control procedure** — Pair control lists matching length + schedule; designate positions i,j as pseudo-repeater; resample to get null distribution
- **Symmetric scoring rule** — Both matched positions = one pseudo-item; second recall of either = repeat recall (excluded); asymmetric scoring inflates apparent effects
- **Procedure summary** — Three components: (i) between-trial comparison, (ii) position matching, (iii) symmetric scoring; isolates true repetition effects

### Datasets
- **Overview** — Four datasets spanning free + serial recall; introduces role of each
- **Lohnas 2014** — Primary free recall; 35 Ss × 4 sessions × 48 lists; control + pure massed + pure spaced + mixed lists; 40 positions/list; lags 0-8 manipulated
- **Broitman 2024** — Secondary free recall; 52 Ss × 6 sessions; mixed lists only (no controls); 12 words/list, half repeated; minimum lag 3; used for generalization without baseline
- **Logan 2021 Exp 2** — Primary serial recall; 24 Ss; 6-letter lists; lags 0-3; simultaneous display; typed recall; tight control over structure
- **Kahana 2000** — Secondary serial recall; 20 Ss × 12 sessions; 11-13 consonants; lags 0-5; learning to criterion (use first recall only); wider lag range

### Model Evaluation
- **Likelihood-based fitting** — Simulate encoding; predict each recall event given previous; sum log-probabilities; lower -logL = better fit
- **Generative assessment** — Fit model, simulate data, apply analyses; match to observed patterns tests specific predictions; failure can falsify model

### Diagnostic Tests
- **General setup** — All use matched symmetric baseline; conditional on eligibility; aggregate within then across participants; mixed-minus-baseline isolates repetition contribution

#### Repetition lag-CRP (outgoing)
- [Figure: schematic] — Repeated items occupy two positions on mental timeline
- **Goal + method** — Test blending vs occurrence-specific; two centered lag-CRPs (first-centered, second-centered); +1 from each center defined
- **Predictions** — Blending → balanced access (reduced separation); occurrence-specific → preserved/enlarged separation favoring first

#### Repetition lag-CRP (incoming)
- **Goal** — Distinguish cue-driven vs reinstatement-driven asymmetry; look one step back from repeater recall
- **Logic** — If asymmetry on incoming → cue-driven; if only on outgoing → reinstatement-driven

#### Cross-occurrence neighbor contiguity
- **Prior work** — Lohnas 2014 measured neighbor-to-neighbor transitions; found elevation
- **Refinements** — Symmetric baseline scoring; recast as centered lag-CRP triggered by neighbor, centered on other occurrence; reveals shape not just bins
- **Predictions** — Blending → above-baseline at other occurrence's neighbors; occurrence-specific → overlay baseline

#### Serial forward-chaining test
- **Goal + method** — After correct report through position i, measure i→i+1 (correct) vs i→j+1 (cross-occurrence error)
- **Predictions** — Blending → elevated j+1 errors; occurrence-specific → no elevation, intact chaining

## Reassessing Empirical Benchmarks Against CMR
- **Verify basic benchmarks** — Before repetition-specific tests, check mixed vs control under baseline; raw SPCs show mid-list advantage for mixed; after baseline correction, mixed=control on SPC, lag-CRP, PFR
- **CMR reproduces benchmarks** — Standard CMR matches SPC, lag-CRP, PFR; predicts minimal mixed-control differences; confirms starting point for repetition-specific tests
- [Figure: raw vs corrected benchmarks] — Shows inflation vanishes under baseline
- [Figure: CMR benchmarks] — Model reproduces patterns

### The Spacing Effect
- **Context on spacing** — Classically a repetition effect; but applies to distinct item pairs too (OR score); "superadditivity" = repeated items benefit more than distinct pairs; attributed to study-phase retrieval
- **Data vs CMR** — Data show steeper slope for mixed vs control (superadditive hint); baseline controls for position/scoring artifacts; CMR captures monotonic advantage but not slope separation
- [Figure: spacing curves] — Data (left) vs CMR (right)

### Cross-Occurrence Neighbor Transitions
- **Goal** — Test whether repetition links neighborhoods of two occurrences
- **Two variants** — First→second: triggers from i+1/i+2, lags centered on j; Second→first: triggers from j+1/j+2, lags centered on i
- **CMR prediction** — Elevated cross-occurrence transitions both directions; above-baseline mass at ℓ=±1,±2
- **Data** — No elevation at other occurrence's neighbors; but selective lag-0 bump in first→second only (not second→first)
- **Interpretation** — No neighborhood knitting; lag-0 effect suggests heightened access to repeated item without blending; foreshadows ICMR-Reinf
- [Figure: neighbor contiguity 2×2] — Data (top) vs CMR (bottom); first→second (left) vs second→first (right)

### The Repetition Lag-CRP
- **Data** — Transitions from repeater favor first occurrence's neighbors; +1 bin first-centered > +1 bin second-centered; separation larger than baseline
- **CMR prediction** — Balanced access (blending); predicts reduced separation relative to baseline; opposite of data; isolates problem: item-based reinstatement routes through composite
- [Figure: rep lag-CRP 2×2] — Mixed vs control × data vs CMR

### Backward/Incoming Variant
- **Data + CMR + interpretation** — Arrivals at repeaters favor first occurrence's backward neighbor; present in mixed and baseline (general serial-order dynamics); CMR yields weak first-over-second; caution: outgoing asymmetry may partly inherit from approach; mixed-minus-baseline on outgoing is cleaner probe
- [Figure: backward rep lag-CRP 2×2]

### Forward Chaining in Serial Recall
- **Data** — After correct report through position i, continuations to i+1 strong; errors to j+1 rare and at baseline; second occurrence doesn't draw responses from correct chain
- **CMR prediction** — Elevated i→j+1 errors; generalizes non-interference to serial recall: whether constrained or unconstrained, routing through single episode
- [Figure: serial rep-CRP 2×2] — Two datasets × data vs CMR

### Summary: Noninterference Across Tasks
- **Synthesis** — Three diagnostics (neighbor contiguity, rep lag-CRP, forward chaining) × two tasks (free, serial) → non-interference; CMR fits benchmarks + spacing but fails interference predictions; motivates mechanistic surgery: NoSPR → OS → Reinf

## CMR-NoSPR: Removing Study-Phase Retrieval
- **Intro + questions** — Removes study-phase reinstatement; at repetition, reinstates fresh non-overlapping context; targets knitting at study, preserves aggregation at retrieval; asks: (i) harm to benchmarks/spacing? (ii) resolve interference signatures?
- **Benchmarks** — Preserved; SPC, lag-CRP, PFR match data; mixed=control under baseline
- [Figure: NoSPR benchmarks]
- **Spacing** — Preserved; contextual variability + test-phase reinstatement sufficient; CMR's study-phase reinstatement not required
- [Figure: NoSPR spacing]
- **Neighbor contiguity** — Fixed; distinct encoding contexts abolish neighborhood knitting; mixed-minus-baseline at zero in both directions
- [Figure: NoSPR neighbor contiguity]
- **Repetition lag-CRP** — Still shows balanced access; eliminating study-phase reinstatement doesn't restore first-over-second; repeater still reinstates composite cue
- [Figure: NoSPR rep lag-CRP]
- **Conclusion** — Study-phase reinstatement not necessary for spacing or benchmarks; fixes neighbor knitting but not rep lag-CRP blending; isolates remaining problem to test-phase → motivates ICMR-OS

## ICMR-OS: Occurrence-Specific Reinstatement
- **Intro + questions** — Goes beyond NoSPR: separates occurrences at both stages; distinct contexts at study (like NoSPR) + traces compete at test (new); selection via trace-sensitivity (no new DOF); asks: (i) benchmarks? (ii) neighbor knitting? (iii) balanced access? (iv) serial recall?
- **Benchmarks** — Preserved; occurrence-specific reinstatement doesn't degrade fits
- [Figure: OS benchmarks]
- **Spacing** — Preserved; model doesn't reproduce steeper mixed-list slope
- [Figure: OS spacing]
- **Neighbor contiguity** — Fixed; inherits distinct contexts + adds selective reinstatement; no above-baseline peaks
- [Figure: OS neighbor contiguity]
- **Repetition lag-CRP (outgoing)** — Fixed: removes balanced access; preserves first-over-second separation; but doesn't boost separation beyond baseline (data show boost)
- **Repetition lag-CRP (incoming)** — Selective reinstatement doesn't affect incoming much; modest first-over-second, weaker than data
- [Figure: OS rep lag-CRP 2×2]
- **Serial recall** — Fixed; maintains i→i+1 chaining; no elevation of i→j+1 errors; matches non-interference in both datasets
- [Figure: OS serial rep-CRP 2×2]
- **Conclusion** — Reconciles RCT with non-interference; preserves benchmarks; eliminates neighbor knitting + balanced access + forward errors; but residual asymmetries unexplained: (i) lag-0 boost, (ii) incoming first-over-second, (iii) boosted outgoing separation beyond baseline → motivates ICMR-Reinf

## ICMR-Reinf: Reinforcement Without Blending
- **Intro + questions** — Reconceptualizes study-phase retrieval as reinforcement not reinstatement; strengthens first-occurrence trace at second presentation; contexts remain distinct; one added parameter (reinforcement factor); asks: (i) benchmarks? (ii) neighbor knitting? (iii) boosted first-over-second? (iv) serial recall?
- **Benchmarks** — Preserved; reinforcement doesn't degrade fits
- [Figure: Reinf benchmarks]
- **Spacing** — Preserved; doesn't reproduce steeper mixed-list slope
- [Figure: Reinf spacing]
- **Note on mid-list boost** — Reinforcement potentially boosts recall for repeated items; more refined model could balance trace strengths differently
- **Neighbor contiguity** — No knitting (inherits distinct contexts + selective reinstatement); plus captures selective lag-0 boost in first→second variant
- [Figure: Reinf neighbor contiguity]
- **Repetition lag-CRP (outgoing)** — Not only avoids balanced access but amplifies first-over-second separation beyond baseline; captures positive repetition-specific asymmetry in data
- [Figure: Reinf rep lag-CRP 2×2]
- **Repetition lag-CRP (incoming)** — Reinforcement increases first-over-second; moves model toward empirical pattern; outgoing remains cleaner probe
- [Figure: Reinf incoming lag-CRP 2×2]
- **Serial recall** — Non-interference preserved; maintains i→i+1; no j+1 error elevation
- [Figure: Reinf serial rep-CRP 2×2]
- **Conclusion** — Keeps occurrences separate at study + selective at test + reinforces first; preserves all benchmarks; captures all diagnostic patterns; price = one parameter; payoff = rehabilitated non-associative study-phase mechanism

## Formal Comparison
- **Setup** — Four variants (CMR, NoSPR, OS, Reinf); common fitting pipeline; metrics: -logL (lower = better), AICw (penalizes parameters)
- **Three patterns** — (1) OS substantially beats CMR everywhere, especially serial recall; (2) Reinf adds reliable advantage in free recall despite extra parameter, tied with OS in serial; (3) NoSPR improves over CMR but insufficient alone
- [Table: Free recall -logL] — Reinf best; Lohnas Δ = -9.20, Broitman Δ = -9.54 vs CMR
- [Table: Serial recall -logL] — OS best; Logan Δ = -41.47, Kahana Δ = -92.37 vs CMR
- **AICw analysis** — Two-step: pairwise OS vs CMR (parameter-matched); three-way in free recall (Reinf adds one parameter)
- [Table: Pairwise AICw] — OS ≈ 1.00 in all four datasets; CMR ≈ 0
- [Table: Three-way AICw] — Reinf ≈ 1.00 in free recall; OS second; CMR not competitive
- **Win rates** — OS beats CMR for 80-100% of participants; Reinf beats OS for 63-64% in free recall, ~50% in serial
- **Division of labor** — OS necessary to eliminate interference (dominates serial recall); Reinf adds first-over-second boost (free recall advantage); NoSPR shows SPR not required; CMR consistently least favored

## General Discussion

### Opening
- **Research question restated** — Temporal contiguity organizes episodic search; question: do repetitions knit neighborhoods or remain separable? Addressed via position-matched, symmetric baseline in free + serial recall

### RCT recap
- **How RCT works** — Items couple to drifting context; reinstatement at study and recall; context as storage address and retrieval cue
- **Item-driven context evolution** — The key commitment; three consequences for repetition: (i) contextual variability (spacing), (ii) study-phase retrieval (knitting at study), (iii) composite reinstatement (blending at test)

### Main findings
- **Framework predicts interference** — Three signatures: neighbor linkage, balanced access, forward errors; data contradict all three under proper baseline
- **Finding 1: Neighbor contiguity** — No knitting; Lohnas 2014 found elevation but potential baseline confound; I found no boost once baseline applied; but lag-0 boost in first→second; NoSPR fixes knitting, Reinf captures lag-0
- **Finding 2: Rep lag-CRP** — First-over-second (not balanced access); Broitman 2024 and Lohnas 2025 corroborate; CMR predicts opposite; OS eliminates balanced access but not boosted separation; Reinf captures boost
- **Finding 3: Serial recall** — No cross-occurrence forward errors; extends classic chaining critique to RCT; CMR predicts elevation; two datasets show none

### What becomes of RCT?
- **Challenge stated** — Item-driven evolution is parsimonious but predicts unobserved interference; what then?
- **Resolution** — Narrow changes reconcile: NoSPR removes SPR, OS allows selective reinstatement; occurrence-specific code replaces item-based; still retrieved-context (evolves by retrieval) but event/instance/position-based

### Connection to positional coding
- **Serial recall literature** — Long emphasized position/timeline codes; my results diagnostic of position-like competition (i+1 preserved, j+1 not elevated); integrated framework for free + serial

### Item features caveat
- **Item features do matter** — Semantic organization, phonological similarity, PLIs; item features undeniably influence organization
- **How to reconcile** — Multiple context layers (semantic, source); additional parameter for item-dependent vs occurrence-specific weighting
- **Tentative endorsement** — No cross-occurrence interference weighs against item-based knitting in standard conditions; caveats: individual differences (minority might show blending), no mixture model evaluation; future work: broader conditions, PLIs, semantic clustering

### Spacing effect implications
- **Context** — Spacing classically a repetition effect; superadditivity (repeated > distinct pairs); attributed to SPR
- **SPR doesn't contribute to spacing in CMR** — Removing SPR preserves spacing; primary consequence of SPR is knitting, not strengthening
- **ICMR-Reinf as rehabilitated SPR** — Strengthens without knitting; captures three patterns (first-over-second, incoming asymmetry, lag-0 boost); fits not substantially improved over OS; example of non-associative SPR

### Closing
- **Bakery example revisited** — RCT predicts blending across visits; results suggest memory sidesteps this; positional/event-based code indexes each occurrence separately; "the bakery down the street" can cue either visit without blending; memory routes through one episode at a time; repeated experience strengthens access without sacrificing specificity
