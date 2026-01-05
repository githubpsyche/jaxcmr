# Same item, different traces: Dissecting the structure of memory for repeated experience

Jordan Gunn, Sean Polyn, Department of Psychology, Vanderbilt University

## Introduction

### The Theoretical Puzzle: Does Memory Blend Repeated Experiences?

Image: A timeline diagram showing a study phase where item C appears at two positions (i and j) in the list. An arrow points from the recall phase back to the study list, with question marks indicating uncertainty about which neighborhood (i±1 or j±1) will be cued next. The diagram poses the central question: when C is recalled, which part of the mental timeline does memory revisit?

Retrieved-context theory (RCT) explains memory search as mental time travel: recalling an item reinstates its associated temporal context, biasing subsequent retrievals toward items studied nearby in time.

The Context Maintenance and Retrieval model (CMR; Polyn, Norman, & Kahana, 2009) implements RCT by coupling items to a slowly evolving temporal context. When an item is studied or recalled, it retrieves and partially reinstates contextual features associated with it, updating the current context state. This item-driven context evolution explains the lag-contiguity effect and other hallmarks of episodic memory search.

But when an item was studied twice, it has TWO associated contexts. Does retrieval blend them or select one?

### Two Competing Accounts

Image: A two-panel diagram contrasting item-based and position-based reinstatement mechanisms. Left panel shows CMR's approach: item C is linked to overlapping contexts at positions i and j; at retrieval, a composite context is reinstated, with arrows fanning out to both neighborhoods. Right panel shows Instance-CMR's approach: item C has two separate traces with distinct, non-overlapping contexts; at retrieval, traces compete, one wins, and only that trace's context is reinstated, with a single arrow to one neighborhood. Visual emphasis on the contrast: blending vs. selection.

**Panel A: Item-based Reinstatement (Standard CMR)**

In CMR, item identity drives context evolution. When C appears at position j, it retrieves context associated with its earlier presentation at i (study-phase retrieval), knitting the two contexts together. At recall, reinstating C produces a composite of both occurrence contexts.

```
ENCODING:
  C at position i → binds to context_i
  C at position j → reinstates context_i, then binds to blended context_j
                    (contexts become overlapping)

RETRIEVAL:
  Recall C → reinstates COMPOSITE of context_i and context_j
          → activates neighbors of BOTH i and j

PREDICTION: Cross-occurrence associative interference
```

**Panel B: Position-based Reinstatement (Instance-CMR)**

In Instance-CMR, position (not item) drives context evolution. Each occurrence encodes to a fresh, non-overlapping context. At recall, occurrence traces compete; the winner's context is selectively reinstated.

```
ENCODING:
  C at position i → binds to context_i (position-specific)
  C at position j → binds to context_j (fresh, non-overlapping)
                    (no study-phase context reinstatement)

RETRIEVAL:
  Recall C → traces COMPETE for selection
          → winning trace's context reinstated
          → activates neighbors of ONE occurrence only

PREDICTION: Occurrence-specific access (no interference)
```

**Reinforcement variant:** At second presentation, strengthen first-occurrence trace WITHOUT reinstating its context. This preserves distinct contexts while enhancing access to the first occurrence.

The key insight: Position-based coding treats each study event as unique, allowing retrieval to route through individual occurrences without blending.

## Datasets

**Free Recall:** Lohnas & Kahana (2014)
- 35 participants, delayed free recall, 48 lists of 40 words
- Control lists: no repetitions
- Mixed lists: 28 singletons + 6 items repeated with 0-8 intervening items

**Serial Recall:** Logan & Cox (2021) Experiment 2
- 24 participants, typed serial recall, 6-letter lists
- Control lists: no repetitions
- Mixed lists: one letter repeated at lags 0-3

Additional datasets (Lohnas 2025; Kahana & Jacobs 2000) confirm generalization. Results available on request.

## Methodological Innovation: Position-Matched Symmetric Baseline

Image: A side-by-side comparison of mixed and control lists using the same visual language as the setup diagram. Left side shows a mixed list timeline with item C appearing at positions 5 and 15, labeled as "Repeater." Right side shows a control list timeline with items X at position 5 and Y at position 15, with a bracket grouping them as "Pseudo-repeater." An equals sign or arrow connects the two, indicating that they are analyzed equivalently. Below, text emphasizes: "Recall of X OR Y counts as recall of pseudo-repeater. Subtract control from mixed to isolate TRUE repetition effects."

Previous analyses compared repeated items to singletons within the same mixed lists — but repetitions change rehearsal and output dynamics for ALL items.

Our approach:
1. Match each mixed-list trial to control-list trials with identical structure
2. Designate position-matched items as pseudo-repeaters
3. Apply SYMMETRIC scoring: recall of either matched position = recall of pseudo-repeater
4. Mixed minus control = repetition effect proper

This rigorous baseline reveals effects previous work missed and eliminates spurious effects from list-composition confounds.

## Repetition Lag-CRP: First-Presentation Bias

### How the Repetition Lag-CRP is Computed

Image: A schematic showing how lag is calculated relative to each presentation of a repeated item. A timeline shows item D studied at positions 4 and 11, and item F studied at position 6. One arrow from position 4 to position 6 indicates "lag = +2 relative to first presentation." Another arrow from position 11 to position 6 indicates "lag = -5 relative to second presentation." The diagram clarifies that the same transition is scored twice, once relative to each study position.

For recalls of repeated items with spacing ≥ 4:
1. Track transitions relative to first presentation (position i) AND second presentation (position j) separately
2. For each lag in [-3, -2, -1, +1, +2, +3]: count possible transitions and actual transitions
3. Conditional probability = Actual / Possible
4. Compare first-centered curve vs second-centered curve

### Results

Image: A 2×4 grid of repetition lag-CRP plots arranged with columns for Data, CMR, Positional CMR, and Reinf Positional CMR, and rows for Mixed lists (top) and Control lists (bottom). Each panel contains two curves: a blue/solid "first-centered" curve showing transition probabilities relative to position i, and an orange/dashed "second-centered" curve showing probabilities relative to position j. X-axis ranges from lag -3 to +3 (excluding 0). Y-axis shows conditional response probability from 0 to approximately 0.3. Key visual pattern: In Data-Mixed, the first-centered curve peaks much higher than second-centered at +1 lag. In CMR-Mixed, the curves nearly overlap. In Positional-Mixed, curves show moderate separation similar to control. In Reinf-Mixed, curves show large separation matching Data.

**Row 1 (Mixed lists):**
- Data: Large separation between curves at +1 lag; first-centered curve substantially higher than second-centered. Shows strong first-presentation bias.
- CMR: Curves nearly overlap; minimal separation. Predicts balanced access (FAILS).
- Positional CMR: Separation matches control baseline. Fixes interference but no boost.
- Reinf Positional CMR: Large separation matching data. Captures first-presentation bias (SUCCEEDS).

**Row 2 (Control lists):**
- All four columns show moderate baseline separation between first- and second-centered curves (serial position effects).
- The KEY comparison: Does mixed-list separation EXCEED control separation?
  - Data: YES (repetition effect)
  - CMR: NO (predicts interference that doesn't exist)
  - Positional: NO (matches baseline — fixes interference)
  - Reinf Positional: YES (matches data — captures the boost)

**Statistical test (controlled +1 transition rate, first vs second):**
- Data: t = 3.70, p = 0.0004
- CMR: t = 1.07, p = 0.15 (not significant — fails to capture effect)
- Reinf Positional CMR: t = 3.12, p = 0.002 (captures effect)

## Cross-Occurrence Neighbor Transitions: No Knitting at Study

This analysis tests whether study-phase retrieval links the neighborhoods of the two occurrences.

Trigger: Recall a neighbor of one occurrence (e.g., position i+1 or i+2)
Measure: Probability of next transition to the OTHER occurrence's neighborhood (centered on j)

CMR predicts: Elevated transitions to j±1, j±2 (study-phase retrieval knits neighborhoods together)
Data shows: Flat curve at baseline (no cross-occurrence linkage)

Image: Three panels arranged horizontally, labeled "Data," "CMR," and "Reinf Positional CMR." Each panel shows a centered lag-CRP with x-axis from -3 to +3 (centered on position j) and y-axis showing conditional response probability. Each panel contains two overlaid curves: mixed lists (solid line) and control lists (dashed line). In the Data panel, the solid and dashed lines overlap almost perfectly, sitting flat near baseline, except for a small bump at lag=0. In the CMR panel, the solid line (mixed) rises above the dashed line (control) at lags ±1 and ±2, showing predicted cross-occurrence elevation. In the Reinf Positional panel, solid and dashed lines overlap like the Data panel, with a small lag=0 bump.

- Panel 1 (Data): Mixed and control curves overlap completely. No elevation at j±1, j±2. Exception: selective elevation at lag=0 (the repeated item itself) — suggests boosted access to the repeater without neighborhood knitting.
- Panel 2 (CMR): Mixed curve elevated above control at j±1, j±2. Predicts cross-occurrence knitting that DATA DOES NOT SHOW.
- Panel 3 (Reinf Positional CMR): Mixed and control curves overlap. Correctly predicts no knitting. Captures the lag=0 boost.

**Implication:** Study-phase context reinstatement (as implemented in CMR) is not supported. Occurrences encode to distinct, non-overlapping contexts.

## Serial Recall: No Cross-Occurrence Forward Errors

In serial recall, after correctly reporting through position i (first occurrence of repeated item), does retrieval erroneously jump to j+1 (second occurrence's forward neighbor)?

CMR predicts: Elevated i→j+1 errors (blended context activates both forward neighbors)
Data shows: No elevation above position-matched baseline

Image: Three panels arranged horizontally, labeled "Data," "CMR," and "Reinf Positional CMR." Each panel shows a bar chart or point plot with x-axis showing transition targets (j+1 and j+2) and y-axis showing conditional probability of erroneous transition. Each panel contains two series: mixed lists (solid/filled) and control lists (dashed/unfilled). In the Data panel, mixed and control bars are approximately equal height — no elevation of cross-occurrence errors. In the CMR panel, mixed bars are noticeably taller than control bars, especially at j+1, showing the predicted (but unsupported) error elevation. In the Reinf Positional panel, mixed and control bars are again approximately equal, correctly matching the data pattern.

- Panel 1 (Data): Mixed and control curves overlap. No elevated cross-occurrence errors. Forward chaining intact.
- Panel 2 (CMR): Mixed curve elevated above control. Predicts errors that DATA DOES NOT SHOW.
- Panel 3 (Reinf Positional CMR): Mixed and control curves overlap. Correctly predicts intact forward chaining.

**Implication:** The non-interference pattern generalizes from free recall to serial recall. Item-based chaining models face the same challenge as item-based context models.

## Model Comparison

Image: A table with 4 columns (one per dataset) and 3 rows (one per model). Column headers identify each dataset: two free recall (Lohnas & Kahana 2014, Lohnas 2025) and two serial recall (Logan & Cox 2021, Kahana & Jacobs 2000). Row labels identify each model: CMR, Positional CMR, Reinf Positional CMR. Cells contain AIC weights formatted to show decisive preferences. The CMR row shows zeros across all datasets. The Positional CMR row shows near-zero for free recall but 1.00 for Logan serial recall. The Reinf Positional CMR row shows 1.00 (bolded) for three of four datasets. Color coding or bolding emphasizes the winning model in each column.

| Model | Free Recall (Lohnas & Kahana 2014) | Free Recall (Lohnas 2025) | Serial Recall (Logan & Cox 2021) | Serial Recall (Kahana & Jacobs 2000) |
|-------|-----------------------------------|---------------------------|----------------------------------|--------------------------------------|
| CMR | 0 | 0 | 0 | 0 |
| Positional CMR | ~0 | ~0 | 1.00 | ~0 |
| Reinf Positional CMR | **1.00** | **1.00** | ~1.00 | **1.00** |

Table: AIC weights (higher = better; weights sum to 1 within each dataset). Models fitted via maximum likelihood to trial-by-trial recall sequences. Reinf Positional CMR is decisively preferred in free recall. Both positional variants dominate in serial recall. All variants fit standard benchmarks (SPC, CRP, PFR) comparably.

**Parameter note:** Positional CMR has the same parameter count as standard CMR. Reinf Positional CMR adds one reinforcement parameter. Despite this penalty, it is strongly preferred.

## Conclusions

### Memory routes through individual occurrences without blending

Three diagnostic tests of cross-occurrence interference all fail to support CMR's predictions:
- Repetition lag-CRP: First-presentation bias, not balanced access
- Neighbor contiguity: No cross-occurrence knitting
- Serial forward chaining: No elevated errors

Position-based context coding — where each study event encodes to a distinct temporal context — eliminates these interference predictions while preserving classic free and serial recall benchmarks.

### First-trace strengthening explains the boost

Beyond eliminating interference, data show ENHANCED access to first-presentation neighbors.

Reinforcement at the second presentation — strengthening the first trace WITHOUT reinstating its context — captures this pattern. This rehabilitates study-phase retrieval as a non-associative strengthening mechanism rather than context blending.

### Take-home

Repeated experience creates distinct memory traces that can be accessed separately. Study-phase retrieval may strengthen earlier traces without knitting their contexts together — preserving what happened when.

## References

Howard, M. W., & Kahana, M. J. (2002). Journal of Mathematical Psychology.
Kahana, M. J., & Jacobs, J. (2000). Psychonomic Bulletin & Review.
Logan, G. D., & Cox, G. E. (2021). Psychological Review.
Lohnas, L. J., & Kahana, M. J. (2014). JEP: Learning, Memory, and Cognition.
Polyn, S. M., Norman, K. A., & Kahana, M. J. (2009). Psychological Review.

---

**More results available on request.** Additional datasets (Lohnas 2025, Kahana & Jacobs 2000) and model variants (CMR-NoSPR, Blend Positional CMR, etc.) confirm these patterns. Ask me about them!
