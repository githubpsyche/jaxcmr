# Same item, different traces: Dissecting the structure of memory for repeated experience

Jordan Gunn, Sean Polyn, Department of Psychology, Vanderbilt University

## Introduction

### The Theoretical Puzzle: Does Memory Blend Repeated Experiences?

Image: A timeline diagram showing a study phase where item C appears at two positions (i and j) in the list. The recall phase shows C being retrieved, with question marks and arrows pointing toward both neighborhoods (i±1 and j±1), posing the question: which neighborhood does retrieval cue?

Retrieved-context theory explains memory search through item-driven context reinstatement: recalling an item reinstates its associated temporal context, cueing neighbors from study.

But when an item was studied twice, it has TWO associated contexts. Does retrieval blend them or select one?

### Two Competing Accounts

Image: A two-panel diagram contrasting item-based and position-based reinstatement mechanisms.

**Panel A: Item-based Reinstatement (Standard CMR)**

```
ENCODING:
  C at position i → context_i
  C at position j → context_j
                    (but study-phase retrieval partially reinstates context_i)
                    (contexts become overlapping/knitted together)

RETRIEVAL:
  Recall C → retrieves COMPOSITE of context_i and context_j
          → activates neighbors of BOTH i and j

PREDICTION: Cross-occurrence associative interference
  - Balanced access to both neighborhoods
  - Cross-occurrence neighbor transitions elevated
  - Forward-chaining errors in serial recall
```

**Panel B: Position-based Reinstatement (Instance-CMR)**

```
ENCODING:
  C at position i → context_i (fresh, position-specific)
  C at position j → context_j (fresh, non-overlapping with context_i)
                    (no study-phase context reinstatement)
                    (contexts remain as distinct as if different items)

RETRIEVAL:
  Recall C → traces COMPETE for selection
          → winning trace's context reinstated
          → activates neighbors of ONE occurrence only

PREDICTION: Occurrence-specific access (no interference)
  - Preferential access to one neighborhood
  - No cross-occurrence neighbor elevation
  - Intact forward chaining in serial recall
```

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

Image: A diagram showing the control analysis procedure. On the left, a mixed list shows item C at positions 5 and 15. On the right, a control list shows items X and Y at the same positions 5 and 15. An arrow indicates that X and Y are treated as a "pseudo-repeater" — recalls of either position count as recalls of the same pseudo-item. Text below: "Subtract control from mixed to isolate TRUE repetition effects."

Previous analyses compared repeated items to singletons within the same mixed lists — but repetitions change rehearsal and output dynamics for ALL items.

Our approach:
1. Match each mixed-list trial to control-list trials with identical structure
2. Designate position-matched items as pseudo-repeaters
3. Apply SYMMETRIC scoring: recall of either matched position = recall of pseudo-repeater
4. Mixed minus control = repetition effect proper

This rigorous baseline reveals effects previous work missed and eliminates spurious effects from list-composition confounds.

## Repetition Lag-CRP: First-Presentation Bias

Image: A 2×4 grid of repetition lag-CRP plots. Columns are: Data, CMR, Positional CMR, Reinf Positional CMR. Rows are: Mixed lists, Control lists.

Each panel shows two curves: first-centered (transitions relative to position i) and second-centered (transitions relative to position j). X-axis: lag (-3 to +3). Y-axis: conditional response probability.

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

Image: Three panels showing neighbor-contiguity analysis (first→second variant: triggers from i+1/i+2, lags centered on j). Each panel overlays mixed (solid line) and control (dashed line) curves.

- Panel 1 (Data): Mixed and control curves overlap completely. No elevation at j±1, j±2. Exception: selective elevation at lag=0 (the repeated item itself) — suggests boosted access to the repeater without neighborhood knitting.
- Panel 2 (CMR): Mixed curve elevated above control at j±1, j±2. Predicts cross-occurrence knitting that DATA DOES NOT SHOW.
- Panel 3 (Reinf Positional CMR): Mixed and control curves overlap. Correctly predicts no knitting. Captures the lag=0 boost.

**Implication:** Study-phase context reinstatement (as implemented in CMR) is not supported. Occurrences encode to distinct, non-overlapping contexts.

## Serial Recall: No Cross-Occurrence Forward Errors

In serial recall, after correctly reporting through position i (first occurrence of repeated item), does retrieval erroneously jump to j+1 (second occurrence's forward neighbor)?

CMR predicts: Elevated i→j+1 errors (blended context activates both forward neighbors)
Data shows: No elevation above position-matched baseline

Image: Three panels showing serial forward-chaining analysis. Each panel overlays mixed (solid line) and control (dashed line) showing probability of transitions to j+1 and j+2 after correct report through position i.

- Panel 1 (Data): Mixed and control curves overlap. No elevated cross-occurrence errors. Forward chaining intact.
- Panel 2 (CMR): Mixed curve elevated above control. Predicts errors that DATA DOES NOT SHOW.
- Panel 3 (Reinf Positional CMR): Mixed and control curves overlap. Correctly predicts intact forward chaining.

**Implication:** The non-interference pattern generalizes from free recall to serial recall. Item-based chaining models face the same challenge as item-based context models.

## Model Comparison

| Model | Free Recall (Lohnas & Kahana 2014) | Free Recall (Lohnas 2025) | Serial Recall (Logan & Cox 2021) | Serial Recall (Kahana & Jacobs 2000) |
|-------|-----------------------------------|---------------------------|----------------------------------|--------------------------------------|
| CMR | 0 | 0 | 0 | 0 |
| Positional CMR | ~0 | ~0 | 1.00 | ~0 |
| Reinf Positional CMR | **1.00** | **1.00** | ~1.00 | **1.00** |

Table: AIC weights (higher = better; weights sum to 1 within each dataset). Reinf Positional CMR is decisively preferred in free recall. Both positional variants dominate in serial recall.

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
