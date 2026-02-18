# VRT Analysis Plan

Empirical evaluation of the CMR account against Video Recognition Task data, mirroring the structure of the simulation plan.

## Framing

The VRT experiment found no selective interference effect.
Task (involuntary vs voluntary) x intervention (podcast vs Tetris) did not produce the predicted interaction — Tetris did not selectively impair involuntary recall while partially sparing voluntary recall.

One candidate explanation, consistent with CMR's architecture but not yet a simulation output: test-phase cues partially override retrieval control by reinstating film context via item-to-context retrieval, reducing the voluntary/involuntary distinction.
The VRT presents 22 film cues during retrieval — exactly the condition under which this hypothesis predicts attenuation.
Mechanistically, cues bypass the start-drift component of retrieval control (they position context directly at a film region) but not tau (choice sensitivity still differs across task conditions).
So the prediction is a *reduced* task gap for cue-driven recalls, not elimination.
Whether this reduction is sufficient to explain the full null result is an empirical question.

The analysis plan asks three questions:

1.  Do the mechanisms the CMR account relies on show up in VRT behavior?
2.  Do cues empirically reduce the voluntary/involuntary distinction?
3.  What does the VRT tell us about paradigm design for future tests?

**Dataset**: 240 trials, 2x2x2 (task x condition x intervention), \~30 per cell.
Task = involuntary/voluntary.
Condition = emotional/neutral.
Intervention = podcast/Tetris.
11 film clips, 22 cues + 68 foils at test.

## The analyses

### Core — The selective interference effect

**Simulation claim**: Interference x control factorial produces graded impairment — competitor encoding fully impairs unguided recall, partially impairs directed recall.

**VRT evaluation**: Document the null.
The task x intervention interaction on total recall and SPC does not show the predicted pattern.
This null is not necessarily disconfirming — the CMR account predicts that test-phase cues attenuate the selective interference effect (see test-phase cues section below), and the VRT includes cues throughout retrieval.
The core selective interference prediction requires a cue-free paradigm.

-   Existing: total_recall_report (task x intervention), spc_report (task x intervention)
-   The null result is the empirical starting point for the rest of this document

### Core — Context reinstatement replaces reconsolidation

**Simulation claim**: Only reminder + competitors produces interference; neither alone is effective.
Interference effectiveness is a continuous function of reinstatement strength.

**VRT evaluation**: Not directly testable.
The VRT does not manipulate reminder presence or reinstatement strength.
All subjects experience the same pre-intervention structure.
Would require a different experiment.

### Decompose control — What protects voluntary recall

**Simulation claim**: Start-drift and tau produce graded protection with different effect shapes.

**VRT evaluation**: The task manipulation is the behavioral manifestation of retrieval control (voluntary = high control, involuntary = low control).
Cannot isolate start-drift from tau behaviorally, but can look for their signatures in the data.
These are tests of CMR's retrieval mechanisms, independent of whether the selective interference effect is present.

-   **Start-drift signature**: PNR by task — voluntary should show primacy bias at early output positions (context biased toward film onset); involuntary should not.
    -   Existing: pnr_report (task)
    -   New: PNR conditioned on cued vs uncued first recall (isolates spontaneous initiation from cue-driven initiation)
-   **Tau signature**: Lag-CRP sharpness by task — voluntary should show sharper contiguity (more concentrated at small \|lag\|) than involuntary.
    -   Existing: lag_crp_report (task, all transition masks)
    -   New: contiguity concentration metric by task (quantifies sharpness difference)
-   **Core CMR signature**: Temporal contiguity in both tasks. Doubly-uncued transitions show purest temporal contiguity uncontaminated by cue-driven positioning.
    -   Existing: lag_crp_report (doubly-uncued by task)

### Decompose interference — What makes interference stronger

**Simulation claim**: Interference varies along MCF encoding strength, context proximity, competitor count, and arousal-context overlap.

**VRT evaluation**: Only the arousal axis is manipulable — condition (emotional/neutral) maps onto arousal-context overlap.
The other three axes (MCF strength, proximity, count) are not parametrically varied in the VRT.

-   **Arousal**: Condition x intervention interaction on total recall and SPC. CMR predicts arousal-matched interference (emotional film + Tetris) shows a flatter SPC than neutral film + Tetris (arousal context broadens interference beyond temporal proximity).
    -   Existing: total_recall_report (condition x intervention), spc_report (condition x intervention)
    -   New: SPC difference (podcast - Tetris) by condition — does the interference profile differ for emotional vs neutral films?
-   **Recency gradient**: SPC shape of intervention effect tests the cross-cutting prediction that interference disproportionately suppresses late film items.
    -   Existing: spc_report (intervention)
    -   New: SPC difference score by serial position (podcast - Tetris)

### Follow-ups — Recognition immunity

**Simulation claim**: Recognition signal unchanged by competitor encoding — architectural immunity via a different retrieval pathway.

**VRT evaluation**: Not testable.
The VRT does not include a recognition test.

### Follow-ups — Test-phase cues

**Simulation claim**: Film cues reinstate context, override retrieval control, blur voluntary/involuntary distinction.
(Planned but unimplemented — the current model does not simulate cues at test.)

**VRT evaluation**: Directly testable empirically — and this section carries the most explanatory weight for the VRT data.
The VRT presents 22 film cues during retrieval.
The hypothesis (from CMR's architecture, not a simulation output): cues reinstate film context via item-to-context retrieval, partially overriding retrieval control and reducing the voluntary/involuntary distinction.
If the data support this, it motivates building the cue-at-test simulation and provides an account of the null selective interference result.

-   **Cue reinstatement**: Cue-centered CRP should show graded temporal gradient from cue position (items near the cue in the film are more likely to be recalled after that cue).
    -   Existing: cue_centered_crp_report
-   **Cue effectiveness**: What fraction of cues elicit matching recalls.
    -   Existing: cue_effectiveness_report
-   **Cues reduce task distinction**: From-cued transitions should show smaller task differences than doubly-uncued transitions. Mechanistic reasoning: cues bypass start-drift (they position context directly) but not tau (choice sensitivity still differs). So the prediction is a reduced task gap for cue-driven recalls, not elimination.
    -   Existing: lag_crp_report (from-cued vs doubly-uncued by task)
    -   New: direct comparison of task effect size in cue-driven vs spontaneous recall (total recall and lag-CRP). The prediction: task differences are larger for spontaneous recalls than for cue-driven recalls.
-   **Spontaneous-only selective interference**: Re-examine total recall restricted to spontaneous (uncued) recalls only. If the selective interference effect is stronger in spontaneous recalls than in the aggregate, cues are partially masking it. Whether this reduction is sufficient to explain the full null result is an empirical question.
    -   Existing: total_recall_report has cue-driven/spontaneous breakdown
    -   New: task x intervention interaction on spontaneous-only recall

### Follow-ups — Paradigm design implications

**Simulation claim**: Model-informed recommendations for better experiments.

**VRT evaluation**: The VRT data directly informs this section.
The analyses above should reveal where the paradigm succeeds (cue-reinstatement effects, temporal contiguity signatures) and where it is limited (cues blur the distinction the experiment is meant to detect).
The analysis identifies which VRT results are interpretable and which are confounded by cue effects.

## Cross-cutting predictions

-   **Recency gradient**: Covered in Decompose interference (SPC difference score by serial position).
-   **Graded immunity**: Partially testable — involuntary vs voluntary gives two points on the gradient. The full gradient (recognition \> voluntary \> involuntary) requires recognition data the VRT does not have.
-   **Null headline result**: Candidate partial explanation via test-phase cues bypassing start-drift but not tau. Architectural reasoning, not yet a simulation result. Whether the reduction is sufficient to explain the full null is an empirical question the test-phase cues analyses address.