# Slide Plan: Lab Group Talk

Reveal.js slide deck presenting the selective interference simulation project to a broader audience.

## What exists

-   **Feb 10 deck** (`notes/2026_02_10/progress_update.qmd`): 1:1 progress update for Deborah. Reveal.js, simple theme, speaker notes. Contains data figures (SPC, lag-CRP, PNR, cue effectiveness) and sim figures (baseline SPC, Sim 1 sweeps, Sim 3 interaction). Reusable as format reference and figure source.
-   **Rendered sim figures** (`simulations/figures/`): calibration (4), interference (3), retrieval (2) — all as .png and .svg.
-   **Data figures** (`notes/2026_02_10/figures/`): SPC by task/condition, lag-CRP by task, PNR by task, cue effectiveness by task/condition.
-   **Missing figures**: eCMR/arousal sweep, core factorial, context overlap demo — being built today in parallel.

## Slide structure

### 1. The puzzle (2-3 slides)

-   **The selective interference effect**: After watching a distressing film, a visuospatial task (e.g. Tetris) reduces intrusive memories but leaves voluntary recall relatively intact.
-   **Why it matters**: Clinical translation — if we understand the mechanism, we can design better interventions.
-   **The standard explanation** (briefly): Separate memory traces (sensory-perceptual vs narrative), consolidation/reconsolidation windows, modality-specific disruption. No single named theory — a cluster of assumptions pervading the literature.

### 2. The alternative (2-3 slides)

-   **One memory system, two retrieval modes**: Context-to-item (free recall, intrusions) vs item-to-context (recognition). Interference arises from competitor encoding in shared temporal context, not separate traces.
-   **eCMR architecture** (one diagram slide): Temporal + emotional context, M_CF and M_FC, encoding and retrieval operations. Keep it schematic.
-   **The paradigm in CMR terms**: Film → delay → reminder → interference encoding → filler → recall.

### 3. Core result: the selective interference effect (2-3 slides)

-   **Interference x control factorial**: Unguided recall fully impaired, directed recall partially impaired, recognition completely spared.
    -   Figure: core factorial result (if rendered today; placeholder if not)
-   **What drives interference — context overlap**: Competitors must share context with film items to interfere. Mere existence is not enough.
    -   Figure: drift rate sweep (existing: `interference_drift_scale.png`)
-   **Graded immunity prediction**: Recognition \> directed \> unguided. Falls out of architecture, not separate stores.

### 4. Decompose the mechanisms (3-4 slides)

-   **What makes interference stronger**: MCF strength, context proximity, competitor count — all graded, all via context overlap.
    -   Figures: existing Sim 1 sweeps (MCF, drift, count)
-   **Arousal broadens interference** (eCMR): Shared arousal context between film and Tetris produces interference beyond temporal proximity.
    -   Figure: arousal sweep (if rendered today; placeholder if not)
-   **What protects voluntary recall**: Start-drift (repositions context) + tau (sharpens choice). Synergistic.
    -   Figure: existing Sim 3 interaction (`retrieval_start_drift_scale.png`, `retrieval_tau_scale.png`, or the interaction heatmap)

### 5. Connecting to data (3-4 slides)

-   **VRT experiment**: 240 trials, 2x2x2, 11 film clips, 22 cues at test. Key result: NO selective interference effect found.
-   **CMR signatures in VRT data**: Temporal contiguity (lag-CRP), serial position effects (SPC), start-drift signatures (PNR by task).
    -   Figures: existing data figures from Feb 10 deck
-   **Why no selective interference?** The VRT presents film cues during retrieval. Cues reinstate film context, bypassing start-drift. This reduces the voluntary/involuntary gap. Hypothesis — not yet simulated.
-   **Evidence**: Cue-match rates higher in involuntary than voluntary condition (cues provide more marginal benefit when no strategic context positioning is underway). No intervention or condition effect on cue match rate (item→context pathway immune to competitor encoding).

### 6. What's next (1-2 slides)

-   Remaining simulations: eCMR arousal (today), core factorial (today), cue-at-test (planned), recognition (planned)
-   Analysis plan: VRT analyses mirroring each simulation section
-   Manuscript: Quarto, simulation notebooks embedded
-   Experiment design: model-informed paradigm improvements

## Implementation details

-   Format: revealjs, simple theme (match Feb 10 deck)
-   Reuse `style.css` from Feb 10 deck (or copy it)
-   Figures: reference `../simulations/figures/` for sim figures, symlink or copy data figures from `2026_02_10/figures/`
-   Speaker notes on every figure slide
-   Use `.smaller` class for text-heavy slides
-   Use `{.r-stretch fig-align="center"}` for figure slides
-   Use `. . .` for incremental reveals where helpful
-   Placeholder slides for figures being rendered today, marked with `[PLACEHOLDER — rendering today]`