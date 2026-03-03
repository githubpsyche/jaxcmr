# Model and Paradigm — Section Notes

## Structural logic

One section ("# Model and Paradigm"), five subsections. Full formal specification of eCMR, interleaved with how the trauma-film paradigm maps to model operations. Organized by what the simulations need, not by the model's internal logic.

**Why merged (vs. separate spec + paradigm mapping):** TalmiEEG demonstrates that treating model specification and task mapping as inseparable eliminates redundancy. Each equation is introduced alongside its functional role in the paradigm. Scale factors (the paradigm-specific multipliers) appear immediately after the equations they modify.

**Relationship to Overview of the Model:** The Overview gives the conceptual argument (why the model predicts selective interference). Model and Paradigm gives the formal machinery (how the model does it). The Overview should be readable without equations; Model and Paradigm should be readable without the Overview. Together they're complementary, not redundant.

**Emphasis:** Sculpted for downstream simulations. Encoding equations emphasize the scale factors that govern interference strength. Retrieval equations emphasize the control parameters. Reminder mechanism gets its own subsection (central to the reconsolidation-replacement argument). Recognition operationalized explicitly (critical for the graded immunity prediction).

## Subsection arc

| Subsection | Role | Key content | Simulation connection |
|---|---|---|---|
| Representations and Memory Architecture | Foundations | $f_i$, $c^T$, $c^E$, $M^{FC}$, $M^{CF}$, pre-experimental initialization, parameter table | All simulations |
| Encoding: Film, Interference, and Filler Phases | Encoding machinery | Context drift, context input, MFC/MCF learning, primacy, emotional boost, scale factors | Decompose Interference sweeps |
| Reminders and Context Reinstatement | Reminder mechanism | Item→context reinstatement without new learning, `reminder_drift_scale` | Context Reinstatement section |
| Retrieval: Three Modes | Retrieval pathways | Context→item (free recall/intrusions), item→context (recognition), unguided vs directed (control parameters) | Selective Interference Effect, What Protects Voluntary Recall, Recognition Immunity |
| Calibration and Paradigm Geometry | Methodology | HK2014 fits, per-subject parameters, paradigm geometry, dependent measures, baseline validation | All simulations |

## Key implementation details

- **Source of equations:** `jaxcmr/selective_interference/cmr.py` (PhasedCMR). Equations verified against code.
- **11 base parameters:** Fitted per-subject to Healey & Kahana 2014 via differential evolution (126 subjects).
- **Scale factors:** Paradigm-specific multipliers on base parameters. Primary manipulation in simulations:
  - $s^{\phi}_\text{int}$ (`interference_mcf_scale`) → competitor encoding strength
  - $s^{\beta}_\text{int}$ (`interference_drift_scale`) → context proximity during interference
  - $s^{\beta}_\text{fill}$ (`filler_drift_scale`) → filler suppresses interference recency
  - $s^{\beta}_\text{break}$ (`break_drift_scale`) → retention interval
  - $s^{\beta}_\text{rem}$ (`reminder_drift_scale`) → reinstatement strength
  - $s^{\beta}_\text{rem,start}$ (`reminder_start_drift_scale`) → pre-reminder context reset
  - $s^{\beta}_\text{start}$ (`start_drift_scale`) → start-of-recall context reinstatement
  - $s^{\tau}$ (`tau_scale`) → competition sharpness
  - $\omega_E$ (`emotion_scale`) → emotional pathway weight
  - $s^{\beta}_\text{emot}$ (`emotion_drift_scale`) → emotional context drift
- **eCMR variant:** Full emotional context channel with separate 2-D emotional context $c^E$, separate memory matrices $M^{FC}_E$ and $M^{CF}_E$. Implemented in `cmr.py`, gated by emotional flag $e_i$.

## Notation conventions (decided)

Following TalmiEEG with extensions for emotional pathway:
- Contexts: $c^T$ (temporal), $c^E$ (emotional)
- Memories: $M^{FC}$, $M^{CF}$ (temporal); $M^{FC}_E$, $M^{CF}_E$ (emotional)
- Drift rates: $\beta_{enc}$, $\beta_{start}$, $\beta_{rec}$, $\beta_{emot}$
- Learning: $\gamma$ (MFC), $\phi_i = \phi_s e^{-\phi_d \cdot i} + 1$ (MCF primacy)
- Competition: $\tau$ (Luce exponent), $\omega_E$ (emotional weight)
- Termination: $\theta_s$, $\theta_r$ (positional stop probability)
- Scale factors: $s^{\beta}_\text{phase}$ (drift), $s^{\phi}_\text{phase}$ (MCF learning)
- Emotional context indexing: $k \in \{0, 1\}$ (neutral, arousal) — distinct from item/context indices $i$, $j$
- Element-wise learning notation (matching TalmiEEG): $\Delta M^{FC}_{ij} = \gamma f_i c^T_j$

## Resolved open problems

1. **Equation notation.** Decided: follow TalmiEEG conventions, use $k$ for emotional context dimension to avoid index confusion. Element-wise notation for learning equations.
2. **Parameter table scope.** Two tables: @tbl-parameters (structures + base parameters) in 4a, @tbl-scale-factors (all scale factors with simulation connections) in 4e.
3. **eCMR emotional context.** Implemented full source-context eCMR in `cmr.py`. Formally specified in 4a (initialization) and 4b (encoding). Equations match code.
4. **Recognition operationalization.** Described in 4d as item-to-context retrieval with architectural immunity. Competitor encoding modifies only competitors' rows, not film items' associations.

## Open problems

1. **Reminder equation notation.** Used `c^T.integrate(...)` notation in 4c — informal. Consider replacing with standard $\rho$/$\beta$ update.
2. **Recognition equation.** Current equation $P(\text{recognized} \mid f_i) = P(i \mid f_i \text{ cues context})$ is conceptual, not formal. May need to operationalize as `outcome_probability(choice)` mapping.

## Session log

- **2026-03-03**: Created section notes file. Structure planned: 5 subsections, merged formal spec + paradigm mapping.
- **2026-03-03**: Implemented full emotional context in PhasedCMR (`cmr.py`), added `emotion_drift_scale` to pipeline. Verified backward compatibility. Drafted all 5 subsections of Model and Paradigm in `index.qmd`. Notation conventions decided. Parameter table split into two: structures+base params (4a) and scale factors (4e).
