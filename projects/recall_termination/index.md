# Recall Termination — Project Plan

## Summary
- Identify empirical constraints on termination in sequential memory search, starting with free recall. Evaluate candidate termination mechanisms within the retrieved-context framework (CMR as the accessibility baseline), then select or propose the mechanism that best satisfies the constraints.

## Objective
- Returns a vetted termination mechanism for free recall, integrated as a swappable policy within CMR’s retrieval process, with per-subject fits and model-selection evidence (AIC/BIC). Expand to serial recall and IRT-based constraints if time permits.

## Scope
- In scope (initial):
  - Free recall only; termination mechanisms as policies within CMR.
  - Termination-agnostic baseline fits using masked likelihood (exclude explicit stop events) to validate CMR as an accessibility model on chosen datasets.
  - Per-subject parameter estimation; across-subject assessment of mechanisms.
- Out of scope (initial):
  - External time budgets or stopping beyond the internal policy (placeholder milestone).
  - Detailed inclusion/exclusion rules (intrusions, PLIs/ELIs) until dataset work begins.
- Future placeholders:
  - Serial recall constraints and mechanism transfer.
  - IRT-derived constraints and time-based stopping.

## Datasets (initial targets)
- Start with Cohen & Kahana (2022).
- Consider adding: Murdock (1962) or newer variable list-length datasets; Lohnas & Kahana (2014) with item repetitions.
- Trial selection specifics and handling of intrusions/repeats: defer until dataset prep.

## Success Criteria
- Baseline validation: CMR (termination-agnostic) provides good fit on selected dataset(s), keeping first recalls in.
- Mechanism selection: a termination policy yields better AIC/BIC than alternatives for a majority of subjects and matches qualitative constraint patterns.
- Reproducibility: results hold under an additional dataset or split.
- Thresholds: exact acceptance thresholds TBD after literature review (M1), with proposals for log-likelihood/AIC deltas and constraint fit metrics.

## Milestones (2 hrs/day, starting 2025‑09‑30)
- M0 — Kickoff and alignment: 2025‑09‑30
- M1 — Literature sweep; enumerate constraints & mechanism families; set evaluation plan: 2025‑10‑14
- M2 — Baseline CMR validation (termination‑agnostic) on Dataset 1; per‑subject fits: 2025‑10‑28
- M3 — Implement swappable termination policies; initial mechanism screening on Dataset 1: 2025‑11‑11
- M4 — Joint fitting (CMR + termination params); model selection (AIC/BIC); across‑subject assessment: 2025‑11‑25
- M5 — Extensions: add Dataset 2 or constraint variants; explore new analyses/mechanism tweaks: 2025‑12‑09
- M6 — Placeholders: IRT constraints and external/time‑budget stopping: 2025‑12‑16
- M7 — Placeholders: Serial recall revisit and transfer test: 2025‑12‑23
- M8 — Write‑up and synthesis: 2026‑01‑06

## Workstreams

### Problem Definition
- [ ] Draft problem statement: termination in free recall within retrieved‑context theory; CMR as accessibility baseline.
- [ ] Enumerate empirical constraints to target; prioritize after literature sweep (M1).
- [ ] Define evaluation goals and selection criteria (LL/AIC/BIC + qualitative constraint matches).

### Termination Mechanisms (swappable policies within CMR)
- Promising options to review and potentially implement:
  - Fixed time‑out / constant hazard.
  - Failure‑count stopping (terminate after N unsuccessful internal attempts).
  - Evidence threshold: stop if max activation/evidence < threshold (static or adaptive).
  - Relative‑benefit/marginal value: stop when expected hit rate falls below a reference.
  - Support‑ratio rule (already present in a variant).
- Decisions on mechanism set: defer until M1; prioritize 2–3 for M3.

### Constraints & Analyses (what mechanisms must match)
- Promising options to formalize (final choice after M1):
  - Distribution of total recalls per trial (by list length).
  - Termination hazard across output position/time; last‑IRT distribution (placeholder until IRT work).
  - Relationship between termination and estimated accessibility (e.g., maximum item support over time).
  - Diminishing returns/marginal value patterns.
  - Pre‑termination transition patterns (e.g., contiguity/lag features before stopping).

### Data & Instrumentation
- [ ] Select Dataset 1 (Cohen & Kahana 2022) and prepare `RecallDataset` fields.
- [ ] Plan Dataset 2 (e.g., Murdock 1962 variable list length) and Dataset 3 (Lohnas & Kahana 2014) as follow‑ups.
- [ ] Defer intrusion/repeat handling specifics until dataset prep.

### Modeling & Algorithms
- Baseline: use termination‑agnostic masked loss via `ExcludeTerminationLikelihoodFnGenerator`; keep first recalls in; per‑subject fits.
- Mechanisms: implement as policies inside CMR outcome probabilities (`stop_probability`) and fit jointly with CMR parameters when screening.
- Repeated recalls: decision deferred; may vary by mechanism.

### Evaluation
- Objectives: log‑likelihood and AIC/BIC per subject; qualitative/quantitative constraint checks.
- Protocol: per‑subject fits; summarize across subjects; optional held‑out split for robustness. Cross‑validation TBD after M1.
- Reporting: plots for list‑length distribution, termination hazard, constraint residuals; comparison table across mechanisms.

### Documentation
- [ ] Baseline validation notes and plots (Dataset 1).
- [ ] Mechanism comparison summary (per‑subject and aggregate).
- [ ] Constraint definitions and how each mechanism addresses them.

## Backlog (prioritized)
- [P0] Literature sweep → enumerate constraints and mechanism families; set acceptance metrics.
- [P0] Prepare Dataset 1 (Cohen & Kahana 2022) for baseline fits.
- [P0] Baseline per‑subject fits with termination‑agnostic masked loss; confirm CMR suitability.
- [P1] Implement 2–3 termination policies (swappable) and run initial screening.
- [P1] Add AIC/BIC selection and reporting; across‑subject summaries.
- [P2] Extend to Dataset 2 and/or constraint variants; explore IRT and serial‑recall placeholders.

## Decision Log
- 2025‑09‑30 — Keep first recalls in baseline fits. Rationale: align with planned analyses. Status: Accepted.
- 2025‑09‑30 — Use ExcludeTerminationLikelihoodFnGenerator for termination‑agnostic baseline. Rationale: isolate accessibility fit. Status: Accepted.
- 2025‑09‑30 — Fit per subject; assess mechanisms across subjects. Rationale: subject‑level variability. Status: Accepted.
- 2025‑09‑30 — Jointly fit CMR + termination parameters for mechanism evaluation. Rationale: interactions matter. Status: Accepted.
- 2025‑09‑30 — Use AIC/BIC for model selection; cross‑validation TBD post‑M1. Status: Accepted.
- 2025‑09‑30 — Implement mechanisms as swappable policies within CMR (internal stopping only initially). Status: Accepted.
- 2025‑09‑30 — Repeated recalls gating: TBD after mechanism scoping. Status: Open.

## Risks & Mitigations
- Dataset availability/format mismatch → Start with one known dataset; keep loaders thin; defer edge handling.
- Constraint ambiguity → Time‑box literature sweep; codify constraints with plots and metrics in M1.
- Mechanism overfitting to Dataset 1 → Add Dataset 2 or held‑out split in M5.
- Parameter identifiability (joint fits) → Use parsimony (AIC/BIC), sensitivity checks, and priors/ranges if needed.

## Open Questions
- [?] Which 2–3 constraints/mechanisms should be P0 after M1?
- [?] Exact acceptance thresholds for constraint matches and AIC/BIC deltas?
- [?] Repeated‑recall policy across mechanisms?
- [?] Add cross‑validation or rely on held‑out splits only?

## References (to review)
- Datasets: Cohen & Kahana (2022); Murdock (1962); Lohnas & Kahana (2014).
- Mechanism families: optimal foraging/marginal value; fixed time‑out/constant hazard; failure‑count stopping; dynamic evidence thresholds; support‑ratio rules.
