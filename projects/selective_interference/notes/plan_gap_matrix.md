# Plan Gap Matrix: Notes vs Current Project State

## How to read this matrix

-   Evaluation context: repository snapshot as of 2026-02-17, using existing code/notebooks/figures only (no new reruns in this pass).
-   Active planning scope: `projects/selective_interference/notes/context_binding.md` and `projects/selective_interference/notes/plan.md`.
-   Redundancy policy in the merged table: each row is a single claim; use multi-source `Source` entries when the same claim appears in both planning files.
-   Status definitions:
-   `Implemented`: direct supporting evidence exists in current artifacts.
-   `Partial`: some supporting evidence exists, but a required component is missing.
-   `Unvalidated`: claim is conceptually specified but has no direct implementation/output evidence yet.
-   `Outdated`: claim conflicts with current project layout/state.
-   Matrix schema (fixed columns): `Source | Claim | Evidence | Status | Gap/Risk | Next Action`.

## Combined Claims vs Current State

| Source | Claim | Evidence | Status | Gap/Risk | Next Action |
|------------|------------|------------|------------|------------|------------|
| `notes/context_binding.md; notes/plan.md` | Competitor encoding in overlapping context is the accepted interference mechanism and Sim 2 baseline handoff for downstream sims. | `jaxcmr/selective_interference/cmr.py:230`; `jaxcmr/selective_interference/pipeline.py:520`; `projects/selective_interference/simulations/figures/interference_mcf_scale.csv`; `projects/selective_interference/simulations/figures/interference_n_interference.csv`; rendered Sim 2 sweep outputs in `projects/selective_interference/simulations/rendered/` | Implemented | Mechanism and sweep baseline exist, but acceptance criteria/handoff are not frozen in planning/manuscript text. | Add "Mechanism established" + "Sim 2 baseline handoff" checkpoints in `projects/selective_interference/notes/plan.md` and cite them in Sim 2 framing in `projects/selective_interference/index.qmd`. |
| `notes/context_binding.md; notes/plan.md` | Reminder reinstatement should drive delayed interference and primarily shift vulnerability targeting, with explicit delayed-condition contrasts delivered in Sim 5. | `jaxcmr/selective_interference/cmr.py:266`; `jaxcmr/selective_interference/cmr.py:279`; `projects/selective_interference/simulations/figures/calibration_reminder_start_drift_scale.csv`; `projects/selective_interference/simulations/figures/calibration_reminder_drift_scale.csv`; no delayed-condition output tables in `projects/selective_interference/simulations/figures/` | Partial | Reminder mechanisms and sweeps exist, but delayed-condition contrasts and a targeting metric are still missing. | Implement Sim 5 with `reminder+competitors`, `no reminder+competitors`, and `reminder only`, and add one SPC-shape targeting metric plus one condition DV panel. |
| `notes/context_binding.md; notes/plan.md` | Recall/recognition dissociation is the core architectural claim and Sim 1 gating deliverable. | No Sim 1 recognition notebook/output in `projects/selective_interference/simulations/rendered/` and no recognition CSV sidecar in `projects/selective_interference/simulations/figures/`. | Unvalidated | Highest-priority narrative anchor remains untested in current artifacts. | Implement Sim 1 recall+recognition outputs and publish one direct mode-comparison figure before downstream interpretation claims. |
| `notes/context_binding.md; notes/plan.md` | Retrieval control (`start_drift` + `tau`) should produce graded protection with explicit controlled-vs-unguided contrasts. | `jaxcmr/selective_interference/cmr.py:75`; `jaxcmr/selective_interference/cmr.py:305`; `projects/selective_interference/simulations/figures/retrieval_start_drift_scale.csv`; `projects/selective_interference/simulations/figures/retrieval_tau_scale.csv` | Partial | Independent sweeps exist, but interaction evidence and task-mode contrasts are not formalized. | Add one `tau × start_drift` interaction summary panel and one controlled-vs-unguided output set as the Sim 3 completion artifact. |
| `notes/context_binding.md; notes/plan.md` | Cue-at-test manipulations should distinguish cueing and retrieval-control accounts (Sim 4). | `jaxcmr/selective_interference/analysis.py:1`; no cue-probability/cue-drift simulation outputs in `projects/selective_interference/simulations/rendered/` or `projects/selective_interference/simulations/figures/`. | Unvalidated | Cue account remains conceptual; no model-output adjudication yet. | Implement Sim 4 cue probability and cue drift sweeps and add one cue x control interaction summary table. |
| `notes/context_binding.md; notes/plan.md` | Arousal-matched competitors should produce arousal-selective interference via the eCMR extension (Sim 6). | `jaxcmr/models_eeg/eeg_ecmr.py:67`; `jaxcmr/models_eeg/eeg_ecmr.py:94`; no arousal-selective outputs in `projects/selective_interference/simulations/figures/`. | Unvalidated | Extension capability exists but is not yet connected to selective-interference artifacts. | Implement minimal Sim 6 high-vs-neutral split scoring outputs and one SPC split panel. |

## Cross-File Coherence Risks

-   The strongest narrative claim (recall/recognition dissociation) is still unvalidated in current simulation outputs.
-   Current outputs are sweep-heavy and mechanism-rich, but manuscript-facing claim closure is under-specified for Sim 1, Sim 4, Sim 5, and Sim 6.
-   Rows that mix mechanism and delivery commitments can drift out of sync unless each row keeps a single explicit artifact target.
-   Reminder mechanism claims are ahead of explicit delayed-condition contrasts, which can overstate causal interpretation.
-   Retrieval-control claims are supported at parameter-sweep level but not yet at explicit task-mode contrast level.
-   Fit metadata drift persists between template/orchestrator and older rendered notebooks (`simulations/fits` vs stale `simulations/results/fits` in rendered artifacts), risking reproducibility confusion.

## Top Unvalidated Claims

-   Recognition immunity is structural and demonstrable in-model.

-   Minimal validation: implement Sim 1 recognition metric output and one recall-vs-recognition comparison figure.

-   Cue-at-test manipulations explain heterogeneous selective-interference findings.

-   Minimal validation: run cue probability and cue drift sweeps, then compare to uncued baseline in one summary figure.

-   Reminder-enabled delayed interference does not require reconsolidation windows.

-   Minimal validation: run the three delayed conditions and summarize with one DV table plus SPC panel.

-   Arousal-matched competitors selectively suppress high-arousal film items.

-   Minimal validation: run one eCMR-based condition set with high/neutral split scoring and one SPC split figure.

-   Retrieval control yields graded immunity across context-to-item tasks.

-   Minimal validation: add explicit controlled-vs-unguided retrieval outputs, not only independent parameter sweeps.

## Recommended Near-Term Sequence

1.  Resolve reproducibility drift by re-rendering sweep notebooks with the current template/orchestrator fit path settings.
2.  Freeze a Sim 2 baseline handoff in `projects/selective_interference/notes/plan.md` and tie it to manuscript framing in `projects/selective_interference/index.qmd`.
3.  Implement Sim 1 (recall/recognition dissociation) as the next gating deliverable.
4.  Implement Sim 5 delayed-condition contrasts to ground reminder-based delayed interference claims.
5.  Implement Sim 4 cue-at-test sweeps to test cueing-vs-control explanations.
6.  Implement minimal Sim 6 eCMR outputs for high/neutral split effects.
7.  Keep planning governance centralized in `projects/selective_interference/notes/plan.md` and `projects/selective_interference/notes/plan_gap_matrix.md`.