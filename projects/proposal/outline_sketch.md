# 3-Page Research Proposal Outline

## Constraints
3 sides A4, portrait, Arial size 10 minimum
Must include: context/objectives/outputs, methodology/data analysis, milestones/timescales

## Paragraph-by-Paragraph Outline

### SECTION 1: CONTEXT AND OBJECTIVES (~0.75 page)

**P1: The Problem (keep current)**
Memory relates past experience to ongoing needs
External cues insufficient → iterative memory search
Current theories fragmented across tasks
Two gaps: can't explain cross-task variation OR control failure/intrusions

**P2: The Project Aim (keep current, minor expansion)**
Goals as explicit, manipulable variables
Predicts cross-task behavior without refitting
Accommodates control failure (accessibility outweighs goal modulation → intrusions)
Challenges accessibility/control separation assumption
Lays groundwork for unified account spanning lab tasks and clinical populations

### SECTION 2: THEORETICAL BACKGROUND (~0.5 page)

**P3: Retrieved-Context Foundations**
CMR framework: temporal context bound to traces, serves as retrieval cue
Explains benchmark phenomena (lag-recency, latency distributions)
But treats recall as direct readout of accessibility, no role for goals

**P4: CMR2 and Its Limitations**
CMR2 introduced post-retrieval monitoring for out-of-list intrusions
Context-matching mechanism rejects candidates whose context doesn't match target list
But addresses only one narrow goal (list membership)
Retrieval goals vary widely: serial order, category, reward priority, etc.

**P5: Our Extension**
Generalize monitoring architecture to arbitrary goal definitions
Goals as explicit states specifying task requirements
Candidate mechanisms: cue composition, competition dynamics, post-selection filtering
Experiments adjudicate between candidates

### SECTION 3: METHODOLOGY (~0.75 page)

**P6: Behavioral Paradigm**
Directed memory-search tasks varying instruction, timing, salience
Within-subject designs: same participants perform multiple task types
Key manipulations: instruction type (free/serial/categorized/reward), instruction timing (pre/post-study), item salience (neutral/reward/emotional)

**P7: Computational Modeling Approach**
Model fitting via maximum likelihood or Bayesian methods
Fit to trial-level data: recall sequences, inter-response times, intrusions
Model comparison via likelihood-based selection (AIC/BIC) or cross-validation
Key test: can a single parameter set (varying only goal) capture cross-task behavior?

**P8: EEG Methodology**
Late positive potentials (LPPs) at encoding: index goal-based prioritization of items during study
Encoding-retrieval similarity (ERS): pattern similarity between study and test epochs indexes goal-dependent reinstatement
EEG constrains model selection: mechanisms make different predictions for when/how neural signatures diverge across conditions

### SECTION 4: SPECIFIC AIMS WITH PREDICTIONS (~1 page)

**Aim 1: Formalize and test goal-conditioned control (~0.33 page)**
Represent goals as explicit states in model
Fit competing variants to existing multi-task datasets
Adjudication logic: cue-composition predicts goal effects on retrieval initiation; competition-modulation predicts effects on accumulation speed; filtering predicts distinct latency signatures (reject-and-retry)
Expected output: Model codebase + paper comparing mechanisms on existing data

**Aim 2: Dissociate goal effects at encoding vs retrieval (~0.33 page)**
New experiments with timing manipulation
Adjudication logic: cue-composition → pre-study instructions matter more; competition/filtering → timing matters less
EEG predictions: LPPs should differ by instruction only when given pre-study; ERS should differ by instruction regardless of timing if goals shape retrieval
Expected output: Behavioral + EEG dataset + paper on encoding/retrieval dissociation

**Aim 3: Emotional intensity and control failure (~0.33 page)**
Vary item salience while holding instruction/timing fixed
Adjudication logic: if intrusions arise from accessibility outweighing goal modulation, emotional items should intrude when goal-incongruent but enhance recall when goal-congruent
EEG predictions: emotional items should show elevated LPPs regardless of goal; goal-congruent emotional items should show enhanced ERS
Expected output: Dataset + paper on emotion-goal interaction + clinical implications framework

### SECTION 5: EXPECTED OUTPUTS (~0.25 page)

**P9: Deliverables**
Open-source model codebase (Python/JAX) with goal-conditioned retrieval mechanisms
Three primary publications (one per aim)
Two behavioral + EEG datasets (Aims 2-3) deposited in public repository
Theoretical framework connecting laboratory findings to clinical phenomena (intrusive memories, rumination)

### SECTION 6: TIMELINE (~0.25 page)

**P10: Year-by-Year Milestones**
Year 1: Model development + fitting to existing data (Aim 1); design and pilot Aim 2 experiments
Year 2: Aim 2 data collection + analysis; model refinement; begin Aim 3 design
Year 3: Aim 3 data collection + analysis; integration paper; clinical extension framework

---

## Files to Modify
[projects/proposal/control_pass_6.md](projects/proposal/control_pass_6.md) - expand from 1-pager to 3-pager following this outline

## Verification
Check page count with Arial 10 formatting
Ensure all required elements present: context/objectives/outputs, methodology/data analysis, milestones/timescales
Export to docx: `quarto render projects/proposal/control_pass_6.md --to docx -o projects/proposal/control_pass_6.docx`
