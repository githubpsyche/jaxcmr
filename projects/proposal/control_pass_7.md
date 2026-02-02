# Remembering for Reasons: Toward a Neurocognitive Model of Goal-Congruent and Incongruent Memory Search

## Context and Objectives

The broad objective of memory is to relate past experience to ongoing needs. However, external cues are often too sparse or misleading to guide retrieval unaided. Humans bridge this gap by initiating an iterative memory search, updating self-generated cues from ongoing context and retrieval outputs to probe stored experience. This internally guided search underpins adaptive behavior, but current theories still tackle directed recall task-by-task, tuning assumptions for each paradigm in isolation. That fragmentation leaves us without a principled account of why identical study experiences yield different recall patterns when goals shift. Nor does it equip us to flexibly explain when control fails, as when emotionally intense memories intrude unbidden.

This project addresses that gap by treating retrieval goals as explicit, manipulable variables within a computational model of memory search. Current frameworks absorb goal-related variance into task-specific parameters, fitting each paradigm in isolation. By representing goals directly, the model predicts how recall should change when goals change, without refitting. The same framework accommodates control failure: when a memory's accessibility outweighs goal-based modulation, it can surface despite intentions, producing intrusions. This challenges a longstanding assumption in memory research: that accessibility and intentional control can be measured and modeled separately. [ME TO ADD: One sentence positioning myself.]

## Theoretical Background

The model builds on retrieved-context frameworks (Howard & Kahana, 2002; Polyn et al., 2009), where gradually changing temporal context is bound to each episodic trace and later serves as the cue for recall. In free recall, participants study a word list and later retrieve items in any order. The temporal-context mechanism explains benchmark phenomena: the tendency to recall nearby items in sequence (lag-recency), characteristic latency distributions, and EEG signatures tied to context reinstatement (Manning et al., 2011). However, these models treat recall patterns as direct reflections of memory accessibility, with no explicit role for retrieval goals.

CMR2 (Lohnas et al., 2015) introduced a post-retrieval monitoring process, explaining the suppression of out-of-list intrusions through an assessed match between items' retrieved context and ongoing context. The model rejects candidates that have won the recall competition but are associated with context states too distinct from the current one. This represents a first step toward modeling goal-directed control. However, it addresses only one narrow goal: retrieve items from the current list, not earlier ones. Retrieval goals vary widely: recall items in serial order, prioritize high-value items, retrieve a specific category. Each must differently configure what counts as a target-appropriate output.

In the proposed framework, a goal state is a learned context vector representing the current retrieval target. Just as temporal context specifies *when* an item was encoded, goal context specifies *what* the retrieval system should prioritize: serial position, semantic category, reward value, or simply list membership. During retrieval, candidate items are evaluated against the active goal state via a similarity computation. Items whose features align with the goal receive higher activation. This representation is flexible: changing the goal vector changes which items are prioritized, enabling cross-task prediction without parameter refitting.

Three mechanisms could implement goal-conditioned control, each with distinct predictions. *Cue composition:* The goal state is blended into the retrieval cue before probing memory. Different goals produce different cues, yielding different activation patterns. Prediction: goal effects should be strongest on the first item recalled and diminish as retrieval-generated cues dominate. *Competition modulation:* The goal state scales evidence accumulation rates. Goal-congruent items race faster to threshold. Prediction: goal effects should appear in inter-response times (IRTs) throughout recall, not just at initiation. *Post-selection filtering:* Items are retrieved via standard competition, then checked against the goal; mismatches are rejected. Prediction: IRTs should show a bimodal distribution—fast accepts (~1s) and slow rejects (~3s)—and EEG should show partial activation for rejected items before suppression.

[FIGURE 1: Panel A shows experimental design as 2×2×3 factorial grid (Instruction × Timing × Salience). Panel B shows mechanism predictions: three columns for each mechanism, rows for (1) where goal effects are strongest in recall sequence, (2) whether instruction timing matters, (3) EEG signature.]

## Methodology [WIP]

For initial model development, we will use publicly available datasets where the same participants performed multiple directed-recall tasks. The Penn Electrophysiology of Encoding and Retrieval Study (PEERS; Kahana lab) provides free recall data from over 300 participants with detailed recall timing. [ME TO ADD: 1-2 additional datasets with within-subject task variation.]

New experiments will manipulate retrieval instruction, instruction timing, and item salience within subjects. Participants (N=40 per experiment) will study 20-item word lists and perform cued recall under different goal instructions. The timing manipulation is critical: half of trials reveal the goal before study (allowing encoding-phase prioritization), half reveal it after (isolating retrieval-phase effects). For Aim 3, items will include neutral words, reward-tagged words (+£0.10 per correct recall), and emotional words drawn from ANEW norms (Bradley & Lang, 1999). EEG will be recorded using a [ME TO SPECIFY: EEG system] at [ME TO SPECIFY: institution].

Model variants will be implemented in Python using JAX for efficient gradient-based optimization. We will fit each variant to trial-level data—recall sequences, inter-response times, intrusion identities—using hierarchical Bayesian estimation (Stan; Carpenter et al., 2017). Model comparison will proceed via WAIC and leave-one-out cross-validation (Vehtari et al., 2017). The critical test: which mechanism best captures the shift in first-recall probability, IRT distributions, and intrusion rates across task instructions without task-specific parameters?

Two electrophysiological markers will constrain model selection. Late positive potentials (LPPs; 400-800ms post-stimulus at Pz) index prioritized encoding of motivationally relevant items (Hajcak et al., 2010). If cue-composition is correct, LPP differences across instruction conditions should appear only when instructions precede encoding. Encoding-retrieval similarity (ERS) measures pattern correlation between study and test epochs (Staresina & Davachi, 2006); higher ERS for goal-congruent items indicates goal-dependent reinstatement. If competition-modulation or filtering operate at retrieval, ERS differences should appear regardless of instruction timing.

## Specific Aims

**Aim 1: Formalize and test goal-conditioned control within episodic memory search.**
We will implement three model variants: (1) cue-composition, where goal states modulate the retrieval cue; (2) competition-modulation, where goals scale accumulation rates; (3) post-selection filtering, extending CMR2's monitoring. Each will be fit to the PEERS dataset and [ME: additional dataset] using hierarchical Bayesian estimation. The adjudication: cue-composition predicts goal effects concentrated at first recall (P(first recall | serial position) should shift with instruction, but IRTs after the first item should not); competition-modulation predicts effects throughout (IRTs for goal-congruent items should be faster at all output positions); filtering predicts bimodal IRTs and elevated intrusion rates when filtering fails.
*Output: Open-source model codebase (GitHub); publication comparing mechanisms.*

**Aim 2: Experimentally dissociate goal effects at encoding versus retrieval.**
We will run N=40 participants in a within-subject design crossing instruction type with instruction timing (pre-study vs. post-study), recording behavior and EEG. The adjudication: if cue-composition, pre-study instructions should yield larger first-recall effects and larger LPP differences at encoding; if competition/filtering, timing should not matter and effects should appear in retrieval-phase ERS.
*Output: Behavioral and EEG dataset (OSF); publication on encoding/retrieval dissociation.*

**Aim 3: Test how emotional intensity interacts with goal-configured control.**
We will run N=40 participants, crossing instruction type with item salience (neutral/reward/emotional). The critical prediction: emotional items should intrude when goal-incongruent (high accessibility overrides weak goal match) but enhance recall when goal-congruent (accessibility and goal alignment converge). EEG predictions: emotional items should show elevated LPPs regardless of goal congruence; goal-congruent emotional items should show enhanced ERS; goal-incongruent emotional items should show LPP elevation but reduced ERS.
*Output: Dataset (OSF); publication bridging cognitive and clinical work; framework paper on intrusive memories.*

## Positioning

[ME TO WRITE: Track record paragraph—publications, preliminary model implementations, relevant expertise.]

This project is timely. CMR2 established that post-retrieval monitoring can be formalized within retrieved-context models, but its scope remains limited to list-membership. Methodological advances now make broader tests feasible: representational similarity analysis quantifies encoding-retrieval overlap from EEG (Kriegeskorte et al., 2008), and hierarchical Bayesian methods enable rigorous comparison of candidate mechanisms (Shiffrin et al., 2008). Existing multi-task datasets are publicly available for initial model development, reducing risk.

[ME TO WRITE: Host institution paragraph—EEG lab, computing resources, named collaborators.]

## Expected Outputs

This project will produce: an open-source computational model implementing goal-conditioned retrieval mechanisms, available on GitHub; three primary publications in journals spanning computational, cognitive, and clinical domains; two behavioral and EEG datasets deposited at the Open Science Framework; and a theoretical framework connecting laboratory findings on goal-directed memory to clinical phenomena including intrusive memories and rumination.

## Timeline

**Year 1:** Implement three model variants. Fit to PEERS and additional datasets. Submit Aim 1 paper. Design Aim 2 experiment; pilot with N=10.

**Year 2:** Collect Aim 2 data (N=40). Analyze behavioral and EEG results. Refine models with neural constraints. Submit Aim 2 paper. Design Aim 3.

**Year 3:** Collect Aim 3 data (N=40). Analyze emotion-goal interactions. Submit Aim 3 paper. Write integration/framework paper on clinical implications.

## References

Bradley, M. M., & Lang, P. J. (1999). Affective norms for English words (ANEW). University of Florida.

Carpenter, B., et al. (2017). Stan: A probabilistic programming language. *Journal of Statistical Software*, 76(1).

Hajcak, G., MacNamara, A., & Olvet, D. M. (2010). Event-related potentials, emotion, and emotion regulation. *Emotion*, 10(4), 498-512.

Howard, M. W., & Kahana, M. J. (2002). A distributed representation of temporal context. *Journal of Mathematical Psychology*, 46(3), 269-299.

Kriegeskorte, N., Mur, M., & Bandettini, P. A. (2008). Representational similarity analysis. *Frontiers in Systems Neuroscience*, 2, 4.

Lohnas, L. J., Polyn, S. M., & Kahana, M. J. (2015). Expanding the scope of memory search. *Psychological Review*, 122(2), 337-363.

Manning, J. R., et al. (2011). Oscillatory patterns in temporal lobe reveal context reinstatement during memory search. *PNAS*, 108(31), 12893-12897.

Polyn, S. M., Norman, K. A., & Kahana, M. J. (2009). A context maintenance and retrieval model. *Psychological Review*, 116(1), 129-156.

Shiffrin, R. M., et al. (2008). A survey of model evaluation approaches. *Cognitive Science*, 32(8), 1248-1284.

Staresina, B. P., & Davachi, L. (2006). Differential encoding mechanisms for subsequent associative recognition and free recall. *Journal of Neuroscience*, 26(36), 9162-9172.

Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413-1432.
