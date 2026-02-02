Remembering for Reasons: A Neurocognitive Model of Goal-Directed Memory Search

The broad objective of memory is to relate past experience to ongoing needs.
However, external cues are often too sparse or misleading to do that unaided.
Humans bridge this gap by launching an internally guided memory search, updating self-generated cues from ongoing context and recent outputs to probe stored experience.
When this control succeeds, it supports adaptive behavior.
When it fails, especially after highly emotional experiences, it can produce persistent goal-incongruent recollections.
These include intrusive memories and rumination linked to disorders such as PTSD and depression.
Yet current computational theories still explain directed recall task-by-task, tuning assumptions for each paradigm instead of specifying how goals configure retrieval across conditions.
As a result, we lack a mechanistic explanation for why identical study experiences can yield different recall patterns under different retrieval goals, or how experience and practice sharpen (or undermine) control.

This project aims to develop and test a computational account of goal-directed memory search that explains how retrieval goals and experience jointly determine recall behavior and electrophysiological brain activity across directed retrieval tasks.
Rather than committing up front to a new “goal controller” mechanism, we take a conservative next step.
We generalize CMR2’s context-monitoring gate (Lohnas et al., 2015) from list-membership discrimination to arbitrary goal definitions such as reward, category, serial position targets, and suppression instructions.
In this framework, a cue-driven retrieved-context generator proposes candidate memories.
A goal-conditioned monitoring and stopping policy determines which candidates become outputs and when search terminates.
We refer to this generalization as Goal-CMR2.
By making goal states explicit only at the level of monitoring and stopping, Goal-CMR2 yields concrete, testable predictions for recall order, latency, termination, and goal-(in)congruent outputs across instructions.
This holds even when baseline accessibility is held constant by design.

Our approach combines behavioral and EEG data from directed retrieval tasks that vary retrieval instruction, instruction timing (pre- vs post-study), and practice history.
We extend the same experimental framework to emotion-driven goal failures.
The model’s core builds on retrieved-context frameworks by Howard and Kahana (2002) and Polyn et al. (2009), where a gradually changing internal context is bound to episodic traces and later serves as the cue for recall.
CMR2 introduced post-retrieval monitoring to suppress prior-list intrusions by rejecting candidates after selection.
The open question is whether an analogous monitoring-and-stopping architecture, generalized to arbitrary goals, is sufficient to explain goal-conditioned retrieval across tasks.
If not, goal states may also need to reshape competition dynamics during candidate generation, for example by changing cue composition or accumulation.
We will use model comparison and targeted experimental manipulations to answer this question.
We will ground the latent control state in EEG markers of reinstatement and monitoring.

Aim 1: Build and test Goal-CMR2 on behavioral signatures of goal-directed recall.
We will formalize Goal-CMR2 with an explicit goal definition, a monitoring rule, and a stopping policy.
We will specify what counts as goal-congruent and how off-goal candidates are rejected.
We will fit the model to existing and newly collected behavioral datasets across directed recall instructions (free, serial, categorized, reward-weighted, suppression/avoidance) and across instruction timing (pre- vs post-study).
We will evaluate recall transitions, latency distributions, and termination times.
We will test whether instruction effects arise mainly through monitoring and stopping or whether the generator must also be goal-conditioned.
If monitoring-only variants fail to capture robust instruction-dependent changes among accepted recalls, we will add the minimal extensions needed.
These may allow goal-conditioned cue weighting or accumulation.

Aim 2: Link EEG to goal-conditioned monitoring and stopping to build a neurocognitive model of control.
We will analyze existing and new EEG recorded during encoding and retrieval.
We will extract trial-wise features for context reinstatement and control demands.
We will use these features to modulate or constrain Goal-CMR2 control parameters such as monitor strength, decision criteria, and stop threshold.
This will enable trial-by-trial prediction of goal-incongruent outputs and termination.
We will test whether EEG-informed models improve out-of-sample prediction relative to behavior-only fits.
We will assess specificity with shuffled-feature controls.
The result is a neurocognitive account in which latent control states are computationally defined and empirically constrained.

Aim 3: Integrate emotion-sensitive retrieved-context dynamics (eCMR/CMR3 lineage) to explain intrusive memory as goal-incongruent retrieval.
Intrusive memories can be operationalized in the laboratory as unwanted, off-goal outputs that occur despite explicit task goals.
We will extend Goal-CMR2 within an emotion-sensitive retrieved-context framework.
We will test whether highly emotional items increase candidate activation or binding, and whether this increases off-goal candidates during search.
We will test whether the same goal-conditioned monitoring and stopping policies explain which candidates become goal-incongruent outputs.
We will apply the Aim 1 instruction and timing framework to emotional and value-tagged materials.
We will quantify whether emotion increases goal-incongruent output rates, biases the content of off-goal recalls, and changes latency and termination dynamics in ways predicted by the unified model.
This connects goal-directed control in the lab to mechanisms relevant for intrusive recollection, without presupposing a separate intrusion route.

Expected outputs and impact.
This project will deliver a reusable Goal-CMR2 model class that generalizes CMR2-style monitoring and stopping to arbitrary goals.
It will deliver behavioral tests that identify which instruction and practice effects can be explained by monitoring alone versus requiring goal-conditioned generation.
It will deliver a neurocognitive extension linking EEG features to latent control parameters.
By treating CMR2 as a successful special case and systematically extending it, the work advances a unified, testable account of goal-directed remembering across tasks and affective contexts.
