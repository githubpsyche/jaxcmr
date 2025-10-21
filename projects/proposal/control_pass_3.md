Theories of voluntary (strategic) and involuntary (spontaneous) recall in episodic memory have often proceeded on separate tracks, limiting mechanistic understanding of how retrieval goals and experience shape recall performance and intrusive recollection.
Accounts that treat retrieval intentionality as a dichotomy often overlook the goal- and state-dependent mechanisms that govern strategic recall, while models of strategic recall frequently neglect the mechanisms that give rise to involuntary intrusions.that might give rise to involuntary intrusions, leaving open questions about when and why recall aligns with task goals versus when it does not.
Building on retrieved-context models of episodic memory, this project aims to develop and test an integrative computational account of retrieval intentionality that explains both voluntary and involuntary recall within a unified framework.

_Our approach is to collect empirical behavioral data and electrophysiological recordings from human participants performing memory search tasks that vary retrieval goals and manipulate emotion and reward conditions, allowing us to iteratively refine a computational model of performance and brain signals in these tasks._
The computational model at the heart of our proposal is a retrieved-context account in which a gradually changing temporal context is encoded into episodic traces and drives cue-based retrieval.
Prior work building on this framework has clarified accounts of how emotion and reward modulate memory dynamics (e.g., eCMR) as well as how strategic recall can be modeled as a task-dependent policy that skilled retrievers learn to optimize.
However, these lines of research have not yet received complete examination within a single, unified architecture that explains how retrieval goals and experience shape both voluntary recall and involuntary intrusions within a single architecture.

Our approach builds on emotion/reward-sensitive retrieved-context models (e.g., eCMR) that specify how temporal context cues retrieval and how emotion and reward modulate memory dynamics. 
We will extend this framework with explicit control (gating/monitoring) and stopping policies that depend on retrieval goals and experience (practice). 
By analyzing existing behavioral and EEG datasets from free-recall tasks with emotional, reward, and suppression manipulations, we will iteratively refine a computational model that explains how retrieval goals and study conditions organize memory search and output.
We will (i) formalize and adjudicate competing accounts of intentionality, (ii) model strategic recall as a task‑dependent policy that adapts with instructions and practice (multi‑trial learning), and (iii) link trial‑wise EEG indices of control and context reinstatement to model parameters that govern voluntary and involuntary recall.

state-dependent nature of control and stopping policies that govern memory search, while models

focused on addressing intrusive memory often fall into two camps: those that treat voluntary and involuntary recall as arising from a shared cue‑driven generator with differences emerging from control and stopping policies, and those that posit a distinct, context‑decoupled route for involuntary outputs.
On the other hand, models of strategic recall have largely neglected the mechanisms that might give rise to involuntary intrusions, leaving open questions about when and why recall aligns with task goals versus when it does not.

Aim 1: Formalize and adjudicate competing accounts of the voluntary/involuntary distinction.
We will construct and compare three model families built on retrieved‑context dynamics: (A) No‑distinction (shared generator; no explicit control), (B) Shared‑generator + control gate (gate strength depends on task goals/state), and (C) Separate‑route (an additional, context‑decoupled generator). Across existing multi‑list free‑recall datasets (neutral, emotional, reward; suppression instructions), we will test each family on intrusion rates/types (PLI/ELI), emotion/reward‑matched intrusion bias, and transition structure (temporal/semantic/emotional contiguity). We will also assess whether adding an item‑independent temporal context drift improves cross‑list intrusion predictions without invoking a separate route. Model comparison will rely on out‑of‑sample prediction, posterior predictive checks, and simulation‑based recovery to ensure identifiability.

Aim 2: Model strategic recall as a task policy that adapts with instructions and practice.
We will characterize retrieval policy as the set of control settings (gate strength, cue weighting for temporal/semantic/emotional features) that aligns recall with task specifications (standard free recall, reward‑weighted recall, suppression instruction). Using existing datasets with explicit rewards and suppression, we will test how policy parameters shift across instructions and change with experience (multi‑trial learning), and whether those shifts generalize across tasks. Analyses will target recall ordering, intrusion propensity, and practice‑related changes in contiguity and emotion/reward weighting. We will quantify where behavior departs from policy predictions and ask whether these departures systematically predict involuntary outputs.

Aim 3: Build a neurocognitive model linking EEG to voluntary and involuntary recall.
Using existing EEG during study and retrieval, we will derive features indexing control/monitoring and context reinstatement. These signals will serve as priors/modulators on model parameters central to intentionality (e.g., gate strength; weights on emotional/reward context; magnitude of item‑independent temporal drift). We will test whether EEG‑informed models improve trial‑wise prediction of intrusions and recall transitions on held‑out data, and whether instruction‑ and practice‑driven policy changes are expressed in EEG‑linked parameters. Specificity will be assessed with shuffled/phase‑randomized controls and cross‑dataset generalization.

Expected Outcomes and Impact. The project will (i) deliver a decisive comparison of accounts of retrieval intentionality, (ii) provide a policy‑level explanation of how task goals and practice reshape strategic recall and its failures, and (iii) ground these mechanisms in EEG. By unifying voluntary and involuntary recall within one retrieved‑context architecture, the work advances mechanistic understanding of intrusive memory and offers a reusable modeling pipeline for emotion/reward‑sensitive recall.

Figure 1. Iterative pipeline: retrieved‑context generator → control gate & task‑dependent policy → predictions for behavior/EEG → model comparison and EEG‑informed parameterization → revised theory of retrieval intentionality.

==

which context partition connects to item layer more

Does emotional attention modulator apply to just emotional mcf or both?
connects to dual representation hypothesis

Does "mood" influence emotional context retrieval?

