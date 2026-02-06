---
title: "Intrusive and Voluntary Memory from a Single System: A Retrieved-Context Account of Selective Interference"
shorttitle: "Intrusive and Voluntary Memory from a Single System"
author:
  - name: Jordan B Gunn
    affiliations:
      - name: University of Cambridge
        department: Department of Psychology
keywords:
  - intrusive memories
  - voluntary memory
  - trauma-film paradigm
  - visuospatial intervention
  - selective interference
  - retrieved-context theory
  - computational modeling
author-note:
  disclosures:
    related-report: "This is an in-progress working draft. Please do not cite, quote, or circulate without the authors' written permission."
floatsintext: true
bibliography: references.bib
---

# Abstract

Post-encoding visuospatial tasks such as playing Tetris after a trauma reminder can reduce subsequent intrusive memories while leaving voluntary memory (e.g., recognition and intentional recall) relatively intact.
This selective interference effect has been taken as evidence for separate "intrusive" versus "voluntary" memory traces or for modality-specific disruption of consolidation or reconsolidation.
Here we develop a single-system account within retrieved-context theory (CMR3) that derives selective interference from generic episodic binding and retrieval dynamics.
In the model, reminders reinstate trauma-associated context, and experiences during the interference task are encoded in that reinstated context, creating many strong competitor episodes.
Later, as ongoing context drifts through the trauma region, uncontrolled context-to-item retrieval is increasingly captured by these competitors, reducing the probability that trauma hotspots cross threshold as intrusions.
In contrast, probe-driven item-to-context retrieval and controlled context-to-item recall can remain largely intact, explaining spared recognition and (under sufficient control) intentional recall.
Simulations of trauma-film paradigms reproduce the intrusion-recognition dissociation and its heterogeneity across designs, and yield testable predictions about timing, cueing, emotional arousal, and retrieval-control manipulations.
This framework reframes visuospatial tasks as one instantiation of a broader competitor-encoding intervention strategy and provides principled boundary conditions for translating lab effects to clinical settings.

# Introduction

<!-- TODO: Write introduction covering:
- Introduction to the trauma-film paradigm and the selective interference effect
- Identification of the research gap: lack of a mechanistic account of selective interference that can explain the full range of findings across paradigms
- Review of empirical findings on selective interference, including the intrusion-recognition dissociation and its heterogeneity across designs
- Review of existing theoretical accounts of selective interference, including dual-trace theories and consolidation/reconsolidation disruption theories, and their limitations
- Introduction to retrieved-context theory (CMR3) and its relevance for understanding selective interference
- Clear statement of the aims and concrete contributions of the current paper
-->

# Overview of the Model

The account rests on six theoretical commitments that together derive selective interference from standard retrieved-context machinery.

First, context-to-item and item-to-context retrieval are architecturally distinct.
Context-driven retrieval (free recall, intrusions) probes $M_{CF}$ with the current context state and selects among all items that share that context, making it vulnerable to competition from any items encoded in overlapping context.
Item-driven retrieval (recognition) probes $M_{FC}$ with a specific item and evaluates the match between the retrieved context and the current context state, a pathway that is not subject to the same competition.
This single architectural distinction is the foundation of the intrusion-recognition dissociation.

Second, interference is implemented by encoding actual competitor items in trauma-adjacent context, not by context drift alone.
When a participant plays Tetris or engages in another visuospatial task after a trauma film, the model treats each salient event during that task as an item encoded via the standard learning rule.
Because these items are encoded in context that overlaps with the film, they become competitors during later context-to-item retrieval.
This reframes visuospatial tasks as one instantiation of a broader strategy: any task that generates strongly encoded events in trauma-adjacent context will interfere with later context-to-item retrieval of film content.

Third, recognition is implemented as item-to-context similarity.
Probing $M_{FC}$ with a test item retrieves the item's associated context; the dot product with the current context state gives the recognition signal.
Context drifts between recognition tests just as it does during free recall, so current context serves as a dynamic reference rather than a frozen snapshot.
Because items are represented as orthogonal vectors, encoding competitor items only modifies the competitor rows of $M_{FC}$ and leaves film items' associations untouched.
The recognition signal for a film item is therefore largely immune to competitor encoding; any residual effect comes only from the small difference in context state at test onset, not from changes to the item-to-context associations themselves.

Fourth, voluntary recall is distinguished from involuntary retrieval by the strength of two control mechanisms.
Starting-context reinstatement biases the initial retrieval context toward the beginning of the film, away from the interference region.
Choice sensitivity (the temperature parameter $\tau$ in the Luce choice rule) sharpens competition, amplifying the advantage of items with the strongest context-to-item support and functionally approximating the context-monitoring mechanism proposed in CMR2.
Together, these mechanisms produce graded immunity across context-to-item tasks: recognition (item-to-context) is most protected, followed by intentional recall with strong control, with unguided recall most vulnerable.

Fifth, reminders of a previously viewed trauma film retrieve associated context through item-to-context associations, driving the current context state back toward the trauma-film region.
Subsequent experiences during an interference task are then bound to this reinstated context, creating competitors even at temporal delay.
This provides a principled alternative to reconsolidation accounts of delayed interference: what matters is the contextual proximity at the time of competitor encoding, not the temporal delay per se.

Sixth, high-arousal items encode and reinstate shared arousal context features.
Using eCMR's emotional source-memory mechanism, high-arousal competitors preferentially interfere with recall of high-arousal study events via shared arousal context, producing arousal-selective interference.

<!-- TODO: Formal model description, equations, parameter table -->

# Simulations

The full trauma-film paradigm involves many interacting components, each involving implementation decisions with ambiguous consequences.
Rather than simulating the complete paradigm in a single step, we present six simulations that progressively build the account.
Early simulations isolate individual mechanisms in simplified settings, using parameter-shifting explorations to show how stronger or weaker mechanism engagement affects outcomes.
Each simulation resolves open questions before later simulations depend on the answers, and each varies one or two parameters while holding others at values established by previous simulations, preventing combinatorial explosion.

All simulations use fitted parameters from @healeySearchingAssociativeMemory2014 free recall data as a base.
The primary visualization for Simulations 2--6 is the serial position curve pooling study and interference items, with formatting to distinguish film and interference regions.

A general prediction across Simulations 1--5 is that interference should show a recency gradient: later-studied film items are disproportionately suppressed because they share more temporal context with competitors encoded immediately after the film phase.
Simulation 6 then contrasts this pattern, because arousal context features operate independently of temporal position and partially flatten the recency gradient for arousal-matched items.

## Simulation 1: Context-Based Competition in Free Recall vs. Recognition

Recognition is more resistant to retroactive interference than free recall, a finding that has been established across many paradigms.
The purpose of this simulation is to show that this dissociation arises naturally from the architecture of the model, without any trauma-specific mechanism.

### Design

Items are encoded in a standard A/B list paradigm.
In the control condition, $N_A$ target items are encoded and then tested.
In the interference condition, $N_B$ competitor items are encoded immediately after the target items.
Both conditions are tested with context-to-item retrieval (free recall) and item-to-context retrieval (recognition).
For free recall, the model reinstates start-of-list context and retrieves freely.
For recognition, the model reinstates start-of-list context and then, for each target item, probes $M_{FC}$ to retrieve the item's associated context, computes the dot product with the current context state as the recognition signal, and integrates the retrieved context (drifting, as during free recall).
The raw recognition signal is the dependent variable, with no decision threshold.

### Results

<!-- TODO: Insert results -->

*Panel A* shows the serial position curve for free recall in the control and interference conditions.
In the interference condition, target items show reduced recall probability while competitor items appear as a new region.
*Panel B* shows the recognition signal plotted by study position for target items in both conditions.
The two curves should nearly overlap, preserving the same serial position structure.
This shows that immunity is not just an aggregate effect: the full position-by-position pattern is unchanged by competitor encoding.

The dissociation is architectural.
Competitor encoding impairs context-to-item retrieval of target items while leaving item-to-context retrieval largely intact, because competitor encoding does not modify the target items' rows of $M_{FC}$.

## Simulation 2: Manipulating Interference Intensity

More engaging interference tasks produce stronger effects on intrusions, and simply increasing task duration shows diminishing returns [@holmesCanPlayingComputer2009; @jamesComputerGamePlay2015].
This simulation explores what intensifies interference under the model's logic, evaluating three candidate factors via dose-response tracking.

### Interference intensifiers

The first factor is $M_{CF}$ encoding strength.
Higher engagement during the interference task is modeled as a boosted $M_{CF}$ learning rate for competitor items, producing stronger context-to-item associations and more competition during retrieval.

The second factor is context proximity.
Competitors encoded with a lower encoding drift rate stay in film-adjacent context longer, producing more context overlap and stronger competition.

The third factor is the number of competitors.
More encoded events mean more competition, but this is subject to a context-drift ceiling: as competitors are encoded sequentially, context drifts away from the film region, and later competitors produce diminishing interference.

### Design

Film hotspot items are encoded with standard parameters, followed by competitor items encoded with interference-specific parameters.
The model is then tested with context-to-item free recall.
Separate parameter sweeps (~10 values each) explore the interference $M_{CF}$ learning rate, the interference encoding drift rate, and the number of competitors.
A context trajectory visualization tracks the similarity between the model's context state and a film-context reference after each competitor encoding step.

### Results

<!-- TODO: Insert results -->

## Simulation 3: Intentional vs. Unintentional Context-to-Item Retrieval

Intentional free recall of film content is sometimes spared despite being context-to-item retrieval [@lau-zhuDoesTetrisReduce2019].
This simulation demonstrates how two retrieval-control mechanisms produce graded immunity for intentional recall.

### Control mechanisms

Starting-context reinstatement biases initial retrieval context toward the start of the film, away from the interference region.
This is the model's existing start-of-retrieval drift mechanism.

Choice sensitivity ($\tau$) sharpens the Luce choice rule, amplifying the advantage of items with the strongest context-to-item support.
This functionally approximates CMR2's context-monitoring mechanism by suppressing weakly matched items, including competitors encoded in partially overlapping context.

### Design

Film and interference items are encoded using parameters established in Simulation 2.
Free recall is then tested under varied control settings: a starting-context reinstatement sweep (~10 values, default $\tau$), a $\tau$ sweep (~10 values, default starting-context reinstatement), and a 2$\times$2 summary crossing low/high values of each mechanism.

### Results

<!-- TODO: Insert results -->

## Simulation 4: How Do Reminder Cues During Recall Affect Performance?

Selective interference findings are inconsistent across VIT-style paradigms that present film cues during the test phase.
This simulation clarifies the role of these cues by showing that they reinstate trauma context, partially overriding the retrieval-control mechanisms established in Simulation 3.

### Design

Film and interference items are encoded with Simulation 2 parameters, and retrieval-control parameters are set to Simulation 3 values.
Film cues are presented before recall attempts by probing $M_{FC}$ with the cue item and integrating the retrieved context.
Three parameter sweeps explore cue drift rate (~10 values), cue probability (~10 values from 0 to 1), and informative combinations of cue parameters with control parameters.

### Results

<!-- TODO: Insert results -->

Film cues reinstate trauma context strongly enough to override retrieval-control mechanisms, flattening the voluntary/involuntary distinction.
Cue-free paradigms yield cleaner tests of the control hypothesis.

## Simulation 5: Delayed Interference with Reminder

Delayed Tetris combined with a reminder reduces intrusions, whereas Tetris without a reminder is less effective [@jamesComputerGamePlay2015].
This simulation shows that context reinstatement via reminder enables delayed interference without requiring a reconsolidation window.

### Design

Film items are encoded with standard parameters.
Context is then drifted to an out-of-list state with maximal drift rate to simulate the delay interval.
In the reminder condition, the model walks through film items in order, probing $M_{FC}$ for each and integrating the retrieved context with a reminder-specific drift rate; this updates context but does not modify $M_{FC}$ or $M_{CF}$.
Competitor items are then encoded using Simulation 2 parameters.
Free recall is tested using control parameters from Simulation 3, with no cues during recall.

Three conditions are compared: reminder followed by competitors, reminder only (no competitors), and competitors only (no reminder, encoding occurs in distant context).
The key manipulation is the reminder drift rate (~10 values), which controls how strongly the reminder reinstates film context.

### Results

<!-- TODO: Insert results -->

Reminder reinstatement is sufficient for delayed interference.
Without a reminder, competitors encoded in distant context are ineffective.
Reminder alone (without competitors) has no effect on later retrieval.

## Simulation 6: Arousal-Specific Interference

Emotional film content produces more intrusions than neutral content [@holmesCanPlayingComputer2009; @brewinNewPerspectiveIntrusive2014].
This simulation extends the account to arousal-specific interference using eCMR.

### Design

The film phase encodes a mixed list of high-arousal and neutral items.
The interference phase encodes only high-arousal competitors using Simulation 2 parameters.
The key independent variable is the ratio of high-arousal to neutral items in the film.
Free recall is scored separately for high-arousal and neutral film items.

Two parameter sweeps explore arousal context strength (~10 values) and arousal ratio (e.g., 20/80, 50/50, 80/20 high-arousal/neutral in the film).

### Results

<!-- TODO: Insert results -->

High-arousal competitors preferentially reduce intrusions of high-arousal film items via shared arousal context, and the effect scales with arousal context strength and with the ratio of high-arousal items in the film.
Critically, arousal-matched interference should show a flatter serial position profile than the recency-weighted pattern seen in Simulations 1--5, because arousal context features bridge film and competitor items independently of temporal position.
Neutral film items retain the standard recency gradient.

# General Discussion

<!-- TODO: Write discussion covering:
- Summary of key findings and contributions of the model
- Implications for understanding the mechanisms underlying selective interference and the intrusion-recognition dissociation
- Comparison of the model's predictions with existing empirical findings and theoretical accounts
- Suggestions for intervention development, including how the model could inform the design of more effective interventions for reducing intrusive memories while preserving voluntary memory
- Discussion of the model's limitations
- Suggestions for future research directions, including experimental tests of the model's predictions and further development of the theoretical framework
-->

# References

::: {#refs}
:::
