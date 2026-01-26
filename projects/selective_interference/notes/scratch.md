# Introduction

Selective interference poses a modeling problem as much as an empirical one: any account must explain (i) why some post-event tasks robustly reduce intrusive memories, (ii) why intentional access is often spared, and (iii) why effects are heterogeneous across designs and measures.
The core aim of this paper is to make that explanatory burden explicit in a single retrieved-context framework, and then use the framework to separate what is *forced* by the data from what is an implementation choice.

To keep the modeling grounded while still aiming for generality, the paper uses an empirical anchor: a Vigilance Retrieval Task (VRT) dataset from our own project, with rich trial- and report-level structure and a scoring pipeline that yields intrusion- and voluntary-memory observables.
We use these analyses to (a) define and validate a mapping from task events to model operations and (b) constrain a plausible parameter regime for retrieval mode and control.
We then use that scaffold to run more data-detached simulations—targeting canonical trauma-film paradigms (e.g., reminder + Tetris vs control) and exploring boundary conditions that can produce both positive and null selective-interference effects.

# Empirical Anchor: VRT Dataset

## Task overview

The VRT dataset pairs an intrusion-analog task with an intentional retrieval task using shared film materials.
For present purposes, the key feature is that both tasks yield scored, film-linked retrieval events that can be mapped onto retrieved-context operations (unguided context→item retrieval vs more controlled, goal-directed retrieval).

## Scoring and dependent variables

Free-text reports are scored using an Autobiographical Interview–style scheme that separates film-linked episodic detail from off-target content and errors.
The current modeling scaffold focuses on three derived measures:

- DV1 (intrusion rate): total number of valid VRT button presses in the first VRT, excluding presses whose reports cannot be linked to any film clip.
- DV2 (intrusion content): number of unique film details reported in the first VRT, excluding repetitions of recurring details.
- DV3 (hotspot overlap; secondary): number (or proportion) of unique film details that appear in both VRTs, used to quantify voluntary access to one’s own hotspots.

## Analysis plan

We treat the scored VRT measures as constraints on both what the model must reproduce and how task events should be represented.
Initial analyses prioritize:

- Group- and condition-level effects on DV1 and DV2 (with robustness checks using alternative “detail” definitions).
- The distributional structure of retrieval events (e.g., clustering by clip, repetition structure, and between-task overlap).
- Task-structure-sensitive organization metrics (e.g., temporal contiguity and semantic organization) that can adjudicate between context-reinstatement vs control-based explanations.

# Retrieved-Context Account

## Retrieval modes and control

## Mapping the VRT into model observables

## Parameterization strategy

# Results

## VRT behavioral constraints

## Model-based account and calibration

# Generalization Simulations

## Simulating selective interference in trauma-film paradigms

## Explaining heterogeneity and null effects

# Discussion
