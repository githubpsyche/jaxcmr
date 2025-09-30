---
title: "Same Item, Separate Contexts: An Instance-based Retrieved-Context Account of Repetition Effects in Free and Serial Recall"
shorttitle: "Same Item, Separate Contexts"
author: 
  - name: Jordan B Gunn
    affiliations:
      - name: Vanderbilt University
        department: Department of Psychology
  - name: Sean M Polyn
    affiliations:
      - name: Vanderbilt University
        department: Department of Psychology
keywords: episodic memory, free recall, serial recall, retrieved-context theory, computational modeling
floatsintext: true
bibliography: references.bib
format:
  apaquarto-docx: default
  apaquarto-html: default
  apaquarto-pdf:
    nocorrespondence: true
    filters:
      - abstract-section
  apaquarto-typst: default
---

# Introduction

First bit just makes the topic interesting. 

Pose: "What is the structure of memory for repeated experience?"

Overview of what we know and what we don't.

Focus: "This highlights the need for more nuanced theories that can explain how repetitions influence the structural organization of memory retrieval."

Present as a promising framework: retrieved-context theory and CMR.

How it has been used to explain free and serial recall generally, and repetition effects specifically.

Along the way, enumerate the key phenomena/domains that the theory has addressed.

Research gap: While CMR represents a substantial advance in computational modeling of key phenomena, existing evaluations of the CMR model as an account of repetition effects have so far been limited in scope.

How the current work builds on previous work by:

1. Adding formal model evaluation, 
2. Simultaneously addressing repetition effects across both free and serial recall, 
3. Focusing scrutiny on how CMR addresses transition rates from repeated items to items studied near either their first or second presentation.

And a parallel paragraph starting with "The present work aims to address these gaps by":

1.  Novel analyses applied to benchmark datasets that identify the independent effects of each study event of a repeated item on the organization of free and serial recall.
2.  Probing limitations of the CMR model in accounting for these effects and pre-existing accounts of how the model captures repetition effects (e.g., @siegel2014retrieved).
3.  Applying the likelihood-based model comparison [@kragel2015neural] and related techniques to evaluate the model's ability to account for these results and examine alternatives that clarify and/or address its limitations.

Then an overview of what we find:

1. In free recall, the temporal contiguity effect is more pronounced relative to the first presentation of a repeated item than to the second.
2. In serial recall, participants are able to consistently transition from recalls of repeated items to their appropriate next items
3. But CMR does not accurately predict these patterns, instead predicting more balanced transition rates to neighbors of *either* presentation of a repeated item.

Then the big take-away that CMR has a gap:

This suggests that CMR's default implementation of retrieved context theory lacks the ability to flexibly focus context-based retrieval and reinstatement on specific study instances of repeated items, even though humans *can*.

Our analyses of the CMR model identify two specific limitations that prevent it from accurately capturing the observed patterns of repetition effects in free and serial recall:

1. **Blended reinstatement of context associated across all instances of a repeated item.** This prevents the model from flexibly focusing contextual updating on the context associated a specific instance of a repeated item.
2. **High representational overlap between contextual representations associated with different traces of repeated items.** This is caused by item-based contextual updating, where context evolves by reinstating an encoded item's pre-existing contextual associations. This results in a high degree of representational overlap between the context associated with different traces of repeated items. Even with instance-based contextual updating, this prevents the model from selectively transitioning to neighbors of a specific instance of a repeated item.

To address these limitations, we propose a new computational model of repetition effects in free and serial recall that builds on the CMR model but incorporates two key modifications:

1. **An instance-based retrieved-context model.** Memory traces rather than items compete for retrieval based on their similarity to the current context, and the model only updates the context associated with the retrieved trace instead of a composite context across all traces.
2. **Event-specific contextual representations.** Instead of (or in addition to) evolving context during encoding based on the encoded item's pre-existing contextual associations, evolve context using an event-specific contextual input that is unique to each study instance. This makes it possible to construct contextual cues that target neighbors of a specific instance of a repeated item.
3. Called InstanceCMR (or whatever).

# Theoretical Framework

## Temporal Dynamics of Free and Serial Recall
Present CMR and retrieved-context theory as a framework for understanding benchmark phenomena in free and serial recall.

## Repetition Effects in Free and Serial Recall
Present revised retrieved-context account of repetition effects in free recall, contrasting with Lohnas (2014) and other relevant work. Two parts of the critique: study-phase retrieval doesn't actually work that well to address spacing effects, and the model predicts associative interference between repeated items.

Discuss effects in serial recall, applying RCT framework to propose a new account of the effects.

distinguish ranschburg effect as likely a response suppression thing outside the scope of this work.

Focusing on: associative interference.

Conversely connect back the problem of associative interference to free recall: just as transition rates from recalls of repeated items to serial neighbors has provided an important constraint for models of serial recall (e.g., ruling out chaining models), we argue that the same is true for free recall.

Propose: CMR's specification shares limitations with the chaining model, instantiating associative interference between contextual representations associated with different instances of a repeated item.

## Instance Theory
Cover literature contrasting instance theory with relevant frameworks.

Focus on analysis by Anderson and otherwise distinguishing classical instance-based model architectures from instance-based retrieval.

Key idea: instance models are best defined in terms of how they retrieve information from memory, rather than their specific implementation: they have the capacity to retrieve specific instances from memory, controlling interference/blending of information across instances.

Key idea: whether a model has this property depends on both the model's architecture and the kinds of representations it uses.

Relate to evidence that humans exhibit this capacity across a variety of tasks that contact memory, including free and serial recall.

## An Instance-based Retrieved Context Model of Repetition Effects

High-level presentation of the model with a diagram and everything.

# Model Structure

Key model variants presented side-by-side, focusing on InstanceCMR and CMR.

# Simulation 1: Conditional Equivalence Between Connectionist and Instance CMR

Establish conditions under which the connectionist and instance-based CMR models are functionally equivalent.

Particularly:

- When recall competition mechanisms are equated
- When using the same contextual inputs
- When study lists lack item repetitions

Present fits to HealeyKahana2014 dataset as empirical evidence along with "latent" MFC and MCF plots. 

# Simulation 2: Addressing Repetition Effects in Free Recall

Focus on LohnasKahana2014 dataset and HowardKahana2005 dataset (maybe different ones).

- Present fits establishing CMR's ability to address several repetition effects in free recall, focusing on benchmarks, spacing effect, and neighbor contiguity effect.

- Test claim (made earlier) that CMR can accomplish this without study-phase reinstatement and explain why. Do it by fitting and evaluating a reduced version of the model that lacks study-phase reinstatement.

- Present repetition lag-crp and show that CMR struggles to account for it.
- Show that only the ICMR variant that both uses event-specific contextual representations and instance-based contextual updating can account for the repetition lag-crp.
- Plus overall fits to the data.
- Plus evaluation of viable alternative candidate models (Specifically: a deficient processing model like CMR-DE [in-house variant], and a repetition suppression model like CMR2).

# Simulation 3: Addressing Repetition Effects in Serial Recall

Focus on KahanaJacobs2000 dataset and hopefully something else.

- Present fits confirming ability to address benchmark phenomena in serial recall, focusing on just serial recall accuracy curve and transposition gradients. 
- Also present fits to the ranchberg effect and our in-house analysis of associative interference.
- Present fits to the transition rate analysis, showing that CMR struggles to account for the transition rates from recalls of repeated items to the appropriate next items.
- Show that only the ICMR variant that both uses event-specific contextual representations and instance-based contextual updating can account for the transition rates.
- Plus overall fits to the data.
- Plus evaluation of viable alternative candidate models (Specifically: a deficient processing model like CMR-DE [in-house variant], and a repetition suppression model like CMR2).

# Discussion Topics

## What do we now know about the structure of memory for repeated experience that we didn't before?
To be precise, we now know that performance on free and serial recall tasks are consistent with a distinctly instance-based account of memory search where participants can retrieve specific study instances of repeated items to guide retrieval instead of always a blended representation of all instances of a repeated item.

We now know that the CMR model doesn't really use study-phase reinstatement all that much when it comes to item repetitions. Theories that emphasize the role of study-phase reinstatement in free recall (e.g., @siegel2014retrieved) but maybe also some Karpicke stuff need a second look in light of this finding. 

If the behavioral evidence that participants do enact study-phase reinstatement is robust, then in turn CMR may need updates to its implementation of study-phase reinstatement to improve the mechanism's efficacy. 

And we have identified/highlighted theoretical constraints that future alternative accounts will have to wrestle with. The repetition lag-crp and related analyses. More broadly, these results highlight the importance of specificity (or whatever we'll call it) as a property of memory retrieval that deserves emphasis in future modeling research.

At a higher level, we've sort of confirmed something that is intuitively obvious: episodic memory can retrieve specific episodes from memory -- even when a retrieval cue is shared across multiple episodes. 

This is a key feature of episodic memory that has been emphasized in the literature, but our work provides a more formalized account of how this works in serial and free recall, and of its behavioral consequences.

## Instance vs Prototype Models
Gist: complements prior work highlighting similarities and drawing distinctions between instance-based and prototype models of memory, helps unify across the different approaches.

ICMR can be a drop-in replacement for CMR in many cases, demonstrating the portability of retrieved context theory across different model architectures.

But also provides a new way to investigate memory search and retrieval in terms of the specificity of context-based retrieval.

Cite that one recent Jamieson Jones paper presenting instance theory as a general theoretical framework for psychology.

## Abandoning the Item-Based Contextual Updating Mechanism?
Main issues: this is a challenge to item-driven contextual evolution? 

Most relevant evidence might be cross-list intrusions.

Review literature on item-driven contextual evolution, at least as far back as that one HowardKahana2002 paper.

What about prior work suggesting that item-driven contextual evolution is a key component of CMR?

Gist: CMR's item-based contextual updating mechanism is a key component of the model's ability to account for cross-list intrusions in free recall. 

Propose: a multi-layer context representation that can be used to represent both item-driven and event-specific contextual representations, with selective prioritization of one or the other depending on the task. 

On the other hand, research in serial recall shows evidence for position-based coding, enabling position-specific cross-list intrusions.

If we interpret InstanceCMR as proposing an event-specific contextual representation, then additional work is needed to clarify how the model can account for position-specific cross-list intrusions.

Alternatively, if we interpret InstanceCMR as proposing an position-specific contextual representation, it may be able to address these limitations without additional work.

## Situating Model in Serial Recall Literature
Discuss CRU, Lohnas2024, and the debate across item-based and position-based coding accounts of serial recall, at least as far back as that KahanaJacobs2000 paper. 

What kind of model is InstanceCMR in this space? Does it make new predictions or contrasting predictions with existing models (particularly CRU)?

Treating Serial Recall and Free Recall as a Single Modeling Domain Providing Complementary Constraints for Memory Models
I really want to highlight this point because it's cool and fun. 

## Beyond
Retrieved context theory has been used to address a variety of domains besides free and serial recall, and of course repetition is a ubiquitous phenomenon in memory research.

At minimum can connect to other domains in free recall involving repetition, such as Karpicke's work on retrieval practice where study-phase reinstatement is a key component of the model.

# References

::: {#refs}
:::