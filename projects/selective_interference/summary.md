
# Modelling Meeting, 10 November 2025

## Background

The grant work that involves me focuses on three linked aims: 

1. **Address the selective interference effect with CMR.** Characterize and evaluate a retrieved-context account of the post-trauma visuospatial dissociation between intrusive involuntary memories and deliberate recall.
2. **Formally contrast dual-representation and unitary recall accounts.** Specify matched model variants for voluntary and involuntary retrieval and compare their predictions under shared task settings.
3. **Clarify and test the DRA's emotion-disrupted integrative encoding claims.** Encode the dual-representation hypothesis that heightened affect strengthens sensory/affect bindings while loosening contextual links, then assess its behavioral implications within CMR.

These aims roughly align, but can pull in different directions depending on implementation choices.
CMR might reproduce the selective interference effect under specific paradigms without giving us leverage to adjudicate dual versus unitary accounts.
A model embodying the dual-representation account could provide different predictions, even about the interference effect, depending on how and which features we implement.
Finally, the significance of disrupted integrative encoding in either the interference effect or dual-representation account can vary based on our other modeling choices.

## Goals at a Glance

Because of that ambiguity, we can't assume progress on one aim satisfies stakeholder expectations for the others.
To keep the strands aligned, we need explicit initial decisions about three key issues before moving the modelling forward:

1. **Select the focal experiment(s) for the selective interference / Tetris effect.**  Different published paradigms emphasize different measurements and outcomes; agreeing on the anchor prevents us from targeting the wrong behavioral signature.
2. **Define the dual-representation commitments for the initial model.** We need to confirm which elements of the DRA belong in the first implementation so that downstream comparisons are meaningful.
3. **Set the theoretical framework and comparison approach.** Clarifying how we conceptualize interference and voluntary vs involuntary recall will determine the modelling toolkit and evaluation strategy.

Here I outline key considerations for each decision, along with questions to guide our discussion.

## Choose Focal Experiment(s) for Selective Interference Effect

**Question**: Which paradigm and dataset should anchor modeling of the Tetris effect?

A range of experiments have examined the selective interference effect, but they differ in task structure, dependent variables, and manipulations. 
So that we don't chase mismatched outcomes, we need to pick one or more paradigms that best align with our modeling capabilities and the project's goals.

- **The behavioral signature should be unambiguous.** Ideally the focal study clearly shows the selective interference effect (visuospatial load reducing intrusions more than voluntary recall) so we know what the models are meant to capture.
- **The measurement approach should be representative.** Dependent variables ideally align with how this phenomenon is typically measured in the literature so our results directly speak to prior findings.
- **The measurement procedures should be rigorous.** The methodology is ideally one the team trusts so we are confident that we're addressing the intended processes. 
- **The measures should map onto model observables.** The dataset's outputs ideally correspond to quantities our modelling frameworks already produce or at least can extend to without forcing speculative bridges that would themselves need validation.
- **Data and procedures should be detailed enough to constrain mechanisms.** Beyond aggregate scores, we prefer enough resolution to distinguish between competing accounts. In particular, tests of retrieved-context assumptions generally benefit from exact study and recall sequences per trial, as well as different presentation orders across trials.

That choice has to balance several constraints, but our best options might not satisfy all of them.
Some paradigms might be representative, demonstrate an unambiguous effect, and map clearly to CMR observables (e.g., free recall + diary intrusions) but lack rigor (Lau-Zhu, Henson & Holmes, 2019 discusses this issue).
Other studies might match our methodological standards and have detailed procedures (e.g., Experiment 1 from the current project) but not show the effect clearly enough for us to target.
Still others might have strong measurements and clear effects but would take more work to relate to CMR outputs or evaluate its key assumptions.
It could be best to target multiple datasets to triangulate these tradeoffs, but we should limit the scope to what we can manage within the project's timeline.
And either way, we'd still prefer to select one _primary_ benchmark for initial model development.

Here are some possible targets to help guide discussion:

-  **Lau-Zhu, A., Henson, R. N., & Holmes, E. A. (2019). Intrusive memories and voluntary memory of a trauma film: Differential effects of a cognitive interference task after encoding.** If data is detailed enough, the range of experiments in this paper could provide strong modeling constraints and help bridge across paradigms. We'd still select one experiment as the primary target.

- **Experiment 1 from the current project**. This dataset has detailed procedures and measurements we trust, and we have immediate access to the data. However, it does not show a clear selective interference effect, which limits its utility as a benchmark for that phenomenon.

- **Free recall vs diary option from a meta-analysis?**. An approach I discussed with Deborah in an earlier meeting contrasts free recall performance with diary intrusion. This method is frequently used in the literature, seems to produce large effects, and aligns well with CMR observables, though extension to diary intrusions may require assumptions. However, the paradigm is vulnerable to explanation in terms of cue mismatch, and extant data may be poor quality.

- **Better options?** Emily or Rick might have other preferred studies that better balance these considerations.

## Choose Dual-Representation Commitments to Implement
**Question**: What parts of the dual representation account should inform the initial model?

We need a shared definition of "dual representation" so we can concretely compare it to unitary alternatives.
In one sentence, the RR offers this summary of the DRA: "The core assumption of the DRA is that there are in effect two memory traces – sensory and contextualised representations – and that pathological encoding of trauma strengthens the former but weakens the latter, leading to dissociations between involuntary and voluntary memory."
More detailed accounts frequently specify a neurobiological substrate, phenomenological consequences, or broader structural commitments.
Imposing these additional may be important for theoretical fidelity or scientific communication.
However, our definition should normally clarify functional commitments that narrow the model's possible behaviors in ways that can be evaluated against data within the project's timeline.

The grant and registered report converge on three functional properties, but the team has discussed and may add possible extensions or variations:

1. **Emotional arousal strengthens binding among sensory and affective elements.** Retrieval cues that overlap on perceptual or feeling dimensions should readily reinstate those elements, yielding high-fidelity recovery of sensory details even without contextual support.
2. **The same arousal weakens links between those elements and spatiotemporal context.** Voluntary recall should omit or scramble where/when details because contextual cues have difficulty reinstating the full episode.
3. **Cues matching sensory or affective features can retrieve the memory even when context mismatches.** Reminders that match only sensory or affective facets should still trigger involuntary recall, even in settings that share none of the original context.

**But what about the selective interference effect?**
Even with these details, DRA's account of the selective interference effect may still be underspecified.
If visuospatial interference reduces intrusion rates without affecting voluntary recall, what is the dual representation explanation for that pattern and how does the explanation contrast with a unitary account?
So far, I've mostly seen descriptions of the effect as supportive evidence for the DRA, but not a detailed mechanistic account that could aid implementation or evaluation.

Does this specification capture the DRA sufficiently for our initial model? Are there other commitments we should include or consider in alternative versions or future extensions?

## Align on Modelling Framework
**Question**: Given our goals, empirical targets, and dual-representation commitments, how should we set up an evaluation of competing accounts of interference and of voluntary vs involuntary recall?

To help guide discussion, I outline CMR's default account of interference effects in general and under two paradigms for measuring the selective interference effect. Then I attempt to relate the dual representation commitments to those frameworks. Then I try to enumerate possible predictions that could distinguish and test the accounts and guide model development. These can hopefully related with everyone else's perspectives and goals to clarify a path forward.

### How CMR Captures Interference

In baseline CMR, 

In the retrieved-context framework, memory traces are not modifiable after encoding. 
Interference emerges

interference is defined with respect 

can arise in a few ways.

- **Competition among items that share retrieval cues.** When multiple items are associated with similar contexts, they can all be partially reactivated during recall attempts, leading to confusion and errors.
- **Disrupted association 