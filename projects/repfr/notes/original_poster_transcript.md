# Same item, different traces: Dissecting the structure of memory for repeated experience
Jordan Gunn, Sean Polyn, Department of Psychology, Vanderbilt University

## Introduction

### Background

Image: A diagram representing the free recall task as a timeline of study and recall phases. A arrow points back from the recall phase to the study phase, indicating retrieval of studied items and transition structure to items studied nearby in time to successively recall more items.

Retrieved context theory (RCT) conceives free recall as a sort of mental time travel, where people revisit past contexts to retrieve associated items (Tulving, 1972).

Each retrieval reactivates and reinstates the context of the retrieved item, biasing subsequent retrievals to nearby items in the study list.

But RCT is most fundamentally reflected in lag-contiguity effect, where items studied close together are more likely to be recalled together (Howard & Kahana, 2002).

### But How Does Mental Time Travel Work for Items With Multiple Study Contexts?

Image: A diagram again representing the free recall task as a timeline of study and recall phases. This time, the same item (C) is shown to be studied twice at different points in the study phase. An arrow points back from a cloud indicating the recall of item C in the recall phase to the two different points in the study phase where this recalled item C was studied at different times. Arrows with question marks point from the two study points of C to the item to their right, indicating uncertainty about how retrieval of that item would affect subsequent retrievals.

### Context Maintenance and Retrieval Model (CMR; Polyn & Kahana, 2009)

Image: A diagram representing the CMR model architecture. It shows a cycle involving an item representation layer, an item-to-context associative memory, a context representation layer, and a context-to-item associative memory. The item representation layer cues item-to-context memory to update the context representation layer, which cues context-to-item memory to guide memory search and retrieve items that get represented back in the item representation layer.

Extends mental time travel account of memory search to explain what temporal context is and  how temporal context evolves.

Temporal context derives its state from the reactivation and partial reinstatement of the temporal context already associated in memory with the current item

### Datasets

For main analyses, used data from Lohnas & Kahana (2014). 
35 participants performed delayed free recall of 48 lists presenting 40 words each, with three relevant conditions: 
Control: No item repetitions
Spaced: 20 Items repeated with 1-8 intervening items
Mixed: 28 items presented once; 6 items presented twice with 0-8 intervening items

For replication, we used data from Howard and Kahana (2005). 
66 participants performed delayed free recall of 15 lists presenting 90 words each. Each item presented three times per trial in two relevant conditions:
Short spacing: Two to six intervening items
Long spacing: six to twenty intervening items

## Repetition Lag-CRP
A bias toward neighors of _first_ presentation!

Image: A diagram representing the free recall task as a timeline of study phase focused on illustrating how lag is calculated separately relative to each presentation of a repeated item. The letter D is studied at study positions 4 and 11 and the letter F is studied at position 6. An arrow from position 4 to position 6 indicates that F is at lag +2 relative to D's first presentation. An arrow from position 11 to position 6 indicates that F is at lag -5 relative to D's second presentation.

Image: A Repetition lag-CRP drawn from Lohnas & Kahana (2014) data. The x-axis represents lag relative to either the first or second presentation of repeated items, ranging from -3 to +3. The y-axis represents recall probability. Two curves are shown: one for lag-CRP relative to first presentation (blue) and one for lag-CRP relative to second presentation (orange). The blue curve shows a pronounced peak at lag +1, indicating a strong bias toward recalling items that were studied immediately after the first presentation of the repeated item. The orange curve also peaks at lag +1, but there is a large gap in lag + 1 transition probability from the first presentation than from the second presentation, indicating that recalls of repeated items are more likely to be followed by items that were studied after their first presentation than after their second presentation.

For recalls for repeated items with spacing >= 4, track relative to their first and second study positions:
1. For lags [-3, -2, -1, 1, 2, 3] if transition with lag is possible
2. For lags [-3, -2, -1, 1, 2, 3] if transition with lag happened
Actual Count / Possible Count = Recall Probability by Lag

### How Much of This is Actually a _Repetition_ Effect?
To measure the proportion of transitions expected in the absence of repeated items, we match the serial positions used in the mixed lists to corresponding items in the control lists, shuffling assignment 100 times. 
The effect disappears under this control analysis, confirming that it’s a repetition effect.

Image: A Repetition lag-CRP control analysis drawn from Lohnas & Kahana (2014) data. The x-axis represents lag relative to either the first or second presentation of repeated items, ranging from -3 to +3. The y-axis represents recall probability. Two curves are shown: one for lag-CRP relative to first presentation (blue) and one for lag-CRP relative to second presentation (orange). Both curves are shaped like a traditional lag-CRP, with peaks at lag +1 and -1 and decaying probabilities for larger lags, but there is no significant difference between the two curves, indicating that the repetition lag-CRP effect observed in the main analysis is not present when controlling for serial position effects.

### Is the Difference Significant?

To summarize the effect and control for serial position effects, we measure +1 lag conditional transition rates relative to each repetition index. 

1. Consider just transitions from repetitions
2. Separately calculate relative to each repetition-index
3. Subtract control analysis values (when available)

Perform subject-level one-tailed paired t-test of scores relative to first study position versus relative to second.

Lohnas & Kahana (2014) Test Result:  t=3.704, p=0.0004

### Is This Dataset Specific?

Same analysis using Howard & Kahana (2005) data:

Image: Same analysis using Howard & Kahana (2005) data. Three lines are drawn because items are presented three times in this experiment. The x-axis represents lag relative to each of the three presentations of repeated items, ranging from -3 to +3. The y-axis represents recall probability. Three curves are shown: one for lag-CRP relative to first presentation (blue), one for lag-CRP relative to second presentation (orange), and one for lag-CRP relative to third presentation (green). The blue curve shows a pronounced peak at lag +1, indicating a strong bias toward recalling items that were studied immediately after the first presentation of the repeated item. The orange and green curves also peak at lag +1, but there is a large gap in lag + 1 transition probability from the first presentation than from the second and third presentations, indicating that recalls of repeated items are more likely to be followed by items that were studied after their first presentation than after their second or third presentations.

(Uncontrolled) +1 Transition Rate Comparison Result: t=2.909, p=0.0025

## But CMR Struggles to Capture the Effect

Fit model parameters based on likelihood of **response sequences** instead of summary statistic MSEs (Kragel et al, 2015; Morton & Polyn, 2016). Find best fits for each subject and simulate trials from experiment design using each fit.

### CMR (Main Analysis)

Image: A Repetition lag-CRP drawn from simulations of the standard CMR model fit to Lohnas & Kahana (2014) data. The x-axis represents lag relative to either the first or second presentation of repeated items, ranging from -3 to +3. The y-axis represents recall probability. Two curves are shown: one for lag-CRP relative to first presentation (blue) and one for lag-CRP relative to second presentation (orange). The first presentation curve peaks at lag +1 a little higher than the second presentation curve, but the difference is much smaller than in the empirical data.

### Control Analysis

Image: A Repetition lag-CRP _control_ analysis drawn from simulations of the standard CMR model fit to Lohnas & Kahana (2014) data. The x-axis represents lag relative to either the first or second presentation of repeated items, ranging from -3 to +3. The y-axis represents recall probability. Two curves are shown: one for lag-CRP relative to first presentation (blue) and one for lag-CRP relative to second presentation (orange). Both curves are similar in shape and differences to the main analysis plot, indicating that the standard CMR model does not simulate a repetition lag-CRP effect that is distinct from serial position effects.

To account for effect, CMR must predict a difference in mixed lists but not control lists. It doesn’t!

So observed difference is mainly serial position effects.

CMR Controlled +1 Transition Rate Result: t=1.07, p=0.1456

### But Good Fits to Other Benchmarks?

Images: A series of four plots showing that the standard CMR model fits other benchmark effects well. The first plot shows serial position curves from empirical data and CMR simulations, demonstrating that the model captures primacy and recency effects. The second plot shows probability of first recall (PFR) curves from empirical data and CMR simulations, indicating that the model accurately predicts which items are likely to be recalled first. The third plot shows lag-CRP curves from empirical data and CMR simulations, illustrating that the model captures the tendency to recall items studied close together in time. The fourth plot shows the spacing effect by plotting recall probability as a function of spacing between repetitions, demonstrating that the model accurately reflects the benefits of spaced repetitions.

### Simulation of Howard & Kahana (2005)

Image: A Repetition lag-CRP drawn from simulations of the standard CMR model fit to Howard & Kahana (2005) data. The x-axis represents lag relative to either the first or second presentation of repeated items, ranging from -3 to +3. The y-axis represents recall probability. Two curves are shown: one for lag-CRP relative to first presentation (blue) and one for lag-CRP relative to second presentation (orange). The first presentation curve peaks at lag +1 slightly higher than the second presentation curve, but the difference is much smaller than in the empirical data.

Uncontrolled +1 Transition Rate Test:
t=0.93, p=0.1771

## Interpreting CMR's Challenge

### Main Explanation: Repeater Traces Don’t Include Forward Contexts

Image: Heatmap of context-item associations formed during study in CMR for a hypothetical list of 8 items where ab item is repeated at positions 3 and 6. The x-axis represents context feature units and the y-axis represents item indices from 1 to 8. The heatmap shows strong associations between study positions and context features for each item presentation. At the repeated item at presentation 6, context strongly overlaps with context from presentation 3, producing high overlap in the kinds of cues that can can retrieve either trace. There is no way to selectively target one presentation's context over the other for retrieval and reinstatement.

In CMR, repeating an item reinstates context associated with its prior presentation, linking current and prior contexts in memory (study-phase retrieval). 

Later, when the repeated item is recalled, an average between these two context states is reinstated to prime the next recall. The next retrieval is primarily driven by the similarity of retrieval context to items’ encoding contexts.

But since first-presentation reinstated context occurs before forward neighbors, biased support for first-presentation neighbors is insufficient to drive effect. Instead, both contexts similarly prime retrieval for either presentation’s forward neighbors.

### Similar Challenge for Trace-Weighting Strategies

Deficient processing (Greene, 1989) and instance-retrieval accounts of repetition effects traditionally explain phenomena such as the spacing effect in terms of diminished access at retrieval to the memory trace corresponding to an item repetition.

But variants of CMR that modulate learning rates in MFC or MCF, or more selectively reinstate trace contexts before retrieval, fail to account for the effect and are dis-preferred during model fitting.

Shared problem with study phase retrieval: driving retrieval with either trace’s context does not substantially bias transitions to either presentation’s +1 neighbors.

Predicted Transition Rate as Function of Context Cue Under a Subject’s Parameters:

Image: Heatmap focusing on recall probability for forward and backward neighbors of repeated items as a function of the type of retrieval cue, each given its own column. In y-axis, three context probes are considered: "Default Composite", "Just First Trace" and "Just Second Trace". Across all three probes, the recall probabilities for forward neighbors of both the first and second presentations of repeated items are similar, indicating that none of the cueing strategies effectively bias retrieval toward a single presentation's neighbors.

Activation scaling (applying an exponent to item activations before retrieval competition) drives larger differences but is similarly dis-preferred under model fitting!

## Discussion
- Across two datasets, after recalling an item presented repeatedly in study lists, participants tended to transition to neighbors of the item’s initial rather than successive study positions.

- Control analyses indicate that bias is due to repetition rather than serial position.

- CMR could not account for these patterns, despite study-phase retrieval and even after considering mechanisms to emulate deficient processing accounts of repetition effects.

- Suggests need for more detailed account of context-based retrieval when items have multiple study contexts

## References

Tulving, E. (1972). Organization of memory.
Howard, M., & Kahana, M. (2002). Journal of mathematical psychology.
Lohnas, L. & Kahana, M. (2014). JEP: LEMC.
Howard, M., & Kahana, M. (2005). Psychonomic bulletin & review.
Morton, N. & Polyn, S. (2016). Journal of memory and language.
Polyn, S., Norman, K., Kahana, M. (2009). Psychological Review.
