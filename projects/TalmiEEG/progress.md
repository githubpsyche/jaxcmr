# EEG-ECMR Progress

Goal: prototype a neurocognitive extension of eCMR that can use Early/Late LPP signals associated with each studied item to improve over eCMR’s baseline prediction of the **emotion-enhanced memory effect**.

# Data Preparation

I initially focused preparation on `Single_Trial_Behavioural_and_EEG_Data_Z.csv` located in the lab drive. However, this dataset was missing study events (and a record of their retrieval success), further reducing my ability to evaluate a model with it.

I located a more complete behavioral dataset, `All_Included_Subjects.csv`. Although it lacked Early/Late LPP scores, it contained full behavioral records for all study events. Missing LPP values were imputed using within-subject, within-trial means for items of the same emotional category. I otherwise confirmed that the dataset was complete: 9 trials per participant, 20 study events per trials with complete records of whether items were recalled or not.

By:
- Combining the complete record of behavioral data with the partial record of early/late LPP readings,  
- Imputing missing data, and  
- converting the results into my HDF5 extension of the EMBAM format established by the Polyn lab, 

I was finally ready to begin analyses and model development.

# Benchmark Analyses

Benchmark analyses serve two purposes: validating data preparation and defining concrete targets for model development.

## Recall Rate by Study Position and Item-Condition / Category-SPC

Our core behavioral analysis to address it on a categorized-serial position curve (category-SPC), plotted cleanly in the “Meeting with Nathaniel Daw” slides on the lab drive. The analysis involves taking the ratio of recalled to total items matching the category label and studied at each serial position.

LEFT: provided in slides. LEFT: generated from preprocessed dataset.   
![][image1]![][image2]  
Error bars reflect a between-subjects 95%-bootstrapped confidence interval at each study position.

The match between shared and generated categorized serial position curves confirms the integrity of data preprocessing.

## LPP by Study Position and Item-Condition

I drew this to help interpret the relationship between early/late LPP and serial position / emotionality. Most model variants augmented by LPP data will make variation across factors of the category-SPC look more like variation across factors of this plot. 

Early LPP on the LEFT and Late LPP on the RIGHT.  
![][image3]![][image4]  
Error bars reflect a between-subjects 95%-bootstrapped confidence interval at each study position.

These plots confirm that both early and late LPP signals vary with item emotionality across study positions. 

## Recall Rate by LPP and Item-Condition

A successful neurally-informed extension of eCMR should not just identify emotional and neutral items, but provide additional information beyond these labels about how and whether emotional and neutral items will be recalled. We thus plot the relationship between LPP values and recall rate across NEGATIVE and NEUTRAL items.

Early LPP on the LEFT and Late LPP on the RIGHT.  
![][image5]![][image6]  
Overall Early LPP x Recall Rate relationship:  
![][image7]  
One line per subject:  
![][image8]

Previous analyses of the relationship between LPP and recall rate seem to have assumed that any relationship would be linear. But these results suggest that this assumption could miss a clear non-linear relationship between the two variables.

But how many observations do we have per bin per subject?  
![][image9]  
Hardly any at all, suggesting that only study events between the LPP bins centered at \-3.25 and 4.86 is relevant for modeling. Between these bins, there’s a clear positive relationship between (early) LPP and recall rate for emotional items but not for neutral items, matching reports in the reference slides:

![][image10]

On the other hand, we may find 

# Loss Functions

Because the dataset does not record the order in which participants recalled items, we cannot compute the full likelihood of recall sequences, which is the best approach for fitting memory search models. This limitation forces us to focus on a simpler question: **does the composition of recalled items match what the model predicts**? 

For now, we ignore recall termination and treat the number of recalled items as given. That means the loss function should measure how well the model reproduces *what* was recalled rather than *how much* or *in what order*.

## Sequence Likelihood-Based Loss 

The most formally precise approach would involve enumerating every possible recall sequence consistent with the observed set of recalled items. The probability of the data under the model would then be the **sum of the probabilities assigned to all compatible recall sequences**. However, the number of possible recall orders increases exponentially with the number of recalled items, making this approach computationally infeasible for all but the smallest datasets.

A potential compromise involves estimating the sequence probability via **Monte Carlo sampling**: drawing a constrained sample of recall sequences consistent with the observed recall set and averaging their model probabilities. This method defines a **speed–accuracy trade-off**: increasing the number of sampled sequences improves the stability of the loss estimate but sharply increases computation time. Because reliable estimation would require a large number of samples per trial, we reserve this approach for more advanced stages of the project.

## Mean Squared Error (MSE) Loss

Given these constraints, we instead adopt a mean squared error (MSE) loss between an observed summary statistic and the same statistic generated by simulated data. This approach is widely used in computational memory modeling when full likelihood estimation is infeasible.

The procedure is as follows:

1. Compute a summary statistic from the experimental data.  
2. Simulate the model repeatedly using a given parameter set.  
3. Compute the same statistic from the simulated outcomes.  
4. Compute the mean squared difference between the observed and simulated statistics.

This MSE provides an interpretable measure of how well a given parameter set allows the model to reproduce the behavioral pattern of interest. While this method does not yield a true likelihood or a maximum-likelihood estimate of parameters—and discards trial-level prediction detail—it offers a practical, scalable, and transparent evaluation metric. Model comparisons can still be made statistically by contrasting best-fitting MSE values across models within subjects.

In the first implementation, the summary statistic was the serial position curve (SPC)—the recall probability of items as a function of study position. This approach provides a straightforward test of whether the model captures overall primacy and recency trends across the list, ignoring differences between emotional and neutral items.

In the end, though, it made sense to focus directly on capturing the Category-SPC introduced in the Benchmark Analyses section. Here, recall probability is computed separately for **negative** and **neutral** items at each study position, allowing the model to be evaluated on its ability to reproduce both serial position effects and the emotion-enhanced memory effect simultaneously. 

## Conclusion

For the current analyses, model fitting minimized the mean-squared error over a combination of the negative and neutral category-SPCs in the data, as visualized in the Benchmark Analyses plots. Once initial modeling is completed, we will shift to a likelihood-based loss function where models are evaluated based on the probability they assign across a random sample of recall sequences consistent with observed recalls in a trial.

# Modeling Plan

The goal of the current modeling pass is to determine whether incorporating LPP data can, in principle, improve a retrieved-context model’s ability to account for the **emotion-enhanced memory effect**. Our strategy is thus to clearly characterize model performance across three scenarios: 

1. **Baseline CMR:** CMR without sensitivity to item emotionality or LPP data.  
2. **Simplified eCMR**: A minimal extension of CMR that makes it sensitive to item emotionality in line with eCMR’s commitments. Rather than incorporate all parameters and mechanisms from eCMR, we only use enough details to achieve adequate capture of our benchmark statistics.  
3. **Neurally Informed Extension**: We use event-level LPP measures as modulators of emotional-attentional strength in addition to simplified-eCMR’s selected parameters.

Right now, we’re still focused on settling on the best specification for simplified eCMR, so various variants of this model have been considered. The neurally-informed extension is still pending.

Furthermore, across model variants, mechanisms controlling recall termination are excluded, with the total number of items instead matched to the number of observed recalls in a matched trial. This reduces the number of parameters to fit and evaluate, making models easier and faster to evaluate and interpret. In general, we seek a narrow space of parameters and mechanisms to evaluate at this stage in the interests of maximizing the interpretability, efficiency, and modularity of our comparison. The trade-off is that some dynamic features of fuller versions of our models are deferred to future modeling stages.

# Simplified eCMR

The **simplified eCMR** variant introduces a minimal emotional-attention mechanism while maintaining tractable parameterization and interpretability. It serves as a bridge between baseline CMR and the more complex neurally informed model planned for later stages. Additional complexity to the model will be added as needed to capture benchmarks valued for this study.

The key simplification embraced is that the **integration rate of emotional context** is fixed to 1.0. This means that the emotional context at each step of either encoding or retrieval is determined entirely by the most recently processed item, removing the need for a separate emotional feature-to-context memory (eMFC).

The emotional context-to-feature memory (eMCF) is represented as a vector where:

- Cells corresponding to neutral items hold a value of 0\.  
- Cells corresponding to emotional items hold a value scaled by an emotional-attention parameter

An additional parameter sometimes used in eCMR—to scale the relative contribution of temporal vs emotional context during retrieval—is redundant under this specification, since the emotional-attention parameter already modulates that balance. With specialized drift rates and feature-to-context memory learning rates also excluded, **simplified eCMR addresses the emotion-enhanced memory effect with only one additional free parameter beyond those specified in baseline CMR**.

## Parameter Sensitivity Tests

To confirm that the model’s extra parameter can address the emotional category-spc, we fit the base model to the dataset then simulate simple-ECMR with different configurations of its arousal parameter, plotting the emotional category-spc for each set of simulations. Here we generated one category-spc for negative items and one for neutral items from the simulated data:  
 ![][image11]![][image12]  
The LEFT plot shows how the recall rates for NEGATIVE items change with configurations of `emotion_scale`. The RIGHT plot shows how the recall rates for NEUTRAL items change.

These results show that scaling up the parameter increases recall rates for negative items while reducing recall rates for neutral items, building confidence that the parameter can improve the baseline model’s ability to address the emotion-enhanced memory effect. However, specific assumptions made by the model about issues like the interaction between primacy and arousal/emotional-attention as well as latent assumptions about effects on response order or initiation could still throw off model fitting.

## Comparison of Fits to Benchmark Statistics

### Negative \+ Neutral Category SPC

Data:  
![][image13]  
Baseline CMR (LEFT) vs Simple-eCMR (RIGHT):  
![][image14]![][image15]  
Finding: Simple-eCMR simulates a consistent emotion-enhanced memory effect; baseline CMR doesn’t. 

### Negative-Category SPC

For direct comparison between data and model simulations, it’s often useful to draw lines directly on top of each other. This becomes a challenge to interpret when an analysis draws two lines per dataset, so we focus on each component of the category-spc individually.

Baseline CMR (LEFT) vs Simple-eCMR (RIGHT):  
![][image16]![][image17]  
Finding: Simple-eCMR does a visibly better job than Baseline CMR of capturing recall rates for NEGATIVE items. Both models do better capturing recall rates for early study positions than for later study positions.

### Neutral-Category SPC

Baseline CMR (LEFT) vs Simple-eCMR (RIGHT):  
![][image18]![][image19]  
Finding: Simple-eCMR’s improvement over Baseline CMR at capturing recall rates for NEUTRAL items is less evident than its improvement at capturing NEGATIVE item recall rates, but there’s still a clear improvement. 

### Regular Serial Position Curve

Baseline CMR (LEFT) vs Simple-eCMR (RIGHT):  
![][image20]![][image21]  
Finding: Simple-eCMR and Baseline CMR produce strikingly similar simulations of the overall serial position effect. Again, we observe gaps in both models’ ability to capture recall rates for late study positions but also for study positions 3-6. 

### Conclusion

Simple-eCMR does a visibly better job than Baseline CMR of capturing recall rates for NEGATIVE items across study positions. Differences are less evident in simulations of recall rates for NEUTRAL items and all item types across study positions. Both models do better capturing recall rates for early study positions than for later study positions. 

Further work could use a likelihood-based loss function to fit models (instead of fitting to a specified summary statistic) and include outcomes of a formal model comparison instead of visual comparison of summary statistics. Additional model development could seek to improve these baselines by investigating reduced recall rates for late study positions but also for study positions 3-6, or by implementing and evaluating more assumptions from the complete eCMR specification.

Despite limitations, Simplified eCMR provides an adequate improvement over baseline eCMR for capturing the emotion-enhanced memory effect.

# Neurally Informed Extension

We now turn to exploring how a neural signature known to carry information about whether negative items will be recalled or not can further augment model performance. Our focus for the moment is an initial prototype that efficiently demonstrates that such improvement using present data is *possible*. Downstream work can attempt to identify the *best* approach.

## Emotion Enhanced LPP

Before settling on a specification, here we seek to clarify what we know about the LPP signal that we’ll be using to extend eCMR:

- Early LPP values are consistently higher for NEGATIVE items than for NEUTRAL items.  
- For NEGATIVE items, LPP values are consistently higher for RECALLED items  
- For NEUTRAL ITEMS, LPP values are do not consistently index whether an item will be recalled

# Side Issues

## Do arousal and serial position additively or interactively influence memory strength?

![][image22]  
The specification for eCMR seems to configure a multiplicative relationship between emotional-attention and primacy-attention. I think that leads to the prediction that the recall rate gap for emotional vs neutral items depends on the compared items' study positions: higher for items studied near the beginning of a list than elsewhere.

Here’s a worked example:

If the primacy-attention value at the first study position is 10 and the list-wide emotional attention scale is 3, then the total attentional boost at that study position is 10\*3=30 for emotional items but just 10 for neutral items. So the difference in attention for emotional vs neutral items is 20\.   
   
At a later study position, the primacy attention value falls to 1 and the emotional-attention scale is still 3\. This makes the total attentional boost at that study position 1\*3=3 for emotional items but just 1 for neutral items. The difference in attention for emotional vs neutral items is just 2\.   
   
That's ten times lower than the emotional vs neutral boost at study position 1\! So according to this specification, the attentional boost that drives the emotion-enhanced memory effect is stronger for early study positions than late study positions. 

The present data and model evaluation pipeline an opportunity to examine whether emotion and primacy independently/additively or interactively/multiplicatively configure emotional context-to-item memory strengths. Fitting and simulating versions of eCMR implementing these two hypotheses produces the following observations:

### Negative \+ Neutral Category SPC

Here we draw separate plots for data and each model variant. 

Data:  
![][image13]  
Multiplicative (LEFT) vs Additive (RIGHT) variants:  
![][image23]![][image15]

### Negative-Category SPC

Here we compare model and data directly in each plot, with one plot per model. 

Multiplicative (LEFT) vs Additive (RIGHT) variants:  
![][image24]![][image25]

### Neutral-Category SPC

Here we compare model and data directly in each plot, with one plot per model. 

Multiplicative (LEFT) vs Additive (RIGHT) variants:  
![][image26]![][image19]

### Conclusion

These simulations suggest that primacy and emotionality *independently* rather than interactively modulate memory strength. The gap between the two variants seems localized to their predictions about recall rates for negative items in early study positions. The multiplicative model tends to predict a much higher recall rate for these items than observed in the data, while the additive model produces no such discrepancy. **We thus use the additive model for simulations going forward, and propose pursuing a paper focused on this distinction as a future project.**

## Improving Trade-Off Between Recency and Primacy

Baseline and eCMR fits seem to give low recall rates to the last few study positions in our serial position curves. One possible reason is that CMR lacks a mechanism to eliminate recency effects without boosting the strength of the primacy effect (through start-of-list context reinstatement). Since the data exhibits neither a strong recency nor a strong primacy effect, the model struggles to capture it.

On the other hand, it could be that eCMR’s complete specification is necessary to avoid this down-modulation of the serial position effect. However, since eCMR’s eMCF only affects recall rates for emotional items, I think it unlikely to help address the down-modulation observed for the neutral-SPC. 

A common approach in retrieved-context models to attenuate the recency effect is to introduce a delay drift rate following the study phase, causing context to drift toward an out-of-list context before retrieval begins. The rationale is that by shifting the contextual cue away from its final study state, recently studied items will receive less contextual overlap and hence less retrieval advantage.

However, this mechanism generally fails to balance primacy and recency because it does not alter the relative similarity between the contextual cue and the study-phase contexts of individual items. Context moves equally far from all studied items, preserving their relative strengths as retrieval cues.

To overcome this limitation, I tested a modified approach where post-study context drifts not toward an external or out-of-list context, but toward all in-list contexts, where the input vector is essentially a ones mask over context units that are linked to any studied item. This adjustment removes temporal directionality from the final cue, effectively making recall initiation reflect just the differences in relative strength across context-to-item associations. 

This didn’t help either, so I’m leaving this problem for later.

# Other Notes

Preference: Warm color for emotional. Black or whatever for neutral.