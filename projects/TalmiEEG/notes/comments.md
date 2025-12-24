> Data prep: please let's eliminate this step by using the files Robin prepared in a specific folder that I pointed you to when I read the document last time.

The file in the folder you pointed me to turned out to be identical to the one I was using, so these steps (merging with behavioral data in `All_Included_Subjects.csv`, imputing missing LPP scores) are still necessary.
I believe that Robin's file deliberately excluded behavioral data that were missing (reliable) LPP scores, so merging with the behavioral data file was needed to restore those missing events.
I've updated the text to clarify the provenance of the files used.

> Primacy buffers: does the orig paper explain about them?
> does robin's method explain?
> I think their recall was not recorded by the authors or they were different items, please check these sources and if not i'll have a look also.

These were excluded because the authors worried that primacy effects would interfere with analyses of the relationship between emotion and memory.
I believe they were excluded before Robin received the data, but the gap is discussed repeatedly across notes in the shared folder.
Recall data for these study events is not available in any of the files I have.

> Missing data: interpolation good, but please list how many trials were missing, per condition, on average + SD.

These details have been added to the text.

> Figure 3: I don't understand why your figure is different from robin's.
> To clarify Robin's figure, this is not raw data - it's data from the mixed model which will have removed some random effects.
> The early LPP values you used may be raw, while his may be z-scores?

Robin and I both used the same Z-normalized LPP values.
However, Robin's figure represents lines of best fit from a mixed effects model predicting recall from LPP and item type, while my figure shows directly observed recall rates within percentile-defined LPP value bins.
The fitted model uses observations across the full range of LPP values to estimate recall rates at value bands with few observations, while my binned approach estimates recall rates only from items within each bin.
Therefore, the two figures differ in how they estimate recall rates at extreme LPP values with few observations.

I believe my figure would align more with Robin's if I used model predictions rather than raw recall rates, or if I improved binning to balance counts across bins.
However, I think the best path forward is configure analyses of the relationship between LPP and recall that avoid binning entirely, as in my LPP by Recall Status and Item-Category plots below.
If your concern is focused more on whether my data practices match Robin's, I can take steps to confirm this by replicating his mixed effects model analysis and plotting procedure.

For reference, Robin included a note about the the Z-transformation applied to LPP values in a corresponding README file:

> Z indicates that the amplitudes have been z-normalised.
> Please note that this normalisation is done by separately for every trial and electrode and involves subtracting the baseline (-200-oms) from each data point after 0 and dividing by the SD of the -200-0ms time window.
> In other words it is similar to a standard ERP baseline correction with the only difference of dividing with the SD of the baseline.
> The z normalisation gives change in amplitude from baseline on a z-scale where M = 0 and SD = 1.

> Figure 5: I don't understand how it's possible to have so many more occurrences of LPP for neutral items than negative ones at every level of LPP and still have higher memory/LPP for negative?

There are more neutral items than negative items in the dataset overall, which leads to higher counts in each LPP bin for neutral items.
However, recall rates are computed as the proportion of recalled items within each bin, so higher counts do not directly translate to higher recall rates.

Once we account for this count imbalance, we still observe 1) that LPP amplitudes for negative items tend to be higher on average than for neutral items and 2) that recall rates scale more strongly with LPP amplitude for negative items than for neutral items.

> pure-attention eCMR: I didn't know how to 'tell' the model that an item was emotional without telling it that the context of the item was emotional.
> how is this solved here?

Rather than 'tell' the model that an item is emotional, pure-attention eCMR posits that the only difference between how emotional and neutral items are encoded for later recall is how much attention they receive during study.
Emotional items receive more attention, which strengthens the associations between temporal context and those items.
This differentiation works similarly to how CMR scales context-to-item associations based on primacy effects, where early list items receive more attention and thus stronger associations.

> two-context eCMR: it's not clear here which context-to-item parameter is increased with emotion, temporal or emotional.

In two-context eCMR, the learning rate for emotional context-to-item associations (not temporal context-to-item associations) is scaled up for emotional items.
This means that when emotional context is reinstated during recall, it more strongly cues emotional items due to the enhanced associations formed during encoding.

> "fixes the emotional context to fully track the most recent item" - what parameter is fixed here?

The drift rate ($\beta_{enc}$) specifically for emotional context is set to 1 during encoding.
This means that emotional context fully updates to reflect the features of the most recently studied item, effectively making it a direct representation of the last item's emotional features.
This is in contrast to standard eCMR, where emotional context can reflect a recency-weighted history of the emotional features of multiple prior items.

> Figures 10-11: " The Two-Context eCMR variant appears to produce a bigger boost to recall rates for the first and last study positions for negative items" - I didn't see this in the graph

You are right.
My comments were about an older version of the plots where differences were apparent.
I'll update the text to remove that claim.
Instead, I think the broad conclusion is that both models are hard to distinguish at all based on our comparisons in this document.

> 4.1.2.5 here you seem to use LPP in the model but how did you do this?
> where is it discussed?
> I didn't understand what's plotted in fig 17.
> might be useful to say 'simulated recall rate' in figures that are from models, like in fig 17 (which otherwise looked too much like fig 6) and also in fig 20 and 29.
> Sorry I'm being thick; I didn't immediately understand why LPP (a measured number) changes when you change model parameters.
> Later, I thought it's because the model changes recall rates at each SPC; this could be stated in the legend - simulated recall rate at each SPC - just as a reminder.

I did not use LPP in these models.
LPP values at each study event do not change model behavior at all in pure-attention eCMR or two-context eCMR.
I generated this plot anyway by simulating each subject's trials (study and recall events) using the model and parameters fitted to those subjects.
I then plotted the relationship between studied items' LPP values and whether those items were recalled in the model's simulations.
This helps us see whether the model can capture the *observed* interactive relationship between emotionality, LPP, and recall, even without using LPP as an input.
The plots show that they cannot.
Despite reproducing an emotion-enhanced memory effect, neither model reproduces the observed relationship between LPP and recall within each item category.
This outcome is key because it motivates the need for a neurally-informed extension that captures this relationship.

> 4.1.3 "Both models struggle to capture the emotion-enhanced memory effect adequately. Most critically, they underestimate recall rates for negative items past study position 10. They also overestimate recall rates for neutral items, particularly in early study positions between 3 and 8." I didn't see this in the graph

You are right.
Those sentences were based on an earlier version of the plots where these discrepancies were apparent.
After some further tuning, I was happily able to eliminate those specific issues.
I've updated the text to remove those claims.

> 4.2 please add equations to clarify what emotion_scale does.
> Good point about multiplicative effects.
> I think I naively thought: if a neutral item is attended at level x (whether primacy, or not) I will make the emotional one 2x.
> That what would matter is that you always boost the emotional item to the same degree relative to what it would have been.
> I didn't care if in SPC4 emotion boost is 3x and SPC 1 30x, because this preserved the relationship of 10x and 1x for neutral.
> I didn't notice that there was a special misfitting to the early positions.
> Is it possible that because I only used the emotion boost for the emotional context, and the primacy is for temporal context, I didn't see this issue?

In the specifications and code I've seen for both eCMR and CMR3, primacy was applied to both emotional context- AND temporal context-to-item learning rates.
So I think it's more likely that the multiplicative interaction is easier to overlook when hand-fitting parameters than when using an automated fitting procedure.
On the other hand, it's also possible that a multiplicative interaction between primacy and emotional attention is actually a really good model, and that this dataset is atypical in showing a misfit in early positions.
This possibility is especially salient given that we lack recall data for the first two list items in each trial, which limits our ability to measure primacy effects directly.
In any case, I'll support these model specification sections with equations going forward to make the implementation decisions clearer.

> 4.3 please add equations.
> Please also explain about the threshold - not sure what that does, and why.

I'll add equations to clarify the implementation of the neurally-informed extension.
I think I can go a little further in explaining the threshold here as well.

> Fig 26, all 4 panels look the same, I am guessing the two models are the same, neurally-informed did nothing?

Yes, my apologies for not adding commentary in the text.
The neurally-informed extensions did not improve fits to category-specific or general serial position curves compared to their regular eCMR counterparts.
This outcome suggests that while the neurally-informed models help capture how LPP relates to recall at the item level, they do not enhance the models' ability to fit the overall emotion-enhanced memory effect as measured by serial position curves.

It is hard to assess how surprising this result is, roughly for the reasons I've already enumerated in the Discussion section.

-   The GMM results don't necessarily imply that adding LPP data will improve fits to summary statistics like category-SPC.
    They don't show that LPP gives us much more information about overall recall rates for emotional vs neutral items than we already get from simply using the labels.
    Instead, the GMM results mainly show that LPP can help us distinguish which emotional items are more likely to be recalled.

-   Our neurally informed extensions are designed to capture the relationship between LPP and recall at the item level in the same way that the GMM does.
    The `lpp_threshold` parameter is fitted to capture the average boost in attention received by emotional items compared to neutral items.
    It thus directly addresses the emotion-enhanced memory effect in practically the same way as the `emotion_scale` parameter configuring emotional attention in regular eCMR.
    I think we confirmed resoundingly that the emotion-enhanced memory effect can be captured by an additive emotional attention mechanism.

-   On the other hand, the `lpp_slope` parameter is not really designed to configure differences in overall recall rates between emotional and neutral items at all.
    Instead, it attempts to capture variability in emotional attention between items as a function of their LPP values.
    It can shift around *which* emotional items are recalled or not, but it does not necessarily change *how many* emotional items are recalled overall compared to neutral items.
    Again, this is the outcome we should have expected coming in based on the GMM results, which show that LPP relates to recall within emotional items but not neutral items.

So as long as we 1) are assessing models based on their ability to predict how much more frequently emotional items are recalled compared to neutral items, and 2) stick to GMM-style hypotheses about the relationship between LPP and recall, we should therefore not expect the neurally-informed extensions to outperform regular eCMR variants.
We will only find differences when we assess models based on their ability to predict which specific emotional items are recalled or not, which is the next step in our modeling pipeline.

> Remaining steps: 1 makes sense; perhaps the fit will look more obvious when it's not about the SPC , although the current approach made sense to me.
> Still, as no evidence was found, we need to show that we didn’t find it even when we used the most rigorous approach.

I think it's maybe misleading to frame the difference between the two approaches as a matter of rigor.
Either can be appropriate depending on the modeling goals and data available, so neither is inherently more rigorous than the other.

The current approach makes *the most* sense if your goal is to assess whether LPP provides additional information about overall recall rates for emotional vs neutral items beyond what eCMR already captures.
But if you have the broader objective to assess whether adding LPP data to eCMR improves its ability to explain why certain items are recalled or not in a given trial, then the current approach is misaligned with that goal.

Of course, I do think the broader objective is more valuable for building psychological theory about how LPP and emotion relate to memory.

> “Refocus evaluation on the complete eCMR specification” – that also made, because we don’t have data here to evaluate which eCMR version is better, might as well use what worked before, unless we have a reason not to – although it made sense to me also to use the simplest version.

For now, I still think we should seek the simplest model that does the job.
Since both pure-attention and two-context eCMR variants are special cases of the complete eCMR specification, any successes we find with those simpler models should generalize to the complete specification.
If we find that neither simpler variant can capture the data adequately, then we can revisit the complete specification later.
I don't think we've yet reached that point!