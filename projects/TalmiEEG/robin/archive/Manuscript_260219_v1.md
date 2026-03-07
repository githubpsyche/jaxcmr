---
title: "Using the Late Positive Potential to constrain emotional Context-Maintenance and Retrieval"
author:
  - name: Robin [Surname TBD]
  - name: Deborah Talmi
bibliography: references.bib
---

## Introduction

Retrieved-context theory [@howard2002distributed] describes the state-of-the-art in terms of the rules that govern the chances that an experience we have now will come to mind later.
It is supported by vast behavioural and neural data.
<!-- FIXME: we will need to add extensive citations to support such a claim. -->
According to this theory, the likelihood of recalling a particular experience is determined by the similarity between the encoding and retrieval "contexts" - abstract constructs that are implemented in brain states.
Initially, the Temporal Context Model only considered similarity between temporally-contiguous brain states.
More recently [@polyn2009context], retrieved-context theory recognised that semantic and experiential similarity also influence which experience would come to mind.
Semantic similarity refers to the tightness of pre-experimental associations between stimuli.
Experiential similarity [@polyn2009context] refers to shared encoding processes during the session itself.
For example, when some stimuli are processed auditorily and others visually, the experiential similarity is greater within, compared to across, presentation modality.
<!-- FIXME: This misrepresents RCT somewhat. It captures the context-binding side of it, but that's generic with respect to context-based accounts of episodic memory. RCT's additional contribution is that items can call back their associated contexts throughout encoding and retrieval. -->
<!-- FIXME: is this really the right way to start an academic paper about this topic? I worry reader should maybe have some clearer foreshadowing re what this paper will be about. -->

A recent variant of retrieved-context theory, emotional Context-Maintenance and Retrieval [@talmi2019retrieved] (eCMR) focuses on the emotional dimension of experiential similarity. 
<!-- FIXME: I don't like the overindexing on "experiential similarity" as a characterization of what emotionality does here.-->
People process emotional experiences differently than neutral ones in terms of appraisal [@sander2005systems], attention [@pourtois2013brain], physiology [@stephens2010autonomic] and feelings.
"Emotional similarity" [@riberto2022neural], therefore, is a type of experiential similarity, which refers to similarity between encoding operations of emotional stimuli.
<!-- FIXME: I don't necessarily agree with this characterization. It's ambiguous across paradigms whether emotional similarity is just experiential similarity rather than semantic similarity given how many expeirments induce emotions via semantic processing (e.g., processing an emotionally compelling sentence or image). I think we should express agnosticism about this detail since eCMR is compatible with either interpretation. Actually, the more I think about it, the more it seems that the distinction provided in the previous paragraph is a little ill-posed. It may be more appropriate to treat experiential similarity as induced by different sources? I guess we should consider the distinction presented by polyn2009context since we're citing them. Even then, it might be neater to avoid this thorny matter completely since, again, eCMR is compatible with any of these perspectives. -->
eCMR follows CMR in the way it simulates experiential similarity.
A stimulus category, such as presentation modality, or emotionality, which is thought to trigger unique processing, is modelled as an additional item feature, called 'source' item feature, with values that differentiate stimuli of each type.
Thus, in eCMR emotional stimuli are modelled with a value of '1' in the source feature, and neutral stimuli with a value of '0'.
<!-- FIXME: this turns out to be an inapt description of eCMR since neutral items aren't marked as similar to each other under this account. Refer to the eCMR paper to get the language right. -->
This feature is used not only to describe which items are similar to each other, but also to model the cognitive repercussion of their unique features.
It is known that emotional items enjoy preferential processing during encoding.
<!-- FIXME: this will certainly need supporting citations. -->
These are simulated in eCMR by tighter associations between the item and context layers in the model, specifically between the source item feature and the source context feature.
<!-- FIXME: This is imprecise -->
The degree of extra processing is governed by the value of parameter $\Phi_{\text{emot}}$.
When it equals 1, emotional and neutral items are modelled as processed equally.
<!-- FIXME: this needs to be clarified not to imply that emotional and neutral items are processed in the same way when the parameter has a value of 1; they will still receive distinct tags in the emotional source features, influencing processing. Should say "as receiving the same amount of processing" or the like.-->
When it is greater than 1, emotional items are modelled as receiving extra processing during encoding.
In order to use eCMR to predict what an individual, or a group, would recall, it could be very useful to set the value of $\Phi_{\text{emot}}$ through empirical measurement [@turner2016more].
<!-- FIXME: I'm curious how the turner2016more entry relates to this.  -->
<!-- FIXME: Awkward wording, and perhaps undermotivated. Maybe need to hype up eCMR some more. -->
An excellent candidate is the late positive potential ERP component (LPP), which increases in amplitude when participants process emotionally-arousing visual stimuli [@schupp2006emotion].
<!-- FIXME: are there any additional citations that can support this claim? Also, this is the first use of "ERP"; we should maybe clarify for non-technical readers, though perhaps it should not be here. LPP is of coruse also subliterature-specific jargon. -->
In this project, our aim was to examine the suitability of the LPP for neuro-cognitive modelling of memory recall.

Changes in LPP amplitudes due to affective significant have been observed at the group level across a range of stimuli and presentation durations.
<!-- FIXME: we need ample citations for this claim -->
<!-- FIXME: change to say "affective significance".  -->
However, to be useful to model an individual participant, it is also important establish how sensitive and specific the LPP is at the level of the individual.
Recently, @schupp2021case have examined this question by using a case-by-case approach in three studies, using a range of emotional induction technique.
<!-- FIXME: "techiques". This sentence could probably be stated more simply. -->
They observed that the LPP was sensitive to emotional in 98% of the cases observed.
<!-- FIXME: sensitive to "emotional" is not correct. Maybe "emotion" or "emotional arousal" was meant here. Will have to check the paper to be sure. -->
Analysis of comparisons which did not differ in emotional arousal allowed them to further determined that the effect is also specific to arousal.
<!-- FIXME: I think this description needs to be more detailed and contextualized against relevant work. How confident can we be that LPP indexes _arousal_ in particular? -->
These results support the use of the LPP to constrain the $\Phi_{\text{emot}}$ parameter.
<!-- FIXME: I'm ambivalent about this logic. There seem to be some hidden assumptions never explicitly stated about what $\Phi_{\text{emot}}$ was intended to represent theoretically in eCMR or at least _could_ represent theoretically in eCMR. If $\Phi_{\text{emot}}$ represents or is is a candidate representation for arousal, that's fine, though theoretically interesting. CMR3 represents arousal as a source feature, not in $\Phi_{\text{emot}}$. This could provide clear motivation for comparisons down the line, and in turn provide stronger theoretical punchiness to what we're exploring here. -->

However, it is not known whether a similar prevalence would be obtained under the presentation conditions more common in memory experiments.
Participants in @schupp2021case's study encoded briefly-presented pictures passively.
By contrast, participants in typical memory recall experiment are aware of an upcoming memory test.
<!-- FIXME: memory recall "experiments" -->
Intentional encoding instructions could dilute the LPP if they bias attention away from intra-stimulus processing towards inter-stimulus relationships, if participants employ encoding strategies that minimise LPP differences, or engage in covert spaced rehearsal of emotional and neutral stimuli.
<!-- FIXME: this seems a bit too meaty to just be one sentence. Intra-stimulus vs inter-stimulus distinction is maybe jargony here for some readers too. Similar for "covert spaced rehearsal". The proposal that intentional encoding instructions could dilute the LPP effect by diverting attention is reasonable but could be conveyed more clearly. -->
The potential impact of encoding mode is evident in findings that incidental encoding instructions alter temporal contiguity effects [@healey2019contiguity].
<!-- FIXME: can we find more evidence for this claim? And can we find more evidence of clear connection to possible disruption of arousal / intra-stimulus processing relevant to LPP as we've framed it? -->
Our first aim, therefore, was to replicate @schupp2021case's findings using a setup more typical in memory research.
In addition to the specific benefit of testing the sensitivity and specificity of the emotional modulation of the LPP during intentional encoding, a replication would also serve to increase confidence in the findings if the same are obtained in another large sample (Schupp and colleagues had a total *n* of 50), tested in a different laboratory, with a different set of stimuli, and where the stimuli are presented for a longer duration (2 seconds instead of 150ms at in Schupp and colleagues' experiments).
<!-- FIXME: this sentence is too long and uses wording atypical of a scientific manuscript, I feel. Might break-up, rethink. -->

Our second aim was to examine the relationship between the emotional modulation of the LPP and the emotional modulation of memory.
Attention and memory are typically coupled, such that when all else is held equal, increased attention during encoding increases participants' ability to recall attended stimuli subsequently.
Increased attention to emotional stimuli during encoding is considered a key driver of enhanced memory for these stimuli.
Accordingly, previous studies report that emotional arousal increased both LPP and memory.
<!-- FIXME: need extensive sourcing for these three sentences above -->
Yet despite the average effect of emotion on both LPP and memory, it is not known whether the relationship between them is consistent either within-subject (across study-test cycles) or between-subjects.
<!-- FIXME: is it really? we need to research relevant work so we can be sure. -->
One possibility is that individuals with increased emotion-dependent LPP differences will exhibit larger emotion-dependent memory differences.
Another, not mutually-exclusive possibility is that individuals would recall best those lists where encoding was accompanied by increased LPP.
<!-- FIXME: this distinction could be clearer, perhaps. -->
The dissociation between within- and between-subject effects in ERPs was recently demonstrated for the left parietal old-new effect, where robust within-subject effects were observed despite null between-subject effects [@macleod2017left].
<!-- FIXME: can we maybe find mroe examples of such ERP patterns in the literature? -->
Simulations with eCMR show that all else held equal, increasing the value of $\Phi_{\text{emot}}$ results in increased recall of emotional stimuli (Figure 1).
<!-- FIXME: we already said this. Now there's a figure? Hmm. -->
If LPP differences provide a good index for the value of this parameter, then the model predicts a positive association between the amplitude of the difference wave and the difference between the number of pictures recalled, either within-subject, between-subjects, or both.

In addition, we wanted to investigate whether the relation between the LPP and emotion enhanced memory is better captured on the single-trial level or when averaged over all the items in a list to decide at what level we should add the LPP to the eCMR.
It could for example be that the LPP is too noisy at the single-trial level to be useful, or that the loss of variability in LPP amplitude between items after averaging leads to a weaker relationship between the LPP and emotion enhanced memory.
It was therefore important to examine at what level the relationship between the LPP and subsequent memory is stronger to inform our modelling decisions.

Finally, we compared different versions of the eCMR to investigate in what way the LPP increases the predictability of emotion enhanced memory.
Specifically, we compared a standard version of the eCMR model with fixed values of $\Phi_{\text{emot}}$ to a version where this value is exchanged for the LPP and another model which included the interaction between both the emotional label (negative vs neutral) and LPP amplitude.
This model comparison has relevance for the interpretation of the LPP ERP effect.
If the LPP reflects attention or working memory processes that are unrelated to emotion then we expect to see similar performance between the LPP only and the LPP x Emotion model.
On the other hand, if the LPP reflects arousal which mainly is elicited by negative items, then we would expect the LPP x Emotion interaction model to outperform both the other models in predicting emotion enhanced memory.
<!-- FIXME: This is probably not the enumeration of comparison results we want to report, and I'm not sure about the theoretical mapping presented here either. Meh. -->

## Method

This secondary data analysis is based on data from @zarubin2020contributions.
A more detailed description of the methods in the original experiment is provided in paper.

### Participants

The forty included participants from the mixed condition in @zarubin2020contributions were re-analysed.
One participant was excluded for missing data from three lists and one participant was excluded due to having less than 16 trials in one of the subsequent memory conditions.
The final sample consisted of 38 participants (28 females, *M*~age~ = 20, *SD*~age~ = 1.07).
Gender and age data was missing for one of the participants.
All participants provided written informed consent and the experiment was approved by Wofford College institutional Review Board [@zarubin2020contributions].

### Materials

There were 22 pictures in each list and nine lists in the whole experiment.
The two first pictures were used as fillers to control for primacy effects and were excluded from data analysis.
The pictures were selected from the International Affective Picture [@lang2005international], the Geneva Affective Picture Database [@danglauser2011geneva], the Emotional Picture Set [@wessa2010emopics], the image pool of @talmi2007contribution and Google Images.
The pictures were either of negative valence or neutral valence.
The neutral condition was divided into semantically unrelated and semantically related neutral pictures in the original study, but they were collapsed in this secondary data analysis to increase the number of trials in the neutral condition.

### Procedure

The participants studied 22 pictures shown consecutively on the screen for 2000ms with a 4000ms inter trial interval, intended to reduce emotional carryover effects [@talmi2012accounting].
The presentation order of both the lists and the images within lists was randomised.
A one minute arithmetic distracter task followed immediately after the study task to reduce rehearsal of the pictures and recency effects.
A verbal recall test followed the distracter task.
The participants were instructed to recall images only from the list that they had just seen.
The participants had three minutes to recall the images from the study phase, but could end earlier if they finished earlier.

### EEG pre-processing

Raw EEG files were downloaded from the open science framework page for the @zarubin2020contributions study.
The EEG was pre-processed with the EEGLAB toolbox in MATLAB [@delorme2004eeglab].
The data was re-referenced offline to the average of the left and right mastoid electrodes and filtered between 0.1 and 40Hz with a band pass FIR filter.
The data was time-locked to the stimulus and segmented into epochs between --2000ms to 3000ms post stimulus onset.
The epochs were baseline corrected using a pre-stimulus time window between --200ms to 0ms.
Customized threshold functions from the FASTER toolbox [@nolan2010faster] was used to detect and reject bad channels and bad epochs based on participant-level z-transformed values over trials and electrodes that exceeded ± 3 (see if you need to expand this later).
Independent component analysis was used to correct for EOG artefacts, by manually removing components classified as blinks, horizontal or vertical eye-movements.
Bad channels which were deleted prior to ICA were interpolated after the ICA cleaning.
In addition, bad channels within single epochs were detected and interpolated with a criteria of ±3 SD.
All subjects contributed at least 16 trials in each of the four ERP conditions.
The mean number of trials were 30.7 (*SD* = 6.08) in the negative remembered condition, 26.1 (*SD* = 5.54) in the negative forgotten condition, 48 (*SD* = 15.33) in the neutral remembered condition and 66.5 (*SD* = 16.93) in the neutral forgotten condition after pre-processing.
Trials that were rejected during EEG preprocessing were excluded from the behavioural analyses as well to ensure that the EEG and behavioural data contained the same trials and to increase the chance of finding a relation between the two measures.

### Behavioural Data Analysis

We investigated if there was an emotion enhanced memory effect by comparing the proportion recalled negative and the proportion recalled neutral items with a paired t-test.

## EEG Data Analysis

### Focal ERP Analysis

We first analysed the electrodes and time windows where the LPP typically is maximal in the literature.
Averaged EEG amplitude was extracted from four centroparietal electrodes, Cz, CP1, CP2 and Pz, in an early time window between 400-1000ms and a late time window between 1000-2000ms post stimulus onset.
We extracted the EEG amplitudes separately for subsequently remembered and subsequently forgotten trials for both the negative and the neutral condition.
All mixed effects model analyses were conducted using the lme4 package in R [@bates2015lme4].

We first performed a linear mixed effects model to analyse if ERP amplitude was predicted by emotion and subsequent memory.
In this model we used ERP amplitude as the dependent variable and fixed effects of Emotion (Negative vs Neutral) and Subsequent memory (Remembered vs Forgotten) as fixed effects and random intercept + random slope effects of participant.
The formula was:

ERP amplitude \~ 1 + Emotion \* Memory + (1 + Emotion \* Memory \| Participant).

The random effect structure needed to be simplified to only include a random intercept effect of participant in the late LPP time window due to a warning of singular fit.
The statistical significance of the fixed effects and their interaction was assessed using $\chi^2$ Wald test as implemented in the car package [@fox2019companion].
Significant interactions were followed-up using pairwise contrasts as implemented in the emmeans package in R [@lenth2024emmeans] applying a Benjamini & Hochberg (false discovery rate, FDR) correction for multiple comparisons.

### Global ERP Analysis

We complemented the focal analysis with a global data-driven analysis including all data points between 0 and 2000ms, the time window when the stimulus was presented.
The reason for including this analysis is that analysing ERP data from a small group of electrode sites and time windows may overlook effects at other electrode sites and time windows.
In this analysis we controlled for multiple comparisons using cluster-based permutation tests as implemented in the FieldTrip toolbox [@maris2007nonparametric; @oostenveld2011fieldtrip].
The first step of this analysis is to perform *t*-tests at every ERP data sample.
Significant samples, uncorrected at $\alpha$ = .05, that were neighbours in time or space, including at least two electrodes, were grouped into clusters.
The cluster level t-value was computed by calculating the sum of the t-values from all samples included in the cluster.
Finally, we compared the cluster level *t*-value in the observed data to the size of the maximal cluster in 5000 permutations, for which the datapoints were randomly swapped between conditions within participants.
The cluster level Monte-Carlo *p*-value was calculated as the proportion of the permutations for which the maximum cluster was larger than the clusters in the observed data.
The exact spatio-temporal distribution of the clusters should be interpreted with caution given that the correction for multiple comparisons are applied at the cluster level rather than at the individual sample level [@sassenhagen2019cluster].

### Bootstrap Analysis

To be useful for the purpose of model constraint, the LPP need to be reliable within subjects.
We therefor performed a bootstrap analysis to assess the reliability of the Negative--Neutral difference within each participants data based.
The rationale of this analysis is that negative stimuli should elicit a more positive LPP than neutral stimuli independent of which specific trials contributed to calculating the ERPs.

The number of trials were first equated for the two conditions with undersampling.
The data included in this analysis was extracted from the same centro-parietal electrode sites as in the focal ERP analysis, i.e.
Cz, CP1, CP2 and Pz.
Bootstrap resampling from single trials was used to create a distribution of 1000 resampled ERPs per condition.
A sliding time window was applied to detect the most positive LPP peak (the mean amplitude of the 100ms time window between 400-1000ms for the early LPP and between 1000-2000ms for the late LPP. Difference scores were calculated by subtracting the amplitude in the neutral condition from the amplitude in the negative condition for each bootstrap sample. Next, the proportion of positive bootstrap differences were calculated for the two time-windows based on the rationale that participants with robust LPP effects should have a large proportion positive bootstrapped differences. We interpreted the emotion enhanced LPP as being reliable if a proportion of 0.9 or more of the bootstrapped differences were positive based on that this cut off has been used in similar analyses in the concealed information test literature [for example @rosenfeld2019p300].

### Relationship between the LPP and Emotion Enhanced Memory

The ERPs were z-transformed using the mean and the standard deviation of a pre-stimulus interval between --200-0ms.
This was done on a single-trial level and separately for all electrodes.
Next, the z-transformed LPP was extracted from the same centroparietal electrodes as in the focal ERP analysis for both the early, 400-1000ms, and the late, 1000-2000ms, time window.

In the between subjects correlation, we calculated the average LPP and memory performance across the 9 lists and computed a negative--neutral difference score in both measures.
These measures were then correlated using Bayesian pearson correlation.

Besides this between subject correlation we also calculated a within-subject correlation.
In this analysis, we correlated the emotion enhanced LPP with the emotion enhanced memory across the 9 lists and calculated the average correlation coefficient across lists for each participant with pearson correlation.
Next, we performed a Bayesian one-sample *t*-test against zero to test if there was a correlation.

Finally, we investigated the relation between memory performance and the LPP on the single-trial level with a binomial generalized linear mixed-effect model where we investigated if Subsequent memory was predicted by Emotion and LPP amplitude.
In this model we used Subsequent memory as the dependent variable and included Emotion (Negative vs Neutral) and LPP amplitude as fixed effects and random intercept + random slope effects of Participant.
The model formula was:

Memory \~ 1 + Emotion \* LPP + (1 + Emotion \* LPP \| Participant).

## eCMR Specification

Building on the behavioral and ERP results reported above, we test mechanistic accounts of how the LPP can modulate encoding within retrieved-context models.
We embed the Early LPP as an item-level modulator of learning strength in eCMR and compare three variants that differ only in how LPP enters the encoding rule.

We use a connectionist CMR architecture with a temporal context state and bidirectional item-context associations.
We implement emotion and Early LPP as modulators of learning strength, and we do not include a separate emotional context layer.
This keeps the retrieved-context dynamics fixed across variants while localizing emotion/LPP effects to a single encoding pathway.

We denote temporal context as $c^{T}$ and item features as $f_{i}$.
$M^{FC}$ maps item features to temporal context, and $M^{CF}$ maps temporal context to item features.

| Symbol                      | Meaning                           |
|:----------------------------|:----------------------------------|
| $c^{T}$                     | temporal context                  |
| $f_{i}$                     | item feature vector (one-hot)     |
| $M^{FC}$                    | feature-to-context associations   |
| $M^{CF}$                    | context-to-feature associations   |
| $\beta_{enc}$               | encoding drift rate               |
| $\beta_{start}$             | start-list drift rate             |
| $\beta_{rec}$               | recall drift rate                 |
| $\alpha$                    | shared support in $M^{CF}$        |
| $\delta$                    | self-support in $M^{CF}$          |
| $\gamma$                    | learning rate for $\Delta M^{FC}$ |
| $\phi_{s}$                  | primacy scale                     |
| $\phi_{d}$                  | primacy decay                     |
| $\tau_{c}$                  | choice sensitivity                |
| $g_{i}$                     | learning strength                 |
| $L_{i}$                     | list-centered Early LPP           |
| $\kappa_{L}, \lambda_{L}$   | LPP main-effect slope, offset     |
| $\kappa_{EL}, \lambda_{EL}$ | LPP-by-emotion slope, offset      |

: Parameters and structures specifying eCMR. {#tbl-params}

Pre-experimental memory is initialized as

$$M_{pre(ij)}^{FC} = \left\{ \begin{matrix}
1 - \gamma & \text{if }i = j \\
0 & \text{if }i \neq j
\end{matrix} \right.\ $$

$$M_{pre(ij)}^{CF} = \left\{ \begin{matrix}
\delta & \text{if }i = j \\
\alpha & \text{if }i \neq j
\end{matrix} \right.\ $$

During encoding, temporal contextual input is retrieved as $c_{i}^{IN} = M^{FC}f_{i}$ and normalized to unit length.
Temporal context updates according to

$$c_{i}^{T} = \rho_{i}c_{i - 1}^{T} + \beta_{enc}c_{i}^{IN}$$

$$\rho_{i} = \sqrt{1 + \beta_{enc}^{2}\left\lbrack \left( c_{i - 1}^{T} \cdot c_{i}^{IN} \right)^{2} - 1 \right\rbrack} - \beta_{enc}\left( c_{i - 1}^{T} \cdot c_{i}^{IN} \right)$$

Feature-to-context learning proceeds via Hebbian updates as:

$$\Delta M_{ij}^{FC} = \gamma f_{i}c_{j}^{T}$$

The primacy-graded attention term is defined as:

$$\phi_{i} = \phi_{s}e^{- \phi_{d}(i - 1)} + 1$$

The complete learning-strength term is defined differently in each model variant (see below).
In the base case, it is:

$$g_{i} = \phi_{emot,i} + \phi_{i}$$

Altogether, context-to-feature associations are updated as:

$$\Delta M_{ij}^{CF} = g_{i}c_{j}^{T}f_{i}$$

At recall onset, temporal context is shifted toward the pre-list state as

$$c_{start}^{T} = \rho_{N + 1}c_{N}^{T} + \beta_{start}c_{0}^{T}$$

At each recall step, temporal context cues item activations according to:

$$A = M^{CF}c^{T}$$

Given this cue, at each recall step, the probability of recalling item $i$ is

$$P(i) = \frac{A_{i}^{\tau_{c}}}{\sum_{k}^{}A_{k}^{\tau_{c}}}$$

After recall, the retrieved item's temporal context is reinstated via $M^{FC}$ and integrated with drift rate $\beta_{rec}$, and the process iterates.
We omit an explicit termination mechanism; instead, we match each trial's observed output length during simulation (see below).

### Model Variants

We compare three variants that differ only in how emotion and Early LPP enter the learning-strength term $g_{i}$ in $\Delta M_{ij}^{CF}$.
Let $e_{i}$ be an indicator that equals 1 for emotional items and 0 for neutral items.

We define the emotion term in the learning strength as $\phi_{emot,i} = \phi_{emot}e_{i}$.
In the emotion-only model, LPP effects are removed by setting $\kappa_{L} = 0$ and $\kappa_{EL} = 0$, yielding

$$g_{i} = \phi_{emot,i} + \phi_{i}$$

In the main-effect model, the interaction term is removed by setting $\kappa_{EL} = 0$, yielding

$$g_{i} = \phi_{emot,i} + \phi_{i} + \kappa_{L}\left( L_{i} - \lambda_{L} \right)$$

In the separate-slope model, both LPP terms are allowed, yielding

$$g_{i} = \phi_{emot,i} + \phi_{i} + \kappa_{L}\left( L_{i} - \lambda_{L} \right) + \kappa_{EL}\left( L_{i} - \lambda_{EL} \right)e_{i}$$

In implementation, negative values of $g_{i}$ are clamped to 0 in all variants.

### Evaluation Approach

We evaluated each model by how well it predicts the unordered set of recalled items on each list (not recall order), because our data is missing recall order information and because our primary behavioral outcomes and benchmarks target item selection rather than order-dependent dynamics.
Let $S_{t}$ denote the set of items recalled on trial $t$, and let $r$ range over recall sequences that contain exactly the items in $S_{t}$ (in any order).
We define the *set likelihood* as

$$P\left( S_{t} \mid \theta \right) = \sum_{r:\, r\ \text{yields}\ S_{t}}^{}P(r \mid \theta),$$

where $\theta$ are the model parameters.

Evaluating this sum exactly is infeasible because the number of sequences that yield a given set grows rapidly with $\left| S_{t} \right|$.
We therefore approximate $P\left( S_{t} \mid \theta \right)$ by Monte Carlo sampling: for each trial we draw $K$ random permutations $r_{1},\ldots,r_{K}$ of $S_{t}$, compute $P\left( r_{k} \mid \theta \right)$, and estimate the sum as $\left| S_{t} \right|!\,\frac{1}{K}\sum_{k = 1}^{K}P\left( r_{k} \mid \theta \right)$.
Because the factorial factor does not depend on $\theta$, we maximize the (unscaled) mean permutation likelihood $\frac{1}{K}\sum_{k}^{}P\left( r_{k} \mid \theta \right)$, which is proportional to the set likelihood and yields identical fits and $\Delta$AIC comparisons.

We maximize the objective with differential evolution [@storn1997differential].
Because the optimizer is stochastic, we run three independent fits per participant and model variant and retain the best-performing fit.
We compare variants using AIC computed from the approximate set log-likelihood.
We then assess qualitative adequacy by simulating each fitted model on the study lists while matching each trial's observed output length and plotting the benchmark summaries.
This set-likelihood approach evaluates models on their ability to reproduce the full recalled sets, providing a stringent test of mechanistic hypotheses.

## Results

### Behavioural Results

Replicating the original analysis, a paired t-test showed that memory performance was significantly higher for negative items (*M* = 0.54, *SD* = 0.1) compared to neutral items (*M* = 0.42, *SD* = .14), *t*(37) = 6.18, *p* \< .001.

[For the record. If we run a one-way repeated measures ANOVA with proportion remembered as the dependent variable and condition (Negative vs Category vs Neutral) as the independent variable, we get a significant effect *F*(1,65) = 42.85, *p* \< .001, partial eta squared = 0.54. Follow up pairwise comparisons shows that Negative items were remembered more than both category items (*t*(37) = 2.46, *p* = .01) and neutral items (*t*(37) = 9.41, *p* \< .001). Memory performance was also higher for category items (semantically related neutral items) than for neutral items (semantically unrelated neutral items; (*t*(37) = 7.29, *p* \< .001). The pattern of results was hence Negative \> Category \> Neutral.]{.mark}

### EEG Results

#### Focal ERP Analysis

A linear mixed effect model was conducted to investigate if Emotion (Negative vs Neutral) and Subsequent memory (Remembered vs Forgotten) predicted ERP amplitude was conducted separately for the early and the late LPP time window.

In the early time window, there was a main effect of Emotion ($\beta$ = .10, $\chi^2$(1) = 6.84, *p* = .009) and a main effect of Subsequent memory ($\beta$ = .25, $\chi^2$(1) = 26.70, *p* \< .001) and a significant Emotion x Subsequent memory interaction ($\beta$ = .20, $\chi^2$(1) = 14.70, *p* \< .001).
Follow-up pairwise contrasts showed that the LPP amplitude was higher for subsequently remembered than for subsequently forgotten negative items (*z* = -5.61, *p* \< .001), while there was no difference in LPP amplitude for subsequently remembered and forgotten neutral items (*z* = -.88, *p* = .379).

In the late time window, there was a main effect of Emotion ($\beta$ = .14, $\chi^2$(1) = 12.72, *p* \< .001), indicating that the LPP amplitude was higher for negative than for neutral items.
There was no significant main effect of Subsequent memory ($\beta$ = .08, $\chi^2$(1) = 2.13, *p* = .144).
There was a trend for a significant Emotion x Subsequent memory interaction ($\beta$ = .09, $\chi^2$(1) = 3.06, *p* = .080).

![Early LPP ERP waveforms (400--1000 ms)](figures/media/image1.png){#fig-1}

![Early LPP topography and ERP differences](figures/media/image2.png){#fig-2}

![Late LPP ERP waveforms (1000--2000 ms)](figures/media/image3.png){#fig-3}

#### Global ERP Analysis

We first analysed the main effect of Emotion (Negative vs Neutral trials) and found a positive cluster (*p* \< .001) which was significant between approximately 380-2000ms with a widespread distribution.
In other words, there was an emotionally enhanced LPP effect which roughly overlapped with the early and late time windows analysed in the focal analysis.

Replicating the focal analysis, there was also a main effect of subsequent memory with more positive ERPs for remembered compared to forgotten trials (*p* \< .001) with a widespread spatial distribution between around 210-1240ms.

In addition, there was an Emotion x Subsequent Memory interaction between 360-1240ms (*p* \< .001) with a widespread distribution, indicating that the negative items subsequent memory effect was larger than the neutral items subsequent memory effect.
In addition, there was a borderline significant widespread cluster (*p* = .029) between 1240-1390ms.
Follow-up cluster permutation tests showed that for negative items, remembered items were related to more positive going ERPs between 220-1420ms (*p* \< .001).
This effect was also widespread.
In contrast, there was no significant subsequent memory (all *p*s \> .19).
ERP effect for neutral items.

Overall, the global analysis replicated the focal analysis and shows that these results hold up in more conservative tests with correction for multiple comparisons.
The analysis also shows that the time windows and electrodes analysed in the focal analysis capture the maximum of the LPP effects well in this data set.

![Global ERP analysis: cluster summary](figures/media/image4.png){#fig-4}

![Global ERP analysis: detailed cluster results](figures/media/image5.png){#fig-5}

#### Bootstrap Reliability

We next performed a bootstrap analysis to test if the emotional LPP effect was reliable within subjects.
This analysis showed that the early LPP was reliable in 73.7% of the participants while the late LPP effect was reliable in 50% of the participants.

![Bootstrap reliability of the emotional LPP effect](figures/media/image6.png){#fig-6}

#### LPP and Emotion Enhanced Memory

To be useful for the purpose of model constraint the LPP will need to correlate with memory performance.
We first investigated if the emotionally enhanced LPP amplitude (negative minus neutral items) correlated with emotionally enhanced memory (negative minus neutral items) between subjects.
Surprisingly, the Bayesian Pearson correlation showed moderate evidence for no correlation for both the early time window (*r* = --.018, BF~01~ = 4.925) and the late time window (*r* = --.028, BF~01~ = 4.885).

Next, we investigated the correlation between emotion enhanced LPP amplitude and emotion enhanced memory within subjects.
For each subject, we correlated the emotion enhanced LPP with the emotion enhanced memory across the 9 lists.
We then calculated the average correlation coefficient for each subject and tested it against zero using a one-sample Bayesian *t*-test.
There was moderate support for no correlation in both the early time window (*t*(37) = --.26, *p* = .795, BF~01~ = 7.658) and the late time window (*t*(37) = --1.26, *p* = .215, BF~01~ = 3.698).

Finally, we conducted binomial generalized mixed effects models to test if Emotion and LPP amplitude predicted memory separately for the two time windows.
In the early time window, there was a main effect of Emotion ($\beta$ = .21, $\chi^2$(1) = 22.68, *p* \< .001), a main effect of LPP ($\beta$ = .08, $\chi^2$(1) = 18.87, *p* \< .001) and an Emotion x LPP interaction ($\beta$ = .06, $\chi^2$(1) = 11.20, *p* \< .001).
The interaction was followed-up by running the same analysis separately for negative and neutral items.
For negative items, there was a significant positive relationship between early LPP amplitude and memory ($\beta$ = .14, $\chi^2$(1) = 25.43, *p* \< .001).
In contrast, there was no significant relationship between early LPP amplitude and memory for neutral items ($\beta$ = .01, $\chi^2$(1) = .48, *p* = .488).
In the late time window, there was a main effect of Emotion ($\beta$ = .23, $\chi^2$(1) = 66.58, *p* \< .001), showing once again that memory was higher for negative than neutral items, but importantly there was neither a main effect of LPP amplitude ($\beta$ = .02, $\chi^2$(1) = 1.98, *p* = .160) nor any interaction between Emotion and LPP amplitude ($\beta$ = .02, $\chi^2$(1) = 2.44, *p* = .118).

In sum, the analyses show that the early LPP is more reliable and related to emotion-enhanced memory on a single-trial level.
Based on this, we will use single-trial data from the early LPP time window to constrain the eCMR model.

![Between-subjects LPP--memory correlation overview](figures/media/image7.png){#fig-7}

![Early LPP between-subjects scatter plot](figures/media/image8.png){#fig-8}

![Late LPP between-subjects scatter plot](figures/media/image9.png){#fig-9}

![Within-subjects LPP--memory correlation](figures/media/image10.png){#fig-10}

### eCMR Model Comparison

The Emotion + LPP (interaction) model provides the best fit by all comparison metrics.
@tbl-aic reports pairwise delta AIC values used in the comparisons above.
Parameter estimates were similar across variants, so inference is driven primarily by relative fit and benchmark behavior.

|   | Emotion + LPP (main effects) | Emotion + LPP (interaction) | Emotion-only |
|:-----------------|:----------------:|:----------------:|:----------------:|
| Emotion + LPP (main effects) | NA | 1.12 \[0.13, 2.11\] | -2.86 \[-4.14, -1.57\] |
| Emotion + LPP (interaction) | -1.12 \[-2.11, -0.13\] | NA | -3.98 \[-5.71, -2.25\] |
| Emotion-only | 2.86 \[1.57, 4.14\] | 3.98 \[2.25, 5.71\] | NA |

: Pairwise delta AIC (row minus column; mean \[95% t-based CI\]). {#tbl-aic}

@fig-11 summarizes empirical benchmarks and their simulations for all fitted variants.
In the Emotion-only simulations, the model captures much of the emotional enhancement of memory in the Category-SPC but shows little separation between recalled and unrecalled negative items in Early LPP. This pattern indicates that emotion labels alone can reproduce the overall category advantage without explaining LPP-linked recall differences.
In the Emotion + LPP (main effects) simulations, the model fits both summary statistics but assigns a larger recalled--unrecalled difference for neutral items and a smaller difference for negative items.
This pattern suggests that a single LPP slope captures the overall association while misallocating the emotion-specific separation.
In the Emotion + LPP (interaction) simulations, the Category-SPC preserves emotional enhancement of memory and the Early LPP panel shows a clearer emotion-dependent separation between recalled and unrecalled items.
Together these benchmarks indicate that combining emotion and LPP with an interaction provides the most faithful account of both the category-level advantage and the emotion-specific LPP--recall pattern.

![Benchmark simulations for all model variants compared to empirical data. The top row corresponds to the empirical benchmarks. Proceeding rows correspond to Emotion-only, Emotion + LPP (main effects), and Emotion + LPP (interaction), respectively. Columns show Category-SPC (left) and the LPP-by-recall diagnostic (right).](figures/media/image11.png){#fig-11}

## Discussion

## References