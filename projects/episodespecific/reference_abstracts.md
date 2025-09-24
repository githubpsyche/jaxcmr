I'm trying to edit a paper for submission for psych review. 
Here I'm doing this with the starting questions:
How are psych review article abstracts normally structured?
What constraints do they offer for my own candidate abstracts?

Here are some examples I analyzed:

# Examples

## 

> It has been suggested that episodic memory relies on the well-studied machinery of spatial memory. This influential notion faces hurdles that become evident with dynamically changing spatial scenes and an immobile agent. Here I propose a model of episodic memory that can accommodate such episodes via temporal indexing. Indices in the model have flexible duration, capable of exhibiting both fixed duration and broadening time fields akin to classical time cells. The latter cannot index episodes beyond short durations and are reminiscent of timing codes in scalar expectancy theory. Contrary to timing repetitive events, the present model focuses on the one-shot indexing of within-episode structure. Hippocampal indices are recruited by a combination of contextual inputs, lateral inhibition, and drive from temporal analogues of grid cells, functioning as an on-demand sequence generator and memory store. Indices learn connections to cortical representations, modulated by an amygdala signal. This architecture relies on biologically plausible, common network motifs, which can replay dynamically changing and spatially structured events, while an agent is immobile, suggests a mechanism for modulating the speed of recall, and can replay disjoint collections (i.e., broken chains) of indices with preserved temporal order. The model is embedded in an extensive review/perspective along two conceptual axes: first, how the model fits in with other accounts of time coding, serial order memory, and flexible temporal cognition and, second, how we can simultaneously reconcile the model framework with classical accounts of episodic memory à la Tulving, as well as with modern reinforcement learning and generative model accounts of hippocampal function.

1. Pose an influential idea
2. Raise a challenge for it.
3. Propose the theoretical solution.
4. Elaborate, focusing on capacities.
5. Shift to description of structure of the paper / other contributions.

No concluding sentence.

## 
> Grief is a reaction to loss that is observed across human cultures and even in other species. While the particular expressions of grief vary significantly, universal aspects include experiences of emotional pain and frequent remembering of what was lost. Despite its prevalence, and its obvious nature, considering grief from a functional perspective is puzzling: Why do we grieve? Why is it painful? And why is it sometimes prolonged enough to be clinically impairing? Using the framework of reinforcement learning with memory replay, we offer answers to these questions and suggest, counterintuitively, that grief may function to maximize future reward. That is, grieving may help to unlearn old habits so that alternative sources of reward can be found. We additionally perform a set of simulations that identify and explore optimal grieving parameters and use our model to account for empirical phenomena such as individual differences in human grief trajectories

First two sentences describe the phenomenon under study.
Third sentence raises research questions and introduces theoretical framing (functional perspectice).
Fourth and fifth sentence proposes theoretical solution.
Sixth sentence describes other / practical / empirical contributions.

## 

> Manipulation plays a critical role in working memory, wherein understanding how items are represented during manipulation is a fundamental question. Previous studies on manipulation have primarily assumed independent representations by default (independent hypothesis). Here, we propose the ensemble hypothesis to challenge this conventional notion, suggesting that items are represented as ensembles undergoing updating during manipulation. To test these hypotheses, we focused on working memory updating in accordance with new information by conducting three delayed-estimation tasks under addition, removal, and replacement scenarios (Study 1). A critical manipulation involved systematically manipulating the mean orientation of all memory stimuli, either increasing (clockwise) or decreasing (counterclockwise) after the updating process. Following the independent hypothesis, memory errors would be similar under both conditions. Conversely, considering the biasing effect of the ensemble on individual representations, the ensemble hypothesis predicts that memories of individual items would be updated, aligning with the ensemble’s change direction. Namely, memory errors would be more positive in the increase-mean condition compared to the decrease-mean condition. Our results supported the ensemble hypothesis. Furthermore, to investigate the mechanisms underlying ensemble computations in updating scenarios, we conducted three ensemble tasks (Study 2) with similar designs to Study 1 and developed a computational model to quantify the contributions of each memory item. The results consistently demonstrated that addition involved complete updating, while removal led to incomplete updating. Across these three research parts, we propose that items are represented as dynamic ensembles during working memory updating processes. Furthermore, we elucidate the computational principles underlying ensembles throughout this process.

First sentence describes the phenomenon under study. 
Second sentence identifies an influential idea about the phenomenon.
Third sentence both raises a challenge for it and proposes an alternative view.
Fourth and fifth and sixth and seventh and eighth and ninth sentence motivate and describe a test of the views(s).
Tenth sentence describes more empirical work and development of a computational modeling.
Eleventh and twelfth summarize results and theoretical contribution.

## 

> Episodic memory is a core function that allows us to remember the events of our lives. Given that many events in our life contain overlapping elements (e.g., similar people and places), it is critical to understand how well we can remember the specific events of our lives versus how susceptible we are to interference between similar memories. Several prominent theories converged on the notion that pattern separation in the hippocampus causes it to play a greater role in processes such as recollection, associative memory, and memory for specific details, while overlapping distributed representations in the neocortex cause it to play a stronger role in domain-specific memory. We propose that studying memory performance on tasks with targets and similar lures provides a critical test bed for comparing the extent to which human memory is driven by pattern separation (e.g., hippocampus) versus more overlapping representations (e.g., neocortex). We generated predictions from several computational models and tested these predictions in a large sample of human participants. We found a linear relationship between memory performance and target–lure pattern similarity within a neural network simulation of inferior temporal cortex, an object-processing region. We also observed strong effects of test format on performance and consistent relationships between test formats. Altogether, our results were better accounted for by distributed memory models at the more linear end of a representational continuum than pattern-separated representations; therefore, our results provide important insight into prominent memory theories by suggesting that recognition memory performance is primarily driven by overlapping representations (e.g., neocortex).

This is pretty similar topic to my current paper.

First sentence introduces topic.
Second sentence converges on a research question within this topic.
Third sentence describes an influential idea related to the research question.
Fourth sentence introduces theoretical proposal.
Following sentences describe work to validate the proposal.
Concluding sentences summarize reuslts and theoretical contribution.

# Reflection

Each abstract begins by giving a sense of the research topic and present thought about it.
Then a shift to describing the paper's theoretical contribution that advances present thought.
Then a shift to describing concrete contributions (models, experiments, reviews, ideas).
Optionally, final sentences summarize the high-level contribution.

Simple enough.
Does current abstract achieve this?

## repfr/abstract_grist3

In repfr/abstract_grist3, the version is:

> Repetition usually improves recall, but each occurrence of an item is embedded in its own episodic context. Retrieved-context theory (RCT) predicts that repetitions knit these contexts together: when an item is encoded or retrieved, it reinstates features from prior occurrences, creating blended cues (Howard & Kahana, 2002; Polyn et al., 2009; Lohnas & Kahana, 2014). Such blending links occurrences across time, but also predicts associative interference across a repeated item’s neighborhoods where one occurrence’s neighbors intrude into another's recall. To test these predictions, we investigated how the temporal contiguity effect -- the tendency to successively recall items that were studied near each other in time -- manifests when items have multiple occurrences. We re-examined six free- and serial-recall datasets with matched control baselines. Three robust patterns emerged that contradict RCT’s interference predictions: (i) no elevated transitions between neighborhoods of different occurrences, (ii) a consistent bias toward the first occurrence’s neighbors rather than balanced access, and (iii) preserved forward chaining in serial recall without cross-occurrence errors. Together these results indicate that repeated items remain separable in memory, rather than linked by associative blending. Episodic memory appears to balance integration and specificity not by fusing occurrence contexts, but by letting distinct traces compete at retrieval.

I think I may have written this for submission to empirically focused venues. My other variants have a similar structure, though I go into more detail about the theoretical proposal elsewhere.

I suppose the problem is that I jump from describing present thought to my evaluation of present though, with the theoretical contribution at the end and describe vaguely.

I'm also dissatisfied with its explanation for how we balance integration with specificity. "by letting distinct traces compete at retrieval" is a good summary of what I found, but not a good self-contained explanation for how episodic memory balances integration with specificity. 

What *would* be my model's explanation? My model more or less rejects the idea that there's any evidence of such integration at encoding or retrieval in these tasks (free or serial recall). Repetition boosts memory for an item simply by adding another episode containing the item to the set of episodes competing for retrieval. The spacing effect emerges when episode traces contain distinct contextual states, broading the range of cues that can prime for retrieval an episode containing the item. There may be other repetition effects that my model can't account for, but they don't seem to arise by knitting together the contextual details of the episodes containing the item.

I suppose it's possible for an item-based cue to retrieve contextual details across items within my model. But it's not something I evaluate just yet. I primarily reject that item-based cues mediate any part of episodic memory search (especially the reinstatement of context following a retrieval) in a way manifested as repetition effects in free or serial recall.

So there's a sense in which my work does not actually address how episodic memory balances integration with specificity. I can either add simulations that make the work achieve this, or I can shift my language a bit.

## Reworking order of presentation

We mainly want to bring the theoretical proposal forward before description of empirical tests.
Here's one way to do it:

Repetition usually improves recall, but each occurrence of an item is embedded in its own episodic context.
Prevailing retrieved‑context accounts propose that repetitions blend the contexts of an item's occurrences, knitting together their neighbhorhoods and inviting cross-occurrence 

Retrieved-context theory (RCT) predicts that repetitions knit these contexts together: when an item is encoded or retrieved, it reinstates features from prior occurrences, creating blended cues.
Such blending links occurrences across time, but also predicts associative interference across a repeated item's neighborhoods where one occurrence’s neighbors intrude into another's recall.
We show that the data point the other way and advance an episodic-specific account in which occurrence contexts remain distinct and separate


 account in which occurrences remain separate and are selected at retrieval by the currently reinstated temporal context.

## Clarifying the addressed research problem

We aren't necessasrily addressing how episodic memory balances integration with specificity.
There's an implied answer by our model though: specificity is achieved by giving experiences containing the same items distinct/separable context representations. Individual episodes can be primed and retrieved using contextual cues, enabling the reinstatement of episodes that match a given episodic cue. On the other hand, an item-based cue can target all episodes containing the item, retrieving information