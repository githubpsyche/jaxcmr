## Science Points

### Feature-Based Clustering May Be Explained Without a Context Signal

Morton & Polyn (2016) evaluated different retrieved-context models of semantic organization in free recall.
Their results did not recapitulate the proposal in Polyn et al. (2009) that a multi-layered context representation that includes both temporal and semantic features is necessary to explain semantic clustering.
Instead, the best performing model use a purely temporal context representation alongside a separate semantic similarity matrix that directly linked items to each other based on their semantic similarity.
This separate semantic similarity matrix was probed by the last item retrieved instead of any context state, and its activations combined with context-based activations to determine the next item recalled.
Otherwise using a context-mediated process to access semantic associations produces errors accounting for recall initiation, consequences for items studied near items that are semantically similar to the last item recalled, as well as the overall magnitude of semantic clustering.

These results are relevant here because they indicate we should contrast phenomena that can be *explained* by a semantic contextual layer (e.g., emotional clustering during recall) with phenomena that *indicate* the use a semantic contextual layer when evaluating retrieved-context accounts of emotional memory.

To do this, we need to look at, for example, how encoding two items with the same emotional features but apart study positions affects associations between neighbors of those items. If a drifting semantic context mediates emotional clustering, then we should see that items near two emotional items are more likely to be recalled together than items near two neutral items, since the neighbors of those emotional items will be associated with a similar emotional context state. 

The key thing here is that RCT predicts different consequences for *encoding* that a merely similarity-based account does not.

On the other hand, there are retrieval-specific commitments in RCT that may be relevant to emotional memory. For example, if successively recalling two items with the same emotional features boosts the likelihood of recalling other items with those features compared to recalling two items with different emotional features, that would indicate a role for a drifting emotional contextual cue during retrieval that maintains emotional features across successive recalls.

### Watching the Difference Between Semantic Similarity and Emotional Similarity

We want to know when and how shared emotional features can explain memory phenomena above and beyond shared semantic features. But many experiments probing emotional memory manipulate emotional content and semantic similarity in confounded ways. For example, if a list-learning experiment works by primarily manipulating whether studied words are emotional or neutral, then it is likely that the emotional words are also more semantically related to each other than the neutral words are to each other. Concomitant neurobiological changes could differentiate whether semantic or emotional similarity is applicable to the memory phenomena of interest, but ideally we could produce behavioral evidence that teases apart the two.

The attention variant of eCMR is a good roadmap for how to specify these unique contributions, because the consequences of two items being associated with overlapping context states are different from the consequences of them each simply having stronger item-context associations. But either way, we want to keep an eye out for ways to clarify the unique contributions of semantic and emotional similarity to memory phenomena.

It looks like this is explicitly addressed in Talmi, Lohnas & Daw 2019, and in other work, which is great. But is still possible that semantic organizational effects could explain apparent variability in phenomena like the emotional list-composition effect, or that embeddings or other techniques for capturing semantic similarity are inadequate for capturing the relevant semantic similarity structure, especially if list composition influences the saliency of different semantic dimensions. Either way though, the theory is simply that emotional features are another dimension of similarity that can be used to organize memory during free recall. I need to see patterns that go beyond this to believe that a retrieved-context account is necessary.

### Does Comparison Between Mixed and Pure Lists Need Position-Matching Controls

In my thesis, position-matching was needed for comparison to pure lists because repeated items have multiple study positions. If recall of *either* control item at the study positions of a repeated item are not counted as a correct recall of both study positions, comparisons will identify repetition effects simply by virtue of the repeated item having more positions to be recalled from, even if repeated items are not processed or recalled differently from singly-presented items.

The more fundamental idea is that an analysis needs to be configured in a way that confirms the null hypothesis when the null hypothesis is true. If when a manipulation has no effect, the analysis still shows an effect, then the analysis is not valid.

Tying back to eCMR, the null hypothesis is simply that emotional items are processed and recalled the same as neutral items. If the study positions of emotional items are randomly distributed across trials and there are many trials, it may not be necessary to position-match when comparing mixed and pure lists, because the chance of emotional items being recalled from their study positions should be the same as the chance of neutral items being recalled from their study positions. But if there are few trials, or if emotional items are not randomly distributed across trials (e.g., if emotional items are always in the first half of a list), then position-matching may still be necessary to avoid confounding serial position effects with emotional effects. Even then, position-matching should be much more straightforward than in the case of repeated items, since each item only has one study position.

### Formal model evaluation valuable

We can fit models directly to data rather than using pre-configured parameter sets to evaluate model performance. This is more work, but provides a more complete sense of how well the model can account for performance, while also providing a more direct sense of which parameters are necessary to account for the data. It's also useful for comparing models to each other, since we can use model comparison metrics that account for model complexity.

### Associative Interference Between Emotional or Neutral Items

Some literature, including some cited by Cohen/Kahana identifies an explored prediction of eCMR that -- due to reinstating overlapping contextual features -- items with the same emotional content should produce associative interference with each other, leading to overgeneral memory that struggles to access specific details of an event. My thesis analysis and framework provide a great starting point for exploring this prediction if you want to go that route.

This all ties back to the question of whether eCMR's or CMR3's commitments to a full-own emotional context layer are worth embracing or not. 