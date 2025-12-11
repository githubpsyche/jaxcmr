In the selective interference effect, a post-film task such as Tetris reduces later intrusive memories while leaving intentional access to the film (for example, recognition) relatively intact. In our last meeting, I presented the context-binding account of post-encoding interference and applied it to interpret findings from the Lau-Zhu et al. (2021) experiment. Here, I briefly recap our key insights from that meeting and then directly explain how they bear on the open questions highlighted by Rik at the end of that discussion. 

# Key Ideas

## The Context-Binding Account of Post-Encoding Interference

In retrieved-context models, each event is encoded by binding it to the current state of a slowly changing context representation. Two kinds of associations are learned: (1) context→item associations that allow a context state to cue items, and (2) item→context associations that allow an item to retrieve its associated context. 

In context→item retrieval, a context state serves as the cue, and items compete for recall based on how strongly they are linked to that state. In item→context retrieval, a probe item is the cue; it retrieves a context state via the item→context associations, and that retrieved context is then used for whatever decision or further retrieval step the task requires.

In standard free recall, retrieval begins from an initial context cue, often the end-of-list context. That context activates items through context→item associations. When an item is recalled, its associated context is retrieved through item→context associations and integrated into the current context state. The updated context is most similar to the states that were active when nearby items were studied. As a result, those nearby items become stronger competitors on the next recall attempt, producing the temporal contiguity effect in free recall.

Post-encoding interference in classic list-learning follows directly from this setup. After a target list has been studied, any post-list events are also encoded, with context states that are adjacent in time to the end of the list. Those post-list items therefore share context features with late list items and with the end-of-list cue that in standard free recall would be used to start free recall. When recall follows an interference task, and is later driven from that cue, it activates both the target items and the post-list items that were bound in nearby context states. Interference arises because the context cue now has several good answers in memory, and post-list items presented during the interference task can win the competition that would otherwise favor the target list. The intensity of such interference depends on how similar the post-list context states are to the target states and how strongly the post-list items are encoded.  
   
\[in typical Tetris experiments there’s a break, then a reminder, then Tetris; a short paragraph on this would be helpful, but the logic remains the same\]

Beyond temporal proximity, items that share pre-experimental contextual associations also tend to drive context toward similar states. For example, events that share meaning or modality features are more likely to be bound to overlapping contexts and to reinstate similar context when they are later retrieved. This means that context-driven recall can be subject to interference that is not purely a function of temporal contiguity: items that were never studied close together in time can still compete at retrieval if they share enough pre-experimental contextual structure.

eCMR and CMR3 build on this by treating emotional features as part of that contextual structure, with CMR3 breaking emotionality down into contextual features for valence and arousal. In CMR3, (1) high-arousal items are encoded more strongly, and (2) items with similar valence or arousal are associated with (and later reinstate) similar context states.  
The first property implies that highly arousing distraction events are stronger sources of interference in context-driven recall. The second implies that this interference will be strongest when the current context is already in a high-arousal region of context space, so that high-arousal distractors and high-arousal targets are cued together and compete within the same emotional subspace.

## Context→Item vs Item→Context Recall

In retrieved-context terms, intrusions measured in the Vigilance–Intrusion Task (VIT) and reported in an intrusion diary can be treated as reflecting an unguided form of context→item retrieval. Ongoing experience can nudge contextual drift, but recall itself happens when the current context state cues trauma-film items strongly enough to cross a response threshold. By contrast, voluntary memory measures in the Lau-Zhu et al. (2021) recognition task are based on item→context associations. Each probe item retrieves its own associated context, and the decision to classify it as ‘old’ or ‘new’ is based on how well that retrieved context matches a specified trauma context.

It follows that post-film interference manipulation can selectively disrupt context→item retrieval without damaging item→context associations. During the post-film task, or after a film reminder is viewed, new events are encoded in context states that lie near the trauma-film. These new traces become additional answers to any context cue in that neighborhood. As everyday context later drifts through this region, it now activates both trauma-film items and post-film items. Because there are more strong competitors, the probability that a given context state cues a trauma item strongly enough to produce an intrusion is reduced. When a film cue is presented as a recognition probe, however, it can still retrieve an appropriate trauma context, leaving recognition performance intact.

I proposed that this distinction between context-driven and item-driven recall specified by retrieved context models easily accounts for selective interference effects as measured in Lau-Zhu et al (2021) and likely many other experiments \[a couple of refs here eg. Holmes 2004 where they used multiple choice in the voluntary test?\]\]. 

## Intentional vs Unintentional Context-Driven Recall

The item- versus context-driven distinction is not sufficient to explain the selective interference effect reported by Lau-Zhu et al. (2021). In several demonstrations of selective interference, the spared measure is intentional free recall of the film, which is itself a context→item task. If all context→item retrieval were equally sensitive to post-film competition, the same interference that reduces intrusions should also impair deliberate free recall.

Retrieved-context models distinguish guided from unguided context-driven recall. Intrusions arise when everyday context drifts near trauma-related states and happens to cue film items above threshold. Intentional free recall instead begins from a context that has been actively steered toward the film session by instructions, cues, or deliberate reconstruction. In model terms, task instructions act as a control signal on the starting context, biasing it toward the beginning (rather than the end) of the trauma-film before the context→item loop begins.

These models also allow participants to constrain intentional context-driven recall by rejecting items whose retrieved context does not match the current goal. In CMR2 and CMR3, when an item wins the retrieval competition, its associated context is retrieved and compared to the current context. If the match is too poor, the item is censored from output and its context is integrated only weakly, limiting drift away from the target context.

When participants are asked to recall trauma film content intentionally in a free recall test,, these two control mechanisms construct and maintain context cues that prioritize film items while suppressing interference from nearby but off-target traces. Taken together, the differential task demands of VRT and diary methods compared to recognition memory tests allow the CMR framework to account for the selective interference effect quite straightforwardly. Matching task demands by using free recall to test intentional memory is more challenging, yet out-of-the-box CMR3 mechanisms offer a ready mechanism for this functional dissociation..

# Addressing Puzzles

The notes above imply answers to each of the puzzles highlighted in Rik’s notes bookending our last meeting.

## Why would Tetris produce more effective interference than a verbal task?

On the context-binding account, what matters is not just semantic overlap with the film, but how many strong competitors are created in the trauma region of context space and how much rehearsal is blocked. Tetris is fast, segmented, and highly engaging, so it generates many strongly encoded post-film traces in the relevant temporal/situational neighborhood and suppresses trauma rehearsal, leading to dense competition with film traces. A verbal task may share more narrative content but, if it induces fewer well-segmented events and allows more mind-wandering and rehearsal, it can end up doing less to crowd the trauma region and so reduce intrusions less.This account means that what matters is not the modality of the interference task but its structure and the emotional impact it has.

## Why can intentional free recall be spared?

The item- vs context-driven distinction in retrieved-context models explains why intrusions can drop while recognition remains intact, but it does not by itself explain why intentional free recall of the film can also be spared, since free recall is still a context→item task. In the retrieved-context framework, the key difference is how context is set and controlled at retrieval. As I laid out above, models like CMR2 specify two control mechanisms—steering the starting context and gating outputs based on context match—that let intentional (but not unintentional) free recall construct and maintain cues that prioritize target items while suppressing interference from nearby but off-target traces. 

## Why might emotional material be more affected than neutral?

If emotional film content is more affected by interference than neutral content, CMR3 accommodates this by treating arousal and valence as explicit dimensions of context. High-arousal events (whether same-valence or not\!) are all encoded more strongly and associated with more similar emotional-context states, so high-arousal post-film events become *especially* strong competitors for high-arousal film items when context is in a high-arousal region. Emotional items can therefore show a disproportionate loss in context-driven accessibility, even when neutral items are also subject to generic temporal interference.This account is predicated on the arousal levels of the trauma event and the  interference task being more closely matched. We tried to  operationalise this in the lab by presenting a trauma-analogue film and a neutral film control, but it is possible that the arousal of the Tetris was more closely matched with the arousal of the neutral film. 

## What Distinguishes Cueing and Control Mechanism Accounts?

The Context→Item vs Item→Context Recall account is directly applicable when the unintentional condition is genuinely context-driven and the intentional condition is genuinely item-driven (for example, intrusions vs recognition). In that case, it predicts selective interference: context→item retrieval (intrusions) is vulnerable to post-film competition in the trauma region, whereas item→context retrieval (recognition) is relatively immune. The same account predicts comparable interference when both tasks sit on the same side of the mapping: two context→item tasks, or two item→context tasks, should be similarly affected if cueing and thresholds are matched.

By contrast, the control mechanism account (steering the starting context and gating outputs based on context match) predicts selective interference even when both tasks are context-driven. Intentional free recall can exploit context control and gating to preserve access to film items despite competition, whereas unguided ‘free recall’ does not. A potential prediction under this account is that intentional free recall should become more intrusion-like (and thus more interference-sensitive) when you add divided attention at test, add time pressure or other stress, or explicitly instruct a “report everything that comes to mind” criterion to weaken gating – externalized free recall.

An account that combines both ideas might predict a graded pattern of immunity: recognition (intentional or unintentional) should be least affected by interference, intentional free recall should show limited but nonzero interference, externalized free recall should show more interference, and unguided ‘free recall’ should be most susceptible. But concreting this prediction would require some creative experimental design.

# Context References

Yonelinas, A. P., Ranganath, C., Ekstrom, A. D., & Wiltgen, B. J. (2019). A contextual binding theory of episodic memory: Systems consolidation reconsidered. Nature Reviews Neuroscience, 20(6), 364-375

Sederberg, P. B., Howard, M. W., & Kahana, M. J. (2008). A context-based theory of recency and contiguity in free recall. Psychological Review, 115(4), 893–912. 

Polyn, S. M., Norman, K. A., & Kahana, M. J. (2009). A context maintenance and retrieval model of organizational processes in free recall. Psychological review, 116(1), 129\.

Lohnas, L. J., Polyn, S. M., & Kahana, M. J. (2015). Expanding the scope of memory search: Modeling intralist and interlist effects in free recall. Psychological review, 122(2), 337\.

Talmi, D., Lohnas, L. J., & Daw, N. D. (2019). A retrieved context model of the emotional modulation of memory. Psychological review, 126(4), 455\.

Cohen, R. T., & Kahana, M. J. (2019). Retrieved-context theory of memory in emotional disorders. bioRxiv, 817486\.  