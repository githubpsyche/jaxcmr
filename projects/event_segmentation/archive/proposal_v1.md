Retrieved-context theory assumes that every study event creates bidirectional links between the active item representation and a drifting internal context state.
Later, presenting an item can reinstate the context in which it was studied, and that reinstated context can in turn cue other items that were encoded in similar states.
In standard CMR-style accounts of event segmentation (e.g., Lohnas, Healey, & Kahana, 2023), an event boundary is handled by changing context itself: temporal context shifts more abruptly, and source context changes to reflect the new event.
That naturally produces higher contextual similarity within events than across boundaries, explaining the usual finding that order memory is better for objects within an event than across an event boundary.

The empirical challenge is that this pattern can reverse at retrieval.
When people have little usable event information at test, they seem to rely on fine-grained local temporal information, and the standard within-event advantage appears.
But when retrieval conditions make event identity or event order more accessible, cross-event judgments can become easier than within-event judgments: a flip effect emerges.
What needs explaining is not just the reversal itself, but why the reversal depends on cue composition at retrieval.

In my mind, three details in existing retrieved-context approaches to event segmentation have to be elaborated to address this finding.

The first is how "events" are represented within the modeling framework.
By representing them just as dimensions of context, we make it awkward to consider how adding them to a probe cue affects recall dynamics.
A better approach might be to represent event identifiers as item features encoded conjunctively with object features so that it is possible to later cue context reinstatement with either an object stimulus, an event stimulus, or both.
Since events correspond to many study events while objects correspond to just one, this change alone makes it easy to think through how having event information available might impact comparison between two same-event probes vs two different-event probes.
(This approach is inspired by a Jin & Kahana in a 2024 project using conjunctive feature representations associated with the same contexts to model associative recognition.)

The second is how temporal order judgments happen. 
Existing approaches either sidestep the mechanism and focus on temporal distance judgments as a proxy, or drive comparison using a distant reference point like current or start-of-list context.
A cleaner, retrieved-context approach might exploit two features of the model to support performance: a) the assymetric forward-going structure of stored associations, and b) the ability to directly retrieve and reinstate an encoded items' associated contexts to search through memory.
These features can be combined into a mechanism that judges temporal order even over long time horizons by checking which item is more accessible when the *other* item's associated context is used as a retrieval cue.
(This approach is inspired by demonstrations by Howard et al, 2025, and Pu et al, 2022).

I'm not sure, but a third fix may be necessary to explain how having event-information in a cue can boost temporal order memory, rather than just interfere with within-event judgments as might otherwise be the default within a retrieved-context model that incorporates the above.
I think that rather than merely having a single temporal context or even dual 'object' and 'event' temporal contexts, we may want a CMR variant where input and memory representations evolve at different time-scales.
At the object level, $F$, $C$, $M^{FC}$ and $M^{CF}$ evolve at each item presentation. 
At the event level, $F$, $C$, $M^{FC}$ and $M^{CF}$ only change when the overall event changes.
This is possibly different from current use of source features and source context because it assumes only one context evolution and learning step across presentations within an "event".
It potentially makes each event a uniquely good cue for successor events within the model's logic, thereby making event information extremely useful compared to object information for resolving temporal order.
(I hedge here though because the boosted temporal order discriminability provided by event information may already be supported in the existing model's logic.)

Combined with our conjunctive feature representation idea, this provides four sets of bidirectional memory associations instead of just the single pair of context-to-item and item-to-context associations I listed earlier. First, there's separate bidirectional context--item associations specific to objects and to events. Bidirectional event-context to event associations help explain how event information in a probe can help resolve temporal order judgments when the events differ. Similarly, bidirectional item-context to item associations provide the ability to resolve temporal order when event information is either unhelpful or unavailable. In addition to these, object-context can support retrieval of events (providing a way for event knowledge to support temporal order judgments even when not present in a probe)and event-context can support retrieval of items (supporting subsequence-focused retrieval or associative recognition). Crucially, there are no direct associations between objects and events; instead, the ability to match objects to events or vice versa is mediated by contextual associations. Similarly, there are no direct retrieval routes between object-context cues and event-contexts or vice versa.

With these details conceptualized, we get a direct route to the flip effect.

When event information is not available, participants steer retrieval with each probe's object-context to assess which is more accessible in memory when starting from the other. Cross-event context similarity slows down and/or disrupts this process, diminishing accuracy.

When a) event information is presented in each probe or can be easily recalled and b) probes belong to distinct events, participants need only judge which event came first. Under the above framework, this is a meaningfully easier task since a sequence of events is necessarily shorter than the sequences of object presentations across the event sequence. Furthermore, depending on how retrieval is set up int he above framework, event cues might reinstate a blend of all associated object-contexts.
This blurring of object-context would make it harder than baseline to resolve within-event temporal order compared to if the event information weren't present at all, adding further pressure to a flip effect.

We can decide on the exact details of the implementation to suit what pattern we want to simulate, but a structure like this is almost definitely sufficient to get us whatever pattern we want and simply demonstrating as much would represent a novel advancement over current state of art. 