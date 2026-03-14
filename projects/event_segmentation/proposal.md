# Proposal
<!-- FIXME: no more subsections; enforce flow and structure by just writing the exposition better -->
<!-- FIXME: whole document is obviously too long and rambling in its -->

Retrieved-context theory gives a natural account of how event boundaries change the contextual structure of a studied episode.
Studied objects become bound to a continuously evolving context state, and later retrieval of an object can reinstate that context and cue other material encoded nearby in time or under similar source conditions.
In standard CMR-style accounts of event segmentation a boundary is implemented by changing context itself: temporal context drifts more sharply, and source context shifts to reflect the new event.
Standard event-segmentation CMR therefore explains contextual structure, not yet the decision rule that would turn that structure into a temporal-order judgment.
<!-- FIXME: there's a cohesion issue between first and second paragraph. Your last sentence centers on "a clear rule for converting that contextual structure into a temporal-order judgment" but then we center on the flip effect. I feel like current language instead foreshadows clarification of a retrieval mechanism that works in CMR. -->

The flip effect is useful precisely because it exposes that missing readout.
<!-- FIXME: this is a bullshit sentence. the flip effect does not make that missing readout "impossible" to ignore. billions and billions of people have gone their whole lives without a single thought about that readout. -->
When encoding context is re-presented at test, or when participants can recover the identities and order of the relevant events, cross-event judgments can become easier than within-event judgments.
So the challenge is not simply to explain a reversal in the sign of the boundary effect.
It is to explain why a change in retrieval conditions can change which level of temporal structure the judgment is based on.
That is why cue composition becomes a modeling problem rather than just a detail of retrieval dynamics.

The standard retrieved-context way of representing events is to let event information live in a source-context dimension that changes when the event changes.
<!-- FIXME: this approach is too repetitive, pretends we didn't already explain the retrieved-context way of representing events. We can't assumes too much familiarity with the event-as-context framing from P1, but still need language here framing that feature for the reader to process the incoming text. -->
But that standard move becomes awkward in temporal-order paradigms that show the flip effect.
<!-- FIXME: we _definitely_ already made this point. -->
<!-- FIXME: what is a "flip effect task"? i don't think it's appropriate to speak of a flip effect task. -->
If event identity exists only as part of latent context, then an object-only probe and an object-plus-event probe are not genuinely different at cue onset.
<!-- FIXME: everything from the sentence below to the end of the paragraph is meandering and unclear. -->
One probe type merely cues later recovery of event information, whereas the other explicitly contains event information at test.
That is a problem because these paradigms manipulate the probe itself, not only the information the probe may later recover.
<!-- FIXME: this final sentence seems vague compared to language in v1 -->

Jin and Kahana addressed a different problem: how to model memory when multiple pieces of information are available together rather than encountered one at a time.
<!-- FIXME: instead of ", namely how", just describe the problem-->
The useful lesson from that work is representational rather than task-specific.
Jointly available information can be coded conjunctively on the feature side, so the probe itself can carry a combined cue rather than relying on latent context to recover everything sequentially.
Borrowed here, that tactic would let an object-only probe and an object-plus-event probe become genuinely different retrieval inputs.
That fixes the probe-representation problem that the standard source-context treatment handles awkwardly, but it still does not explain why cross-event judgments should become easier.
<!-- FIXME: logic here is unclear -->

<!-- FIXME: this whole section feels almost vestigial now. i can't tell if it's because you wrote it poorly or if that's how it is-->

The stronger proposal is therefore a hierarchical multiscale retrieved-context model that supplies the missing cross-event route.
There is a fast object scale that updates on every studied object and supports fine within-event order information, and there is a slower event scale that changes mainly when events shift and supports coarse event-order information.
These are not two independent systems placed side by side.
They are coupled parts of the same architecture.

On this view, an event shift changes the event-scale state, and that higher-level change perturbs the object-scale state.
This gives a cleaner way to think about boundary disruption.
Object-level temporal context is not disrupted because arbitrary extra drift was injected at the boundary.
It is disrupted because the higher-level event representation changed, and that change propagates downward into the object-scale context.
The model therefore acquires its own event-level reference points while still preserving object-level temporal continuity within events.

Once the model has both scales, the flip becomes much easier to state.
When retrieval is driven mainly by the object scale, either because the probe is object-only or because event retrieval is weak, the model has to judge order from fine object-level temporal information.
In that regime, the object-level computation should work best for within-event pairs, because local temporal continuity is strongest there and boundary-induced disruption is weakest there.

The crucial change comes when event information is strongly available at test.
For same-event pairs, event retrieval does not solve the problem because both probes localize to the same event-level state.
The judgment still has to be made at the harder object scale.
Shared event information may make same-event probes slightly less distinct, but that is not the main source of the flip.

For cross-event pairs, the situation is different in kind rather than degree.
If the two probes localize to different event-level states, the model can answer from event order rather than fine item order.
That is a simpler and more reliable computation: it asks which event came first, not which of two nearby object states was the better predecessor of the other.
The main lever behind the flip is therefore a new cross-event facilitation route.
The hierarchical model matters because it gives the system a real event-order computation that can override the classic object-level pattern when event information is available.

This implies an adaptive retrieval policy.
The model should first try to localize each probe at the event scale.
If the probes are cleanly separated into different event states, it can answer from event order and stop there.
If not, because both probes map to the same event or event retrieval is weak, it should fall back on object-level temporal information.
At that point order is judged from the directional structure of object-level temporal associations rather than from raw context similarity alone.

This also keeps the account psychologically reasonable.
Event information is not treated as a nuisance that always contaminates item judgments, and fine item-level reconstruction is not run when coarse event-level information already solves the problem.
The system uses whichever level of retrieved structure is informative enough for the task.

The first implementation does not need full object-to-event and event-to-object retrieval in both directions.
It is enough for explicit event information at test to engage the event scale directly, while event shifts during study perturb the object scale through coupling.
Stronger cross-scale retrieval may matter later if the aim expands to item-only recovery of event identity or richer source-memory behavior, but it is not required for the first temporal-order model.
The next modeling step is therefore to define coupled object- and event-scale updates, specify how event-state change perturbs object context at boundaries, and implement an event-first retrieval gate that falls back to object-level order computation only when event-level separation is insufficient.
