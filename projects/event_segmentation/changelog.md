# Proposal changelog

## Active constraints

This section is forward-looking.
It is the authoritative constraint ledger for `proposal.md` and this changelog.
Do not maintain a separate `constraints.md`.
If a stale `constraints.md` appears in the editor or working tree, treat it as obsolete unless the user explicitly changes policy.
These are live drafting rules, not retrospective claims about what the manuscript has already achieved.

1. Use one sentence per line when editing prose.
2. Do not claim that standard event-segmentation CMR already explains temporal-order judgments.
3. Standard event-segmentation CMR explains how event boundaries reshape contextual structure.
4. Do not infer easier temporal ordering directly from higher within-event contextual similarity.
5. Temporal-order performance requires an explicit readout or decision mechanism.
6. The opening must separate contextual structure from the additional readout needed for temporal-order judgment.
7. The flip must be anchored to concrete retrieval conditions rather than vague "some conditions" language.
8. Do not use shorthand labels like "event-as-context" before introducing the standard modeling move in plain language.
9. Do not use shorthand labels like "the Jin and Kahana move" before identifying the source problem and the representational tactic being borrowed.
10. Do not attribute to cited work a stronger claim than the work actually provides.
11. Treat conjunctive cueing as a representational bridge that makes probe composition modelable.
12. Do not present conjunctive cueing as a complete account of the flip effect.
13. Treat the hierarchical multiscale account as the main proposal.
14. Present the hierarchical multiscale account as the leading account, but not as fully settled in every detail.
15. The proposal must explicitly explain how the model produces the flip effect.
16. The main lever in that explanation must be a cross-event facilitation route via event-level order information, not only within-event impairment.
17. Inline drafting comments should not remain in the final approved manuscript.
18. Before future edits, identify which constraints are relevant to the lines being changed.
19. After future edits, report the exact lines changed and which constraints the edit was meant to satisfy.
20. Keep the memo as continuous exposition unless the user explicitly asks for internal subsections.
21. Do not repeat the immediately preceding paragraph's main claim unless the next paragraph depends on a more specific distinction.
22. A paragraph may briefly recall prior material, but should not re-argue it without a new inferential payoff.
23. Paragraphs should end on their main inferential claim rather than on a vague restatement.
24. Do not remove inline user comments from `proposal.md` without explicit user approval.
25. Retrospective entries must not claim that a disputed front-half issue is fully resolved while the current manuscript still contains inline comments challenging it.

## Retrospective entries

Original draft archived at `projects/event_segmentation/archive/proposal_v1.md`.
Current version at `projects/event_segmentation/proposal.md`.

Entries 1-8 cover the full memo.
The revision keeps the same core problem from v1, namely how to explain the flip effect in a retrieved-context framework, but it changes the argumentative center of gravity.
The new draft expands the background, sharpens the critique of standard event-as-context modeling, demotes conjunctive cueing from leading solution to representational bridge, introduces a hierarchical multiscale architecture as the main proposal, and adds a dedicated section explaining exactly how the model produces the flip.
All entries are ordered by location in the original draft (v1).

### 1 (v1 lines 1-3): background and flip-effect framing were too compressed and required further cohesion repair

The v1 opening already had the right raw ingredients: bidirectional item-context associations, standard CMR-style boundary drift, the classic within-event advantage, and the observation that the pattern can reverse at retrieval.
The problem was compression.
Background, empirical pattern, and modeling demand were collapsed into two dense paragraphs, which made the argument feel more like a sketch than a memo.
The reader had to infer why the flip was theoretically difficult rather than being walked through it.
The current draft expands this material into a fuller opening that now distinguishes contextual structure from temporal-order readout.
It says explicitly that standard retrieved-context theory explains how boundaries reshape contextual structure, but does not yet provide a rule for turning that structure into a temporal-order judgment.
The opening has now been revised again so the flip is the reason that missing readout matters, rather than a second topic introduced after the opening setup.
The flip is now anchored to concrete retrieval conditions, namely the re-presentation of encoding context at test or successful recovery of event identity or event order.
This corrects the earlier theoretical overclaim and moves the opening closer to a cohesive transition, although the inline comments show that this repair is still under review rather than treated as permanently settled.

### 2 (v1 lines 5-6): the critique of standard event representation was shortened, clarified, and made less repetitive

V1 already argued that treating event details as context is awkward, but the prose was informal and the criticism remained partly gestural.
The phrase "a _context_ feature in model" was also ungrammatical, and the paragraph did not quite isolate the real issue.
The previous revision sharpened the criticism, but it still repeated points the opening had already made and relied too much on shorthand before clearly stating the standard modeling move.
The current draft now recalls that move in one sentence, deletes redundant material about what it is useful for, and turns quickly to the actual mismatch.
That mismatch is now stated more sharply: if event identity lives only in latent context, then an object-only probe and an object-plus-event probe are not genuinely different at cue onset.
The paragraph now ends on the further point that these paradigms manipulate the probe itself, not only the information later recovered from it.
This makes the critique more accessible, less repetitive, and more directly tied to the experimental contrast.

### 3 (v1 line 8): the Jin and Kahana bridge was narrowed, compressed, and made more faithful to the cited work

The Jin and Kahana paragraph in v1 was conceptually useful but too loose in execution.
It contained drafting errors, and it did not clearly separate what their work contributes from what it does not.
The previous revision improved the prose but still overreached by invoking "the Jin and Kahana move" before identifying the problem their paper addressed.
It also jumped too quickly to object-plus-event feature coding as if that were the paper's own claim.
The current draft now compresses the bridge into a cleaner sequence.
It first states the problem Jin and Kahana addressed, namely memory with jointly available information.
It then identifies the borrowed lesson as a representational tactic, namely conjunctive coding on the feature side.
It next states our extension separately, namely that the tactic can make object-only and object-plus-event probes genuinely different inputs in a temporal-order model.
It ends by stating the limit of that borrowing explicitly: this solves the probe-representation problem, but not yet the problem of why cross-event judgments become easier.
This makes the bridge easier to follow, more compact, and more faithful to the cited work.

### 4 (v1 lines 10-13): the object-only versus object-plus-event distinction was retained but reduced to the probe-composition problem

V1 made the distinction between two probe types explicit, which was valuable.
It stated that an object-only cue should mainly reinstate the context of that object's study event, while an object-plus-event cue should retrieve both object-specific and shared event-level information.
But in v1 this distinction sat inside a longer argument that still treated the conjunctive-cue account as though it might be the main explanation.
The previous revision demoted that distinction but still left it as a semi-independent section, which made the bridge feel vestigial.
The current draft keeps the distinction only in its leanest functional form.
One probe type merely cues later recovery of event information, whereas the other explicitly contains event information at test.
That is enough to state the probe-composition problem.
It is no longer allowed to sit on the page as a quasi-separate explanation of the flip.
This keeps the bridge while preventing it from competing structurally or conceptually with the hierarchical account.

### 5 (v1 lines 15-16): the provisional retrieval mechanism was removed as the main story

V1 proposed a retrieval rule based on forward asymmetry at the object level.
Participants were supposed to judge order by asking which of two objects was the better predecessor of the other when one object's associated context was used as a cue.
That idea was useful for preserving a CMR-native directional notion of order, but it was still too weak and underspecified to carry the full flip.
In particular, it did not explain why event-associated retrieval should actively improve cross-event judgments rather than simply making same-event judgments noisier.
The new draft therefore removes this object-level mechanism from the center of the theory.
It is retained only as the fallback computation used when retrieval cannot be solved at the event level.
The memo now states that object-level temporal information remains important, but only after the system has failed to obtain a clean event-level separation.
This is a substantive theoretical change.
V1 treated object-level predecessor structure as the main retrieval story.
The new draft treats it as the second-stage computation in a hierarchical system.

### 6 (v1 lines 17-22): the incomplete flip-effect explanation was replaced by a dedicated section

V1 reached the crucial moment in the argument and then stopped.
It explained why same-event probes might become more confusable under conjunctive cueing, and it began to claim that different-event probes would be further differentiated, but the paragraph ended mid-sentence.
Even before the truncation, the mechanism was still one-sided because it mainly described within-event impairment and did not yet provide a clear cross-event facilitation route.
The revised memo adds a dedicated section called "How this model can actually produce the flip."
That section walks through three cases in prose.
First, object-only or weak-event retrieval yields the classic within-event advantage because judgments depend mainly on object-scale temporal continuity.
Second, same-event pairs remain hard even under strong event retrieval because event information does not distinguish the probes.
Third, cross-event pairs become easier when probes localize to different event-level states, because the system can answer from event order rather than fine item order.
This is the single most important addition to the manuscript.
It makes explicit that the main lever behind the flip is a new cross-event facilitation route, not merely degraded within-event resolution.

### 7 (v1 lines 10-22): the main proposal shifted from conjunctive cueing to a hierarchical multiscale architecture

Although v1 gestured toward joint item-event cueing, it did not yet define the stronger architecture needed to make the flip plausible.
There was no event-level system with its own state dynamics, no explicit explanation of how event order could be represented, and no account of how boundary disruption at the item level should arise from higher-level event change.
The new draft adds a full section called "The stronger proposal."
It introduces two coupled scales: a fast object scale that updates on every object and supports fine within-event order, and a slower event scale that changes mainly when events shift and supports coarse event order.
It also introduces structured coupling, so changes in event state perturb object context at boundaries.
This gives the memo a new center of gravity.
The main proposal is no longer that object-plus-event cueing blurs retrieved states.
It is that a coupled object/event architecture gives the model its own event-level reference points and therefore a real way to compute event order.

### 8 (v1 lines 10-22): boundary disruption, retrieval policy, and scope are now explicit

Several important commitments were implicit or absent in v1.
Boundary disruption was still effectively inherited from the standard "extra drift" story.
The decision rule for when event information should or should not be used was not stated.
The memo also did not say what was in scope for a first computational implementation versus what could wait.
The new draft addresses all three.
First, it states that boundary disruption at the object scale should be generated by change in the event scale, not by arbitrary noise or hand-added drift.
Second, it adds a dedicated closing section on "Retrieval policy and scope," where the model first tries to separate the probes at the event level, answers from event order if successful, and falls back to object-level temporal information otherwise.
Third, the closing section narrows the first implementation target by saying that full object-to-event and event-to-object retrieval is not yet required.
These additions make the proposal more decision-ready.
They turn v1 from a promising conceptual direction into a memo that specifies what the first model should actually try to build and what can remain future work.
