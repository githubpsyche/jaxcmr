# Abstract Revision — S5 and S6

## Current abstract (as in index.qmd)

> (1) Repetition strengthens memories while tying each occurrence to an evolving temporal context.
> (2) The core idea of retrieved‑context theory (RCT) is that this evolution is item‑based: encoding or retrieval "calls back" and blends items' prior contexts into the ongoing state.
> (3) Such blending links occurrences across time, but also engenders associative interference by overlapping traces and diffusing retrieval across occurrence contexts.
> (4) Comparing against symmetrically-scored control baselines, we identify patterns consistent across free‑ and serial‑recall datasets contradicting this account: (i) no elevated transitions between neighborhoods of different occurrences, (ii) biased transitions from repeated items toward first-occurrence neighbors, and (iii) preserved forward chaining in serial recall without cross-occurrence errors.
> (5) In formal comparisons, retrieved-context models account for these patterns and improve overall fits only when repetitions (a) reinstate unique, non-overlapping contextual features and (b) produce traces that compete separately for later retrieval.
> (6) These results argue that episodic memory for repeated experience balances integration and specificity not through contextual blending, but differentiated competition.

S1–S4 are settled. S5 and S6 are the targets. S5 reports the modeling result; S6 interprets it. The two sentences need to land the same message: item-based reinstatement — the thing that makes RCT RCT — is the problem, and removing it is the fix.

## Constraints

### Focal problems

#### S5: "retrieved-context models" may be misleading

S5 says "retrieved-context models account for these patterns... only when." But RCT's signature departure from predecessors (TCM, other drifting-context models) is item-based context reinstatement — exactly what Instance-CMR removes. Instance-CMR keeps drifting temporal context, item-context associations, and context-based retrieval, but these are pre-RCT features. Under the distinction emphasized in the slides and in S2 of the abstract itself, Instance-CMR may not qualify as a retrieved-context model. Calling it one softens the challenge: the reader hears "RCT works when you tweak it" rather than "RCT's signature mechanism is the problem."

#### S5: "account for these patterns" overstates

Instance-CMR is the null hypothesis about repetition — no repetition-specific machinery. It eliminates the false predictions of cross-occurrence interference (S4 patterns i and iii) and improves overall fits, but cannot capture the positive first-presentation bias (S4 pattern ii). "Account for these patterns" claims it handles all three. "Only when" implies (a)/(b) are sufficient; they're necessary but not sufficient for everything.

#### S6: "balances integration and specificity" overreaches

The work shows item-based reinstatement is counterproductive during sequential memory search (free and serial recall). It does not address how episodic memory achieves integration across occurrences — the framework still allows item-based cueing to retrieve contexts associated with a repeated item, but the work doesn't test that route. "Differentiated competition" names the architecture but only explains the specificity side (independent traces preserve episode-specific information); it doesn't explain integration. The result is more accurately a void where RCT's blending answer used to be than a complete alternative account of integration/specificity.

### S5

1. **Null-hypothesis framing.** Instance-CMR is the null hypothesis about repetition within RCT — not a proposed successor.
2. **Convey the mechanism difference.** How context evolution changes: independently of item identity, not through item-based reinstatement.
3. **Signal formal modeling.** Must read as modeling work for Psych Review.
4. **No content overlap with S6.** S5 reports the modeling result; S6 interprets.
5. **No ICMR-Reinf.** Muddles the core story.
6. **One sentence, one job.** No overextension.
7. **No redundant summative claims.** S4 has the empirical verdict; S6 has the theoretical one.
8. **First use of "model" grounded by context.** "Formal comparisons" + "retrieved-context models" handles this.
9. **Both mechanisms required.** (a) independent context evolution + (b) competing traces.
10. **Don't soften the challenge.** RCT's signature departure from predecessors is item-based reinstatement — exactly what Instance-CMR removes. The abstract must name the removal explicitly so the reader hears "RCT's signature mechanism is the problem," not "RCT works when tweaked." But the naming can live in S6 (interpretation) rather than S5 (result), since (a)/(b) already implies the contrast with S2's "item-based" framing.
11. **Don't label the winning model as RCT.** Instance-CMR removes RCT's defining feature. Calling it a "retrieved-context model" contradicts the challenge. Describe what the winning model does, not what family it belongs to.
12. **Active voice.** Passive constructions ("are accounted for") weaken the sentence.
13. **No predicate redundancy.** Don't name the removal AND describe the replacement in the same predicate chain (e.g., "removing item-based reinstatement... instead (a) reinstating..."). If the removal is named, put it in the subject (identifying *which* models) rather than the predicate (competing with (a)/(b) for the same job).
14. **Don't overstate what the null model captures.** Instance-CMR eliminates false interference predictions and improves overall fits, but doesn't capture the first-presentation bias. Don't say "account for these patterns" — it claims all three. Frame as eliminating interference / improving fits, not as explaining the full data pattern.

### S6

1. **Scope must match the work — and "contradict" is defensible.** The work shows item-based reinstatement is counterproductive during sequential recall. It doesn't address how episodic memory achieves integration across occurrences — no integration/specificity claim. However, "contradict retrieved-context theory" is NOT overstating: RCT's distinguishing commitment over predecessors (TCM, other drifting-context models) is item-based reinstatement. Non-repetition successes (contiguity, recency, asymmetry) are equally explained by predecessors. Repetition is the *only* paradigm where item-based reinstatement makes predictions the predecessors don't — so contradicting its repetition predictions contradicts the thing that makes RCT a distinct theory.
2. **Close the abstract's actual arc.** S1–S3 set up blending → S4 data contradicts → S5 models confirm → S6 delivers theoretical interpretation. Don't pivot to a different question.
3. **"Differentiated competition" worth keeping** if scoped correctly — names the architecture that replaces blending.
4. **Must work as a Psych Review closing sentence.** Memorable, lands the takeaway.

## Proposed draft

> (5) In formal comparisons, the predicted interference vanishes and fits improve under a null model where context evolves independently of item identity and repetitions produce distinct traces that compete separately for retrieval.
> (6) These results challenge retrieved-context theory, suggesting that memory search for repeated experience is organized not by item-based context reinstatement, but instead differentiated competition.

**Division of labor:** S5 reports the modeling result — interference gone, fits better, null model, both mechanisms. S6 delivers the verdict — challenges the theory, names the mechanism rejected, names the replacement.

**S5:** "context evolves independently of item identity" directly negates S2's "this evolution is item-based" (C2). "A null model" frames Instance-CMR correctly (C1, C8). "Distinct traces that compete separately for retrieval" captures the second mechanism (C9). All active verbs: vanishes, improve, evolves, produce (C12). Doesn't label winner as RCT (C11). "Interference vanishes and fits improve" accurate without overstating (C14). "In formal comparisons" signals modeling (C3).

**S6:** "challenge retrieved-context theory" names the theory (C10). "Suggesting" is subordinate to "challenge" — single logical move, not two jobs (S6-C4). "Not by item-based context reinstatement, but instead differentiated competition" names both the rejected mechanism and the replacement (S6-C2, S6-C3), echoing S2 directly. "Memory search for repeated experience" scopes to what the work tested (S6-C1).

## Rejected drafts

**S5:**

| Draft | Problem |
|-------|---------|
| "Item-based reinstatement proves counterproductive: a model that treats each occurrence as an independent study event, with traces competing without blending, outperforms the standard account; preferential strengthening of first occurrences captures residual asymmetries." | Too complex (colon + semicolon). Redundant. Mentions ICMR-Reinf. |
| "A retrieved-context model that treats each occurrence as an independent study event, with no item-based blending or other repetition effects, outperforms the standard account in formal model comparisons." | "model" without setup. No mechanism difference. |
| "Retrieved-context theory accounts for these patterns only when item-based reinstatement is removed: each occurrence must bind independently, with traces competing for retrieval rather than blending." | Overlaps S6. No modeling signal. |
| "In formal model comparisons, retrieved-context theory accounts for these patterns only when context evolves independently of item identity rather than through item-based reinstatement." | Drops trace-competition mechanism. |
| "In formal model comparisons, retrieved-context theory accounts for these patterns only when context evolves independently of item identity, with each occurrence stored as a separate competing trace." | Rewrites (a)/(b) unnecessarily. |
| "In formal model comparisons, retrieved-context theory accounts for these patterns only when repetitions (a) reinstate unique, non-overlapping contextual features and (b) produce traces that compete for retrieval." | "theory" less precise than "models"; no overall-fit signal. |
| "In formal comparisons, retrieved-context models account for these patterns and improve overall fits only when item-based reinstatement is removed and repetitions (a) reinstate unique, non-overlapping contextual features and (b) produce traces that compete separately for later retrieval." | Still labels winning model as "retrieved-context models" — contradicts the challenge. |
| "In formal model comparisons, these patterns are accounted for and overall fits improve only when item-based reinstatement is removed and repetitions (a) reinstate unique, non-overlapping contextual features and (b) produce traces that compete separately for later retrieval." | Passive voice weakens the sentence. |
| "Formal model comparisons show that removing item-based reinstatement accounts for these patterns and improves overall fits, with repetitions instead (a) reinstating unique, non-overlapping contextual features and (b) producing traces that compete separately for later retrieval." | "Removing item-based reinstatement" is redundant next to (a)/(b) — both sit in the predicate chain describing the same mechanism change from two angles. Awkward. |
| "In formal comparisons, models without item-based reinstatement account for these patterns and improve overall fits, with repetitions (a) reinstating unique, non-overlapping contextual features and (b) producing traces that compete separately for later retrieval." | Moving removal to subject reduces syntactic redundancy but semantic redundancy persists — "without item-based reinstatement" and (a) still describe the same change negatively/positively. |
| "In formal comparisons, models account for these patterns and improve overall fits only when repetitions (a) reinstate unique, non-overlapping contextual features and (b) produce traces that compete separately for later retrieval." | "Account for these patterns" overstates — Instance-CMR eliminates interference (i, iii) but can't capture first-presentation bias (ii). "Only when" implies sufficiency. |
| "In formal comparisons, the predicted interference vanishes and overall fits improve only when item-based linkage is abandoned such that repetitions (a) reinstate non-overlapping contextual features and (b) produce traces that compete separately for retrieval." | "Item-based linkage" echoes S3 (consequence) not S2 (mechanism). "Model" doesn't appear (C8). "Is abandoned" is passive (C12). No null-model framing (C1). |
| "In formal model comparisons, the predicted interference vanishes and overall fits improve only under a null model that abandons item-based reinstatement such that repetitions (a) reinstate non-overlapping contextual features and (b) produce traces that compete separately for retrieval." | Too wordy (~42 words). |

**S6:**

| Draft | Problem |
|-------|---------|
| "We argue that episodic memory for repeated experience balances integration and specificity not through contextual blending, but differentiated competition." | Overreaches — work doesn't address integration. |
| "These results argue that episodic memory for repeated experience balances integration and specificity not through contextual blending, but differentiated competition." | (Current in index.qmd) Same overreach. |
| "We argue that memory search for repeated items is governed not by contextual blending, but by differentiated competition." | Correct scope, but "contextual blending" names the consequence rather than the mechanism. "Item-based reinstatement" is more precise and echoes S2. Also, if S5 no longer names the removal, S6 should. |
| "We argue that memory search for repeated items is governed not by item-based reinstatement, but by differentiated competition." | Correct mechanism naming, but doesn't hammer the takeaway — doesn't name RCT or convey that a major theory is challenged. |
| "These results argue that memory search through repeated experience is governed not through contextual blending, but differentiated competition." | Doesn't overstate, but doesn't name RCT or item-based reinstatement — fails to hammer the challenge. |
| "These results contradict retrieved-context theory and argue that memory search through repeated experience is governed not by contextual blending, but differentiated competition." | Two jobs joined by "and" dilutes the punch (S6-C4). |
| "These results contradict retrieved-context theory: memory search through repeated experience is governed not by contextual blending, but by differentiated competition." | Too wordy. Colon structure mirrors S2 ("RCT is X: [elaboration]"). |
