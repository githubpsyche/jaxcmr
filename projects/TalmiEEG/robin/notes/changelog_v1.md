# Manuscript changelog

Previous version archived at `robin/archive/Manuscript_260219_v1.md`.

## v2 — 2026-03-07

Introduction revisions addressing theoretical accuracy, factual precision, and minor errors. Changes verified against Talmi, Lohnas & Daw (2019) and Schupp & Kirmse (2021).

### FIXME 1: Retrieved-context theory description is too generic

The opening paragraph described RCT purely in terms of context-binding at encoding: items encoded in similar contexts are more likely to be recalled together. That much is true, but it is generic to any context-based account of episodic memory. What distinguishes RCT — and what gives it its name — is that recalled items *reinstate* their associated encoding contexts, which then serve as retrieval cues for further items. Without foregrounding this mechanism, the paragraph undersells the theory and makes it indistinguishable from simpler encoding-similarity accounts.

I replaced the original four sentences (lines 13–15 of v1: "According to this theory, the likelihood of recalling a particular experience is determined by the similarity between the encoding and retrieval 'contexts'... Initially, the Temporal Context Model only considered similarity between temporally-contiguous brain states. More recently, retrieved-context theory recognised that semantic and experiential similarity also influence which experience would come to mind.") with three new sentences that explicitly describe the drifting-context representation, the retrieved-context mechanism, and temporal contiguity effects as a consequence. The Polyn et al. (2009) extension to nontemporal dimensions of context (semantic and experiential similarity) is now introduced in a single sentence rather than spread across four. 

The revised text now captures both sides of RCT — context-binding at encoding *and* context-reinstatement at retrieval — which is necessary for the reader to understand why eCMR's emotional context dimension matters for recall dynamics, not just encoding similarity. My hope is that this updated language still builds up to the presentation of eCMR where experiential similarity (supported by either pre-experimental associations or shared encoding processes) is central.

### FIXME 2: eCMR characterisation overindexes on "experiential similarity"

The original sentence — "eCMR focuses on the emotional dimension of experiential similarity" — reduces eCMR to a similarity story. In fact, eCMR extends CMR in two distinct ways: it introduces an emotional dimension of context (the category-only variant, which captures emotional similarity) *and* it allows emotion to modulate encoding strength via the $\Phi_{\text{emot}}$ parameter (the attention-category variant, which captures preferential attention). Characterising eCMR as being about experiential similarity alone omits the second mechanism entirely.

I replaced the sentence with: "eCMR extends CMR in two ways: by introducing an emotional dimension of context and by allowing emotion to modulate encoding strength, capturing the effects of emotional similarity and preferential attention, respectively."

This gives the reader a two-part roadmap for the eCMR description that follows, and it correctly represents both mechanisms from TLD19 (pp. 459–460) rather than collapsing them into one.

### FIXME 3: Source-feature description implies a mechanism that wouldn't make neutral items similar

The original text described emotionality as a single source-feature dimension where emotional items receive a value of 1 and neutral items a value of 0. The problem is that this description implies a mechanism under which neutral items receive *no* source-context binding — a feature value of 0 contributes nothing when projected into source context. But eCMR does not work this way. TLD19 (p. 463, 465) is explicit that both categories have their own source context representation: "recall of an item from either emotional state (neutral or emotional) will support recall of items from the same emotional state." The mechanism described in v1 and the behaviour it was meant to explain were in conflict.

I replaced the two sentences with: "Emotionality is represented as a source context dimension: emotional items share a common emotional source context, and neutral items share a distinct neutral source context, so that recalling an item from either category promotes further recall of items from the same category."

This correctly reflects TLD19's symmetric two-category source scheme, where both emotional and neutral items have non-zero source representations that promote within-category recall. The asymmetry that drives emotional memory advantages comes not from the source features (which are symmetric) but from the attention-category variant's $\Phi_{\text{emot}}$ parameter, which is introduced in the sentences that follow.

### FIXME 4: Preferential processing claim lacks citations

The sentence "It is known that emotional items enjoy preferential processing during encoding" made a strong empirical claim with no supporting references. This is a well-established finding, but it needs to be anchored in the literature — especially because the next sentence describes how eCMR formalises it, and the reader should be able to trace the empirical motivation.

I added four citations: Anderson (2005), MacKay et al. (2004), Pourtois et al. (2013), and Schupp et al. (2006). These were chosen to cover the key lines of evidence cited by TLD19 (p. 460): priority binding (MacKay), attentional dynamics (Anderson), amygdala-driven enhanced sensory processing (Pourtois), and ERP evidence for early selective processing of emotional stimuli (Schupp). Two new .bib entries were created for Anderson (2005) and MacKay et al. (2004); the other two were already in the bibliography.

The claim is now properly supported, and the citations span the evidence base that TLD19 themselves used to motivate the attention-category variant.

### FIXME 5: Mechanism description is imprecise

The original sentence — "These are simulated in eCMR by tighter associations between the item and context layers in the model, specifically between the source item feature and the source context feature" — is imprecise in two ways. First, "tighter associations between the item and context layers" is vague about directionality. Second, localising the effect to "the source item feature and the source context feature" is incorrect: in the attention-category variant, $\Phi_{\text{emot}}$ scales the learning rate for the full context-to-item association matrix $M^{CF}$ (TLD19, Eq. 4: $\Delta M^{CF} = \phi_i L^{CF} \mathbf{f}_i \mathbf{c}_i^T$), not just the source subspace.

I replaced this with: "eCMR simulates this by scaling the learning rate for associations from context to item features ($M^{CF}$) during encoding, so that emotional items form stronger bindings to their encoding context."

The revised sentence correctly identifies the direction of association ($M^{CF}$, context-to-item), the mechanism (scaling the learning rate), and the consequence (stronger bindings to encoding context), without incorrectly restricting the effect to the source features.

### FIXME 6: $\Phi_{\text{emot}} = 1$ does not mean "processed equally"

The original sentence — "When it equals 1, emotional and neutral items are modelled as processed equally" — is misleading. Even when $\Phi_{\text{emot}} = 1$, emotional and neutral items still differ: emotional items carry distinct tags in the emotional source features, which influence context updating and retrieval. What $\Phi_{\text{emot}} = 1$ actually means is that emotional items receive no *additional encoding strength* beyond what neutral items receive — the learning rate multiplier is the same for both.

I changed "processed equally" to "receive the same encoding strength."

This is a small but important distinction. "Processed equally" could be read as implying no difference whatsoever between emotional and neutral items, which would contradict the source-feature mechanism described two sentences earlier. "Receive the same encoding strength" correctly scopes the claim to the learning rate parameter without implying that all processing is identical.

### FIXME 7: Motivation for constraining $\Phi_{\text{emot}}$ is weak

The original sentence — "In order to use eCMR to predict what an individual, or a group, would recall, it could be very useful to set the value of $\Phi_{\text{emot}}$ through empirical measurement" — is hedging ("it could be very useful") and fails to motivate *why* constraining the parameter matters theoretically. It reads as a practical convenience rather than a scientific contribution.

I replaced it with: "Constraining $\Phi_{\text{emot}}$ with an independent neural measure, rather than treating it as a free parameter, would provide a more principled account of individual differences in emotional memory and a stronger test of the theory."

The revised sentence makes two arguments: (1) the move from free parameter to neural constraint is epistemically stronger, and (2) it enables the model to account for individual differences — which is the central aim of this paper. This frames the LPP-constrained eCMR as a theoretical advance rather than a modelling convenience.

### FIXME 8: "affective significant" (typo)

The phrase "affective significant" is ungrammatical — an adjective where a noun is needed.

I changed it to "affective significance."

Straightforward typo fix.

### FIXME 9: Schupp & Kirmse sentence is awkward and contains a typo

The sentence "Recently, @schupp2021case have examined this question by using a case-by-case approach in three studies, using a range of emotional induction technique" has two problems: "technique" should be "techniques," and the sentence is unnecessarily wordy with redundant prepositional phrases ("by using... in three studies, using a range of...").

I replaced it with: "Recently, @schupp2021case examined this question using a case-by-case approach across three studies with different emotional stimulus categories."

The revision fixes the typo, tightens the phrasing, and more accurately characterises the studies — Schupp & Kirmse varied stimulus *categories* (predator fear, disease avoidance, sexual reproduction), not "emotional induction techniques."

### FIXME 10: "sensitive to emotional" is incomplete and possibly wrong

The sentence "the LPP was sensitive to emotional in 98% of the cases observed" is missing a noun after "emotional" and leaves the reader guessing what the LPP was sensitive to. The word "emotional" alone could mean emotional content, emotional arousal, or emotional valence — and the distinction matters because Schupp & Kirmse (2021) specifically tested high- vs. low-arousing stimuli within emotional categories.

I replaced it with: "the LPP was larger for high- compared to low-arousing stimuli in 98% of cases." This mirrors the paper's own phrasing (p. 10: "47 and 50 out of 51 individual tests indicated significantly larger EPN (92%) and LPP (98%) amplitudes to emotional stimuli high in arousal").

The revised sentence is both grammatically complete and factually precise. It specifies the comparison (high vs. low arousal) and the direction of the effect (larger), which matters for the next sentence about arousal specificity.

### FIXME 11: "experiment" (typo)

The phrase "typical memory recall experiment" should be plural — the sentence is referring to memory experiments in general, not a single experiment.

I changed it to "typical memory recall experiments."

Straightforward typo fix.

### Bibliography additions

I added two new .bib entries to support the citations introduced at FIXME 4: `anderson2005affective` (Anderson, 2005, *Journal of Experimental Psychology: General*) and `mackay2004emotion` (MacKay et al., 2004, *Memory & Cognition*). Total references increased from 28 to 30. All 30 resolve correctly in the rendered output.
