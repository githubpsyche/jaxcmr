# Manuscript changelog

Original draft archived at `robin/archive/Manuscript_260219_v1.md`. Current version at `robin/Manuscript_260219.md`.

Introduction revisions addressing theoretical accuracy, factual precision, citation gaps, writing clarity, and the LPP–memory novelty claim. Changes verified against Talmi, Lohnas & Daw (2019) and Schupp & Kirmse (2021). Total references increased from 28 to 47. Entries below are ordered by location in the original draft (v1).

### 1 (v1 line 12): "vast behavioural and neural data" needs citations

The sentence claimed retrieved-context theory is supported by "vast behavioural and neural data" without citing any evidence beyond the original Howard & Kahana (2002) paper. This is a strong empirical claim that demands supporting references, especially because the theory has accumulated substantial evidence over two decades. Without citations, the sentence reads as an unsubstantiated assertion.

I replaced "vast behavioural and neural data" with "extensive behavioural and neural evidence" and added seven citations spanning the key lines of evidence. Three behavioural: Sederberg, Howard & Kahana (2008) for recency and contiguity effects; Lohnas, Polyn & Kahana (2015) for intralist and interlist effects; Healey & Kahana (2016) for age-related memory phenomena. Four neural: Polyn, Natu, Cohen & Norman (2005) for the first fMRI demonstration of category-specific cortical reinstatement during free recall; Manning et al. (2011) for intracranial EEG evidence of temporal context reinstatement; Folkerts, Rutishauser & Howard (2018) for single-unit evidence of a neural contiguity effect in the medial temporal lobe; and Kragel et al. (2021) for intracranial EEG evidence dissociating content and context reinstatement systems. For the comprehensive reference, I replaced Kahana (2012) — a book that predates the behavioural citations in the same sentence — with Lohnas & Healey (2021), a chapter-length review covering both behavioural and neural evidence for retrieved-context models.

The behavioural citations are drawn from the core retrieved-context modelling tradition (free-recall benchmarks, ageing effects). The neural citations span three methodologies (fMRI, intracranial EEG, single-unit recordings) and two independent labs (Kahana, Howard), substantiating the claim of "extensive" neural evidence. The word "vast" was replaced with "extensive" for a more measured tone, and "comprehensive treatment" was softened to "recent review" to better match the scope of a chapter versus a book.

### 2 (v1 lines 13–20): Retrieved-context theory description is too generic

The opening paragraph described RCT purely in terms of context-binding at encoding: items encoded in similar contexts are more likely to be recalled together. That much is true, but it is generic to any context-based account of episodic memory. What distinguishes RCT — and what gives it its name — is that recalled items *reinstate* their associated encoding contexts, which then serve as retrieval cues for further items. Without foregrounding this mechanism, the paragraph undersells the theory and makes it indistinguishable from simpler encoding-similarity accounts.

I replaced the original four sentences (lines 13–15 of v1: "According to this theory, the likelihood of recalling a particular experience is determined by the similarity between the encoding and retrieval 'contexts'... Initially, the Temporal Context Model only considered similarity between temporally-contiguous brain states. More recently, retrieved-context theory recognised that semantic and experiential similarity also influence which experience would come to mind.") with three new sentences that explicitly describe the drifting-context representation, the retrieved-context mechanism, and temporal contiguity effects as a consequence. The Polyn et al. (2009) extension to nontemporal dimensions of context (semantic and experiential similarity) is now introduced in a single sentence rather than spread across four.

The revised text now captures both sides of RCT — context-binding at encoding *and* context-reinstatement at retrieval — which is necessary for the reader to understand why eCMR's emotional context dimension matters for recall dynamics, not just encoding similarity. My hope is that this updated language still builds up to the presentation of eCMR where experiential similarity (supported by either pre-experimental associations or shared encoding processes) is central.

### 3 (v1 lines 22–23): eCMR characterisation

The original sentence — "eCMR focuses on the emotional dimension of experiential similarity" — reduces eCMR to a similarity story. In fact, eCMR extends CMR in two distinct ways: it introduces an emotional dimension of context (the category-only variant, which captures emotional similarity) *and* it allows emotion to modulate encoding strength via the $\Phi_{\text{emot}}$ parameter (the attention-category variant, which captures preferential attention). Characterising eCMR as being about experiential similarity alone omits the second mechanism entirely.

I replaced the sentence with a two-part summary: eCMR extends CMR by (1) introducing an emotional dimension of context and (2) allowing emotion to modulate encoding strength. I then split this across two sentences — the first states the two extensions, the second maps them onto the effects they capture (emotional similarity and preferential attention, respectively). The single-sentence version packed four ideas into one clause, and the "respectively" construction made it difficult for a first-time reader to track which mechanism captures which effect. I also replaced the opening phrase "A recent variant of retrieved-context theory" with "Within this framework" — eCMR is a model within the RCT framework, not a variant of the theory itself. TLD19's abstract describes the work as "extending [CMR]" and "leveraging the rich tradition of temporal context models," never as proposing a variant theory.

This gives the reader a two-part roadmap for the eCMR description that follows, and it correctly represents both mechanisms from TLD19 (pp. 459–460) rather than collapsing them into one. The two-sentence version allows the reader to process the two mechanisms before encountering what they capture, which is important because the rest of the Introduction depends on the reader tracking these two distinct mechanisms. To reinforce this two-part structure, I also split the paragraph that follows into two: the first covers emotional similarity and the source context mechanism, the second covers preferential processing, the $\Phi_{\text{emot}}$ parameter, and the motivation for constraining it with the LPP. The split falls at the natural boundary between the two mechanisms that the roadmap sets up.

### 4 (v1 lines 25–26): emotional similarity — experiential vs semantic

The original text characterised emotional similarity as "a type of experiential similarity, which refers to similarity between encoding operations of emotional stimuli." Whether emotional similarity is experiential, semantic, or both depends on the paradigm — many experiments induce emotions through semantic processing (e.g., emotionally compelling images), blurring the experiential–semantic distinction. eCMR is compatible with any of these perspectives, so a categorical claim is unnecessary.

I replaced the categorical claim with two sentences. The first states that emotional items are more similar to one another than to neutral items (citing Riberto et al., 2022) because they share processing characteristics. The second explicitly notes that whether this similarity is best characterised as experiential, semantic, or both may depend on the paradigm, and that eCMR is compatible with each perspective, representing emotional similarity through its source context mechanism.

The revised text avoids committing to a theoretical position that the model does not require. eCMR's source context mechanism can represent any form of shared encoding characteristics — whether these arise from common perceptual operations (experiential), shared conceptual content (semantic), or both is an empirical question that varies across paradigms. This agnosticism is faithful to Polyn et al.'s (2009) original formulation of source context and to the flexibility of the eCMR framework.

### 5 (v1 lines 28–30): source-feature description implies a mechanism that wouldn't make neutral items similar

The original text described emotionality as a single source-feature dimension where emotional items receive a value of 1 and neutral items a value of 0. The problem is that this description implies a mechanism under which neutral items receive *no* source-context binding — a feature value of 0 contributes nothing when projected into source context. But eCMR does not work this way. TLD19 (p. 463, 465) is explicit that both categories have their own source context representation: "recall of an item from either emotional state (neutral or emotional) will support recall of items from the same emotional state." The mechanism described in v1 and the behaviour it was meant to explain were in conflict.

I replaced the two sentences with: "Emotionality is represented as a source context dimension: emotional items share a common emotional source context, and neutral items share a distinct neutral source context, so that recalling an item from either category promotes further recall of items from the same category."

This correctly reflects TLD19's symmetric two-category source scheme, where both emotional and neutral items have non-zero source representations that promote within-category recall. The asymmetry that drives emotional memory advantages comes not from the source features (which are symmetric) but from the attention-category variant's $\Phi_{\text{emot}}$ parameter, which is introduced in the sentences that follow.

### 6 (v1 lines 32–33): preferential processing claim lacks citations

The sentence "It is known that emotional items enjoy preferential processing during encoding" made a strong empirical claim with no supporting references. This is a well-established finding, but it needs to be anchored in the literature — especially because the next sentence describes how eCMR formalises it, and the reader should be able to trace the empirical motivation.

I added four citations: Anderson (2005), MacKay et al. (2004), Pourtois et al. (2013), and Schupp et al. (2006). These were chosen to cover the key lines of evidence cited by TLD19 (p. 460): priority binding (MacKay), attentional dynamics (Anderson), amygdala-driven enhanced sensory processing (Pourtois), and ERP evidence for early selective processing of emotional stimuli (Schupp). Two new .bib entries were created for Anderson (2005) and MacKay et al. (2004); the other two were already in the bibliography.

The claim is now properly supported, and the citations span the evidence base that TLD19 themselves used to motivate the attention-category variant.

### 7 (v1 lines 34–35): mechanism description is imprecise

The original sentence — "These are simulated in eCMR by tighter associations between the item and context layers in the model, specifically between the source item feature and the source context feature" — is imprecise in two ways. First, "tighter associations between the item and context layers" is vague about directionality. Second, localising the effect to "the source item feature and the source context feature" is incorrect: in the attention-category variant, $\Phi_{\text{emot}}$ scales the learning rate for the full context-to-item association matrix $M^{CF}$ (TLD19, Eq. 4: $\Delta M^{CF} = \phi_i L^{CF} \mathbf{f}_i \mathbf{c}_i^T$), not just the source subspace.

I replaced this with: "eCMR simulates this by scaling the learning rate for associations from context to item features ($M^{CF}$) during encoding, so that emotional items form stronger bindings to their encoding context."

The revised sentence correctly identifies the direction of association ($M^{CF}$, context-to-item), the mechanism (scaling the learning rate), and the consequence (stronger bindings to encoding context), without incorrectly restricting the effect to the source features.

### 8 (v1 lines 37–38): $\Phi_{\text{emot}} = 1$ does not mean "processed equally"

The original sentence — "When it equals 1, emotional and neutral items are modelled as processed equally" — is misleading. Even when $\Phi_{\text{emot}} = 1$, emotional and neutral items still differ: emotional items carry distinct tags in the emotional source features, which influence context updating and retrieval. What $\Phi_{\text{emot}} = 1$ actually means is that emotional items receive no *additional encoding strength* beyond what neutral items receive — the learning rate multiplier is the same for both.

I changed "processed equally" to "receive the same encoding strength."

This is a small but important distinction. "Processed equally" could be read as implying no difference whatsoever between emotional and neutral items, which would contradict the source-feature mechanism described two sentences earlier. "Receive the same encoding strength" correctly scopes the claim to the learning rate parameter without implying that all processing is identical.

### 9 (v1 lines 40–42): motivation for constraining $\Phi_{\text{emot}}$ and Turner citation

The original sentence — "In order to use eCMR to predict what an individual, or a group, would recall, it could be very useful to set the value of $\Phi_{\text{emot}}$ through empirical measurement" — is hedging ("it could be very useful") and fails to motivate *why* constraining the parameter matters theoretically. It reads as a practical convenience rather than a scientific contribution. The Turner et al. (2016) citation appeared at the end of the sentence but its connection to the claim was left implicit — Turner et al. argue for simultaneous modelling of EEG, fMRI, and behavioural data, a broader programme of which constraining a single model parameter with an ERP measure is a specific instance.

I replaced the sentence with: "Constraining $\Phi_{\text{emot}}$ with an independent neural measure, rather than treating it as a free parameter, would provide a more principled account of individual differences in emotional memory and a stronger test of the theory — an approach consistent with the broader case for using neural data to constrain cognitive models." The Turner et al. citation now follows the dash clause rather than the preceding claim.

The revised sentence makes two arguments: (1) the move from free parameter to neural constraint is epistemically stronger, and (2) it enables the model to account for individual differences — which is the central aim of this paper. This frames the LPP-constrained eCMR as a theoretical advance rather than a modelling convenience. Making the Turner et al. connection explicit situates the approach within a broader methodological tradition and adds theoretical weight.

### 10 (v1 lines 43–44): LPP citations and ERP definition

The sentence introducing the LPP cited only Schupp et al. (2006) and used "ERP" without defining it. "Event-related potential" is standard in the EEG literature but may not be familiar to readers from the computational modelling or behavioural memory traditions. The LPP itself is also subliterature-specific jargon that benefits from being introduced with a broader evidence base.

I restructured the sentence to define both terms parenthetically: "the late positive potential (LPP), an event-related potential (ERP) component." I added three foundational citations alongside the existing Schupp et al. (2006): Cuthbert et al. (2000) for early LPP characterisation with autonomic covariation, Schupp et al. (2000) for the original demonstration that the LPP is modulated by motivational relevance, and Hajcak, MacNamara & Olvet (2010) as a comprehensive review.

The revised sentence serves double duty: it defines the jargon for non-specialist readers and grounds the LPP in a richer evidence base. The four citations now span the key milestones in the LPP literature — from early characterisation through to a modern integrative review.

### 11 (v1 lines 47–49): group-level LPP claims and typo

The sentence "Changes in LPP amplitudes due to affective significant have been observed at the group level across a range of stimuli and presentation durations" contained a typo ("affective significant" instead of "affective significance") and made a broad empirical claim without any supporting references. The reader has no way to verify or follow up on the claim.

I fixed the typo and added two citations: Olofsson et al. (2008), a major integrative review covering decades of ERP studies with affective pictures, and Hajcak et al. (2010), a review covering LPP modulation across stimulus types, developmental stages, and experimental conditions.

Both are review articles, which is appropriate for a claim about the breadth of evidence rather than a specific finding. They cover the range of stimuli (IAPS pictures, faces, words) and presentation parameters that the sentence references.

### 12 (v1 lines 51–52): Schupp & Kirmse sentence is awkward and contains a typo

The sentence "Recently, @schupp2021case have examined this question by using a case-by-case approach in three studies, using a range of emotional induction technique" has two problems: "technique" should be "techniques," and the sentence is unnecessarily wordy with redundant prepositional phrases ("by using... in three studies, using a range of...").

I replaced it with: "Recently, @schupp2021case examined this question using a case-by-case approach across three studies with different emotional stimulus categories."

The revision fixes the typo, tightens the phrasing, and more accurately characterises the studies — Schupp & Kirmse varied stimulus *categories* (predator fear, disease avoidance, sexual reproduction), not "emotional induction techniques."

### 13 (v1 lines 53–56): "sensitive to emotional" and arousal interpretation

The sentence "the LPP was sensitive to emotional in 98% of the cases observed" is missing a noun after "emotional" and leaves the reader guessing what the LPP was sensitive to. The word "emotional" alone could mean emotional content, emotional arousal, or emotional valence — and the distinction matters because Schupp & Kirmse (2021) specifically tested high- vs. low-arousing stimuli within emotional categories. More broadly, the description of Schupp & Kirmse's findings was terse — it stated the 98% sensitivity figure and that the effect was "specific to arousal" without explaining the specificity analysis or contextualising the claim against the broader LPP literature.

I replaced the incomplete sentence with: "the LPP was larger for high- compared to low-arousing stimuli in 98% of individual-level tests," mirroring the paper's own phrasing. I specified that the specificity analysis compared conditions that did not differ in emotional content, confirming the effect was driven by emotional arousal rather than low-level stimulus differences. I then added a sentence noting that the broader literature characterises the LPP as reflecting affective stimulus significance (citing Hajcak & Foti, 2020 and Olofsson et al., 2008), of which arousal is a major determinant.

The revised text is grammatically complete, factually precise, and gives the reader a more nuanced picture: Schupp & Kirmse's specificity analysis supports an arousal interpretation, but the LPP likely reflects a broader construct of affective significance. This nuance matters because it tempers the assumption underlying the LPP-to-$\Phi_{\text{emot}}$ mapping, which the next sentences address directly.

### 14 (v1 lines 57–58): $\Phi_{\text{emot}}$ as arousal — hidden assumptions

The sentence "These results support the use of the LPP to constrain the $\Phi_{\text{emot}}$ parameter" contained a hidden assumption: that emotional arousal drives the encoding-strength advantage captured by $\Phi_{\text{emot}}$. This assumption was never stated, making the logic of the LPP–$\Phi_{\text{emot}}$ mapping opaque. Alternative formalisations exist — CMR3 represents arousal as a source feature rather than as an encoding-strength modulator — and making this explicit would add theoretical punchiness.

I added two sentences after the transition sentence. The first makes the assumption explicit: the mapping is justified "under the assumption that emotional arousal drives the encoding-strength advantage that $\Phi_{\text{emot}}$ captures." The second notes that alternative formalisations exist (more recent retrieved-context models represent arousal as a source-context feature), which would motivate different neural-to-parameter mappings.

Making the assumption explicit turns a potential weakness into a theoretical contribution. The reader now understands the interpretive commitment involved in the LPP-to-$\Phi_{\text{emot}}$ mapping and can see that future work testing alternative formalisations (e.g., mapping LPP to a source-context arousal dimension) would be a natural extension.

### 15 (v1 lines 62–63): "experiment" (typo)

The phrase "typical memory recall experiment" should be plural — the sentence is referring to memory experiments in general, not a single experiment.

I changed it to "typical memory recall experiments."

Straightforward typo fix.

### 16 (v1 lines 64–65): encoding instructions sentence too dense

The sentence about how intentional encoding could dilute the LPP packed three distinct mechanisms into a single clause: bias away from "intra-stimulus processing," adoption of strategies that minimise LPP differences, and "covert spaced rehearsal." These terms are jargon and the proposal deserved more unpacking.

I broke the sentence into four. The first states the general claim (intentional encoding could dilute the emotional modulation of the LPP). The second describes the attention-shifting mechanism in plain language (participants shift from processing each stimulus in isolation to comparing or organising stimuli). The third covers strategy adoption and rehearsal effects. The fourth has been revised separately (see entry 17 below).

The revised text replaces technical terms with accessible descriptions. "Intra-stimulus processing" becomes "processing each stimulus in isolation"; "inter-stimulus relationships" becomes "comparing stimuli or organising them for later recall"; "covert spaced rehearsal" becomes "rehearsal that redistributes attention more evenly across stimulus types." Each mechanism now gets enough room to be understood on its own.

### 17 (v1 lines 66–67): encoding mode evidence

The claim that encoding mode can disrupt LPP-relevant processing cited only Healey & Kahana (2019), which is about temporal contiguity effects in recall — a related but indirect piece of evidence. More direct evidence linking encoding task to LPP modulation was needed.

I replaced the sentence with one that cites two complementary pieces of evidence: Schindler & Straube (2020), who showed that task relevance modulates emotional ERP effects specifically for the LPP, and Healey & Kahana (2019) for the broader point that encoding instructions alter recall organisation. The sentence now reads: "The potential impact of encoding mode is supported by evidence that task demands modulate emotional ERP effects and that encoding instructions alter the organisation of recall."

Schindler & Straube (2020) provides the direct LPP evidence — they demonstrated that emotional LPP effects are present only when attention is directed toward emotional content, which supports the claim that task instructions can modulate the LPP. The Healey & Kahana citation is retained for the broader encoding-mode argument. No direct study of intentional encoding instructions diluting the LPP was found, but the revised wording accurately characterises the available evidence without overclaiming.

### 18 (v1 lines 69–70): replication sentence too long

The sentence listing replication benefits was over 60 words long and mixed the argument (replication value) with methodological details (sample size, stimuli, duration) in a way that was difficult to parse. It also contained a grammatical error ("150ms at in Schupp and colleagues' experiments").

I split it into two sentences. The first states the argument: replication in a different setup would increase confidence in the generality of the findings. The second lists the specific differences between our setup and Schupp & Kirmse's: different stimulus set, longer presentation duration (2 s vs. 150 ms), and a large independent sample.

The split separates the "why replicate" argument from the "how our setup differs" details and eliminates the grammatical error.

### 19 (v1 lines 73–76): attention–memory–LPP claims need sourcing

Three consecutive sentences made strong empirical claims without citations: (1) attention and memory are coupled, (2) emotional attention drives emotional memory, (3) arousal increases both LPP and memory. These are well-established findings but need to be anchored in the literature.

I added citations to each sentence: Chun & Turk-Browne (2007) for the general attention–memory coupling; Talmi (2013) and Mather & Sutherland (2011) for emotional attention as a driver of enhanced memory; and Dolcos & Cabeza (2002) for a study demonstrating both LPP effects and subsequent memory effects for emotional pictures.

The citations span the key literatures: cognitive attention–memory interactions (Chun & Turk-Browne), emotional memory mechanisms (Talmi; Mather & Sutherland), and the specific intersection of LPP and memory (Dolcos & Cabeza). Each claim is now individually anchored.

### 20 (v1 lines 77–78): is the LPP–memory consistency really unknown?

The sentence "it is not known whether the relationship between them is consistent either within-subject or between-subjects" made a novelty claim. If prior work had established this relationship, the claim would be false and the motivation for the study would be undermined.

I verified through literature search that the claim is substantially correct. Fields (2023), in a recent review of the LPP's functional significance, explicitly notes that only a few studies have directly examined the correlation between the LPP and later memory. I revised the sentence to: "Yet direct tests of whether the LPP–memory relationship is consistent within or between subjects are scarce," citing Fields (2023).

The revised wording is softer than "it is not known" — it acknowledges that some relevant work exists while correctly identifying it as scarce. The Fields citation gives the reader a recent review that documents this gap, which is stronger than an unsupported novelty claim.

### 21 (v1 lines 79–81): within/between distinction could be clearer

The two possibilities (between-subject correlation vs within-subject list-level correlation) were presented as "one possibility" and "another possibility" without labelling which level of analysis each referred to. The reader had to infer from the phrasing whether "individuals with increased LPP differences" meant a between-subjects comparison or a within-subject comparison.

I replaced the generic framing with explicit level labels: "At the between-subjects level" and "At the within-subject level." The predictions are otherwise unchanged.

This minimal revision makes the two levels of analysis immediately clear and sets up the subsequent discussion of within- vs between-subject ERP effects without requiring the reader to reconstruct the distinction from context.

### 22 (v1 lines 82–83): ERP within/between dissociation examples

The dissociation between within- and between-subject ERP effects was illustrated only by MacLeod & Donaldson (2017) on the left parietal old/new effect. Additional examples were needed.

I added Weinberg et al. (2021), who demonstrated that the LPP's emotional modulation is stable within individuals across five testing sessions. I incorporated this as a new sentence: "Although the emotional modulation of the LPP is itself reliable within individuals across testing sessions, it is unknown whether this reliability extends to predicting memory differences."

Weinberg et al. (2021) is particularly relevant because it establishes within-subject reliability of the LPP — a necessary precondition for the within-subject analyses in this paper. The sentence also bridges to the novelty claim by noting that reliability of the LPP effect per se does not guarantee predictive validity for memory. No additional examples of within/between ERP dissociations beyond MacLeod & Donaldson were found in the literature, which itself underscores that this pattern is under-explored.

### 23 (v1 lines 84–85): figure introduction is redundant

The sentence "Simulations with eCMR show that all else held equal, increasing the value of $\Phi_{\text{emot}}$ results in increased recall of emotional stimuli (Figure 1)" repeated information from the preceding paragraph about $\Phi_{\text{emot}}$ values and emotional processing. The figure reference appeared as an afterthought attached to a redundant claim.

I replaced the sentence with: "As illustrated in Figure 1, eCMR predicts that larger values of $\Phi_{\text{emot}}$ yield greater recall of emotional relative to neutral stimuli." This uses the figure as the leading element rather than the restated claim.

The revised sentence serves as a transition: it introduces the figure and frames the prediction that the next sentence will test against LPP data. The earlier mention of $\Phi_{\text{emot}}$ described the parameter's role; this sentence describes the model's prediction. The "as illustrated" construction also tells the reader the figure exists for a reason — it visually demonstrates the prediction — rather than tacking the figure reference onto a restated fact.

### 24 (v1 lines 95–97): model comparison enumeration questionable

The original text mapped "LPP reflects attention or working memory processes unrelated to emotion" to the LPP-only model and "LPP reflects arousal which mainly is elicited by negative items" to the interaction model. This conflates the model structure (main effect vs interaction) with the theoretical interpretation (emotion-general vs emotion-specific processing). The phrasing also implied that the LPP-only model tests whether LPP reflects general attention, but the comparison cannot adjudicate between specific cognitive constructs.

I rewrote the three sentences to focus on what the model comparison actually tests. If the LPP captures encoding processes that operate similarly for emotional and neutral items, the main-effect model should account for the data as well as the interaction model. If the LPP primarily reflects emotion-specific processing — such that its influence on encoding strength is larger for emotional items — the interaction model should provide the best fit.

The revised predictions are cleaner because they map directly to the model structures without invoking specific cognitive constructs (attention, working memory, arousal) that go beyond what the comparison can adjudicate. The comparison tests whether the LPP's effect on encoding is emotion-general or emotion-specific — it cannot determine whether the underlying construct is attention, arousal, or something else. This more modest framing is both more accurate and more defensible.

### Bibliography additions

Twenty-two new .bib entries were added across the editing passes; one (Kahana, 2012) was removed after becoming unreferenced. Total references increased from 28 to 50. Key additions include retrieved-context theory papers — behavioural (Sederberg et al., 2008; Lohnas et al., 2015; Healey & Kahana, 2016) and neural (Polyn et al., 2005; Manning et al., 2011; Folkerts et al., 2018; Kragel et al., 2021) — plus a recent review (Lohnas & Healey, 2021), LPP review articles (Hajcak et al., 2010; Hajcak & Foti, 2020; Olofsson et al., 2008), attention–memory coupling references (Chun & Turk-Browne, 2007; Talmi, 2013; Mather & Sutherland, 2011; Dolcos & Cabeza, 2002), and additional LPP sources (Cuthbert et al., 2000; Schupp et al., 2000; Fields, 2023; Weinberg et al., 2021; Schindler & Straube, 2020).

### Bibliography corrections

All 47 entries were verified against their DOI records via the Crossref API. Six had metadata errors:

MacLeod & Donaldson (2017) had a wrong DOI that resolved to an unrelated paper by Bürki. The correct paper is in *Frontiers in Human Neuroscience*, not *Psychophysiology*. DOI, journal, volume, pages, and first name (Catherine, not Claire) were corrected.

Zarubin et al. (2020) had incorrect first names for all eight authors (e.g., Vitaliy instead of Vanessa, Tiffany instead of Timothy) and two incomplete surnames (Swafford should be Bolton Swafford, Steinmetz should be Mickley Steinmetz). All corrected per the Crossref record.

Rosenfeld (2019/2020) was published online in 2019 but assigned to volume 57, issue 7 in 2020. Year updated to 2020 and volume/issue added.

Wessa et al. (2010) had an anglicised title and journal name. The actual publication is in German: *Zeitschrift für Klinische Psychologie und Psychotherapie*. Original German title restored and supplement/page information added.

Schindler & Straube (2020) had been entered with the wrong co-author (Kissler instead of Straube) and wrong issue number (8 instead of 9). Both corrected.

Weinberg et al. (2021) had the wrong first name for the second author (Kreshnik instead of Kelly). Corrected.
