# Manuscript changelog

Original draft archived at `robin/archive/Manuscript_260219_v1.md`. Current version at `robin/Manuscript_260219.md`.

Entries 1–32 cover the introduction: theoretical accuracy, factual precision, citation gaps, writing clarity, and the LPP–memory novelty claim. Changes verified against Talmi, Lohnas & Daw (2019), Schupp & Kirmse (2021), and Barnacle et al. (2018). Entries 33–34 cover the Method and EEG Data Analysis sections: heading hierarchy, a drafting note, and batch grammar/formatting corrections. Entries 35–38 cover the Model Variants, Evaluation Approach, and eCMR Model Comparison results sections: the model set changed from three LPP-parameterization variants to three architecturally distinct models (CMR, eCMR, LPP-eCMR), nesting flooring and BMS were added to the evaluation, and the results were rewritten with corrected post-flooring numbers. Entries 39–50 refine the Introduction's model comparison framing, Model Variants prose, and eCMR Model Comparison results: removing premature supplementary commitments, previewing the qualitative dimension of the comparison, correcting the eCMR description, adding positive characterization of CMR, justifying the comparison's scope via the output-order constraint, refining Model Variants prose, adding interpretation standards to the Evaluation Approach, rewriting the results to fix tautological claims and sentence structure, replacing the model comparison table and benchmark figure, fixing rendering issues (running header, inline math, missing bib entries), labeling CIs, adding the category SPC qualitative-evidence paragraph, removing an overstated claim, completing the manuscript-wide colon sweep, and splitting the benchmark figure into separate SPC and LPP-by-recall figures with empirical data. Entries 51–54 add CMR+LPP (k=10, single-context, LPP-only encoding modulation) as a fourth model: Introduction preview, Model Variants 2×2 design, expanded comparison table and narrative, and updated diagnostic figures. Entries 55–56 reframe the eCMR vs LPP-eCMR comparison around the population-individual distinction and add the Discussion section. All entries are ordered by location in the original draft (v1).

### 1 (v1 line 12): "vast behavioural and neural data" needs citations

The sentence claimed retrieved-context theory is supported by "vast behavioural and neural data" without citing any evidence beyond the original Howard & Kahana (2002) paper. This is a strong empirical claim that demands supporting references, especially because the theory has accumulated substantial evidence over two decades. Without citations, the sentence reads as an unsubstantiated assertion.

I replaced "vast behavioural and neural data" with "extensive behavioural and neural evidence" and added seven citations spanning the key lines of evidence. Three behavioural: Sederberg, Howard & Kahana (2008) for recency and contiguity effects; Lohnas, Polyn & Kahana (2015) for intralist and interlist effects; Healey & Kahana (2016) for age-related memory phenomena. Four neural: Polyn, Natu, Cohen & Norman (2005) for the first fMRI demonstration of category-specific cortical reinstatement during free recall; Manning et al. (2011) for intracranial EEG evidence of temporal context reinstatement; Folkerts, Rutishauser & Howard (2018) for single-unit evidence of a neural contiguity effect in the medial temporal lobe; and Kragel et al. (2021) for intracranial EEG evidence dissociating content and context reinstatement systems. I also added Lohnas & Healey (2021), a chapter-length review covering both behavioural and neural evidence for retrieved-context models, as the review reference for the sentence ("for a recent review, see...").

The word "vast" was replaced with "extensive" for a more measured tone.

### 2 (v1 lines 13–20): Retrieved-context theory description is too generic

The opening paragraph described RCT purely in terms of context-binding at encoding: items encoded in similar contexts are more likely to be recalled together. That much is true, but it is generic to any context-based account of episodic memory. What distinguishes RCT (and gives it its name) is that recalled items _reinstate_ their associated encoding contexts, which then serve as retrieval cues for further items. Without describing this mechanism, the paragraph undersells the theory and makes it indistinguishable from simpler encoding-similarity accounts.

I replaced the original four sentences with three new sentences that explicitly describe the drifting-context representation, the retrieved-context mechanism, and temporal contiguity effects as a consequence. The Polyn et al. (2009) extension to nontemporal dimensions of context is now introduced in two sentences rather than spread across four, using Polyn et al.'s own terminology ("semantic associations," "source features" / "source context") rather than the "semantic similarity" / "experiential similarity" taxonomy of the original draft. The original labels implied a clean binary between pre-experimental and session-specific similarity, but the boundary blurs whenever semantic processing drives encoding experiences (see entry 4). Using Polyn et al.'s source-context vocabulary avoids the binary and introduces terminology the reader will encounter again when eCMR's source context mechanism is described.

The revised text covers both sides of RCT (context-binding at encoding and context-reinstatement at retrieval), which is needed for understanding why eCMR's emotional context dimension matters for recall dynamics, not just encoding similarity.

### 3 (v1 lines 22–23): eCMR characterisation

The original sentence ("eCMR focuses on the emotional dimension of experiential similarity") reduces eCMR to a similarity story. In fact, eCMR extends CMR in two distinct ways: it introduces an emotional dimension of context (the category-only variant, which captures emotional similarity) *and* it allows emotion to modulate encoding strength via the $\Phi_{\text{emot}}$ parameter (the attention-category variant, which captures preferential attention). Characterising eCMR as being about experiential similarity alone omits the second mechanism entirely.

I replaced the sentence with a two-part summary: eCMR extends CMR by (1) introducing an emotional dimension of context and (2) allowing emotion to modulate encoding strength. I then split this across two sentences: the first states the two extensions, the second maps them onto the effects they capture (emotional similarity and preferential attention, respectively). I also replaced "A recent variant of retrieved-context theory" with "Within this framework," since eCMR is a model within the RCT framework, not a variant of the theory itself. TLD19's abstract describes the work as "extending [CMR]" and "leveraging the rich tradition of temporal context models," never as proposing a variant theory.

The two-part summary correctly represents both mechanisms from TLD19 (pp. 459–460). I also split the paragraph that follows into two at the boundary between the mechanisms: the first covers emotional similarity and source context, the second covers preferential processing, $\Phi_{\text{emot}}$, and the LPP motivation.

### 4 (v1 lines 25–26): emotional similarity, experiential vs semantic

The original text characterised emotional similarity as "a type of experiential similarity, which refers to similarity between encoding operations of emotional stimuli." Whether emotional similarity is experiential, semantic, or both depends on the paradigm; many experiments induce emotions through semantic processing (e.g., emotionally compelling images), blurring the experiential–semantic distinction. eCMR is compatible with any of these perspectives, so a categorical claim is unnecessary.

I removed the categorical claim and the experiential/semantic labels entirely. The revised text states that emotional items share processing characteristics and are more similar to one another than to neutral items (citing Riberto et al., 2022), then moves directly to eCMR's source context mechanism. The experiential/semantic taxonomy was also removed from the opening paragraph's description of Polyn et al. (2009) (see entry 2), so there is no longer a binary to apply (or misapply) to emotional similarity.

The revised text describes emotional similarity in terms of shared processing characteristics and represents it through eCMR's source context mechanism, faithful to Polyn et al.'s (2009) original formulation.

### 5 (v1 lines 28–30): source-feature description implies a mechanism that wouldn't make neutral items similar

The original text described emotionality as a single source-feature dimension where emotional items receive a value of 1 and neutral items a value of 0. The problem is that this description implies neutral items receive *no* source-context binding, since a feature value of 0 contributes nothing when projected into source context. But eCMR does not work this way. TLD19 (p. 463, 465) is explicit that both categories have their own source context representation: "recall of an item from either emotional state (neutral or emotional) will support recall of items from the same emotional state." The mechanism described in v1 and the behaviour it was meant to explain were in conflict.

I replaced the two sentences with: "Emotionality is represented as a source context dimension: emotional items share a common emotional source context, and neutral items share a distinct neutral source context, so that recalling an item from either category promotes further recall of items from the same category."

This correctly reflects TLD19's symmetric two-category source scheme, where both emotional and neutral items have non-zero source representations that promote within-category recall. The asymmetry that drives emotional memory advantages comes not from the source features (which are symmetric) but from the attention-category variant's $\Phi_{\text{emot}}$ parameter, which is introduced in the sentences that follow.

### 6 (v1 lines 32–33): preferential processing claim lacks citations

The sentence "It is known that emotional items enjoy preferential processing during encoding" made a strong empirical claim with no supporting references. This is well-established but needs citations, especially because the next sentence describes how eCMR formalises the claim.

I added four citations: Anderson (2005), MacKay et al. (2004), Pourtois et al. (2013), and Schupp et al. (2006). These were chosen to cover the key lines of evidence cited by TLD19 (p. 460): priority binding (MacKay), attentional dynamics (Anderson), amygdala-driven enhanced sensory processing (Pourtois), and ERP evidence for early selective processing of emotional stimuli (Schupp). Two new .bib entries were created for Anderson (2005) and MacKay et al. (2004); the other two were already in the bibliography.

The citations span the evidence base TLD19 used to motivate the attention-category variant.

### 7 (v1 lines 34–35): mechanism description is imprecise

The original sentence ("These are simulated in eCMR by tighter associations between the item and context layers in the model, specifically between the source item feature and the source context feature") is imprecise in two ways. First, "tighter associations between the item and context layers" is vague about directionality. Second, localising the effect to "the source item feature and the source context feature" is incorrect: in the attention-category variant, $\Phi_{\text{emot}}$ scales the learning rate for the full context-to-item association matrix $M^{CF}$ (TLD19, Eq. 4: $\Delta M^{CF} = \phi_i L^{CF} \mathbf{f}_i \mathbf{c}_i^T$), not just the source subspace.

I replaced this with: "eCMR simulates this by scaling the learning rate for associations from context to item features ($M^{CF}$) during encoding, so that emotional items form stronger bindings to their encoding context."

The revised sentence correctly identifies the direction of association ($M^{CF}$, context-to-item), the mechanism (scaling the learning rate), and the consequence (stronger bindings to encoding context), without incorrectly restricting the effect to the source features.

### 8 (v1 lines 37–38): $\Phi_{\text{emot}} = 1$ does not mean "processed equally"

The original sentence ("When it equals 1, emotional and neutral items are modelled as processed equally") is misleading. Even when $\Phi_{\text{emot}} = 1$, emotional and neutral items still differ: emotional items carry distinct tags in the emotional source features, which influence context updating and retrieval. What $\Phi_{\text{emot}} = 1$ actually means is that emotional items receive no *additional encoding strength* beyond what neutral items receive; the learning rate multiplier is the same for both.

I changed "processed equally" to "receive the same encoding strength."

"Processed equally" could be read as implying no difference whatsoever, which would contradict the source-feature mechanism described two sentences earlier. "Receive the same encoding strength" scopes the claim to the learning rate parameter.

### 9 (v1 lines 40–42): motivation for constraining $\Phi_{\text{emot}}$ and Turner citation

The original sentence ("In order to use eCMR to predict what an individual, or a group, would recall, it could be very useful to set the value of $\Phi_{\text{emot}}$ through empirical measurement") hedges ("it could be very useful") and fails to motivate *why* constraining the parameter matters theoretically. The Turner et al. (2016) citation appeared at the end but its connection to the claim was left implicit. Turner et al. argue for simultaneous modelling of EEG, fMRI, and behavioural data; constraining a single model parameter with an ERP measure is a specific instance of that programme.

I replaced the sentence with: "Constraining $\Phi_{\text{emot}}$ with an independent neural measure, rather than treating it as a free parameter, would provide a more principled account of individual differences in emotional memory and a stronger test of the theory." The Turner et al. citation now follows the claim it supports rather than appearing after an unrelated hedge.

The revised text makes two arguments: (1) the move from free parameter to neural constraint is epistemically stronger, and (2) it enables the model to account for individual differences, which is the central aim of this paper.

### 10 (v1 lines 43–44): LPP citations and ERP definition

The sentence introducing the LPP cited only Schupp et al. (2006) and used "ERP" without defining it. "Event-related potential" is standard in the EEG literature but may not be familiar to readers from the computational modelling or behavioural memory traditions. The LPP itself is also subliterature-specific jargon that benefits from being introduced with a broader evidence base.

I restructured the sentence to define both terms parenthetically: "the late positive potential (LPP), an event-related potential (ERP) component." I added three foundational citations alongside the existing Schupp et al. (2006): Cuthbert et al. (2000) for early LPP characterisation with autonomic covariation, Schupp et al. (2000) for the original demonstration that the LPP is modulated by motivational relevance, and Hajcak, MacNamara & Olvet (2010) as a comprehensive review.

The revised sentence defines both terms for non-specialist readers and grounds the LPP in a broader evidence base, spanning from early characterisation (Cuthbert et al., Schupp et al.) to a modern integrative review (Hajcak et al.).

### 11 (v1 lines 47–49): group-level LPP claims and typo

The sentence "Changes in LPP amplitudes due to affective significant have been observed at the group level across a range of stimuli and presentation durations" contained a typo ("affective significant" instead of "affective significance") and made a broad empirical claim without any supporting references. The reader has no way to verify or follow up on the claim.

I fixed the typo and rewrote the opening clause to foreground the LPP's construct: "The LPP is broadly characterised as reflecting affective stimulus significance." I added two review citations: Hajcak & Foti (2020), a recent review of the LPP's functional significance, and Olofsson et al. (2008), a major integrative review covering decades of ERP studies with affective pictures.

Both are review articles covering the range of stimuli (IAPS pictures, faces, words) and presentation parameters the sentence references.

### 12 (v1 lines 51–52): Schupp & Kirmse sentence is awkward and contains a typo

The sentence "Recently, @schupp2021case have examined this question by using a case-by-case approach in three studies, using a range of emotional induction technique" has two problems: "technique" should be "techniques," and the sentence is unnecessarily wordy with redundant prepositional phrases ("by using... in three studies, using a range of...").

I replaced it with: "Recently, @schupp2021case examined this question using a case-by-case approach across three studies with different emotional stimulus categories."

The revision fixes the typo and tightens the phrasing. Schupp & Kirmse varied stimulus *categories* (predator fear, disease avoidance, sexual reproduction), not "emotional induction techniques."

### 13 (v1 lines 53–56): "sensitive to emotional" and arousal interpretation

The sentence "the LPP was sensitive to emotional in 98% of the cases observed" is missing a noun after "emotional" and leaves the reader guessing what the LPP was sensitive to. The word "emotional" alone could mean emotional content, emotional arousal, or emotional valence, and the distinction matters because Schupp & Kirmse (2021) specifically tested high- vs. low-arousing stimuli within emotional categories. The description of their findings was also terse: it stated the 98% sensitivity figure and that the effect was "specific to arousal" without explaining the specificity analysis.

I replaced the incomplete sentence with: "the LPP was larger for high- compared to low-arousing stimuli in 98% of individual-level tests," following the paper's own phrasing. I also made the specificity analysis description more precise: rather than the vague claim that the effect is "specific to arousal," the revised text states that "a complementary specificity analysis confirmed that this effect was driven by emotional content rather than low-level stimulus differences."

The revision presents the evidence in broad-to-specific order: general sensitivity first, then individual-level reliability.

### 14 (v1 lines 57–58): $\Phi_{\text{emot}}$ as arousal, hidden assumption

The sentence "These results support the use of the LPP to constrain the $\Phi_{\text{emot}}$ parameter" contained a hidden assumption: that emotional arousal drives the encoding-strength advantage captured by $\Phi_{\text{emot}}$. The assumption was never stated, making the logic of the LPP–$\Phi_{\text{emot}}$ mapping opaque.

I replaced the sentence with one that explicitly grounds the connection in the preferential-processing argument developed earlier in the Introduction: "Because eCMR attributes the emotional encoding advantage to preferential processing during study, and the LPP indexes this processing, these results support using the LPP to constrain $\Phi_{\text{emot}}$."

The LPP–$\Phi_{\text{emot}}$ connection is now a consequence of the paper's own argument (eCMR posits extra encoding strength via preferential processing; the LPP indexes that processing), not an unsupported assertion.

### 15 (v1 lines 46, 61, 73): aim connectivity

The broad aim (v1 line 46) introduces four sub-aims across the Introduction, but the hierarchical relationship among them is left implicit. The encoding-mode sentence (v1 line 61) does not reference the broad aim. The second aim (v1 line 73) begins a new paragraph with no bridge to either the first aim or the broad aim.

Three changes address this. (1) The encoding-mode sentence was replaced with: "For the LPP to serve this constraining role, however, the same individual-level reliability must hold under conditions typical of memory experiments. It is not yet known whether it does." (2) A roadmap sentence was added after the broad aim: "We pursued this aim in three steps: by establishing individual-level reliability of the LPP under memory-experiment conditions, by testing whether LPP variation predicts emotional memory, and by investigating how best to incorporate the neural measure into the model." (3) The second aim opening was changed to: "Beyond individual-level reliability, our second aim was to examine whether the emotional modulation of the LPP predicts the emotional modulation of memory."

The roadmap decomposes "suitability" into three questions that the sub-aims address in sequence. "This constraining role" and "Beyond individual-level reliability" create explicit backward references, so the aim hierarchy is legible as each sub-aim appears.

### 16 (v1 lines 62–63): "experiment" (typo)

The phrase "typical memory recall experiment" should be plural, since the sentence refers to memory experiments in general.

I changed it to "typical memory recall experiments."

Straightforward typo fix.

### 17 (v1 lines 64–65): encoding instructions sentence too dense

The sentence about how intentional encoding could dilute the LPP packed three distinct mechanisms into a single clause: bias away from "intra-stimulus processing," adoption of strategies that minimise LPP differences, and "covert spaced rehearsal." These terms are jargon and the proposal deserved more unpacking.

I broke the sentence into four. The first states the general claim (intentional encoding could dilute the emotional modulation of the LPP). The second describes the attention-shifting mechanism in plain language (participants shift from processing each stimulus in isolation to comparing or organising stimuli). The third covers strategy adoption and rehearsal effects. The fourth has been revised separately (see entry 18 below).

The revised text replaces technical terms with accessible descriptions. "Intra-stimulus processing" becomes "processing each stimulus in isolation"; "inter-stimulus relationships" becomes "comparing stimuli or organising them for later recall"; "covert spaced rehearsal" becomes "rehearsal that redistributes attention more evenly across stimulus types." Each mechanism now gets enough room to be understood on its own.

### 18 (v1 lines 66–67): encoding mode evidence

The claim that encoding mode can disrupt LPP-relevant processing cited only Healey & Kahana (2019), which is about temporal contiguity effects in recall, a related but indirect piece of evidence. More direct evidence linking encoding task to LPP modulation was needed.

I replaced the sentence with one that cites two complementary pieces of evidence: Schindler & Straube (2020), who showed that task relevance modulates emotional ERP effects specifically for the LPP, and Healey & Kahana (2019) for the broader point that encoding instructions alter recall organisation. The sentence now reads: "The potential impact of encoding mode is supported by evidence that task demands modulate emotional ERP effects and that encoding instructions alter the organisation of recall."

Schindler & Straube (2020) provides the direct LPP evidence: they showed that emotional LPP effects are present only when attention is directed toward emotional content. No direct study of intentional encoding instructions diluting the LPP was found, but the revised wording avoids overclaiming.

### 19 (v1 lines 69–70): replication sentence too long

The sentence listing replication benefits was over 60 words long and mixed the argument (replication value) with methodological details (sample size, stimuli, duration) in a way that was difficult to parse. It also contained a grammatical error ("150ms at in Schupp and colleagues' experiments").

I split it into two sentences. The first states the argument: replication in a different setup would increase confidence in the generality of the findings. The second lists the specific differences relative to Schupp & Kirmse: different stimulus set, longer presentation duration (2 s vs. 150 ms), and a larger sample. "The present study" avoids implying that the authors designed the experiment (the Method section clarifies this is a secondary analysis of Zarubin et al.'s data).

All factual details were verified against the Schupp & Kirmse (2021) PDF: 150 ms presentation duration (p. 3), sample sizes of 16, 18, and 17 across three studies.

### 20 (v1 lines 73–76): attention–memory–LPP claims need sourcing

Three consecutive sentences made strong empirical claims without citations: (1) attention and memory are coupled, (2) emotional attention drives emotional memory, (3) arousal increases both LPP and memory. These are well-established findings but need to be anchored in the literature.

I added citations to each sentence: Chun & Turk-Browne (2007) for the general attention–memory coupling; Talmi (2013) and Mather & Sutherland (2011) for emotional attention as a driver of enhanced memory; and Dolcos & Cabeza (2002) for a study demonstrating both LPP effects and subsequent memory effects for emotional pictures.

The citations span the key literatures: cognitive attention–memory interactions (Chun & Turk-Browne), emotional memory mechanisms (Talmi; Mather & Sutherland), and the specific intersection of LPP and memory (Dolcos & Cabeza). Each claim is now individually anchored.

### 21 (v1 lines 77–78): is the LPP–memory consistency really unknown?

The sentence "it is not known whether the relationship between them is consistent either within-subject or between-subjects" made a novelty claim. If prior work had established this relationship, the claim would be false and the motivation for the study would be undermined.

I verified through literature search that the claim is substantially correct. Fields (2023), in a recent review of the LPP's functional significance, explicitly notes that only a few studies have directly examined the correlation between the LPP and later memory. I revised the sentence to: "Yet direct tests of whether the LPP–memory relationship is consistent within or between subjects are scarce," citing Fields (2023).

The revised wording is softer than "it is not known": it acknowledges some relevant work exists while identifying it as scarce. The Fields citation gives a recent review that documents this gap.

### 22 (v1 lines 79–81): within/between distinction could be clearer

The two possibilities (between-subject correlation vs within-subject list-level correlation) were presented as "one possibility" and "another possibility" without labelling which level of analysis each referred to. The reader had to infer from the phrasing whether "individuals with increased LPP differences" meant a between-subjects comparison or a within-subject comparison.

I replaced the generic framing with explicit level labels: "At the between-subjects level" and "At the within-subject level." The predictions are otherwise unchanged.

The explicit labels make the two levels of analysis clear and set up the subsequent discussion of within- vs between-subject ERP effects.

### 23 (v1 lines 82–83): ERP within/between dissociation examples

The dissociation between within- and between-subject ERP effects was illustrated only by MacLeod & Donaldson (2017) on the left parietal old/new effect. Additional examples were needed.

I added Weinberg et al. (2021), who demonstrated that the LPP's emotional modulation is stable within individuals across five testing sessions. I incorporated this as a new sentence: "Although the emotional modulation of the LPP is itself reliable within individuals across testing sessions, it is unknown whether this reliability extends to predicting memory differences."

Weinberg et al. (2021) establishes within-subject reliability of the LPP, a precondition for the within-subject analyses in this paper. The sentence bridges to the novelty claim: reliability of the LPP effect does not guarantee predictive validity for memory.

### 24 (v1 lines 85–87): figure sentence restates known information; prediction sentence overlong

The figure sentence (v1 line 85, "Simulations with eCMR show...") restates the effect of $\Phi_{\text{emot}}$ already established earlier in the Introduction. The prediction sentence (v1 line 87) was also overlong (42 words) with nested "the difference between... the difference" phrasing.

I deleted the figure sentence entirely and replaced the prediction sentence with: "If the LPP indexes $\Phi_{\text{emot}}$, the model predicts a positive association between the emotional modulation of the LPP and the emotional recall advantage, at the within-subject level, the between-subjects level, or both (Figure 1)."

The Figure 1 reference is now attached to the prediction rather than the restated claim.

### 25 (v1 lines 95–97): model comparison enumeration questionable

The original text mapped "LPP reflects attention or working memory processes unrelated to emotion" to the LPP-only model and "LPP reflects arousal which mainly is elicited by negative items" to the interaction model. This conflates the model structure (main effect vs interaction) with the theoretical interpretation (emotion-general vs emotion-specific processing). The phrasing also implied that the LPP-only model tests whether LPP reflects general attention, but the comparison cannot adjudicate between specific cognitive constructs.

The construct-based predictions are removed entirely. The current version no longer frames the comparison as testing emotion-general vs emotion-specific processing. Instead, the introduction presents the comparison as testing whether embedding an established neural-behavioral relationship (the GLMM's emotion-specific LPP effect) in a process model improves prediction. This avoids the construct-adjudication problem: the comparison speaks to model prediction, not to which cognitive construct the LPP reflects.

### 26 (v1 lines 57–58, continued): LPP–$\Phi_{\text{emot}}$ bridge overstates construct validity

Entry 14 replaced the bare assertion "These results support the use of the LPP to constrain $\Phi_{\text{emot}}$" with a sentence that makes the logical chain explicit: eCMR posits preferential processing → the LPP indexes this processing → therefore LPP constrains $\Phi_{\text{emot}}$. The explicit chain was an improvement, but "indexes this processing" claims a tighter measurement relationship than the LPP literature supports. Contemporary reviews characterise the LPP as reflecting affective or motivational significance broadly (Hajcak & Foti, 2020; Olofsson et al., 2008), and Fields (2023, already cited at line 59) notes that the LPP's functional significance remains an open question.

I replaced "indexes this processing" with "is sensitive to the enhanced processing of motivationally significant stimuli," changed "support using" to "motivate using," and specified "LPP variation" rather than "the LPP." The revised sentence states what the review literature can defend (sensitivity to motivationally significant processing) and frames the constraining move as motivated rather than entailed by the evidence. No new citations are needed; the broader construct is already established at line 37 with Hajcak & Foti (2020) and Olofsson et al. (2008).

### 27 (v1 lines 82–83, continued): LPP psychometric properties citation

Entry 23 cited Weinberg et al. (2021) for within-individual reliability of the LPP's emotional modulation across testing sessions, a test-retest argument. I added Moran, Jendrusina & Moser (2013), who showed the LPP has good-to-excellent internal consistency (split-half reliability rs = .54–.79) and stabilizes within approximately eight trials. The two citations now cover complementary psychometric properties: internal consistency (Moran et al.) and test-retest reliability (Weinberg et al.). No other text changes; the addition strengthens the evidence base for the claim that precedes the novelty claim about LPP–memory prediction.

### Bibliography additions

Twenty-three new .bib entries were added across the editing passes. Total references increased from 28 to 51. Key additions include retrieved-context theory papers, behavioural (Sederberg et al., 2008; Lohnas et al., 2015; Healey & Kahana, 2016) and neural (Polyn et al., 2005; Manning et al., 2011; Folkerts et al., 2018; Kragel et al., 2021), plus a recent review (Lohnas & Healey, 2021), LPP review articles (Hajcak et al., 2010; Hajcak & Foti, 2020; Olofsson et al., 2008), attention–memory coupling references (Chun & Turk-Browne, 2007; Talmi, 2013; Mather & Sutherland, 2011; Dolcos & Cabeza, 2002), and additional LPP sources (Cuthbert et al., 2000; Schupp et al., 2000; Fields, 2023; Weinberg et al., 2021; Schindler & Straube, 2020; Moran et al., 2013).

### Bibliography corrections

All 50 entries were verified against their DOI records via the Crossref API. Six had metadata errors:

MacLeod & Donaldson (2017) had a wrong DOI that resolved to an unrelated paper by Bürki. The correct paper is in *Frontiers in Human Neuroscience*, not *Psychophysiology*. DOI, journal, volume, pages, and first name (Catherine, not Claire) were corrected.

Zarubin et al. (2020) had incorrect first names for all eight authors (e.g., Vitaliy instead of Vanessa, Tiffany instead of Timothy) and two incomplete surnames (Swafford should be Bolton Swafford, Steinmetz should be Mickley Steinmetz). All corrected per the Crossref record.

Rosenfeld (2019/2020) was published online in 2019 but assigned to volume 57, issue 7 in 2020. Year updated to 2020 and volume/issue added.

Wessa et al. (2010) had an anglicised title and journal name. The actual publication is in German: *Zeitschrift für Klinische Psychologie und Psychotherapie*. Original German title restored and supplement/page information added.

Schindler & Straube (2020) had been entered with the wrong co-author (Kissler instead of Straube) and wrong issue number (8 instead of 9). Both corrected.

Weinberg et al. (2021) had the wrong first name for the second author (Kreshnik instead of Kelly). Corrected.

### 28: Construct-validity cleanup, propagating entries 25/26

Entries 25 and 26 addressed the construct-validity concern about equating the LPP with specific cognitive processes: entry 25 removed "arousal" and "attention/working memory" labels from v1's model comparison predictions, and entry 26 softened "indexes this processing" to "is sensitive to the enhanced processing of motivationally significant stimuli." Two instances of the same issue survived those edits.

The LPP introduction sentence (v1 line 44, entry 10) was restructured and given additional citations but preserved v1's "emotionally arousing visual stimuli." The word "arousing" can be read as implying the LPP specifically indexes arousal, the same premature construct commitment that entries 25 and 26 addressed elsewhere. I changed "emotionally arousing" to "emotionally significant," aligning with the "affective stimulus significance" characterisation established in the following paragraph (Hajcak & Foti, 2020).

The prediction sentence (v1 line 87, entry 24) was rewritten as "If the LPP indexes Φ_emot," which claims a direct measurement relationship between a neural component and a model parameter. This is stronger than what the LPP literature supports and conflicts with the softer framing adopted by entry 26. I changed "If the LPP indexes Φ_emot" to "If trial-level LPP variation tracks encoding-related processing relevant to Φ_emot." The conditional structure is preserved, but the claim is now about predictive relevance rather than indexing equivalence.

Entry 26 also left "motivate using LPP variation to constrain" (v1 line 58), which presupposes that the constraining succeeds. I changed "motivate using" to "motivate testing whether... can usefully constrain," framing the LPP–Φ_emot link as the hypothesis under test rather than a premise.

### 29: Barnacle dissociation as motivation for the modelling approach

Barnacle et al. (2018, from Talmi's lab) found that the emotional modulation of the LPP was equally strong in pure and mixed lists under intentional encoding conditions, while the emotional memory advantage was context-dependent. This dissociation is relevant to the modelling rationale: if the LPP's encoding-related modulation is context-independent, then a model representing both encoding strength and retrieval context is needed to explain when LPP variation translates into a memory advantage.

I inserted two sentences after the novelty claim (entry 21, "Yet direct tests... are scarce") that cite this dissociation and frame the modelling approach as a way to bridge the gap. "Under intentional encoding conditions" was added to strengthen the connection to Aim 1, which asks whether the LPP's emotional modulation survives the same kind of memory-task conditions.

This entry originally attributed the finding to Zarubin et al. (2020). That was a misattribution: both papers study pure/mixed lists with ERP measures, but Zarubin's LPP analyses focus on the Dm effect (remembered vs forgotten), not the main emotional modulation across list types. Zarubin explicitly credits Barnacle for the context-independent ERP finding (p. 174). Only Barnacle should be cited here. Barnacle et al. (2018) was added to references.bib.

### 30: Reframe model comparison framing

V1 (lines 93–97) framed the comparison as three architecturally identical eCMR variants differing only in LPP parameterization: "a standard version of the eCMR model with fixed values of Φ_emot" versus a main-effect and an interaction model. The current version compares three architecturally distinct models: CMR (no emotion), eCMR (emotion labels), and LPP-eCMR (emotion labels + LPP interaction). This required rewriting the introduction's final two paragraphs.

The first paragraph now opens with "we used these results to constrain a process model," establishing the GLMM findings as input to the modelling rather than a hypothesis to re-test. It then introduces the three models in short separate sentences and states the comparison question: whether embedding the LPP in a mechanistic retrieval model improves prediction beyond category labels alone. An earlier revision of this passage framed the comparison as testing "whether LPP adds predictive value for emotional items," which re-stated the GLMM finding (the EEG Results section already established that trial-level LPP predicts recall for emotional but not neutral items). The current framing avoids this circularity by treating the LPP–memory relationship as established and asking whether it translates into improved process-model prediction.

The second paragraph describes the comparison logic at two levels: eCMR and LPP-eCMR share identical retrieval dynamics, so differences between them isolate the LPP term; the comparison between CMR and eCMR tests the broader structural contribution of the emotional context layer.

### 31: Narrow the novelty claim

V1 line 61 said "it is not known whether a similar prevalence would be obtained," referring to whether the LPP's individual-level emotional modulation would replicate under memory-experiment conditions. Entry 15 preserved this as "It is not yet known whether it does." The claim was always too sweeping because Barnacle et al. (2018) had already shown that the LPP's emotional modulation survives intentional encoding at the group level. The gap became more visible after entry 29 inserted the Barnacle dissociation in the adjacent paragraph.

The fix narrows the claim: "Although emotional modulation of the LPP has been observed at the group level in intentional-encoding paradigms followed by free recall [@barnacle2018context], it is not yet known whether comparable individual-level sensitivity is preserved." This acknowledges the group-level precedent while preserving the novelty of Aim 1, which is about individual-level reliability, something Barnacle did not test (no case-by-case analysis).

### 32: Sharpen the Schupp description

V1 line 56 said Schupp & Kirmse (2021) "examined this question," where "this question" was whether the LPP differentiates emotional from neutral processing at the individual level (v1 line 55). But Schupp tested high- versus low-arousing pictures within three behavior systems (predator fear, disease avoidance, sexual reproduction), a related but not identical contrast to the negative-versus-neutral comparison used in the present study. "Examined this question" implied an exact match.

Replaced with "examined individual-level sensitivity to emotional content," which accurately describes what Schupp tested without implying the contrast was the same as the present study's. The rest of the Schupp paragraph is unchanged: the 98% figure, the specificity analysis, and the bridging sentence all remain accurate.

### 33: Heading hierarchy and drafting note

The `## EEG Data Analysis` heading (v1 line 149) was at the same level as `## Method`, making it a sibling section rather than a subsection. This was inconsistent with `### Behavioural Data Analysis`, which sat under Method. Demoted `## EEG Data Analysis` to `###` and its subsections from `###` to `####`.

Removed a drafting note at v1 line 137: "(see if you need to expand this later)," an author-facing reminder left in the EEG pre-processing description.

### 34: Grammar, usage, and formatting in Method and EEG sections

Batch corrections to the Method and EEG Data Analysis sections (all unchanged from v1). No changes to meaning or analysis descriptions.

Subject-verb agreement: "functions... was" → "were" (v1 line 137), "mean number... were" → "was" (v1 line 142), "number of trials were" → "was" (v1 line 186), "the LPP need" → "needs" (v1 line 182).

Missing/wrong words: "provided in paper" → "provided in @zarubin2020contributions" (v1 line 103), "less than" → "fewer than" (v1 line 108), "a criteria" → "a criterion" (v1 line 140), "two first" → "first two" (v1 line 116), "International Affective Picture" → "International Affective Picture System" (v1 line 117), "therefor" → "therefore" (v1 line 183), "participants data based" → "participant's data" (v1 line 183), "based on that this cut off" → "because this cutoff" (v1 line 190).

Capitalisation: "institutional Review Board" → "Institutional Review Board" (v1 line 111), "pearson" → "Pearson" (v1 lines 199, 202).

Terminology consistency: "subjects" → "participants" (v1 line 141).

Formatting: added spaces before "ms" per SI convention throughout; fixed "between X to Y" → "from X to Y" preposition mismatch; broke up the run-on bootstrap paragraph (v1 line 190) into separate sentences with a paragraph break before the reliability threshold; "behavioral" → "behavioural" in eCMR Specification for British English consistency.

Redundancy: "fixed effects of... as fixed effects" → removed duplicate (v1 line 159).

Style: "could end earlier if they finished earlier" → "could stop sooner if they had no more items to report" (v1 line 128).

### 35 (v1 lines 298–314): Model Variants rewrite

V1 described three eCMR variants that differed only in how the LPP entered the encoding-strength term $\phi_{emot,i}$: emotion-only (no LPP, $\kappa_L = \kappa_{EL} = 0$), main-effect ($\kappa_{EL} = 0$), and separate-slope (both LPP terms). All three shared the same eCMR architecture with the emotional context layer. The model comparison was therefore limited to the question of how LPP enters encoding strength, not whether the emotional context layer itself matters.

The current version replaces these with three architecturally distinct models. CMR ($k = 9$) has no emotional context layer and no emotion parameters. eCMR ($k = 10$) adds the emotional context layer with one parameter ($\phi_{emot}$) controlling emotion-modulated encoding strength via category labels. LPP-eCMR ($k = 11$) adds one parameter ($\kappa_{EL}$) scaling the Early LPP's contribution to emotional encoding strength, restricted to emotional items by the $e_i$ indicator. A main-effect LPP term is omitted because the GLMM found no LPP–recall association for neutral items, and trial-mean-centering of $L_i$ absorbs any threshold offset.

The change reflects the expanded comparison structure: the manuscript now tests both whether the emotional context layer matters (CMR vs eCMR) and whether LPP adds to category labels (eCMR vs LPP-eCMR). V1's additional parameterizations (main-effect, separate-slope, and pathway-breadth variants) are not carried over.

### 36 (v1 lines 332–334): Evaluation Approach additions

V1 described model comparison using only AIC: "We compare variants using AIC computed from the approximate set log-likelihood." Two additions were made.

First, nesting flooring: "For nested model pairs, we floor the child model's per-subject NLL at the parent model's value to correct for occasional optimizer failures that would otherwise violate the nesting guarantee." This is necessary because differential evolution can fail to find the global optimum, producing cases where a nested child model has worse NLL than its parent — a logical impossibility. Flooring enforces the nesting constraint without altering results for participants where the optimizer succeeded.

Second, Bayesian model selection (Stephan et al., 2009) with protected exceedance probabilities (Rigoux et al., 2014) was added alongside AIC to characterize population-level model frequencies. BMS estimates the probability that each model is the most frequent in the population, complementing AIC weights which summarize aggregate fit. Two new citations were added to the bibliography.

### 37 (v1 lines 420–441): eCMR Model Comparison results rewrite

V1 reported: "The Emotion + LPP (interaction) model provides the best fit by all comparison metrics." It included a 3×3 pairwise ΔAIC table comparing three LPP variants (emotion-only, main-effect, interaction) with mean [95% t-based CI] values, and a composite benchmark figure (@fig-11) showing Category-SPC and LPP-by-recall diagnostics for all three.

These results were replaced entirely. The model set changed from three LPP variants to CMR, eCMR, and LPP-eCMR. The AIC computation was corrected (a prior revision had fixed a bug in the aggregate AIC formula). Nesting flooring was applied (entry 36).

The current results section has four paragraphs. The first reports that eCMR achieved strictly lower NLL than CMR for 89.5% of participants (mean ΔAIC = 4.06, 95% CI [1.36, 6.77]) with BMS exceedance probability and expected frequency. The second reports that LPP-eCMR matched or beat eCMR for every participant (73.7% strictly better, remainder tied after flooring; AICw = .997 vs .003; BMS xp = .761 vs .226), but the protected exceedance probability remained at chance (.333), and the median per-subject ΔAIC favored eCMR (−1.21 [−2.37, −0.05]). The third presents the cat_lpp_by_recall diagnostic as genuine generalization: LPP-eCMR reproduces the recalled/unrecalled LPP separation for emotional items that eCMR cannot, despite being fit to set likelihood rather than to this statistic. The fourth interprets the LPP's value as enabling mechanistically grounded neural-behavioral predictions rather than improving aggregate fit metrics.

The table was updated to 3×3 median ΔAIC with 95% bootstrap CIs (replacing mean with t-based CIs). The figure caption was updated for three models.

### 38 (current line 75): Sentence-structure cleanup in Introduction

The sentence introducing CMR, eCMR, and LPP-eCMR used semicolons and an em dash to pack three model descriptions and a parenthetical GLMM reference into a single sentence: "CMR provides a floor with no emotion mechanism; eCMR adds an emotional context layer driven by category labels; LPP-eCMR adds one parameter that maps the GLMM finding—trial-level Early LPP predicts recall for emotional but not neutral items—onto the per-item encoding strength." This was split into four short sentences: one per model plus a comparison-question sentence. Preference: avoid semicolons and em dashes for listing model descriptions.

### 39 (current line 78): Remove premature supplementary commitment; preview qualitative test

The rewrite of the model comparison framing (entry 30) included "Additional LPP parameterizations and an architectural extension are reported in the supplementary model comparison." This forward reference commits the paper to reporting content whose inclusion has not been decided. Removed. If a supplementary comparison is added later, the forward reference can be restored.

The comparison question sentence was also extended with a second clause: "and whether the resulting model generates qualitatively distinct predictions about neural-behavioral relationships." V1 had explicit alternative predictions (if emotion-general → main-effect; if emotion-specific → interaction), which were removed because they conflated model structure with cognitive constructs (entry 25). The added clause restores some of the old draft's hypothesis-driven character by previewing that the comparison has a qualitative dimension beyond aggregate fit, without invoking specific constructs. The two clauses now map to the paper's two types of evidence: aggregate fit metrics (equivocal) and qualitative predictions about untrained statistics (decisive).

### 40 (current line 76): eCMR one-mechanism reduction in Introduction

The model comparison reframing (entry 30) introduced "eCMR adds an emotional context layer driven by category labels." This reduces eCMR to its similarity mechanism, the same error corrected in entry 3. Entry 3 established that eCMR extends CMR in two ways: an emotional context layer (emotional similarity) and emotion-modulated encoding strength (preferential attention). The manuscript's own line 20 states both mechanisms. Revised to: "eCMR adds an emotional context layer and emotion-modulated encoding strength, using only category labels." The Model Variants section (line 311) was already correct.

### 41 (current line 75): Positive characterization of CMR in Introduction

The model comparison reframing (entry 30) introduced "CMR, which has no emotion mechanism, provides a floor." This describes CMR solely by absence, obscuring the value of the comparison. Lines 12–18 of the Introduction describe retrieved-context theory's mechanisms (context drift, bidirectional associations, context reinstatement, temporal contiguity) but never use the "CMR" acronym — it first appears at line 20 in the eCMR description. Revised to: "CMR captures temporal-context encoding and retrieval but has no emotion mechanism, and so provides a floor." This reminds the reader that CMR is a full process model, making the floor comparison meaningful rather than trivial.

### 42 (current line 85): Output-order justification for comparison scope

Added: "Comparing alternatives that differ in retrieval dynamics would require output-order information that the present data lack." This justifies why the comparison is limited to models sharing retrieval dynamics (CMR → eCMR → LPP-eCMR). The point appeared in a previous draft but was not carried over during the rewrite (entry 30). The Evaluation Approach section (line 332) already notes that the data lack recall-order information in the context of motivating set likelihood, but this is the first place the constraint is invoked to justify the comparison's scope.

### 43 (current lines 304–322): Model Variants prose refinements

Three refinements to the Model Variants text (entry 35). First, bold pseudo-headings (`**CMR** (*k* = 9).`, `**eCMR** (*k* = 10).`, `**LPP-eCMR** (*k* = 11).`) were converted to paragraph-opening topic sentences. V1 used flowing prose; the entry 35 rewrite introduced bold labels as sub-subsection markers. The current version integrates model names into natural topic sentences. Second, "This is the most parsimonious LPP parameterization: no threshold parameters and no main-effect LPP term" was replaced with data-driven motivation: a main-effect LPP term is omitted because the GLMM found no LPP–recall association for neutral items, and trial-mean-centering absorbs any threshold offset. Third, "Additional LPP parameterizations and a pathway-breadth manipulation are reported in the supplementary model comparison" was removed for the same reason as entry 39: it commits the paper to content whose inclusion has not been decided.

### 44 (v1 lines 332–334, continued): Interpretation standards in Evaluation Approach

Entry 36 added AIC and BMS as comparison tools but did not state interpretation standards. Added two sentences after the BMS description: a ΔAIC CI excluding zero indicates a reliable advantage, and an exceedance probability above .95 provides strong population-level evidence while values near chance indicate indistinguishable models. These standards make the Results section's interpretive claims self-sufficient.

### 45 (v1 lines 420–441, continued): eCMR Model Comparison results refinements

Six refinements to the results text (entry 37). First, the opening sentence "Adding the emotional context layer improved fit" was replaced with a framing sentence that names the comparison being tested. Second, "LPP-eCMR achieved equal or better NLL than eCMR for every participant" was removed because nesting flooring (entry 36) guarantees this outcome; the sentence now leads with the informative 73.7% strictly-better figure. Third, "Among all eCMR parameterizations tested (supplementary model comparison)..." was removed for the same reason as entries 39 and 43: it commits to content whose inclusion has not been decided. Fourth, colons and an em dash that extended sentences in the results paragraphs were split into separate sentences, consistent with the preference established in entry 38.

### 46 (v1 lines 420–441, continued): Table and figure overhaul

The pairwise median ΔAIC table was replaced with a model-level summary table reporting free parameters, mean NLL, aggregate AIC weights, and BMS expected frequencies and exceedance probabilities. The old composite benchmark figure (image11.png) was replaced with a Quarto subfigure layout showing three diagnostics (category SPC, LPP-by-recall for negative items, LPP-by-recall for neutral items) for each of the three models. The neutral-items column demonstrates that none of the models predict an LPP–recall association for neutral items, consistent with the GLMM null result. Table and figure labels were updated from `tbl-aic`/`fig-11` to `tbl-comparison`/`fig-benchmarks`.

### 47: Rendering fixes

Added `shorttitle: "LPP and emotional memory model validity"` to the YAML front matter so the running header fits on one line (APA ≤50 characters). Fixed broken inline math in the ΔAIC confidence intervals (lines 448–450): escaped dollar signs (`\$`) from the docx conversion were rendering literally; replaced with proper inline math delimiters. Added author affiliations required by apaquarto and two missing BibTeX entries (Stephan et al. 2009, Rigoux et al. 2014). Added `\usepackage{soul}` to support the pre-existing `{.mark}` highlighted span.

### 48 (eCMR Model Comparison, continued): CI labeling and category SPC paragraph

The bracketed intervals after ΔAIC values were verified to be 95% CIs (t-based for means, bootstrap for medians) but this was not stated in the text. Added "(95% CI [...])" to the median ΔAIC report and normalized escaped brackets `\[...\]` to plain brackets for consistency. The figure's category SPC column had no prose support: a new paragraph explains that CMR, lacking an emotional context layer, predicts nearly identical recall for negative and neutral items, while eCMR and LPP-eCMR reproduce the empirical separation. The LPP-by-recall discussion is now introduced as a second diagnostic, giving the two benchmark tests parallel framing. Two FIXMEs resolved (CI semantics, qualitative-evidence structure); the manuscript-wide em-dash/colon review FIXME is deferred.

### 49 (eCMR Model Comparison + Introduction): Overstated claim and colon sweep

Deleted the sentence "The emotional context mechanism is therefore necessary to account for the category-level recall advantage." The preceding sentences already establish that CMR fails and eCMR/LPP-eCMR succeed; the interpretive gloss overstated what a three-model comparison can demonstrate. The manuscript-wide em-dash/colon review (deferred in entry 48) was completed. One candidate was found: the eCMR introduction sentence on line 31 combined a definition and its consequence after a colon in 40+ words. The colon was replaced with a period. All other colons and the single em-dash (a parenthetical appositive) were judged legitimate. Both remaining FIXMEs resolved.

### 50 (eCMR Model Comparison, continued): Separate benchmark figures with empirical data

The single 3×3 benchmark figure (`@fig-benchmarks`) was split into two figures: `@fig-catspc` for category-specific serial position curves and `@fig-lpp-recall` for the LPP-by-recall diagnostic. Entry 46 replaced the old composite image with a Quarto subfigure layout but kept both diagnostics in a single figure and did not include empirical data. Each figure now displays the empirical data alongside model predictions in a 2×2 grid, giving the reader the reference pattern before evaluating model adequacy. Subcaptions were removed; the overall caption identifies panels by grid position. The LPP-by-recall figure shows only negative items (the condition where models diverge). Model fits were updated to the refactored eCMR variants (eCMREmotionOnly, eCMREmotionTimesLPP). Three in-text cross-references were updated to point to the relevant figure.

### 51 (Introduction): CMR+LPP added to model comparison preview

Added CMR+LPP to the four-model preview paragraph. The comparison question now includes whether trial-level LPP can substitute for category labels. The paragraph about eCMR and LPP-eCMR sharing dynamics was broadened to describe the 2×2 structure: emotional context layer (absent/present) × LPP encoding modulation (absent/present). CMR+LPP and eCMR share k=10 but differ in architecture and information source.

### 52 (Model Variants): Four-model 2×2 design with CMR+LPP

Changed from "three models that form a nested hierarchy" to "four models that cross two factors." Added CMR+LPP (k=10) description with equation φ_{emot,i} = κ_L · L_i — single-context architecture, LPP modulates temporal MCF encoding for all items, no category labels. Revised eCMR opening to contrast with CMR+LPP ("takes a different approach: rather than continuous neural measures..."). Added nesting structure paragraph: two nested pairs (CMR→CMR+LPP, eCMR→LPP-eCMR), CMR+LPP and eCMR not nested.

### 53 (eCMR Model Comparison): Expanded to four-model comparison

Table expanded from 3 to 4 rows with CMR+LPP (NLL 213.84±15.73, AICw≈0, BMS r=.089, XP<.001). BMS numbers updated for the 4-model comparison (CMR XP .009, eCMR XP .269, protected XP .250 each). Added two new results paragraphs: (1) LPP alone adds nothing over CMR (ΔAIC 0.33 [−0.77, 1.44], CI includes zero); (2) critical comparison at k=10 — eCMR dramatically outperforms CMR+LPP (ΔAIC 3.73 [1.23, 6.22], median 1.52 [0.18, 2.86], both CIs exclude zero). Added synthesis paragraph: emotional context architecture essential, LPP cannot replace it, LPP supplements but does not substitute. Updated eCMR vs LPP-eCMR BMS numbers to reflect 4-model comparison (XP .721 vs .269, protected XP .250).

### 54 (eCMR Model Comparison, continued): CMR+LPP in diagnostic figures

Both `@fig-catspc` and `@fig-lpp-recall` expanded from 2×2 to 3×2 (or 2×3) layout with CMR+LPP panels added. Cat_spc caption updated: CMR+LPP produces faint separation via indirect LPP–category correlation. LPP-by-recall caption updated: both CMR+LPP and LPP-eCMR reproduce the recalled/unrecalled separation. Qualitative discussion updated: CMR+LPP noted as producing weak category separation in SPC, and both LPP-using models noted as reproducing LPP-by-recall pattern. Concluding paragraph reframed LPP's role as supplement not substitute.

### 55 (Results + Introduction): Population-individual distinction elevated from caveat to characterization

Covers four locations. In the individual-level results paragraph, the adversative "however" was removed ("At the individual level, however..." → "At the individual level, ...") and the heterogeneity pattern named explicitly: "a subset of participants benefit substantially from the LPP term, driving the aggregate advantage, while for most participants the improvement is too small to offset the cost of one additional parameter." In the synthesis paragraph, "modest aggregate improvement that is not reliable at the individual level" was replaced with "aggregate fit advantage that is driven by a subset of participants rather than being uniform across individuals." In the concluding paragraph, "improving aggregate fit metrics... enabling the model to generate richer, mechanistically grounded predictions about neural-behavioral relationships" was replaced with two sentences: "not in replacing existing model structure or uniformly improving fit. It is in characterizing which individuals benefit from the neural constraint and in generating mechanistic predictions that were not used as fitting targets." In the Introduction, a sentence was added after the comparison-question paragraph previewing the per-subject question: "Because the cognitive model is fit to each participant independently, it can reveal whether the LPP's contribution to prediction is consistent across individuals or concentrated in a subset." All numerical values unchanged.

### 56 (Discussion): Discussion section added

Discussion section added between the eCMR Model Comparison heading and References. Five paragraphs. (1) Population-level convergence: the GLMM fixed effects (β=.14, p<.001 for emotional; β=.01, p=.488 for neutral) and the cognitive model's aggregate comparison metrics (AICw=.997, BMS XP=.721) both aggregate across participants and both favor the LPP-informed model. (2) Individual-level heterogeneity: per-subject ΔAIC and protected exceedance probability show the group-level advantage is not uniform; the GLMM's maximal random-effects structure accommodates analogous between-participant variation via random slopes. (3) Qualitative generalization: the LPP-by-recall diagnostic provides complementary evidence independent of numerical fit, since the models are fit to set likelihood and the LPP-by-recall pattern is an untrained summary statistic. (4) Methodological contribution: workflow for incorporating neural measures into generative models, with the observation that a reliable group-level neural-behavioral association does not guarantee individual-level prediction improvement; set-likelihood scope noted as a contributing factor. (5) Limitations: per-subject fitting without partial pooling, limited per-subject data (9 lists), hierarchical fitting as a future direction. No new citations.