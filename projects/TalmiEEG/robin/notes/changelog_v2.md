# Manuscript changelog

Previous versions archived at `robin/archive/Manuscript_260219_v1.md` and `robin/archive/Manuscript_260219_v2.md`. For v1→v2 changes, see `robin/notes/changelog_v1.md`.

## v3 — 2026-03-07

Introduction revisions addressing citation gaps, theoretical precision, writing clarity, and the LPP–memory novelty claim. Seventeen FIXME comments from v2 resolved. Total references increased from 30 to 47.

### FIXME 1 (v2 line 13): "vast behavioural and neural data" needs citations

The sentence claimed retrieved-context theory is supported by "vast behavioural and neural data" without citing any evidence beyond the original Howard & Kahana (2002) paper. This is a strong empirical claim that demands supporting references, especially because the theory has accumulated substantial evidence over two decades. Without citations, the sentence reads as an unsubstantiated assertion.

I replaced "vast behavioural and neural data" with "extensive behavioural and neural evidence" and added five citations spanning the key lines of evidence: Sederberg, Howard & Kahana (2008) for recency and contiguity effects; Lohnas, Polyn & Kahana (2015) for intralist and interlist effects; Healey & Kahana (2016) for age-related memory phenomena; Manning et al. (2011) for intracranial neural evidence of context reinstatement; and Kahana (2012) as a comprehensive reference.

These citations are drawn from the core retrieved-context modelling tradition and include both behavioural (free-recall benchmarks, ageing effects) and neural (intracranial oscillatory) evidence. The book reference (Kahana, 2012) serves as a pointer for readers who want the full treatment. The word "vast" was replaced with "extensive" for a more measured tone.

### FIXME 2 (v2 line 21): eCMR summary sentence too long

The sentence introducing eCMR — "extends CMR in two ways: by introducing an emotional dimension of context and by allowing emotion to modulate encoding strength, capturing the effects of emotional similarity and preferential attention, respectively" — packs four ideas into a single clause. For a reader encountering eCMR for the first time, the "respectively" construction makes it difficult to track which mechanism captures which effect.

I split the sentence into two. The first states that eCMR extends CMR by introducing an emotional dimension of context and by allowing emotion to modulate encoding strength. The second explains what each mechanism captures: emotional similarity among items and preferential attention to emotional stimuli, respectively.

The two-sentence version conveys the same information but allows the reader to process the two mechanisms before encountering what they capture. This is particularly important because the rest of the Introduction depends on the reader tracking these two distinct mechanisms.

### FIXME 3 (v2 line 24): emotional similarity — experiential vs semantic

The original text characterised emotional similarity as "a type of experiential similarity, which refers to similarity between encoding operations of emotional stimuli." The FIXME raised a genuine theoretical tension: whether emotional similarity is experiential, semantic, or both depends on the paradigm. Many experiments induce emotions through semantic processing (e.g., emotionally compelling images), blurring the experiential–semantic distinction. The FIXME further noted that eCMR is compatible with any of these perspectives and suggested expressing agnosticism.

I replaced the categorical claim with two sentences. The first states that emotional items are more similar to one another than to neutral items (citing Riberto et al., 2022) because they share processing characteristics. The second explicitly notes that whether this similarity is best characterised as experiential, semantic, or both may depend on the paradigm, and that eCMR is compatible with each perspective, representing emotional similarity through its source context mechanism.

The revised text avoids committing to a theoretical position that the model does not require. eCMR's source context mechanism can represent any form of shared encoding characteristics — whether these arise from common perceptual operations (experiential), shared conceptual content (semantic), or both is an empirical question that varies across paradigms. This agnosticism is faithful to Polyn et al.'s (2009) original formulation of source context and to the flexibility of the eCMR framework.

### FIXME 4 (v2 line 34): how does Turner (2016) relate?

The Turner et al. (2016) citation appeared at the end of a sentence about constraining $\Phi_{\text{emot}}$ with neural data, but the connection between the paper and the claim was left implicit. Turner et al. argue for simultaneous modelling of EEG, fMRI, and behavioural data — a broader programme of which constraining a single model parameter with an ERP measure is a specific instance. Without making this connection explicit, the citation appears arbitrary.

I appended a dash clause to the sentence: "— an approach consistent with the broader case for using neural data to constrain cognitive models." The citation now follows this clause rather than the preceding claim about individual differences.

The revised sentence makes the relevance of Turner et al. transparent: the specific move of constraining $\Phi_{\text{emot}}$ with LPP is an instance of the general programme they advocate. This also adds theoretical weight to the approach by situating it within a broader methodological tradition.

### FIXME 5 (v2 line 36): additional LPP citations and ERP definition

The sentence introducing the LPP cited only Schupp et al. (2006) and used "ERP" without defining it. "Event-related potential" is standard in the EEG literature but may not be familiar to readers from the computational modelling or behavioural memory traditions. The LPP itself is also subliterature-specific jargon that benefits from being introduced with a broader evidence base.

I restructured the sentence to define both terms parenthetically: "the late positive potential (LPP), an event-related potential (ERP) component." I added three foundational citations alongside the existing Schupp et al. (2006): Cuthbert et al. (2000) for early LPP characterisation with autonomic covariation, Schupp et al. (2000) for the original demonstration that the LPP is modulated by motivational relevance, and Hajcak, MacNamara & Olvet (2010) as a comprehensive review.

The revised sentence serves double duty: it defines the jargon for non-specialist readers and grounds the LPP in a richer evidence base. The four citations now span the key milestones in the LPP literature — from early characterisation through to a modern integrative review.

### FIXME 6 (v2 line 40): group-level LPP claims need citations

The sentence "Changes in LPP amplitudes due to affective significance have been observed at the group level across a range of stimuli and presentation durations" made a broad empirical claim without any supporting references. The reader has no way to verify or follow up on the claim.

I added two citations: Olofsson et al. (2008), a major integrative review covering decades of ERP studies with affective pictures, and Hajcak et al. (2010), a review covering LPP modulation across stimulus types, developmental stages, and experimental conditions.

Both are review articles, which is appropriate for a claim about the breadth of evidence rather than a specific finding. They cover the range of stimuli (IAPS pictures, faces, words) and presentation parameters that the sentence references.

### FIXME 7 (v2 lines 43–46): LPP and arousal — how confident?

The description of Schupp & Kirmse's (2021) findings was terse — it stated the 98% sensitivity figure and that the effect was "specific to arousal" without explaining the specificity analysis or contextualising the claim against the broader LPP literature. The FIXME asked how confident we can be that the LPP indexes arousal in particular, given that the broader literature characterises it more broadly.

I expanded the description in two ways. First, I specified "individual-level tests" (rather than just "cases") and added that the specificity analysis compared conditions that did not differ in emotional content, confirming the effect was driven by emotional arousal rather than low-level stimulus differences. Second, I added a sentence noting that the broader literature characterises the LPP as reflecting affective stimulus significance (citing Hajcak & Foti, 2020 and Olofsson et al., 2008), of which arousal is a major determinant.

This gives the reader a more nuanced picture: Schupp & Kirmse's specificity analysis supports an arousal interpretation, but the LPP likely reflects a broader construct of affective significance. This nuance matters because it tempers the assumption underlying the LPP-to-$\Phi_{\text{emot}}$ mapping, which the next sentences address directly.

### FIXME 8 (v2 line 47): $\Phi_{\text{emot}}$ as arousal — hidden assumptions

The sentence "These results support the use of the LPP to constrain the $\Phi_{\text{emot}}$ parameter" contained a hidden assumption: that emotional arousal drives the encoding-strength advantage captured by $\Phi_{\text{emot}}$. This assumption was never stated, making the logic of the LPP–$\Phi_{\text{emot}}$ mapping opaque. The FIXME further noted that alternative formalisations exist — CMR3 represents arousal as a source feature rather than as an encoding-strength modulator — and suggested making this explicit for theoretical punchiness.

I added two sentences after the transition sentence. The first makes the assumption explicit: the mapping is justified "under the assumption that emotional arousal drives the encoding-strength advantage that $\Phi_{\text{emot}}$ captures." The second notes that alternative formalisations exist (more recent retrieved-context models represent arousal as a source-context feature), which would motivate different neural-to-parameter mappings.

Making the assumption explicit turns a potential weakness into a theoretical contribution. The reader now understands the interpretive commitment involved in the LPP-to-$\Phi_{\text{emot}}$ mapping and can see that future work testing alternative formalisations (e.g., mapping LPP to a source-context arousal dimension) would be a natural extension. This is precisely the "theoretical punchiness" the FIXME called for.

### FIXME 9 (v2 line 53): sentence too dense; jargon

The sentence about how intentional encoding could dilute the LPP packed three distinct mechanisms into a single clause: bias away from "intra-stimulus processing," adoption of strategies that minimise LPP differences, and "covert spaced rehearsal." The FIXME noted that these terms are jargon and the proposal deserved more unpacking.

I broke the sentence into four. The first states the general claim (intentional encoding could dilute the emotional modulation of the LPP). The second describes the attention-shifting mechanism in plain language (participants shift from processing each stimulus in isolation to comparing or organising stimuli). The third covers strategy adoption and rehearsal effects. The fourth has been revised separately (see FIXME 10 below).

The revised text replaces technical terms with accessible descriptions. "Intra-stimulus processing" becomes "processing each stimulus in isolation"; "inter-stimulus relationships" becomes "comparing stimuli or organising them for later recall"; "covert spaced rehearsal" becomes "rehearsal that redistributes attention more evenly across stimulus types." Each mechanism now gets enough room to be understood on its own.

### FIXME 10 (v2 line 55): more evidence for encoding mode claim

The claim that encoding mode can disrupt LPP-relevant processing cited only Healey & Kahana (2019), which is about temporal contiguity effects in recall — a related but indirect piece of evidence. The FIXME asked for more direct evidence linking encoding task to LPP modulation.

I replaced the sentence with one that cites two complementary pieces of evidence: Schindler & Kissler (2020), who showed that task relevance modulates emotional ERP effects specifically for the LPP, and Healey & Kahana (2019) for the broader point that encoding instructions alter recall organisation. The sentence now reads: "The potential impact of encoding mode is supported by evidence that task demands modulate emotional ERP effects and that encoding instructions alter the organisation of recall."

Schindler & Kissler (2020) provides the direct LPP evidence the FIXME asked for — they demonstrated that emotional LPP effects are present only when attention is directed toward emotional content, which supports the claim that task instructions can modulate the LPP. The Healey & Kahana citation is retained for the broader encoding-mode argument. No direct study of intentional encoding instructions diluting the LPP was found, but the revised wording accurately characterises the available evidence without overclaiming.

### FIXME 11 (v2 line 58): replication sentence too long

The sentence listing replication benefits was over 60 words long and mixed the argument (replication value) with methodological details (sample size, stimuli, duration) in a way that was difficult to parse. The FIXME suggested breaking it up.

I split it into two sentences. The first states the argument: replication in a different setup would increase confidence in the generality of the findings. The second lists the specific differences between our setup and Schupp & Kirmse's: different stimulus set, longer presentation duration (2 s vs. 150 ms), and a large independent sample.

The split separates the "why replicate" argument from the "how our setup differs" details. The prose also no longer includes the parenthetical about "150ms at in Schupp and colleagues' experiments" which contained a grammatical error ("at in").

### FIXME 12 (v2 line 64): attention–memory–LPP claims need sourcing

Three consecutive sentences made strong empirical claims without citations: (1) attention and memory are coupled, (2) emotional attention drives emotional memory, (3) arousal increases both LPP and memory. These are well-established findings but need to be anchored in the literature.

I added citations to each sentence: Chun & Turk-Browne (2007) for the general attention–memory coupling; Talmi (2013) and Mather & Sutherland (2011) for emotional attention as a driver of enhanced memory; and Dolcos & Cabeza (2002) for a study demonstrating both LPP effects and subsequent memory effects for emotional pictures.

The citations span the key literatures: cognitive attention–memory interactions (Chun & Turk-Browne), emotional memory mechanisms (Talmi; Mather & Sutherland), and the specific intersection of LPP and memory (Dolcos & Cabeza). Each claim is now individually anchored.

### FIXME 13 (v2 line 66): is the LPP–memory consistency really unknown?

The sentence "it is not known whether the relationship between them is consistent either within-subject or between-subjects" made a novelty claim that the FIXME questioned. If prior work had established this relationship, the claim would be false and the motivation for the study would be undermined.

I verified through literature search that the claim is substantially correct. Fields (2023), in a recent review of the LPP's functional significance, explicitly notes that only a few studies have directly examined the correlation between the LPP and later memory. I revised the sentence to: "Yet direct tests of whether the LPP–memory relationship is consistent within or between subjects are scarce," citing Fields (2023).

The revised wording is softer than "it is not known" — it acknowledges that some relevant work exists while correctly identifying it as scarce. The Fields citation gives the reader a recent review that documents this gap, which is stronger than an unsupported novelty claim.

### FIXME 14 (v2 line 69): within/between distinction could be clearer

The two possibilities (between-subject correlation vs within-subject list-level correlation) were presented as "one possibility" and "another possibility" without labelling which level of analysis each referred to. The reader had to infer from the phrasing whether "individuals with increased LPP differences" meant a between-subjects comparison or a within-subject comparison.

I replaced the generic framing with explicit level labels: "At the between-subjects level" and "At the within-subject level." The predictions are otherwise unchanged.

This minimal revision makes the two levels of analysis immediately clear and sets up the subsequent discussion of within- vs between-subject ERP effects without requiring the reader to reconstruct the distinction from context.

### FIXME 15 (v2 line 71): more ERP within/between dissociation examples

The dissociation between within- and between-subject ERP effects was illustrated only by MacLeod & Donaldson (2017) on the left parietal old/new effect. The FIXME asked for additional examples.

I added Weinberg et al. (2021), who demonstrated that the LPP's emotional modulation is stable within individuals across five testing sessions. I incorporated this as a new sentence: "Although the emotional modulation of the LPP is itself reliable within individuals across testing sessions, it is unknown whether this reliability extends to predicting memory differences."

Weinberg et al. (2021) is particularly relevant because it establishes within-subject reliability of the LPP — a necessary precondition for the within-subject analyses in this paper. The sentence also bridges to the novelty claim by noting that reliability of the LPP effect per se does not guarantee predictive validity for memory. No additional examples of within/between ERP dissociations beyond MacLeod & Donaldson were found in the literature, which itself underscores that this pattern is under-explored.

### FIXME 16 (v2 line 73): "we already said this — now there's a figure?"

The sentence "Simulations with eCMR show that all else held equal, increasing the value of $\Phi_{\text{emot}}$ results in increased recall of emotional stimuli (Figure 1)" repeated information from the preceding paragraph about $\Phi_{\text{emot}}$ values and emotional processing. The figure reference appeared as an afterthought attached to a redundant claim.

I replaced the sentence with: "As illustrated in Figure 1, eCMR predicts that larger values of $\Phi_{\text{emot}}$ yield greater recall of emotional relative to neutral stimuli." This uses the figure as the leading element rather than the restated claim.

The revised sentence serves as a transition: it introduces the figure and frames the prediction that the next sentence will test against LPP data. The earlier mention of $\Phi_{\text{emot}}$ described the parameter's role; this sentence describes the model's prediction. The "as illustrated" construction also tells the reader the figure exists for a reason — it visually demonstrates the prediction — rather than tacking the figure reference onto a restated fact.

### FIXME 17 (v2 line 85): model comparison enumeration questionable

The original text mapped "LPP reflects attention or working memory processes unrelated to emotion" to the LPP-only model and "LPP reflects arousal which mainly is elicited by negative items" to the interaction model. The FIXME questioned both the enumeration and the theoretical mapping. The phrasing also implied that the LPP-only model tests whether LPP reflects general attention, but this conflates the model structure (main effect vs interaction) with the theoretical interpretation (emotion-general vs emotion-specific processing).

I rewrote the three sentences to focus on what the model comparison actually tests. If the LPP captures encoding processes that operate similarly for emotional and neutral items, the main-effect model should account for the data as well as the interaction model. If the LPP primarily reflects emotion-specific processing — such that its influence on encoding strength is larger for emotional items — the interaction model should provide the best fit.

The revised predictions are cleaner because they map directly to the model structures without invoking specific cognitive constructs (attention, working memory, arousal) that go beyond what the comparison can adjudicate. The comparison tests whether the LPP's effect on encoding is emotion-general or emotion-specific — it cannot determine whether the underlying construct is attention, arousal, or something else. This more modest framing is both more accurate and more defensible.

### Bibliography additions

Seventeen new .bib entries were added to support the citations introduced above. Total references increased from 30 to 47. Key additions include foundational retrieved-context theory papers (Sederberg et al., 2008; Lohnas et al., 2015; Healey & Kahana, 2016; Manning et al., 2011; Kahana, 2012), LPP review articles (Hajcak et al., 2010; Hajcak & Foti, 2020; Olofsson et al., 2008), and attention–memory coupling references (Chun & Turk-Browne, 2007; Talmi, 2013; Mather & Sutherland, 2011; Dolcos & Cabeza, 2002).
