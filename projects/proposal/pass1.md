SPECIFIC AIMS
Comprehension is an interaction between memory access and coherence maintenance: readers build situation models by retrieving and reinstating prior information and then evaluating how it fits with the current input [@mckoon1992inference; @kintsch1998comprehension; @zwaan1995dimensions]. Because access to prior information is controlled by the mechanisms of episodic memory search, theories that leave those mechanisms underspecified cannot predict the timing and selectivity of inference. Conversely, access alone is insufficient for understanding; a coherence gate must determine whether retrieved content is integrated. Yet research on episodic memory and on narrative comprehension has often proceeded on separate tracks, limiting theory and preventing unified accounts that tie recall dynamics to inference‑based understanding [@kahana2020computational]. A retrieved‑context framework (TCM/CMR) specifies how internal context evolves and is reinstated to cue retrieval, yielding quantitative predictions for availability over time; coupled with an explicit coherence gate, it provides the needed mechanistic bridge [@howard2002distributed; @polyn2009context].

The goal of this project is to develop and test a computational model of narrative comprehension and memory in which retrieved‑context dynamics determine access to prior ideas and interact with a coherence mechanism to determine their integration into an evolving situation model. In this account, inference is not reducible to reinstatement: rather, selective context reinstatement supplies candidates in proportion to overlap and recency, and a coherence‑evaluation process (sensitive to causal/protagonist structure, goals, and discourse cues) gates integration under capacity constraints [@graesser1994constructing; @van1999landscape]. This integration explains which links are formed, when, and with what behavioral consequences in both recall sequences and tests of understanding, without invoking an ad hoc inference engine separate from memory.

<!-- 
This is strong, and yet I sense that the first sentence and second sentence here repeat a lot of the same details. It may be better to say something less redundant.

One thing I miss is the characterization of an influential alternative viewpoint that the present work challenges. Previous drafts of this paragraph cited what might or might not have been a strawman (an influential proponent of the view was not be identified). This draft abandons that kind of contrastive demonstration completely. Is this for the best? 

Another thing I miss from the reference doc is a line that occurs after the concrete statement of empirical consequences of this work to address. In the reference doc, the sentence was "This theoretical unification will lay the foundation for a comprehensive model of human memory that will provide new insights into the continuity of working memory and long-term memory deficits in memory disorders and cognitive impairments in aging."
This seems to assert broader high-impact consequences of the proposed work for both psychological theory and interpreting relevant disorders and impairments -- "real-world problems". 
A proposal for a model of reading comprehension and memory can probably spin up a similar sentence to prosecute the urgency and transformative nature of the proposed work.
 -->

Our approach is to collect behavioral data in narrative recall and comprehension (free recall sequences, online bridging/causal probes, and delayed question answering) and to iteratively refine a computational model (Fig. 1) of both performance and latent reinstatement dynamics. The core model is a long‑standing, well‑specified retrieved‑context account (TCM/CMR) in which a gradually evolving context representation is bound to each studied unit and reinstated to cue what comes next [@howard2002distributed; @polyn2009context]. We adapt this machinery to sentences/propositions and show how its temporal targeting and overlap sensitivity reinterpret benchmark narrative phenomena (contiguity of recalls, boundary effects on order, causal centrality in memory and Q&A) without invoking a separate inference engine [@sederberg2008context; @healey2019contiguity; @zacks2007event; @dubrow2016temporal; @graesser1994constructing]. The experiments are designed to challenge the framework, adjudicating selective‑ vs blended‑reinstatement variants and quantifying how coherence constraints gate inference [@mckoon1992inference; @kintsch1998comprehension; @van1999landscape]. We will deploy this approach to accomplish the following aims:

<!-- If last paragraph needed to give a sense of the research problem and why its study is impactful and urgent, I think here we need give a sense of the urgency and impact of our research approach given the state of the art in the field. -->

<!-- Downstream: maybe meta-cognitive control mechanisms specified here can be lined back to the study of free/serial recall as mechanisms that can explain how people monitor and organize performance on these tasks, or (more significantly spun) how such mechanisms may be engaged more broadly in memory processes beyond text comprehension. -->

<!-- 
I don't like the "without a separate inference engine".
Furthermore, I don't think it's a good idea to imply that inference can be characterized entirely in terms of memory coactivation.
Simply recalling a piece of information related to the current information and associating the two in memory is a necessary but insufficient condition for inference -- to understand how the piece of information retrieved relates to what is being encoded now. 
The goal of our proposal is to dissect the distinct contributions of memory and inference to the formation of coherent text representations, to clarify how how retrieval configures inference and vice versa to support comprehension.
Next, I don't care about the adjudication of selective‑ vs blended‑reinstatement variants in this proposal. 
That was a central theme of my thesis, but it's a less focal issue in this investigation. 
The contributions from my thesis relevant for this proposal have to deal with its experimental and analysis methodology rather than distinctions between the specific models considered and evaluated.
This paragraph repeats content from earlier paragraph, a list of phenomena/manipulations to address. We need to think about whether that's a good use of space or whether the two ideas can be more clearly distinguishes to avoid creating the appearance of redundancy.

There's a phrase "he experiments are designed to challenge the framework", but "the framework" has not been meaningfully identified by this point. I assume it's referring to the notion that inference is independent of memory, but that is a straw man without evidence confirming that the view is influential in 2025 and defines a substantive alternative theory with concrete theoretical consequences that can diverge from RCT.
 -->

Aim 1: Test the selective‑reinstatement account of inference in narratives.
In controlled stories where key content is repeated or replaced with overlap pairs (semantic similarity; shared protagonist via coreference; explicit causal links), we will measure sentence‑level sequence signatures (dual‑center lag‑CRPs; cross‑occurrence neighbor analyses; forward‑chaining analogs) and inference behavior (bridging/causal probes during and after reading). Selective‑reinstatement RCT predicts first‑over‑second preferences and graded integration (causal > protagonist > semantic) tied to probe success; blended CMR predicts balanced access and broader cross‑neighborhood linkage. Preliminary narrative‑recall analyses and Semantic‑CMR fits included.

<!-- 
First, I don't care about the adjudication of selective‑ vs blended‑reinstatement variants in this proposal. 
That was a central theme of my thesis, but it's a less focal issue in this investigation even if my findings may be 
We are testing a retrieved-context account of narrative comprehension and 
 of accessibility in narrative comprehension
-->

Aim 2: Demonstrate the joint constraints between memory and inference using a linking model.
We will develop a linking model from latent context state to both next‑step recall/inference and probe accuracy/RT, estimating shared parameters for reinstatement strength and coherence sensitivity on joint datasets (sequences + probes). We will design double‑dissociation tests that separate memory availability (reinstatement makes information accessible) from inference commitment (coherence gate accepts/rejects the link), and use cross‑validation and posterior predictive checks to arbitrate selective vs blended architectures [@kahana2020computational; @wilson2019ten]. Preliminary feasibility analyses for joint fitting included.

Aim 3: Establish how narrative structure modulates reinstatement and inference (event boundaries and beyond).
We will orthogonally manipulate event boundaries (scene/time/space changes), goal/intent shifts, and cohesion cues while holding overlap fixed. The model predicts boundary‑gated reinstatement—attenuated across‑boundary integration rescued by strong causal/goal links—whereas blended reinstatement predicts broader cross‑boundary linkage [@zacks2007event; @dubrow2016temporal; @graesser1994constructing]. We will quantify structure × overlap interactions on both sequence signatures and comprehension outcomes. Exploratory boundary‑sensitive sequence analyses included.

Figure 1. Iterative cycle of model‑derived paradigm development and behavioral data collection (recall sequences, probes, Q&A), enabling empirical testing of model variants to distinguish selective vs blended reinstatement and to quantify coherence gating in narrative comprehension.