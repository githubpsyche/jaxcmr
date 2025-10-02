SPECIFIC AIMS

Comprehension is memory‑in‑use: readers construct situation models by retrieving and reinstating information from earlier in the text when current input overlaps with it [@mckoon1992inference; @kintsch1998comprehension; @zwaan1995dimensions]. Because access to prior information is controlled by the mechanisms of episodic memory search, a theory that leaves those mechanisms underspecified cannot predict which nonadjacent details will be available when and, therefore, which inferences will be drawn. Yet research on episodic memory and on narrative comprehension has proceeded on separate tracks, limiting theory and preventing unified accounts that tie recall dynamics to inference‑based understanding [@kahana2020computational]. A retrieved‑context framework (TCM/CMR) specifies how internal context evolves and is reinstated to cue retrieval, offering the needed mechanistic bridge [@howard2002distributed; @polyn2009context].

<!-- 
I don't think it's a good idea to imply that inference is realized simply as selective reinstatement of episodic context. 
Simply recalling a piece of information related to the current information and associating the two in memory is a necessary but insufficient condition for inference -- to understand how the piece of information retrieved relates to what is being encoded now. 

The sentence "Comprehension is memory‑in‑use: readers construct situation models by retrieving and reinstating information from earlier in the text when current input overlaps with it" also miscasts comprehension as almost entirely involving retrieving relevant information to match the current moment. "Access to prior information" as you describe it can only be a necessary but not a sufficient step in comprehension, so the language here needs reivsion too.

You allude that reinstatement could be "conditioned on overlap and coherence constraints"; maybe something along these lines works. 
But as described, this stuff seems ancillary / easy to miss and may need clarity about the role of these constraints in encoding or retrieval. 
One idea based on minimalist accounts of narrative comprehension in the review I'm sharing (it would maybe help to delv into current research on the same papers and/or refer to these details to extent available in room) is that context reinstates information according to dynamics specified by RCT, but inference is a process that operates over idea units active in context (maybe in proportion to their activity), evaluating the relevance of retrieved information to the current information and gating memory integration to enforce coherence to extent constrained by memory limitations. 
I don't know if this is the idea we should propose or if we should propose an idea so concretely. 
Either way, we must seek to tackle this distinction head-on in an integrated model and robust research program.
Help me figure out how the strategy around this.

Also, I wonder if the last sentence can be stronger. Why retrieved-context theory instead of another memory model? Presumably because it's such a successful framework, has been specified in enough detail to offer highly specific predictions about performance that leave it amenable to evaluation, and it natively addresses how memory can transform a set of items with varying relatedness to one another into a structured representation when they are processed sequentially and then search this memory representation either freely or in a directed fashion to recall items and their contextual details. 
 -->

The goal of this project is to develop and test a computational model of narrative comprehension and memory that implements selective context reinstatement under retrieved‑context theory (RCT) to explain both recall structure and inference/question‑answering. This model challenges the implicit assumption that inference is a separate, strategy‑first process: instead, inference is realized as selective reinstatement of episodic context conditioned on overlap and coherence constraints [@graesser1994constructing; @van1999landscape]. By unifying episodic search with situation‑model construction, the project provides a process‑level account of how semantic/protagonist/causal overlaps and narrative structure (e.g., event boundaries, goal shifts) determine which links are formed, when, and with what behavioral consequences [@zacks2007event; @dubrow2016temporal].

<!-- 
I feel like we need a broader account of the scope of this investigation than just recall structure and inference/question-answering. "Recall structure" is sufficiently broad for the free recall side and even the narrative recall DV. In the comprehension space, there are lots of other DVs relevant for measuring inference success/understanding than just question-answering. We need to work toward a more convincing account of how we'll probe and distinguish inference-making artifacts from memory artifacts to really have a rigorous design.

The rest seems subtly but substantially flawed as an account of the current literature and 
It is not clearly true that it is a common "implicit assumption" to treat inference as a distinct, strategy‑initiated process decoupled from episodic retrieval.
If we want to describe a common but dubious belief about how comprehension works, then work convincingly indicative this belief should be cited and possibly described along with the claim.
Instead though, I think it's quite openly debated rather than implicitly assumed in the text comprehension literature that inference is a memory-dependent behavior. For example, an influential ongoing disagreement in the literature is between constructionist and minimalist accounts of memory which respectively argue that memory availability does or does not reliably gate inference-drawing during reading comprehension. I'll share my review of this literature again to remind you. However, you may need to pursue follow-up research yourself to clarify what the current state of the art is if the research discussed in my draft does not cover state-of-the-art research activity.

As before, I don't think it's a good idea to imply that inference is realized simply as selective reinstatement of episodic context. 
Simply recalling a piece of information related to the current information and associating the two in memory is a necessary but insufficient condition for inference -- to understand how the piece of information retrieved relates to what is being encoded now. 
You allude that reinstatement could be "conditioned on overlap and coherence constraints"; maybe something along these lines works. 
But as described, this stuff seems ancillary / easy to miss and may need clarity about the role of these constraints in encoding or retrieval. 
One idea based on minimalist accounts of narrative comprehension in the review I'm sharing (it would maybe help to delv into current research on the same papers and/or refer to these details to extent available in room) is that context reinstates information according to dynamics specified by RCT, but inference is a process that operates over idea units active in context (maybe in proportion to their activity), evaluating the relevance of retrieved information to the current information and gating memory integration to enforce coherence to extent constrained by memory limitations. 
I don't know if this is the idea we should propose or if we should propose an idea so concretely. 
Either way, we must seek to tackle this distinction head-on in an integrated model and robust research program.
Help me figure out how to correspondingly strategize this specific aims section.

Finally, we should work on our characterization of *how* we propose to accomplish our research goals. In this paragraph, you suggest we'll achieve by "unifying episodic search with situation‑model construction". I don't necessarily see this as the objective

Finally, I again think we need to do better than frame proposed work as achieving its goals "by reaching across the divide between episodic memory search and discourse comprehension". It is redundant and meaningless to say that I seek to a principled unification of episodic memory search and discourse comprehension "by reaching across the divide between episodic memory search and discourse comprehension" -- the reaching is implied by the goal. We need to say something more concrete about our approach here that forecasts how we're transforming the field with refined theory and methodology. 
 -->

Our approach is to collect behavioral data in narrative recall and comprehension (free recall sequences, online bridging/causal probes, and delayed question answering) and to iteratively refine a computational model (Fig. 1) of both performance and latent reinstatement dynamics. The core model is a long‑standing, well‑specified retrieved‑context account (TCM/CMR) in which a gradually evolving context representation is bound to each studied unit and reinstated to cue what comes next [@howard2002distributed; @polyn2009context]. We adapt this machinery to sentences/propositions and show how its temporal targeting and overlap sensitivity reinterpret benchmark narrative phenomena (contiguity of recalls, boundary effects on order, causal centrality in memory and Q&A) without invoking a separate inference engine [@sederberg2008context; @healey2019contiguity; @zacks2007event; @dubrow2016temporal; @graesser1994constructing]. The experiments are designed to challenge the framework, adjudicating selective‑ vs blended‑reinstatement variants and quantifying how coherence constraints gate inference [@mckoon1992inference; @kintsch1998comprehension; @van1999landscape]. We will deploy this approach to accomplish the following aims:

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