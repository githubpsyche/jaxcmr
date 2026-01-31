Repetition strengthens memories while tying each occurrence to evolving temporal context.
According to retrieved‑context theory (RCT), this evolution is item‑based: encoding or retrieving an item blends contextual features specific to it into the ongoing state.
Such blending links occurrences across time, but also engenders associative interference as recall cues distribute activation more evenly across each occurrence's neighbors.
The Context Maintenance and Retrieval (CMR) and other models formalize RCT and capture many classic benchmarks.
However, repetition tests typically focus on individual domains and use ambiguous control comparisons.

We re-examined six free- and serial-recall datasets.
Whereas associative interference in CMR predicts balanced cross-occurrence transition rates across tasks, we identify three contradictory patterns:
(i) no surplus transitions between neighbors belonging to different occurrences,
(ii) a bias from repeated items toward neighbors of their first occurrence, and
(iii) preserved forward chaining from repeated items in serial recall.

To probe these discrepancies, we introduce Instance‑CMR, a trace-based framework that stores item-context pairings separately and is configurable to either reproduce CMR's behavior or test alternative assumptions.
In the successful configuration, repetitions reinstate unique, non-overlapping contextual features and traces compete during retrieval for reinstatement.
Together these mechanisms eliminate interference and improve sequence‑likelihood fits without adding parameters.
Instance‑CMR thus unifies free‑ and serial‑recall phenomena and generalizes RCT beyond a composite framework.
Episodic memory, we argue, balances integration and specificity not by blending contexts, but by letting distinct contexts compete at retrieval.

---

Repetition both strengthens memories and links them to an evolving temporal context.
According to retrieved‑context theory (RCT), this evolution is item‑based: encoding or retrieving an item blends the composite of its prior contexts into the ongoing state.
Such blending links occurrences across time, but also engenders associative interference as recall cues distribute activation more evenly across each occurrence's neighbors.
The Context Maintenance and Retrieval (CMR) model formalizes RCT in a connectionist architecture and captures classic benchmarks.
However, repetition tests typically focus on individual domains and use ambiguous control comparisons.

We re-examined six free- and serial-recall datasets.
Whereas associative interference in CMR predicts balanced cross-occurrence transition rates across tasks, we identify three contradictory patterns:
(i) no surplus transitions between neighbors belonging to different occurrences,
(ii) a bias from repeated items toward neighbors of their first occurrence, and
(iii) preserved forward chaining from repeated items in serial recall.

To probe these discrepancies, we introduce Instance‑CMR, a trace-based framework that stores item-context pairings separately and is configurable to either reproduce CMR's behavior or test alternative assumptions.
In the successful configuration, repetitions reinstate unique, non-overlapping contextual features and traces compete during retrieval for reinstatement.
Together these mechanisms eliminate interference and improve sequence‑likelihood fits without adding parameters.
Instance‑CMR thus unifies free‑ and serial‑recall phenomena and generalizes RCT beyond a composite framework.
Episodic memory, we argue, balances integration and specificity not by blending contexts, but by letting distinct contexts compete at retrieval.

Episodic memory, we argue, balances integration with specificity by allowing distinct contexts to vie for reinstatement.
---

Repetition both strengthens memories and links them to an evolving temporal context.
According to retrieved‑context theory (RCT), this evolution is item‑based: encoding or retrieving an item blends the composite of its prior contexts into the ongoing state.
Such blending links occurrences across time, but also engenders associative interference as recall cues distribute activation more evenly across each occurrence's neighbors.
The Context Maintenance and Retrieval (CMR) model formalizes RCT and captures many classic benchmarks.
However, repetition tests typically focus on individual task domains and use ambiguous control comparisons.

We re-examined six datasets spanning free and serial recall.
Whereas CMR predicts balanced cross-occurrence transitions in these tasks, we identify three contradictory patterns:
(i) no surplus transitions between neighbors belonging to different occurrences,
(ii) a bias from repeated items toward neighbors of their first occurrence, and
(iii) preserved forward chaining from repeated items in serial recall.

We introduce Instance‑CMR, a trace-based framework that stores item-context pairings separately and is configurable to either reproduce CMR's behavior or test alternative assumptions.
In the successful configuration, (a) repetitions reinstate unique contextual features, and (b) traces compete during retrieval for reinstatement.
Together these mechanisms eliminate interference and improve sequence‑likelihood fits without adding parameters.
These findings demonstrate RCT's architectural portability but challenge its blending assumptions, casting repetition memory as competition among distinct episodic contexts.
This reframing unifies free‑ and serial‑recall phenomena and invites a broader reconsideration of how episodic memory balances integration with specificity.


We re-examined six free‑ and serial‑recall datasets.
Whereas CMR predicts balanced cross‑occurrence transitions, we uncovered three contradictory patterns:
---

Only the configuration combining non‑overlapping context reinstatement with competition between traces for retrieval eliminates interference and satisfies key constraints, yielding higher sequence‑likelihood fits without extra parameters.


---

Repetition both strengthens memories and links them to an evolving temporal context [@delaney2010spacing].
According to retrieved‑context theory (RCT), this evolution is item‑based: encoding or retrieving an item blends the composite of its prior contexts into the ongoing state [@howard2002distributed; @siegel2014retrieved].
Such blending links occurrences across time, but also engenders associative interference as recall cues spread activation more evenly across neighbors from each occurrence.
The Context Maintenance and Retrieval (CMR) model formalizes these mechanisms and captures many classic benchmarks [@polyn2009context;@healey2019contiguity], but repetition tests frequently have narrow scope and apply ambiguous control comparisons.

We re‑analysed six free‑ and serial‑recall datasets with position‑balanced controls. 
Whereas CMR predicts balanced cross‑occurrence transitions, we uncovered three contradictory signatures:
(i) no surplus transitions between neighbors belonging to different occurrences,
(ii) a bias from repeated items toward neighbors of their first occurrence, and
(iii) preserved forward chaining from repeated items in serial recall.

We introduce Instance‑CMR, an implementation of RCT that stores each item-context pairing separately and can either reproduce CMR's behavior or test alternative assumptions.
Only the configuration combining non‑overlapping context reinstatement with competition between traces for retrieval eliminates interference and satisfies key constraints, yielding higher sequence‑likelihood fits without extra parameters.
These findings demonstrate RCT's architectural portability but challenge its blending assumptions: repetition memory is better conceived as competition among distinct, episode-specific contexts.
This reframing unifies free‑ and serial‑recall phenomena and invites a broader reconsideration of how episodic memory balances integration with specificity.




To address this, we explore the Context Maintenance and Retrieval (CMR) model, which implements RCT in a connectionist architecture and captures many classic benchmarks [@polyn2009context;@healey2019contiguity].
Across six free‑ and serial‑recall datasets, we identify three diagnostic patterns that challenge these blending assumptions:
(i) no surplus transitions between neighbors drawn from different occurrences,
(ii) a bias from repeated items toward neighbors of their first appearance, and
(iii) intact forward chaining from repeaters in serial recall.

We introduce Instance‑CMR, an RCT implementation that stores item-context pairs separately and can either reproduce CMR's behavior or test alternative assumptions.


The Context Maintenance and Retrieval (CMR) model implements RCT in a connectionist architecture and captures many classic benchmarks [@polyn2009context;@healey2019contiguity].
Yet it 

Because that composite is used at both encoding and retrieval, two sources of associative interference arise: (i) the new trace shares overlapping context features with earlier traces, and (ii) the reinstated composite cue activates neighbours of every occurrence equally, predicting balanced cross‑occurrence transitions.
Across six free‑ and serial‑recall datasets we instead observe three diagnostic signatures: (i) no surplus transitions between neighbours from different repetitions, (ii) a bias toward neighbours of the first presentation, and (iii) preserved forward chaining from repeated items in serial recall.
We introduce Instance‑CMR, which stores each item–context pairing separately and explores two independent mechanisms: whether repetitions reinstate overlapping versus non‑overlapping contextual features, and whether retrieval adds or competes among traces. Only the configuration combining non‑overlapping reinstatement with competitive retrieval eliminates interference and accounts for all three empirical signatures, providing higher sequence‑likelihood fits than standard CMR without adding parameters.
These results demonstrate that contextual reinstatement need not entail composite blending; repetition memory is better conceptualised as competition among distinct, episode‑specific contexts. By bridging free‑ and serial‑recall phenomena, the work clarifies how episodic memory balances integration with specificity and offers a general tool for testing alternative context‑retrieval mechanisms.

---

Repetition both strengthens memories and links them to an evolving temporal context that reflects recent experience [@delaney2010spacing].
According to retrieved‑context theory (RCT), this evolution is item‑based: encoding or retrieving an item blends the composite of its prior contexts into the ongoing state, linking occurrences across time [@howard2002distributed; @siegel2014retrieved].
The Context Maintenance and Retrieval (CMR) model implements RCT in a connectionist architecture and captures many classic benchmarks [@polyn2009context;@healey2019contiguity].
Yet across six free‑ and serial‑recall datasets we identify three diagnostic patterns that challenge this composite framework: 
(i) no surplus transitions between neighbors drawn from different occurrences, 
(ii) a bias from repeated items toward neighbors of their first appearance, and 
(iii) intact forward chaining from repeaters in serial recall.

We introduce Instance‑CMR, which stores each item–context pairing separately and explores two independent mechanisms: whether repetitions reinstate overlapping versus non‑overlapping contextual features, and whether distinct episode traces compete or blend for reinstatement at retrieval.
Only the configuration combining non‑overlapping reinstatement with competitive retrieval eliminates interference and accounts for all three empirical signatures, providing higher sequence‑likelihood fits than standard CMR without adding parameters.
These results demonstrate that contextual reinstatement need not entail composite blending; repetition memory is better conceptualised as competition among distinct, episode‑specific contexts. By bridging free‑ and serial‑recall phenomena, the work clarifies how episodic memory balances integration with specificity and offers a general tool for testing alternative context‑retrieval mechanisms.

We introduce Instance‑CMR, an implementation of RCT that stores item-context pairs separately and can either reproduce CMR's behavior or test alternative assumptions.
Standard CMR fails because overlap at encoding and composite reinstatement at retrieval produces associative interference that predicts balanced cross‑occurrence transitions.
When repetitions instead reinstate non-overlapping contextual features and traces compete at recall for reinstatement, Instance‑CMR eliminates associative interference and satisfies key constraints, yielding better fits without extra parameters.
These results demonstrate RCT's architectural portability but challenge its blending assumptions; instead repetition memory is better conceived as competition among distinct, non‑overlapping contexts.
By bridging free‑ and serial‑recall phenomena, this work clarifies how episodic memory balances integration with specificity and provides a general tool for testing alternative context‑retrieval mechanisms.

---

Repetition both strengthens memories and links them to an evolving temporal context that reflects recent experience [@delaney2010spacing].
According to retrieved‑context theory (RCT), this evolution is item‑based: encoding or retrieving an item blends the composite of its prior contexts into the ongoing state, linking occurrences across time [@howard2002distributed; @siegel2014retrieved].
The Context Maintenance and Retrieval (CMR) model implements RCT in a connectionist architecture and captures many classic benchmarks [@polyn2009context;@healey2019contiguity], yet repetition tests have been narrow in scope and used ambiguous control comparisons.
Yet it assumes that repetition produces contextual overlap at encoding and reinstates a composite context at retrieval, which leads to associative interference that predicts balanced cross‑occurrence transitions.
Across six free‑ and serial‑recall datasets we identify three diagnostic signatures that challenge this framework: 
(i) no surplus transitions between neighbors of different occurrences,
(ii) a bias from repeaters toward first-occurrence neighbors, and
(iii) preserved forward chaining from repeaters in serial recall.

To isolate the source of discrepancy: we develop Instance-CMR, an RCT implementation that stores item-context pairs separately and can either reproduce CMR's behavior or test alternative assumptions.
When repetitions reinstate non-overlapping contextual features and traces compete at recall for reinstatement, Instance‑CMR eliminates associative interference and satisfies key constraints, yielding higher sequence-likelihood fits without extra parameters.




Standard CMR fails these because overlap at encoding and composite reinstatement at retrieval produce associative interference that predicts balanced cross‑occurrence transitions.

We introduce Instance‑CMR, an implementation of RCT that stores item-context pairs separately and can can either reproduce CMR's composite behaviour or test alternative assumptions.
When repetitions instead reinstate non-overlapping contextual features and traces compete at recall for reinstatement, Instance‑CMR eliminates interference and satisfies key constraints, yielding higher sequence-likelihood fits without extra parameters.
These results demonstrate RCT's portability across architectures but challenge its blending assumptions, recasting repetition memory as the competitive selection between episode‑specific, non‑overlapping contexts.

---

Yet across six free‑ and serial‑recall datasets we identify three diagnostic patterns that conflict with this account:
(i) no surplus transitions between neighbors drawn from different occurrences, 
(ii) a bias from repeated items toward neighbors of their first appearance, and 
(iii) intact forward chaining from repeaters in serial recall.


Repeated items produce contextual overlap at encoding and reinstates a composite context at retrieval

Because CMR assumes that repeated items produce contextual overlap at encoding and reinstates a composite context at retrieval, these patterns suggest that CMR's blending assumptions are too strong.


We present Instance‑CMR, an RCT implementation that stores item-context pairs separately and can either reproduce CMR's behavior or test alternative assumptions.
Under CMR's blending assumptions, repetition produces contextual overlap at encoding and composite reinstatement at retrieval produces associative interference that predicts balanced cross‑occurrence transitions.
When repetitions instead reinstate non-overlapping contextual features and traces compete at recall for reinstatement, Instance‑CMR eliminates associative interference and satisfies key constraints, yielding better fits without extra parameters.
These results demonstrate RCT's portability across architectures but challenge its blending assumptions; instead repetition memory is better conceived as competition among distinct, non‑overlapping contexts.
This reframing unifies free‑ and serial‑recall phenomena and invites a broader reconsideration of how episodic memory balances integration with specificity.


<!-- one wonders if we need this whole sentence -->
<!-- However, repetition tests have typically focused on a single task domain and relied on control comparisons that leave open whether a composite framework accurately characterizes repetition memory. -->
<!-- Has good content but seems wordy. Why repeat "repetition"? -->