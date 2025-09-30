Retrieved‑context theory (RCT) explains how episodic cues guide memory search by letting each studied or recalled item inject its contextual features into a continually evolving representation. 
Most implementations adopt a connectionist architecture -- typified by the Context Maintenance and Retrieval model (CMR) -- where item–context weights accumulated across events blend in composite stores.
This dissertation demonstrates that RCT is architecturally portable, shows that task success hinges on algorithmic rather than architectural choices, and applies these insights to identify and resolve a failure of its item-based contextual evolution mechanism to avoid associative interference across repetitions.

Chapter 1 introduces Instance‑CMR (ICMR), an instance‑based reformulation that stores every item–context pairing as a separate trace. 
When its parameters collapse those traces into a composite contextual input, ICMR can reproduce CMR's behaviour exactly, dissolving architecture as a theoretical boundary for RCT.

Chapter 2 leverages this equivalence to compare CMR, a connectionist model designed for free recall, with the Context Retrieval and Updating model (CRU), an instance‑based serial recall model.
A factorial model‑selection analysis across free‑ and serial‑recall datasets shows that the models' divergent successes stem not from architectural differences but from algorithmic factors like item confusability, context‑drift rules, primacy/recency mechanisms, and retrieval‑termination criteria. 
The exercise yields a common suite of benchmarks and model features across task domains that the final chapter exploits.

Chapter 3 tests an assumption shared by most retrieved‑context models: that repeated items always reinstate overlapping contextual features, a rule that produces associative interference across occurrence contexts.
Re-analyzing six free- and serial-recall datasets, I find no evidence of such interference, contradicting predictions by either CMR or CRU.
Within ICMR, allowing repetitions to reinstate non‑overlapping features and letting traces compete during retrieval for reinstatement eliminates the interference while preserving fits to key benchmarks, all without adding parameters.

Collectively, the three chapters show that episodic memory can balance integration and specificity by storing distinct traces and adjudicating among them at retrieval, thereby retaining RCT's strengths while overcoming the limitations of composite, connectionist implementations.