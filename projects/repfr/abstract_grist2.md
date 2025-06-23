Repetition both strengthens memories and links them to an evolving temporal context that reflects recent experience [@delaney2010spacing].
According to retrieved‑context theory (RCT), this evolution is item‑based: encountering an item reactivates and blends the composite of its prior contexts into the ongoing state, linking moments through overlapping features [@howard2002distributed; @siegel2014retrieved].
The Context Maintenance and Retrieval (CMR) model implements RCT in a connectionist architecture and captures many memory benchmarks [@polyn2009context;@healey2019contiguity], yet repetition tests have typically examined a single task domain and applied control analyses that under‑count baseline transitions.

Re-examining six free‑ and serial‑recall datasets, we uncovered three diagnostic signatures:
(i) no surplus transitions between neighbors of different occurrences,
(ii) biased transitions from repeaters toward first-occurrence neighbors, and
(iii) preserved forward chaining from repeaters in serial recall.
Standard CMR fails these because it blends occurrence contexts during encoding and upon retrieval, producing associative interference that predicts balanced cross‑occurrence transitions.

We introduce Instance‑CMR, an implementation of RCT that encodes item-context associations as separate episode traces.
Configured with the same blending assumptions, Instance-CMR functionally reproduces standard CMR.
When repetitions instead reinstate non-overlapping contextual features and traces compete at recall for reinstatement, Instance-CMR eliminates interference and satisfies key constraints, yielding higher sequence likelihood fits across datasets without adding parameters.
These results demonstrate RCT's portability across architectures but challenge its blending assumptions, recasting repetition memory as selection between episode‑specific, non‑overlapping contexts.
