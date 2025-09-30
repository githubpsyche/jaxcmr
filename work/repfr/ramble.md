Rambly update on thesis stuff. Re-found a procedure for reproducing the abilities of a MINERVA-2 instance model within a connectionist memory. I think an old Hintzman paper already presents close to the same idea though:

1. In the linear memory, assign each memory trace a unique output feature. Use Hebbian learning directly rather than concatenating input and output features into vectors.
2. Outside the linear memory, maintain a mapping vector that links these unique output features to behavioral responses, which may overlap across distinct traces (such as when study items are repeated).

Probing the linear memory with an input pattern generates trace activations equivalent to a pure instance memory. The mapping vector then identifies traces corresponding to the same item. Activation scaling or recall competition can occur directly on trace activations or on a response support vector derived from them.

This relationship between trace activations and responses can be interpreted as adding an extra layer to the connectionist memory:
- When each input pattern instance associates with a single output, the extra layer can simply be a vector.
- For complex output patterns, an explicit second linear memory layer is required.
- On the other hand, if each input pattern instance corresponds uniquely to distinct responses, the pooling or mapping step is unnecessary.

This extra layer can be defined as a vector when we assume input_pattern instances are associated with a single output. But if input_pattern instances are associated with complex output_patterns, then you actually do need an explicit second layer of the linear memory with weight patterns associated with each "trace" unit. And alternatively, if each input_pattern instance is paired with distinct responses, you don't even need this conversion/pooling step. (This is already what CMR's MFC does when lists don't have item repetitions.)

I think this equivalence is significant for a few reasons:

1. Explaining the link between connectionist and instance model. It confirms that anything an instance model can do, a connectionist memory can do, and explains how: you just need a final pooling step/layer. Previously, we primarily demonstrated the converse: that an instance memory can reproduce linear memory behavior. The broader literature (especially Jamieson/Jones) left this link between connectionist and instance models ambiguous, perhaps to promote the instance framework.
2. Improved performance trade-offs. Instance models defined like this in connectionist terms instead of with an explicit MINERIVA-II representation are more compact, so they're faster to fit/evaluate/simulate. 
3. Relating Positional CMR to instance theory. My version of CMR with event/positional-based codes (Positional CMR) already does this! I've been wondering how instance theory fit into the model, but with one caveat, it's already an instance model just implemented in a connectionist memory as outlined here. 

The notable caveat involves pre-experimental associations in Positional CMR:

- Positional CMR assigns distinct pre-experimental associations to each study position, pooling these with experimental associations rather than treating them as separate competitors. 
- This can be posed the "correct" way for instance-based retrieved-context models to incorporate pre-experimental associations. For example, it can be suggested in our in-review paper as a pathway for CRU variants to incorporate CMR’s pre-experimental memory features and effectively address both free and serial recall.
- In our version of Instance CMR designed to be identical with CMR, we still give these pre-experimental associations their own set traces, but we don't need to in the presented version. 