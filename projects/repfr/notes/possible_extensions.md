You're right, let me refocus on genuinely new ideas. New Model Candidates:
1. Spacing-Modulated Reinforcement
Scale reinforcement by the P1-P2 lag:

mfc_reinf = base_reinf * f(spacing)
Where f() could be logarithmic, linear, or learned. This directly incorporates spacing effects into the reinforcement mechanism rather than treating them as separate phenomena.
2. Bidirectional Asymmetric Reinforcement
At P2 encoding, boost BOTH positions but asymmetrically:
P1 gets reinf_strength
P2 gets reinf_strength * asymmetry_factor (where asymmetry < 1)
This could improve spacing effect fit (P2 also benefits) while maintaining P1 bias. Could test with fixed asymmetry to avoid extra parameters.
3. Retrieval-Strength Weighted Encoding
At P2, weight the reinforcement by how strongly P1 is currently activated:

reinf_amount = base_reinf * activation(P1) / (activation(P1) + activation(P2))
More reinforcement when P1 is contextually relevant, less when P2 dominates. This naturally varies with spacing/context.
4. Separate Item Identity Layer
Add a lightweight item-level layer alongside position layer:
Position layer: distinct contexts (current positional CMR)
Item layer: shared identity that accumulates activation across presentations
Item layer contributes to activation but NOT to context update
This could provide repetition benefit without context interference.
5. Decay + Restrengthen Model
P1's trace decays during lag, P2 presentation restores it:

P1_strength = initial_strength * decay(spacing) + restore_amount
Natural spacing effects from the decay function, P1 bias from restoration being tied to P1 specifically. Which of these directions feels worth developing further?