"""
Compound Cueing Analysis - REVISED
Let's trace through the math more carefully.

The issue: Mixed cueing ({j-2, i-1}) has HIGHER support than pure ({i-2, i-1}).

Why? Because in mixed cueing, the j-2 recall injects some similarity to c_j
into the context, which ADDS to the total support.

Let's trace this step by step.
"""

import numpy as np


def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def update_context(c_old, c_input, beta):
    dot = np.dot(c_old, c_input)
    rho = np.sqrt(1 + beta**2 * (dot**2 - 1)) - beta * dot
    c_new = rho * c_old + beta * c_input
    return normalize(c_new)


def similarity(c1, c2):
    return np.dot(c1, c2)


# Use a simple deterministic model to trace the math
# Assume contexts drift with lag according to exponential decay


def context_similarity_by_lag(lag, decay=0.1):
    """
    Similarity between contexts at positions p and p+lag.
    Assume S(c_p, c_{p+lag}) = exp(-decay * |lag|)
    """
    return np.exp(-decay * abs(lag))


def analyze_compound_cueing(spacing, beta=0.5, decay=0.1):
    """
    Analyze compound cueing with a simplified analytical model.

    Setup:
    - Item repeated at positions i and j, with j = i + spacing
    - c_i and c_j are the contexts at those positions
    - After recalling item at position p, context is updated toward c_p

    We analyze:
    - Pure: {i-2, i-1} -> context ends up near c_{i-1}
    - Mixed: {j-2, i-1} -> context is first pulled toward c_{j-2}, then toward c_{i-1}
    """

    print(f"\n{'='*70}")
    print(f"ANALYTICAL TRACE: spacing = {spacing}, beta = {beta}, decay = {decay}")
    print(f"{'='*70}")

    # Similarities between c_i/c_j and various neighboring contexts
    # S(c_i, c_{i-1}) = exp(-decay * 1) = high
    # S(c_i, c_{i-2}) = exp(-decay * 2) = somewhat high
    # S(c_j, c_{j-1}) = exp(-decay * 1) = high
    # S(c_j, c_{j-2}) = exp(-decay * 2) = somewhat high
    # S(c_i, c_j) = exp(-decay * spacing) = low (for large spacing)
    # S(c_i, c_{j-1}) = exp(-decay * (spacing-1)) = low
    # S(c_i, c_{j-2}) = exp(-decay * (spacing-2)) = low

    sim_lag1 = np.exp(-decay * 1)  # S(c_p, c_{p±1})
    sim_lag2 = np.exp(-decay * 2)  # S(c_p, c_{p±2})
    sim_cross = np.exp(-decay * spacing)  # S(c_i, c_j)

    print(f"\nContext similarities:")
    print(f"  S(c_i, c_{{i-1}}) = {sim_lag1:.4f}")
    print(f"  S(c_i, c_{{i-2}}) = {sim_lag2:.4f}")
    print(f"  S(c_i, c_j) = {sim_cross:.4f}")
    print(f"  S(c_j, c_{{j-1}}) = {sim_lag1:.4f}")
    print(f"  S(c_j, c_{{j-2}}) = {sim_lag2:.4f}")

    # After recall sequence, context is approximately:
    # c_final ≈ (1-beta) * c_after_first + beta * c_last_recalled
    #
    # For {i-2, i-1}:
    #   After first: c ≈ pulled toward c_{i-2}
    #   After second: c ≈ (1-β)*c_{i-2} + β*c_{i-1} (simplifying)
    #
    # Key: both c_{i-2} and c_{i-1} are similar to c_i, not to c_j

    # Let's compute effective similarities to c_i and c_j after each sequence

    # Approximation: final context is weighted avg of recent recalled contexts
    # c_final ≈ w1 * c_{last} + w2 * c_{second_last} where w1 > w2
    w1 = beta  # weight of most recent
    w2 = (1 - beta) * beta  # weight of second-most-recent (rough approx)

    print(f"\nApprox weights: w1 (recent) = {w1:.4f}, w2 (older) = {w2:.4f}")

    # PURE: {i-2, i-1}
    # S(c_final, c_i) ≈ w1 * S(c_{i-1}, c_i) + w2 * S(c_{i-2}, c_i) = w1 * sim_lag1 + w2 * sim_lag2
    # S(c_final, c_j) ≈ w1 * S(c_{i-1}, c_j) + w2 * S(c_{i-2}, c_j)
    #                 ≈ w1 * exp(-decay*(spacing+1)) + w2 * exp(-decay*(spacing+2))

    sim_i_pure = w1 * sim_lag1 + w2 * sim_lag2
    sim_j_pure = w1 * np.exp(-decay * (spacing + 1)) + w2 * np.exp(
        -decay * (spacing + 2)
    )

    print(f"\nPURE cueing {{i-2, i-1}}:")
    print(f"  S(c_final, c_i) ≈ {sim_i_pure:.4f}")
    print(f"  S(c_final, c_j) ≈ {sim_j_pure:.4f}")

    # MIXED: {j-2, i-1}
    # S(c_final, c_i) ≈ w1 * S(c_{i-1}, c_i) + w2 * S(c_{j-2}, c_i)
    #                 = w1 * sim_lag1 + w2 * exp(-decay*(spacing-2))
    # S(c_final, c_j) ≈ w1 * S(c_{i-1}, c_j) + w2 * S(c_{j-2}, c_j)
    #                 = w1 * exp(-decay*(spacing+1)) + w2 * sim_lag2

    sim_i_mixed = w1 * sim_lag1 + w2 * np.exp(-decay * (spacing - 2))
    sim_j_mixed = w1 * np.exp(-decay * (spacing + 1)) + w2 * sim_lag2

    print(f"\nMIXED cueing {{j-2, i-1}}:")
    print(f"  S(c_final, c_i) ≈ {sim_i_mixed:.4f}")
    print(f"  S(c_final, c_j) ≈ {sim_j_mixed:.4f}")

    # Compare support for different tau values
    print(f"\n{'='*70}")
    print("SUPPORT COMPARISON")
    print(f"{'='*70}")

    for tau in [1.0, 2.0, 3.0]:
        support_pure = max(0, sim_i_pure) ** tau + max(0, sim_j_pure) ** tau
        support_mixed = max(0, sim_i_mixed) ** tau + max(0, sim_j_mixed) ** tau
        ratio = support_pure / support_mixed if support_mixed > 0 else float("inf")

        print(f"\ntau = {tau}:")
        print(
            f"  Pure support:  {sim_i_pure:.4f}^{tau} + {sim_j_pure:.4f}^{tau} = {support_pure:.6f}"
        )
        print(
            f"  Mixed support: {sim_i_mixed:.4f}^{tau} + {sim_j_mixed:.4f}^{tau} = {support_mixed:.6f}"
        )
        print(f"  Ratio (pure/mixed) = {ratio:.4f}")

    return {"pure": (sim_i_pure, sim_j_pure), "mixed": (sim_i_mixed, sim_j_mixed)}


# MAIN ANALYSIS
print("\n" + "#" * 70)
print("# KEY INSIGHT")
print("#" * 70)
print(
    """
The issue is that MIXED cueing {j-2, i-1} gives HIGHER total support because:

- In PURE {i-2, i-1}: 
  - High similarity to c_i (good!)
  - Very low similarity to c_j (wasted)
  
- In MIXED {j-2, i-1}:
  - High similarity to c_i (from i-1 recall)
  - PLUS some residual similarity to c_j (from j-2 recall)

The j-2 recall in MIXED adds support to the c_j term that PURE doesn't have.

CMR: The extra support from j-2 helps linearly
ICMR: The extra support from j-2 is sharpened, but still helps

The ratio (pure/mixed) is < 1 in BOTH models, but:
- If CMR: ratio should be stable
- If ICMR: ratio might change with tau
"""
)

# Run analysis for different spacings
analyze_compound_cueing(spacing=10)
analyze_compound_cueing(spacing=20)

# Now let's think about the CORRECT comparison
print("\n" + "#" * 70)
print("# BETTER COMPARISON: {i-1, i-2} vs {i-1, j-2}")
print("#" * 70)
print(
    """
The user's formulation has i-1 FIRST, then i-2 or j-2.
This means the MOST RECENT recall is i-2 or j-2.

Let me reconsider with this order...
"""
)


def analyze_user_formulation(spacing, beta=0.5, decay=0.1):
    """
    User's formulation: {i-1, i-2} vs {i-1, j-2}
    Note: i-1 is first, i-2 or j-2 is second (and most recent).
    """
    print(f"\n{'='*70}")
    print(f"USER'S FORMULATION: spacing = {spacing}")
    print(f"{'='*70}")

    sim_lag1 = np.exp(-decay * 1)
    sim_lag2 = np.exp(-decay * 2)

    w1 = beta  # weight of most recent (i-2 or j-2)
    w2 = (1 - beta) * beta  # weight of older (i-1)

    # SCENARIO A: {i-1, i-2} - most recent is i-2
    # S(c_final, c_i) ≈ w1 * S(c_{i-2}, c_i) + w2 * S(c_{i-1}, c_i) = w1 * sim_lag2 + w2 * sim_lag1
    # S(c_final, c_j) ≈ w1 * S(c_{i-2}, c_j) + w2 * S(c_{i-1}, c_j) ≈ very small

    sim_i_A = w1 * sim_lag2 + w2 * sim_lag1
    sim_j_A = w1 * np.exp(-decay * (spacing + 2)) + w2 * np.exp(-decay * (spacing + 1))

    print(f"\nScenario A: {{i-1, i-2}} - most recent = i-2")
    print(f"  S(c_final, c_i) ≈ {sim_i_A:.4f}")
    print(f"  S(c_final, c_j) ≈ {sim_j_A:.4f}")

    # SCENARIO B: {i-1, j-2} - most recent is j-2
    # S(c_final, c_i) ≈ w1 * S(c_{j-2}, c_i) + w2 * S(c_{i-1}, c_i)
    #                 = w1 * exp(-decay*(spacing-2)) + w2 * sim_lag1
    # S(c_final, c_j) ≈ w1 * S(c_{j-2}, c_j) + w2 * S(c_{i-1}, c_j)
    #                 = w1 * sim_lag2 + w2 * exp(-decay*(spacing+1))

    sim_i_B = w1 * np.exp(-decay * (spacing - 2)) + w2 * sim_lag1
    sim_j_B = w1 * sim_lag2 + w2 * np.exp(-decay * (spacing + 1))

    print(f"\nScenario B: {{i-1, j-2}} - most recent = j-2")
    print(f"  S(c_final, c_i) ≈ {sim_i_B:.4f}")
    print(f"  S(c_final, c_j) ≈ {sim_j_B:.4f}")

    print(f"\n{'='*70}")
    print("SUPPORT COMPARISON")
    print(f"{'='*70}")

    for tau in [1.0, 2.0, 3.0]:
        support_A = max(0, sim_i_A) ** tau + max(0, sim_j_A) ** tau
        support_B = max(0, sim_i_B) ** tau + max(0, sim_j_B) ** tau
        ratio = support_A / support_B if support_B > 0 else float("inf")

        print(f"\ntau = {tau}:")
        print(
            f"  A {{i-1,i-2}}: {sim_i_A:.4f}^{tau} + {sim_j_A:.4f}^{tau} = {support_A:.6f}"
        )
        print(
            f"  B {{i-1,j-2}}: {sim_i_B:.4f}^{tau} + {sim_j_B:.4f}^{tau} = {support_B:.6f}"
        )
        print(f"  Ratio A/B = {ratio:.4f}")
        print(
            f"  --> {'A>B' if support_A > support_B else 'B>A' if support_B > support_A else 'A=B'}"
        )


analyze_user_formulation(spacing=10)
analyze_user_formulation(spacing=20)
