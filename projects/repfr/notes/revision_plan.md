# Revision: repfr.qmd → Psychological Review Article

**Target**: Psychological Review
**Approach**: Issue by issue, using reverse outlining method

---

## Revision Method

1. **Reverse outline the section** — For each paragraph, write:
   - Topic: What the paragraph is about
   - Argument: How it advances the overall argument

2. **Diagnose structural issues** — Identify:
   - Paragraphs with no argumentative payoff (cut or revise)
   - Gaps in the logical flow
   - Redundancies
   - Wrong ordering

3. **Propose new outline** — Restructure based on diagnosis

4. **Draft new paragraphs** — Then reverse outline each at sentence level:
   - For each sentence: topic + argument move
   - Assess each sentence against the paragraph's goal
   - Tighten sentences that delay the goal

5. **Iterate** — Edit based on sentence-level analysis rather than rewriting from scratch

---

## Introduction: Proposed New Structure

| Para | Goal | Content |
|------|------|---------|
| 1 | Why RCT matters | Scene-setting (temporal regularities), introduce RCT, explain bidirectionality as core innovation |
| 2 | What it explains | List successes: contiguity, asymmetry, distractor-robust recency, prior-list intrusions, serial recall |
| 3 | The pivot | Same mechanism predicts cross-occurrence interference (blending at study + retrieval) |
| 4 | Our finding | Tested with refined methodology; predictions unsupported; CMR produces them |
| 5 | Our contribution | Failure traces to blending assumption; reformulating reinstatement fixes it |
| 6 | Road map (optional) | Paper structure |

**Cut from original:**
- Bakery example (doesn't map to paradigm)
- Spacing mechanisms paragraph (no payoff)
- Detailed model variant descriptions in intro (move to Methods)

---

## Current Issue: Remove Chapter 1 Dependencies

The manuscript references "Chapter 1" and "Instance-CMR (Chapter 1)" in multiple places, which breaks standalone reading for a journal article.

### Locations to fix:
- **Line 49**: "Chapter 1's Instance-CMR"
- **Line 105**: "Instance-CMR (Chapter 1)"
- **Line 113**: "trace-sensitivity sharpening that already governs trace weighting during retrieval competition in Chapter 1"
- **Line 123**: "Instance-CMR is functionally equivalent to standard CMR (as established in Chapter 1)"
- **Line 126**: "CMR-equated Instance-CMR from Chapter 1"

### Fix approach:
1. Replace "Chapter 1" references with brief inline explanations or citations to published work
2. Add a self-contained explanation of Instance-CMR's trace architecture in the Modeling Framework section
3. Explain trace-sensitivity parameter without assuming prior chapter

### Key content needed:
Instance-CMR stores each item-context pairing as a separate memory trace rather than aggregating into composite associations. This trace-based architecture allows:
- Separate storage of each occurrence
- Competition among traces at retrieval
- Selective reinstatement of individual occurrence contexts

The "trace-sensitivity" parameter controls how sharply traces compete — high values approach winner-take-all selection, low values blend across traces.

---

## Issues identified but not yet addressed:
- Introduction lacks compelling narrative arc
- Missing statistical tests on empirical patterns
- No effect sizes reported
- Incomplete methods (exclusion criteria, demographics)
- Discussion defensive about item effects

We'll tackle these one at a time after fixing Chapter 1 dependencies.
