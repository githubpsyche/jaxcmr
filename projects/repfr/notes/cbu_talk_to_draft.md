# Reflecting on cbu.pptx vs. index.qmd — Connecting the Talk to the Draft

## What the talk crystallized

### 1. The core narrative is tighter in the talk than the draft

The talk's arc is clean: **RCT's item-based reinstatement → predicts interference → three tests show no interference → episode-specific variant fixes it → but it's a floor, not a full account → residual effects point to reinforcement over knitting.** The draft covers all of this, but the same arc is distributed across ~1250 lines with the ablation (CMR-NoSPR) occupying its own full section. The talk skips CMR-NoSPR entirely and goes straight from Standard CMR failing to CMR-ES succeeding. The paper needs the ablation for completeness, but the talk demonstrated how to frame the *story* without it — the ablation is a supporting detail, not a plot point.

### 2. "Floor, not full account" is a strong framing

Slide 33 calls CMR-ES a "floor" for cross-occurrence interaction, with remaining repetition effects unexplained. The draft makes this point (Instance-CMR resolves interference but doesn't capture the boosted first-over-second separation), but never names it so crisply. The word "floor" (or the concept) could be introduced at the transition from Instance-CMR to ICMR-Reinf (~line 936) to tighten the motivation.

### 3. The "focal contribution" slide clarified the theoretical stakes

Slide 10 explicitly quotes Kahana (2020) to identify what's *new* about RCT relative to predecessors: earlier drifting-context models explained forgetting and recency but never linked context evolution to item retrieval. Slide 11 names the payoff: forward asymmetry in the lag-CRP + distraction resistance. The draft covers this (lines 69-76) but less pointedly. The introduction could benefit from a sentence or two that says, in effect: "Item-based reinstatement is RCT's signature claim — what sets it apart from prior drifting-context accounts — so testing it when items repeat is not a niche question but a direct probe of the theory's core mechanism."

### 4. The talk's takeaways map cleanly onto the abstract

The abstract (lines 29-34) already captures the talk's message well: "differentiated competition" not "contextual blending." The closing paragraph (lines 1246-1253) should land on the same message but currently doesn't — it's flagged as out of date.

## Gaps and mismatches between talk and draft

### A. The closing paragraph (lines 1244-1253)

The `<!-- out of date -->` comment at line 1244 is the most obvious loose end. The talk's takeaways (slide 33) offer a clear template:

- **Blending is rejected** across three diagnostics and two tasks
- **Removing blending improves fits** but serves as a floor — not a full account of repetition
- **Residual effects** (0-bin boost, first-occurrence bias) point to reinforcement over knitting
- **Methodological contribution**: symmetric baseline cleanly isolates repetition effects

The draft's current closing tries a bakery-analogy redemption ("A later mention of 'the bakery down the street' can cue either visit, but it need not blend them"). The bakery analogy wasn't in the talk and the current paragraph reads as thesis-era prose. A tighter close could state the theoretical upshot directly: memory for repeated experience balances integration and specificity through differentiated competition over episode-specific traces, not through blending item-linked composites.

### B. The spacing-effect discussion (lines 1218-1242)

The talk barely touches spacing — it's not in the takeaways, and the talk only shows spacing data in the standard-CMR vs data context (not for CMR-ES). The draft devotes ~25 lines to discussing how study-phase retrieval doesn't contribute to the spacing effect and how superadditivity might be reinterpreted. This section feels somewhat disconnected from the main narrative. It may be worth either:
- Tightening it into a concise paragraph that makes one point (study-phase retrieval in CMR doesn't drive spacing; contextual variability + test-phase reinstatement suffices)
- Or moving it earlier into the results as a brief observation rather than giving it prominence in the discussion

### C. The item-features/PLI paragraph (lines 1195-1216)

This limitation/future-directions section doesn't appear in the talk. If audience questions at CBU didn't push on this, it might be appropriately scoped as-is. But if questions *did* push on it — e.g., "how do you account for semantic clustering or PLIs without item features?" — you may want to sharpen or expand it based on what you learned. This is the section where the draft is most clearly conceding ground, and the talk's clean narrative doesn't help resolve it.

### D. Naming inconsistency

The talk uses "CMR-ES" (Episode-Specific). The draft uses "Instance-CMR" / "ICMR-OS" / "ICMR-Reinf" / "CMR-NoSPR". These are different audiences, but worth checking: is the final paper going with "Instance-CMR" everywhere? The draft introduction (line 59) uses both "Instance-CMR" and refers to the general concept as "episode-specific." The talk's "CMR-ES" is simpler, but "Instance-CMR" carries more theoretical weight (linking to instance theory).

### E. The discussion's opening (lines 1114-1131)

These ~17 lines restate what RCT is and how contiguity works. The talk didn't need this because the audience was primed. But in the paper, this re-exposition reads somewhat like a second introduction. It might be tightened to ~5 lines that just remind the reader of the specific commitment being tested (item-based reinstatement → composite cue) rather than re-explaining the full framework.

## Concrete edits (in priority order)

### 1. Rewrite the closing paragraph (lines 1244-1253)
Delete the `<!-- out of date -->` comment and the bakery-analogy paragraph. Replace with a direct theoretical close that:
- States that across three diagnostics and two tasks, item-based context reinstatement produces interference not found in data
- Names the resolution: episode-specific traces indexed by temporal position, selected by competition, not blended via item identity
- Acknowledges the floor/reinforcement distinction: removing blending is necessary; reinforcement without blending captures residual first-occurrence advantages
- Ends on the takeaway: episodic memory for repeated experience achieves both integration and specificity through differentiated competition, not contextual blending

### 2. Tighten the discussion opening (lines 1114-1131)
Currently ~17 lines re-explaining RCT from scratch. Compress to ~5-7 lines that remind the reader only of the specific commitment being tested: item-based reinstatement produces composite cues that blend across occurrences. The full RCT tutorial belongs in the introduction, not the discussion.

### 3. Introduce "floor" framing at the ICMR-OS → ICMR-Reinf transition (~line 936)
After summarizing what Instance-CMR resolves and what residual asymmetries it leaves, add a sentence like: "Instance-CMR thus serves as a floor for cross-occurrence interaction — a null model that eliminates interference but does not yet account for any positive repetition-specific effects." This motivates ICMR-Reinf more crisply.

### 4. Sharpen the "focal contribution" framing in the introduction (lines 69-76)
After explaining that RCT departs from earlier drifting-context theories by allowing items to reinstate context, add a sentence making the stakes explicit: something like "Item-based reinstatement is the signature departure of retrieved-context theory from its predecessors — the mechanism that simultaneously explains forward asymmetry, distraction resistance, and repetition-mediated spacing benefits. Testing it when items repeat therefore probes the theory's central claim, not a peripheral detail."

### 5. Tighten the spacing-effect discussion (lines 1218-1242)
This ~25-line block makes one main point (study-phase retrieval in CMR doesn't drive spacing; contextual variability + test-phase reinstatement suffices) but takes too long to get there. Compress to ~10-12 lines: lead with the result (CMR-NoSPR preserves spacing), state the implication (study-phase retrieval's main consequence in CMR is knitting, not strengthening), note the superadditivity gap remains open for future work.

### 6. Item-features/PLI section (lines 1195-1216) — leave as-is
No audience pressure on this. The section appropriately flags the limitation and points to future work. No changes needed.

### 7. Naming — no change proposed
"Instance-CMR" / "ICMR-OS" / "ICMR-Reinf" / "CMR-NoSPR" carry theoretical weight appropriate for the paper. The talk's "CMR-ES" was audience-appropriate shorthand. Keep the paper's naming unless you want to simplify.

## Files to modify
- `projects/repfr/index.qmd` — all edits above are to this file

## Verification
- Re-render the document (`quarto render projects/repfr/index.qmd`) and check that the discussion flows from the results without re-introducing tutorial material
- Confirm the closing paragraph lands the abstract's "differentiated competition" message
