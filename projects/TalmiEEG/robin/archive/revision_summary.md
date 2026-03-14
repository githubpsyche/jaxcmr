# Manuscript revision summary

Dear Robin and Deborah,

I've suggested a lot of updates to the Introduction, Method, Model Variants, Evaluation Approach, and eCMR Model Comparison sections. The Discussion is still to be written and I'm still hoping to do more work on the modelling, as current results force us to report that the LPP does not meaningfully improve predictive accuracy at the subject level. A per-entry changelog is available for the detailed reasoning behind each change, but seems extreme at this stage. So I wrote this summary here.

The revisions fall into six areas.

## 1. Theoretical framework

Several descriptions in the Introduction have been brought into closer alignment with the source papers (primarily Talmi, Lohnas & Daw, 2019, and Polyn et al., 2009).

- **Retrieved-context theory** now foregrounds the reinstatement mechanism — items reinstating their encoding contexts at retrieval — since this is what distinguishes RCT from simpler encoding-similarity accounts. The extension to nontemporal dimensions uses Polyn et al.'s own terminology ("source features," "source context") rather than the experiential/semantic binary, which blurs in practice when semantic processing drives encoding experiences.
- **eCMR** is characterised as two extensions of CMR — an emotional dimension of context and an encoding-strength modulator ($\Phi_{\text{emot}}$) — rather than the similarity aspect alone. This matters because the rest of the manuscript depends on the reader tracking both mechanisms: the emotional context layer drives the category SPC diagnostic, and $\Phi_{\text{emot}}$ drives the LPP-by-recall diagnostic.
- **Source features** are described as symmetric: both emotional and neutral items have non-zero source-context representations. The original description (emotional = 1, neutral = 0) implied that neutral items lack source-context binding, which conflicts with TLD19's two-category scheme.
- **$\Phi_{\text{emot}}$ mechanism** now specifies the direction of association ($M^{CF}$, context-to-item) and clarifies that $\Phi_{\text{emot}} = 1$ means "same encoding strength" rather than "processed equally," since the source-feature mechanism still distinguishes the two categories even when $\Phi_{\text{emot}}$ is 1.
- **LPP–$\Phi_{\text{emot}}$ link** is framed as a hypothesis under test rather than a premise. The original text presupposed that the LPP directly indexes $\Phi_{\text{emot}}$; the revised text says the LPP "is sensitive to the enhanced processing of motivationally significant stimuli" and frames the constraining move as something we are "testing whether" rather than asserting.

## 2. Aim structure and novelty claims

The connection between the broad aim and the sub-aims has been made explicit.

- A **roadmap sentence** after the broad aim decomposes "suitability" into three questions: reliability, prediction, and model incorporation. The sub-aim openings now reference this decomposition.
- The **novelty claim** has been narrowed. Barnacle et al. (2018) already showed that the LPP's emotional modulation survives intentional encoding at the group level. The novelty of Aim 1 is individual-level reliability, which Barnacle did not test. The revised text acknowledges the group-level precedent while preserving what is genuinely new.
- The **Barnacle dissociation** (LPP modulation is context-independent but the memory advantage is context-dependent) is now cited as motivation for the modelling approach: a model representing both encoding strength and retrieval context is needed to explain when LPP variation translates into a memory advantage.

## 3. Evidence base and bibliography

Several claims seemed to deserve additional supporting references. The reference count has gone from 28 to over 50. The main additions cover three areas: the RCT evidence base (7 citations spanning behavioural and neural work, plus a recent review), LPP psychometric and functional review literature (Hajcak et al., 2010; Hajcak & Foti, 2020; Olofsson et al., 2008; Fields, 2023; Weinberg et al., 2021; Moran et al., 2013), and attention-memory coupling (Chun & Turk-Browne, 2007; Talmi, 2013; Mather & Sutherland, 2011; Dolcos & Cabeza, 2002).

## 4. Method and EEG sections

The heading hierarchy was corrected (EEG Data Analysis was a sibling of Method; demoted to subsection). A drafting note was removed. Batch grammar and formatting corrections were applied: subject-verb agreement, missing words, capitalisation, British English normalisation, SI-convention spacing. No changes were made to analysis descriptions or statistical procedures.

## 5. Model comparison redesign

This is the largest structural change.

**Model set**: The original draft compared three eCMR variants that differed only in how the LPP entered the encoding-strength term (emotion-only, main-effect, interaction). All three shared the emotional context layer, so the comparison could not test whether that layer itself mattered. The current version compares three architecturally distinct models: CMR ($k = 9$, no emotion mechanism), eCMR ($k = 10$, emotional context + category-based encoding strength), and LPP-eCMR ($k = 11$, adds one LPP parameter). This tests both whether the emotional context layer matters and whether trial-level LPP adds to category labels.

**Evaluation**: Nesting flooring corrects occasional optimizer failures that would otherwise violate the nesting guarantee. Bayesian model selection (Stephan et al., 2009) with protected exceedance probabilities (Rigoux et al., 2014) was added alongside AIC. Interpretation standards are stated explicitly: a ΔAIC CI excluding zero indicates a reliable advantage; an exceedance probability above .95 provides strong population-level evidence.

**Results**: Rewritten with corrected post-flooring numbers. The key finding is a tension: LPP-eCMR dominates aggregate AIC weights (.997) and BMS expected frequency, but the protected exceedance probability remains at chance and the median per-subject ΔAIC favours eCMR. Qualitative diagnostics resolve this tension — LPP-eCMR reproduces the recalled/unrecalled LPP separation for emotional items (genuine generalisation to an untrained statistic) that eCMR cannot.

**Figures**: The single composite benchmark figure was split into two: one for category-specific serial position curves and one for the LPP-by-recall diagnostic. Each now displays the empirical data alongside model predictions, giving the reader the reference pattern before evaluating model adequacy. Model fits were updated to the refactored eCMR variants.

**Construct validity**: The original draft framed the comparison as testing whether the LPP reflects emotion-general vs emotion-specific processing. This conflated model structure with cognitive constructs. The revised framing asks whether embedding an established neural-behavioural relationship in a process model improves prediction, without claiming to adjudicate which construct the LPP reflects.

## 6. Writing quality

Typos were fixed throughout (11 in the Introduction, 10 in Method/EEG). Dense sentences were unpacked — the encoding-instructions sentence, the replication sentence, and the prediction sentence were each split into shorter units. Semicolons and em-dashes used to stretch sentences were replaced with periods. A manuscript-wide colon review found one candidate for splitting (the eCMR introduction sentence) and judged the rest legitimate. British English was normalised.

## What remains

- **Discussion**: Not yet written.
- **Three FIXMEs**: (1) A figure cross-reference that should use Quarto's scheme. (2) Whether the handling of semantically related and unrelated items should be justified in the modelling sections. (3) A decision about departing from a particular approach that needs motivation.
- **Behavioural Results and EEG Results sections**: Not touched by this revision pass.

Thank you, Robin, for the strong foundation your draft provides. These are suggestions — please let me know wherever you'd prefer a different approach or would like something changed back.

Best,
Jordan
