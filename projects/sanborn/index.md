Perfect—here's a 7‑slide, 5‑minute deck that follows your outline exactly, with figure call‑outs and a tight script.

---

## Slide 1 — Bayesian inference (what & why) — **40s**

**On the slide**

* **Bayes' rule:** (p(h\mid d)\propto p(d\mid h)p(h)).
* **Decisions as expectations:** compute ( \mathbb{E}[f(h)\mid d] ) over the posterior.
* **Toy example:** denoise (x) to recover (x^*) (posterior mean).
  *(Show Eq. 4 and the expectation idea from Eq. 7.)* 

**Say**
“Bayesian inference updates prior beliefs with data via the likelihood. Many tasks reduce to a posterior **expectation**—e.g., the best reconstruction of a noisy stimulus is ( \mathbb{E}[x^*\mid x] ).” 

---

## Slide 2 — Exemplar models (process view) — **40s**

**On the slide**

* Store exemplars (X^*={x^*_j}); compute **similarity** (s(x,x^*)).
* **General response rule:**
  [
  \hat f(x)=\frac{\sum_j f_j, s(x,x^*_j)}{\sum_j s(x,x^*_j)}
  ]
* Identification/categorization as special cases (Luce–Shepard choice).
  *(Reference Eqs. 1–3 for identification, categorization, and the general rule.)* 

**Say**
“The standard exemplar mechanism: activate stored instances by similarity and take a similarity‑weighted average of their associated responses.” 

---

## Slide 3 — What the paper claims: exemplars **≈** importance sampling — **50s**

**On the slide**

* **Key equivalence:** If exemplars ( (x^*_j,z_j) \sim p(x^*,z)) (the prior) **and** (s(x,x^*) \propto p(x\mid x^*)) (the likelihood), then
  [
  \hat f(x)=\frac{\sum_j f(x^*_j,z_j), p(x\mid x^*_j)}{\sum_j p(x\mid x^*_j)}
  ]
  = **self‑normalized importance sampling** estimate of ( \mathbb{E}[f(x^*,z)\mid x] ). *(Eq. 3 ↔ Eq. 12).*
* **When it works/fails:** depends on prior–posterior **overlap** (weight variance). *(Show **Fig. 1–2, p. 5**.)* 

**Say**
“Exemplars drawn from the prior and weighted by likelihood implement Monte‑Carlo **importance sampling** of the posterior expectation. It's stable when prior and posterior overlap (Fig. 2A) and brittle when they don't (Fig. 2B).” 

---

## Slide 4 — Domain demo 1: Perception & Generalization — **50s**

**On the slide**

* **Perceptual magnet effect (vowels).** With ~10–50 exemplars, importance‑weighted averaging denoises toward category centers and matches MDS perceptual maps. *(Show **Fig. 3, p. 8**.)* 
* **Universal law of generalization.** Sampling “consequential regions” and weighting by consistency reproduces exponential‑like gradients across six priors with 20–100 samples. *(Show **Fig. 4, p. 10**.)* 

**Say**
“In perception, exemplar‑as‑IS predicts category‑biased compression without hard labels. In generalization, the same sampler recovers near‑exponential gradients under diverse priors.” 

---

## Slide 5 — Domain demo 2: Everyday prediction & Concept learning — **45s**

**On the slide**

* **Predicting the future** (movie grosses, lifespans, etc.): memory‑ or computation‑limited sampling (5–50 exemplars) tracks Bayesian medians and human judgments; see **Fig. 5, p. 12** and **Table 1, p. 13**. 
* **Number game** (6,412 hypotheses): sampling 20–50 rules from the prior approximates the full Bayesian model and captures **individual variability**; see **Figs. 6–7, pp. 14–16**. 

**Say**
“Small, weighted samples are enough to approximate both optimal medians and structured hypothesis‑space reasoning, including person‑to‑person differences from sampling noise.” 

---

## Slide 6 — Domain demo 3: Reconstruction from memory — **35s**

**On the slide**

* **Category‑biased recall**: memory‑limited models (≤10 exemplars) best match human bias; bias shrinks for extremes because the current observation is included as an exemplar. *(Show **Fig. 8, p. 18**; note MSE vs memory capacity in panel B.)* 

**Say**
“Online recruitment of exemplars explains both category‑biased reconstruction and the characteristic **reduced bias** for extreme items; interestingly, more memory can **worsen** fit.” 

---

## Slide 7 — Take‑aways (and limits to flag) — **40s**

**On the slide**

* **Big idea:** Exemplar models are **rational process models**—a psychologically plausible implementation of approximate Bayes via importance sampling. 
* **Works when:** data are sparse/noisy and prior–posterior **overlap** is decent (see Fig. 2). 
* **Watch the assumptions:** (i) exemplars ≈ **prior samples** (or use proposal (q) with (p/q) reweighting), (ii) similarity ≈ **likelihood**; violations break the equivalence. 
* **Scope:** Static hypothesis spaces; for dynamic problems use **particle filters** (sequential IS) as the natural extension. 

**Say**
“A few well‑weighted examples can go a long way toward Bayesian behavior—but only under the right overlap and mapping assumptions; otherwise move to better proposals or sequential samplers.” 

---

### Timing guide (sum ≈ 5:00)

1. 40s · 2) 40s · 3) 50s · 4) 50s · 5) 45s · 6) 35s · 7) 40s.

---

## Slide assets (drop these into your deck)

* **Fig. 1–2 (p. 5)**: intuitive IS picture and overlap/variance.
* **Fig. 3 (p. 8)**: perceptual magnet.
* **Fig. 4 (p. 10)**: six generalization gradients.
* **Fig. 5 (p. 12)** + **Table 1 (p. 13)**: everyday prediction; error summary.
* **Figs. 6–7 (pp. 14–16)**: number game + individual variability.
* **Fig. 8 (p. 18)**: memory reconstruction vs capacity/noise. 

---

## Anticipate Q&A (one‑liners)

* **“Is similarity really a likelihood?”** Only if the mapping (s(x,x^*)\propto p(x\mid x^*)) holds; otherwise you need a proposal (q) and weights (p/q) (Eq. 10). 
* **“What if my experiences aren't prior‑representative?”** That breaks the prior‑sampling assumption; again treat memory as (q) and reweight. 
* **“What about learning with changing structure?”** Use **particle filters** for dynamic hypothesis spaces; the paper points to clustering, change‑point detection, RL, and sentence processing examples. 

---

If you want, I can paste these bullets straight into a slide deck with speaker notes and figure captions, but you can also lift them as‑is into your template.
