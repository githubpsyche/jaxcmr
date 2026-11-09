# Example fit records

These examples use the existing `data/Dupertuys2026.h5` observation file, unchanged.
Its SHA-256 is `93ac35ad7f17c907ad369438c0899be45d4f6728250eccfe7ae3757d2fe2468a`.
The observation file is identical to the ECMR3 project's copy. No dataset extract
is maintained for these templates.

## Ordinary CMR demonstration

`Dupertuys2026_CMR_demo.json` was produced by the default procedure in
`templates/fitting_scipyde.ipynb` during preparation of these examples on
2026-11-09, using JAXCMR 0.2.0rc1. Its metadata records the data hash and selected
rows, factory, components, likelihood, fixed and free parameters, starting
baseline, optimizer settings and recalculated objective.

It pools all 492 lists, estimates encoding and recall drift within [0.05, 0.95],
uses the remaining README baseline parameters, and excludes termination from
the likelihood. SciPy differential evolution uses at most 30 generations,
population multiplier 8, one restart, seed 0 and no polishing. Early stopping
uses the fitter's recorded tolerances. This is a short procedural example, not
evidence of parameter recovery or a completed scientific fit.

The simulation templates repeat the shared parameter values over the original
participant identifiers and force response counts to those observed. They
regenerate their outputs whenever run; changing a fit does not silently load
simulations made from an older fit.

## Historical eCMR comparison

The following records were copied without changing their contents or metadata
from the sibling ECMR3 checkout (`emcr3/results/fits/`) on 2026-11-09:

| Example file | Original filename |
|---|---|
| `Dupertuys2026_eCMR_source_only.json` | `Dupertuys2026_eCMR_source_only_phi_primacy_pooled_evosax_fixed_term_best_of_1.json` |
| `Dupertuys2026_eCMR_broad.json` | `Dupertuys2026_eCMR_broad_phi_primacy_pooled_evosax_fixed_term_best_of_1.json` |

Both pool the same experimental trials (`list_type >= 1`), use the same
termination-excluded likelihood and no-stop policy, and estimate ten parameters.
They differ in whether emotional modulation also acts on temporal context
(`phi_emot_modulates_temporal`: 0 for source-only, 1 for broader eCMR).
The comparison chapter uses their recorded objectives; it does not refit them,
floor their fitness values or treat them as participant-level estimates.

Original ECMR3 records remain in place and unchanged. Download copies are staged
only for the book build; these research inputs are excluded from package distributions.
