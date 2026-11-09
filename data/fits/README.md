# Example fits prepared with the current library

All three records were prepared on 2026-11-09 with the existing observation file
`data/Dupertuys2026.h5`, unchanged. Its SHA-256 is
`93ac35ad7f17c907ad369438c0899be45d4f6728250eccfe7ae3757d2fe2468a`; it matches the ECMR3 project's copy.
No dataset extract is maintained.

## Models and objective

All fits pool the 492 lists, use `NoStopTermination` and
`ExcludeTerminationLikelihoodLoss`, and disable repeated recalls. Learning
occurs before context updating. The nine shared continuous parameters are
encoding, start and recall drift; learning rate; item and shared support;
primacy scale and decay; and choice sensitivity.

- `Dupertuys2026_CMR_demo.json` estimates these nine ordinary CMR parameters.
- `Dupertuys2026_eCMR_source_only.json` additionally estimates emotional strength,
  acting on source associations.
- `Dupertuys2026_eCMR_broad.json` additionally applies that emotional strength
  additively to temporal association learning.

Both eCMR variants use multiplicative primacy modulation of emotional
associations, emotional encoding drift of 1 and emotional recall drift tied
to temporal recall drift. The latter follows the current model's default when
`emotion_recall_drift_rate` is omitted. The legacy `emotion_drift_rate` is absent.
`phi_emot_modulates_temporal` distinguishes the two variants;
`modulate_temporal_emotion_by_primacy` is false in both.

Likelihood scores recall order conditional on observed response counts.
Simulations force those counts, using ten repetitions per list and seed 1.
Diagnostic SPC, PNR and lag-CRP select the 491 complete twelve-item lists.
Matching recall length is imposed, not a prediction of stopping behavior.

## Searches and selected results

All use 64-bit calculations and SciPy differential evolution: a maximum of
1,000 generations, population multiplier 15, three restarts, relative tolerance
0.0001, absolute tolerance 0, crossover 0.9, mutation range 0.5–1.0,
Latin-hypercube initialization and no polishing. Rates lie between float64
machine epsilon and one minus epsilon; support, primacy and sensitivity bounds
are [epsilon, 100]; emotional-strength bounds are [epsilon, 10].

Each seed below initializes three independent restarts. The better seed's result
is supplied; the CMR template defaults to that selected seed. `search_comparison`
retains both results, iteration/convergence metadata and fitting times.

| Model | Seed 0 NLL | Seed 1 NLL | Selected seed | NLL spread |
|---|---:|---:|---:|---:|
| Ordinary CMR | 7202.114181 | 7202.029122 | 1 | 0.085059 |
| Source-only eCMR | 7169.430814 | 7169.426270 | 1 | 0.004544 |
| Broader eCMR | 7164.357873 | 7164.407215 | 0 | 0.049342 |

The same model has the lower objective in both seed sets. The selected eCMR fits differ by 10.137 AIC units,
favoring broader eCMR. This is an illustrative comparison, not a definitive
model-selection result. All returned objectives were recalculated before saving;
an additional evaluation through the generic event-by-event likelihood path
agreed with each selected objective (`independent_nll`).

## Behavioral and boundary review

The CMR simulation captures realistic first-position recall and the forward
lag-CRP peak. It still overpredicts some early/middle positions and underpredicts
late-list recall. The forward peak is slightly too large. All three models'
SPC, PNR and CRP were inspected on matched selections; likelihood fit does not
guarantee agreement with every summary.

The following values are within 1% of a bound measured as a fraction of the
full search interval. With broad ranges, this is a screening rule, not evidence
that a fitted parameter is pinned to its boundary or is well identified.

| Model | Parameter | Selected value | Bounds |
|---|---|---:|---:|
| Ordinary CMR | `primacy_scale` | 0.0841712 | 2.22045e-16–100 |
| Source-only eCMR | `primacy_scale` | 0.135032 | 2.22045e-16–100 |
| Broader eCMR | `primacy_scale` | 0.12976 | 2.22045e-16–100 |

## Reproduction and historical records

`templates/fitting_scipyde.ipynb` is the procedure used for all three models.
For an eCMR run, take `parameters` from the record's `fit_configuration`, set
`model_name` to its `model`, and use its `model_factory`, `components`,
`loss_function`, `fit_algorithm` and `simulation_function` for the corresponding
import-path settings. Set `seed` to the recorded selected seed and use the
matching example filename. All other search defaults match the notebook.
The `fits` columns contain the fitted values; they are not the input bounds.

All three records identify JAXCMR 0.2.0rc1 and source SHA-256
`b4ecd25b7eedf5a9cbd38c70df4191338444b0b2d27c73702d6d908454b74d81`. This fingerprint hashes sorted package-relative
Python filenames and their bytes, each followed by a zero byte. Each record
also stores Python, JAX, NumPy and SciPy versions, precision, input identity,
selected rows and full configuration. Recorded objectives belong to that
implementation; the package version alone is insufficient when local source
changes without a release.

The original ECMR3 result files remain unchanged in `emcr3/results/fits/`.
They used older context updating and an older emotional-drift interpretation.
Loading their parameter arrays into the current model does not reproduce their
original likelihoods. Retain their original semantics for historical reproduction,
or refit under current equations. These new book examples replace the earlier
copied records; no historical compatibility model is added to JAXCMR.

Download copies are staged for the book build. These research inputs remain
excluded from package distributions.
