# Regression data

`recall.json` is synthetic: four eight-position study lists, with adjacent
repetitions, different recall orders, an empty response, and a unique-item list.
Recall entries are one-based **study positions**, not item identifiers; zero is
padding. The two list geometries exercise preparation and selection changes.
No private files or downloaded datasets are required. A missing fixture is an error.

The gradient test also retains one explicitly written 40-position adjacent-repeat
example (responses at positions 31 and 29), because its previously established
likelihood of 5.9108604179346145 is a historical regression value. Its original
independent derivation is not documented here; it checks continuity with earlier
results, not independent validation of the CMR equations. It does not read the
full Lohnas–Kahana dataset.
