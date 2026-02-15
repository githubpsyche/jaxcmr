# Issues

## Model coupling in preparation.py and sweep.py

`prepare_single_subject` and `run_from_cache` manipulate model
internals not covered by the `MemorySearch` protocol:

- `_mcf_learning_rate` (derived array, not a raw param)
- `encoding_drift_rate`, `start_drift_rate`, `mcf_sensitivity`
- `.replace()` for swapping rates mid-sequence
- `.context.integrate()`, `.items`, `.mfc.probe()` in reminder loop

This means these functions only work with concrete CMR instances,
not arbitrary MemorySearch implementations.

**Clean fix:** Add model-level methods that encapsulate the
scale-and-encode pattern:

```python
model = model.encode_items(items, drift_scale=0.5, mcf_scale=0.3)
model = model.reinstate_start_context(drift_scale=1.0)
model = model.run_reminder(cue_items, drift_scale=0.8)
```

This would let preparation.py and sweep.py operate purely through
the protocol, and push rate-scaling logic into the model where it
belongs.

**Status:** Accepted as design debt. Revisit when model API is
next under active development.
