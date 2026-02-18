# Sim 2: Manipulating Interference Density — Implementation Plan

## Goal

Explore what intensifies interference from competing events. Show that MCF encoding strength, context proximity, and competitor count each modulate how much competitor encoding suppresses film-item recall, and characterize dose-response shapes and ceilings. This simulation is worked first because its findings about interference intensifiers inform the configuration of all subsequent simulations.

## Design

Two-phase synthetic paradigm (no h5 data). All items are neutral (no emotional features).

- **Film phase**: Encode `N_film` items with fitted HealeyKahana2014 parameters (per-subject)
- **Interference phase**: Encode `M` competitor items with interference-specific parameters
- **Test**: `start_retrieving()` → `simulate_free_recall()` → score film items recalled

### Key constants

- `N_film = 16` — matches HealeyKahana2014 list length (parameters were fit to 16-item lists)
- `M_max = 16` — maximum competitor count across all conditions
- `list_length = N_film + M_max = 32` — model capacity (items, context size, MFC, MCF)
- `experiment_count = 10` — replications per subject per condition

### Model initialization

`CMR(list_length=32, parameters=subject_params)` with `NoStopTermination` (as in the existing notebook) so recall continues until all recallable items are attempted. Context size = 33 (32 items + 1 start-of-list unit). Items are a 32×32 identity.

### Two-phase encoding

```
Film phase: experience(1), ..., experience(N_film) with fitted parameters

Parameter swap via model.replace():
  encoding_drift_rate → interference_drift_rate
  _mcf_learning_rate  → flat array of interference_mcf_lr

Interference phase: experience(N_film+1), ..., experience(N_film+M)
  (zero-padded to M_max; experience(0) is a no-op per cmr.py:151)
```

### Recall

`start_retrieving()` reinstates start-of-list context via fitted `start_drift_rate` (not swept — that's Sim 3). Then `simulate_free_recall(model, list_length, rng)`.

### Scoring

- **Film items recalled**: Count recalls in 1..N_film
- **SPC**: Recall probability at each study position (1..N_film+M)

## Parameter sweeps

### Sweep 1: Interference MCF learning rate

**What varies**: `interference_mcf_lr` — flat MCF learning rate for all interference items (replaces primacy-modulated `_mcf_learning_rate` at positions ≥ N_film)

**What's fixed**: `interference_drift_rate` = fitted `encoding_drift_rate`, `M = M_max`

**Values**: ~10 values from baseline (primacy-decayed MCF rate at position 16, from fitted `primacy_scale` and `primacy_decay`) up to ~5× baseline

**Question**: Does stronger encoding of competitors produce more interference with film recall?

### Sweep 2: Interference encoding drift rate

**What varies**: `interference_drift_rate` — encoding drift rate during the interference phase

**What's fixed**: `interference_mcf_lr` = informative value from Sweep 1, `M = M_max`

**Values**: ~10 values from near-zero (competitors stay in film context) to ~1.0 (maximal drift per step)

**Question**: How does context proximity between competitors and film items affect interference? Where does the ceiling emerge?

### Sweep 3: Competitor count

**What varies**: `M` — number of interference items (implemented via zero-padding)

**What's fixed**: `interference_drift_rate` and `interference_mcf_lr` from Sweeps 1-2

**Values**: ~10 values from 1 to M_max (e.g. 1, 2, 4, 6, 8, 10, 12, 14, 16)

**Question**: Do more competitors produce more interference? Where do diminishing returns emerge due to context drift?

### Sweep 4: Context trajectory (diagnostic, not a separate sweep)

Record `dot(normalize(context.state), reference)` after each encoding step in the interference phase, for several values of `interference_drift_rate`. Reference = context state at end of film phase. Visualizes how quickly context leaves the film region.

## Visualization

### Figures 1-3: Parameter sweep SPCs

- X: Study position (1..32). Y: Recall probability
- Lines: one per swept value, colored light→dark
- Vertical dashed line at position 16 separating "Film" / "Interference"
- Region labels above plot

### Figure 4: Context trajectory

- X: Interference encoding step (1..M_max). Y: Similarity to film-end context
- Lines: one per drift rate value

### Summary DV panels (one per sweep)

- X: Swept parameter value. Y: Film items recalled (mean ± 95% CI across subjects)

## Implementation

### Library code: `jaxcmr/selective_interference/`

#### `paradigm.py`

Core function:

```python
def simulate_film_and_interference_trial(
    model: MemorySearch,
    film_items: Integer[Array, " n_film"],
    interference_items: Integer[Array, " m_max"],
    interference_drift_rate: Float_,
    interference_mcf_lr: Float_,
    rng: PRNGKeyArray,
) -> tuple[MemorySearch, Integer[Array, " recall_events"]]:
```

1. Encode film items via `lax.fori_loop` + `model.experience()`
2. `model.replace(encoding_drift_rate=..., _mcf_learning_rate=...)` for interference
3. Encode interference items via `lax.fori_loop` (zero-padded)
4. `start_retrieving()` + `simulate_free_recall()`

Also: a vmappable wrapper that creates a CMR from per-subject parameters, runs the trial, and returns recalls.

#### `context_tracking.py`

```python
def track_context_trajectory(
    model: MemorySearch,
    items: Integer[Array, " n_items"],
    reference: Float[Array, " context_size"],
) -> Float[Array, " n_items"]:
```

Uses `lax.scan` to encode items one at a time, recording `dot(normalize(state), normalize(reference))` after each step.

#### `plotting.py`

- `plot_interference_spc()` — multi-line SPC with film/interference boundary
- `plot_context_trajectory()` — similarity vs encoding step
- `plot_summary_dv()` — swept parameter vs film items recalled

### Notebook: `projects/selective_interference/code/sim2_interference_density.ipynb`

1. Imports and setup
2. Fit / load fitted parameters (WeirdCMRNoStop on HealeyKahana2014)
3. Define sweep values
4. Sweep 1: MCF learning rate → SPC + summary DV
5. Sweep 2: Encoding drift rate → SPC + summary DV
6. Sweep 3: Competitor count → SPC + summary DV
7. Sweep 4: Context trajectory diagnostic
8. Discussion of findings and parameter values to carry forward

### Per-subject simulation

For each sweep value, for each subject × replication: create CMR with subject's fitted parameters and `list_length=32`, run `simulate_film_and_interference_trial()`, collect recalls. Aggregate across replications and subjects for SPC + summary DVs.

Vmappable across subjects × replications (same pattern as `parameter_shifted_simulate_h5_from_h5` but without h5 data structures).

## Key code paths

| What | Where |
|------|-------|
| `CMR.__init__` | `jaxcmr/models/cmr.py:40` |
| `experience()` / `experience_item()` | `cmr.py:136` / `cmr.py:105` |
| `_mcf_learning_rate` (primacy gradient) | `cmr.py:80-82`, property at `cmr.py:94` |
| `start_retrieving()` | `cmr.py:157` |
| `simulate_free_recall()` | `jaxcmr/simulation.py:143` |
| `TemporalContext.integrate()` | `jaxcmr/components/context.py:45` |
| `LinearMemory.associate()` / `.probe()` | `jaxcmr/components/linear_memory.py:30` / `:47` |
| `init_mcf()` / `init_mfc()` | `linear_memory.py:91` / `:69` |
| `NoStopTermination` | `jaxcmr/components/termination.py` |
| `plot_spc()` | `jaxcmr/analyses/spc.py:72` |
| Existing parameter-shifting demo | `projects/selective_interference/code/start_drift_rate_parameter_shifting.ipynb` |

## Verification

1. **Sanity check**: Film-only condition (M=0) should reproduce standard SPC shape from HealeyKahana2014 fits
2. **MCF sweep**: Increasing MCF learning rate should monotonically decrease film items recalled
3. **Drift rate sweep**: Lower drift rates should produce more interference (competitors stay closer to film context)
4. **Count sweep**: More competitors should reduce film recall, with diminishing returns visible
5. **Context trajectory**: Should show exponential-like decay of similarity with encoding steps, steeper for higher drift rates
6. **Recency gradient**: Interference should disproportionately suppress late film items (positions near the boundary) more than early film items
