# unit_based: pooled spike-train OFF detection

Detects OFF periods from sorted-unit spike trains, pooled per
`(subject, probe, structure)`, as a spike-based alternative to `morphological`.

Detection is structure-level (no spatial windowing): all units in a structure are
merged into one multiunit spike train and run through an
[`on_off_detection`](../../../../on_off_detection) method. The pooled detector is
the substrate for two things that *are* used downstream: the interactive
`unit-based-off-tuner` (`interactive.py`) and the banded spatially-resolved
detector (`banded.py`), which reuses its macro-state bouts and runs the same
algorithm within depth bands.

The whole-structure-pooled OFF-rate layer (per-condition rates, homeostasis
contrasts, and the agreement analysis vs `morphological`) was removed on 2026-08-12
as non-manuscript; see the note under [CLI](#cli).

## Algorithms (`--algo`)

| algo | source | notes |
| --- | --- | --- |
| `threshold` | Ji & Wilson 2007 / Chen et al. 2009 | histogram auto-threshold + gap merge; vectorized/fast; tolerates lower firing rates |
| `hmmem` | Chen et al. 2009 | 2-state Poisson-GLM HMM with spike-history term |
| `sticky` (default) | Li & La Camera 2025 | 2-state Poisson HMM with a self-transition floor (`min_dwell`) + a near-silence OFF constraint (`off_rate_max`); numba-accelerated. Default `off_rate_max=0.0` = OFF is the silent (zero-count) state |

The near-silence constraint is the key lever against over-detection: pooling
~200 units means the population is rarely fully silent, so an unconstrained HMM
labels mere *dips* as OFF. `off_rate_max` caps the OFF-state firing rate during
EM. A cap-scheme calibration across 21 cortical structures
(scripts `unit_based_cap_sweep.py` / `unit_based_offrate_distributions.py`,
removed 2026-08-12 with the pooled analysis layer; recoverable from git history)
showed a fixed absolute cap does not generalize: `P(silent 10 ms bin)` spans
~50× across cortex, so any absolute Hz value (or count-percentile) sits at a
wildly different operating point per structure. The default is therefore
`off_rate_max=0.0` (cap0): OFF is the silent (zero-count) state,
parameter-free and structure-invariant. cap0 gave the best median F1 vs
morphological BLAS and the deepest, most BLAS-like OFFs; `min_dwell` (not a rate cap)
governs minimum OFF duration. `off_rate_max>0` is still accepted as an absolute Hz
cap but does not generalize (figures survive on NFS under
`detection_mode=pooled-sticky/offrate_calibration/`). Detection scope is
cortex-only.

Firing-rate handling: only `hmmem` (Chen 2009 GLM-HMM) needs ~100 Hz; below
`const.MIN_POOLED_FR` it falls back to `sticky`. `sticky`+cap0 and `threshold`
run at any rate; a low-FR test (2026-06-17) showed sticky+cap0 is robust and
clearly better than `threshold` at >=~60 Hz, roughly tied below. Structures below
`const.LOW_CONFIDENCE_POOLED_FR` (~100 Hz; few units / sparse population) are not
excluded but flagged `low_confidence=True` in `detection_info.pickle`
(alongside `n_units` and `pooled_fr_nrem`) for downstream filtering. Any genuine
HMM *fit* failure still falls back to `threshold`. The algorithm actually used
per pass is recorded in `detection_info.pickle`.

## Flow

1. `pipeline/detect_full.do_structure(subject, probe, structure, algo=...)`
   - loads units via `cnpix_local_sleep.units.load_structure_sorting` (`unit_quality="all"`, MUA)
   - builds NREM and NOD-Wake bout sets from the statistical condition
     hypnograms (`Full.Conservative` / `NOD.Wake`)
   - runs `on_off_detection.OnOffModel` once per macro-state pass
   - maps OFFs to the shared `Off` schema (`loading.on_off_df_to_off_frame`) and
     writes one condition-agnostic `offs.parquet` under
     `method=unit_based/.../detection_mode=pooled-{algo}/`

## Schema fills (pooled OFFs have no spatial blob)

`on_off_df_to_off_frame` fills the `Off` schema as follows: `start_time`,
`end_time`, `duration` are direct; `span`/`lo`/`hi`/`max_span` come from the
structure depth extent (constant, so `span_rel2max == 1`); `area` is a bin-count
proxy (`duration / binsize`); trace amplitudes, onset/offset propagation, and
laminar areas are `NaN`. Pooled OFFs carry no real depth footprint; for that,
use the banded variant below.

## Banded (spatially-resolved) variant: PROVISIONAL

`banded.py` (+ `banded_eval.py`) drives `on_off_detection.SpatialOffModel`: instead
of one structure-wide pooled train, it pools units within depth bands (a
multi-scale ladder by default), detects OFFs per band, and merges them within/across
bands, so the `Off` rows carry a real `lo`/`hi`/`span`/`center_of_mass_depth`
(morphology fields stay `NaN`). On disk it is `detection_mode=banded-{algo}`, distinct
from `pooled-{algo}` and from morphological spatial OFFs.

- `banded.detect_structure_banded(...)`: bands via `fixed_tiled` (ladder of
  `band_sizes` µm, tiled from `tile_start`) or `greedy_fr`; params `shared`
  (sticky+cap0) or `adaptive` (per-band cap, `build_per_band_params`); `time_window`
  `shared` (consensus core) / `longest` (union).
- `banded_eval` + `scripts/banded_unit_based_validation.py`: score vs manual labels
  and morphological (spatial-to-spatial) with the `cnpix_local_sleep.evaluation` kernels;
  `notebooks/banded_unit_based_validation.ipynb` for raster overlays.

First validation (high recall, over-detects) + mechanics:
`docs/reports/banded_unit_based_off_detection.md`.

## CLI

```bash
unit-based-offs detect CNPIX12-Santiago imec0 M2 --algo sticky
unit-based-offs detect-experiment --algo sticky      # all included structures
unit-based-offs detect-banded CNPIX12-Santiago imec0 M2 --algo sticky
unit-based-offs detect-banded-experiment --algo sticky --jobs 16
unit-based-offs precompute-interactive CNPIX12-Santiago imec0 M2
```

`aggregate-rates` and `plot` were removed on 2026-08-12 along with
`aggregate.py` / `compare.py` / `plots.py`: whole-structure-pooled OFF rates and
mua-vs-unit-based agreement are not reported in the manuscript. Their outputs
(`summarized_unit_based_rate_offs.parquet`, `unit_based_rate_contrasts.parquet`,
`detection_agreement.parquet`, `rate_agreement.parquet`) remain on NFS but can no
longer be regenerated in-tree.

## Inclusion

`data/structure_config.csv` (loaded via `sps_conf.load_method_inclusion("unit_based")`).
