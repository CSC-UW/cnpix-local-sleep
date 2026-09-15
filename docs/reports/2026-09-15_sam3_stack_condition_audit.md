---
title: SAM3 annotation stacks — what each condition directory actually contains
updated: 2026-09-15
---

# SAM3 annotation stacks: condition-directory audit

Question answered: do the image stacks stored under `condition=Late.NOD` and
`condition=Early.NOD` hold the mixed Wake+NREM window of that name, or the
pure-wake `*.Wake` window?

Method: for each of the 26 `annotation-grid` (subject, probe) pairs, read
`timestamps.zarr` for the `Late.NOD` and `Early.NOD` stacks and compute the
fraction of stack samples covered by the bare condition hypnogram vs. the `.Wake`
one (`hyp.load_statistical_condition_hypnograms`). NREM seconds in the bare
window are from the same hypnograms.

## Result

| stack dir | written | content matches | pairs |
| --- | --- | --- | --- |
| `condition=Late.NOD` | 2026-02 (notebook, pre-refactor) | `Late.NOD.Wake` (coverage 1.000; bare `Late.NOD` 0.83–1.00) | 24 of 26 |
| `condition=Late.NOD` | 2026-02 | neither (see Giuseppe) | CNPIX7-Giuseppe imec0, imec1 |
| `condition=Early.NOD` | 2026-07 (`scripts/write_sam3_stacks.py`) | bare `Early.NOD` (mixed) | 26 of 26; differs from `Early.NOD.Wake` in 5 pairs |

Pairs where `Early.NOD` != `Early.NOD.Wake` (NREM seconds inside the bare
1 h window; fraction of stack covered by `Early.NOD.Wake`): CNPIX3-Valentino
imec0 (701 s; 0.805), CNPIX4-Doppio imec0/imec1 (23 s; 0.994), CNPIX16-Walter
imec0 (44 s; 0.988), CNPIX18-Pier imec0/imec1 (37 s; 0.990). For the other
21 pairs the two windows are identical.

So the two NOD stack families were built by different rules:

- `Late.NOD` directories hold **`Late.NOD.Wake`** content. The directory name is
  wrong, not merely truncated. Manual labels and model labels for these stacks
  are correctly filed under `condition=Late.NOD.Wake`;
  `cnpix.evaluation.config.stack_condition()` maps between the two names.
- `Early.NOD` directories hold **`Early.NOD`** (mixed) content, because the
  July script passes the hypnogram key verbatim and the writer selects
  `hgs[condition]` (`stacks/write.py::load_data`,
  `trace_io.open_preprocessed_traces_as_xarray`). Whether the wake-only window
  was intended is an open question for the user; if it was, the five affected
  pairs need regenerating with `--conditions Early.NOD.Wake`.

## CNPIX7-Giuseppe

Giuseppe's condition hypnograms (`shared_s3/novel_objects_deprivation/CNPIX7-Giuseppe/imec*.condition_hypnograms.parquet`)
were regenerated on 2026-08-12; every other subject's date from 2026-03-09. Both
Giuseppe stack batches predate that, so its stacks no longer line up with the
current windows:

| probe | stack | stack span (s) | current `Late.NOD.Wake` window (s) | coverage |
| --- | --- | --- | --- | --- |
| imec0 | Late.NOD | 99278–107847 | 103514–107850 | 0.033 |
| imec1 | Late.NOD (837 chunks) | 92142–107657 | 103509–107738 | 0.007 |
| imec1 | Early.NOD | 92142–99629 | 92142–96678 | 0.925 |

Giuseppe has manual `Late.NOD.Wake` labels on both probes and SAM3 model labels
on both, all painted/inferred on these stale stacks. Any evaluation that
restricts by the current hypnogram (rather than by the stack's own
`timestamps.zarr`) will effectively drop Giuseppe's wake data. Regenerating the
stacks would orphan the labels, so this is flagged, not fixed.

## Related housekeeping done the same day

- `sam_stack_tarballs/` restructured from flat `{subject}_{probe}_{condition}.tar.gz`
  to a mirror of the stack tree
  (`{subject}/method=sam3/probe={probe}/condition={condition}.tar.gz`); 104
  archives moved unchanged; README added there; `write_sam3_stacks.py` updated.
- README added at `shared_s3/offproj/novel_objects_deprivation/` describing the
  manual-label layout and symlink scheme.
- Verified all 104 legacy date-folder SAM3 label files under
  `nobak/.../method=sam3/{05-09-2026,05-26-2026}/` byte-identical to their
  per-recording `model=` copies (`cmp`, twice), then deleted the date folders.
  `samoffs.config.SAM3_MODELS` is now keyed by `model=` id and the one-shot
  migration script is gone.

## Why `Late.NOD` and `Early.NOD` stacks differ

Before 2026-03-09, offproj translated its dotted condition names to
`wisc_ecephys_tools` keys through `const.CONDITION_NAME_MAP`, in which
`"Late.NOD" -> "late_nod_wake"` and `"Early.NOD" -> "early_nod_wake"`. The
February stacks were written under that alias, so `condition=Late.NOD` holds the
wake-only window. The 2026-03-09 rename made `Late.NOD`/`Early.NOD` native keys
for the mixed windows, added `*.Wake` keys, and dropped the alias map; the July
script passed `Early.NOD` verbatim into the new semantics. Same code path,
different meaning of the name. The manuscript's NOD contrast
(`const.CONTRASTS["NOD.Incline"]`) was carried over as the `.Wake` pair.
