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

Two-sided check against the **current** hypnogram files: `in` = fraction of
stack samples inside the window; `of` = fraction of 1 s bins of the window that
contain a stack sample. A stack holds 3596–3600 s of a 3600 s window (the last
partial 4 s chunk is dropped), so `of` ≥ 0.9989 is a full match.

| stack dir | written | content matches (current definition) | pairs |
| --- | --- | --- | --- |
| `condition=Late.NOD` | 2026-02 (notebook, pre-refactor) | `Late.NOD.Wake`: in = 1.0000, of ≥ 0.9989; bare `Late.NOD`: in 0.83–1.00 | 24 of 26 |
| `condition=Late.NOD` | 2026-02 | the 2026-03-09 `Late.NOD.Wake` exactly (in = 1.0000), current 0.03 / 0.007 | CNPIX7-Giuseppe imec0, imec1 |
| `condition=Early.NOD` | 2026-07 (`scripts/write_sam3_stacks.py`) | bare `Early.NOD`: in = 1.0000, of ≥ 0.9991 | 25 of 26 |
| `condition=Early.NOD` | 2026-07 | the 2026-06-20 `Early.NOD` (in = 1.0000), current 0.92 | CNPIX7-Giuseppe imec1 |
| `condition=Early.REC.NREM` | 2026-02 | `Early.REC.NREM`: in = 1.0, of ≥ 0.9992 | 26 of 26 |

Pairs where the current `Early.NOD` != `Early.NOD.Wake` (NREM seconds inside the
bare 1 h window; fraction of stack covered by `Early.NOD.Wake`): CNPIX3-Valentino
imec0 (701 s; 0.805), CNPIX4-Doppio imec0 and imec1 (23 s; 0.994), CNPIX16-Walter
imec0 (44 s; 0.988), CNPIX18-Pier imec0 and imec1 (37 s; 0.990). That is six
pairs across four subjects (an earlier draft said five). For the other 19
non-Giuseppe pairs the two windows coincide, and their `Early.NOD` stacks match
the current `Early.NOD.Wake` two-sidedly (of ≥ 0.9991). Regenerating
`Early.NOD.Wake` stacks would therefore touch seven pairs: those six plus
CNPIX7-Giuseppe imec1.

Only Giuseppe's condition hypnograms changed after 2026-03-09 (all other
subjects' parquets still carry that mtime), so for every other pair "current"
and "as of stack generation" are the same file for the July stacks; the
February stacks predate the 2026-03-09 regeneration, but the two-sided match
above is measured against the current files, so they match the current
definition regardless of what they were generated from.

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
were re-edited on 2026-06-20 (twice) and 2026-08-12 (`hypnogram_ephyviewer_edits.csv`
edited that day); the superseded versions survive as `*.bak-2026-06-20`,
`*.bak-2026-06-20-2` and `*.bak-2026-08-12`. Tested against each version, the
February `Late.NOD` stacks match the 2026-03-09 `Late.NOD.Wake` exactly
(in = 1.0000 on both probes) and the July imec1 `Early.NOD` stack matches the
second 2026-06-20 `Early.NOD` exactly. Against the current windows:

| probe | stack | stack span (s) | current `Late.NOD.Wake` window (s) | coverage |
| --- | --- | --- | --- | --- |
| imec0 | Late.NOD | 99278–107847 | 103514–107850 | 0.033 |
| imec1 | Late.NOD (837 chunks) | 92142–107657 | 103509–107738 | 0.007 |
| imec1 | Early.NOD | 92142–99629 | 92142–96678 | 0.925 |

Giuseppe has manual `Late.NOD.Wake` labels on both probes and SAM3 model labels
on both, all painted/inferred on these stale stacks. Regenerating the stacks
would orphan the labels, so on 2026-09-15 the stacks and their labels were
instead renamed to carry the hypnogram version they match:
`condition=Late.NOD.Wake.hyp2026-03-09` (stacks, manual labels, both `model=`
trees, both probes) and `condition=Early.NOD.hyp2026-06-20` (imec1 stack).
Glob-based label discovery filtered on `Late.NOD.Wake` now returns 17 pairs
without Giuseppe.

## Renames performed 2026-09-15

- 24 `condition=Late.NOD` stack directories -> `condition=Late.NOD.Wake`
  (every pair whose stack matches the current window two-sidedly).
- `cnpix.evaluation.config.STACK_CONDITION` / `stack_condition()` removed; the
  evaluation condition is the stack directory name.
- Giuseppe: 9 directories renamed to the versioned names above.
- No tarballs were affected (none exist for `Late.NOD.Wake`).

## Size of a whole-SD stack

Current `SD` windows span 3.8–5.9 h across the 26 pairs (median ≈ 5.4 h; `SD.Wake`
3.5–5.9 h). A 1 h stack is 400–490 MB on disk (Otto `Late.NOD.Wake`: 452 MB,
of which the full-resolution AP level is 327 MB), so a whole-SD stack is
roughly 1.7–2.7 GB per pair, about 2.4 GB at the median, or ≈ 60 GB for the
cohort; tarballs ≈ 0.7×. Chunks stay 4 s, so ≈ 4 900 chunks per stack. The
label arrays scale the same way and are the larger cost if saved uncompressed:
the Otto `Late.NOD.Wake` manual NPZ is 1.64 GB for 900 chunks (`np.savez`),
which would be ≈ 9 GB for a whole SD; `np.savez_compressed` brings that to a
few MB (the model NPZs are 1.6–2.4 MB).

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
