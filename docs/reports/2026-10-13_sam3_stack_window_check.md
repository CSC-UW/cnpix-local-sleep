---
title: SAM3 stack window membership after the prb_sync timebase fix
updated: 2026-10-13
---

# SAM3 stack window membership after the prb_sync timebase fix

Question: after the 2026-10-06 prb_sync fix
(`gfys_workspace/docs/reports/2026-10-06_prb_sync_regeneration.md`), do the
existing annotation stacks still hold the samples of their current condition
window, and what would re-rendering them cost the labels painted on them?
No stack, label or timestamps array was changed. Decision (Graham,
2026-10-13): no stack is re-rendered, and the stacks and their labels stay as
they are.

## Scope

All 185 stack directories under
`/Volumes/npx_nfs/nobak/samoffs/novel_objects_deprivation/*/probe=*/condition=*/`:
26 (subject, probe) pairs x 7 conditions (182 stacks), plus CNPIX7-Giuseppe's
three `STALE.json` directories, listed separately at the end.

The kickoff
(`gfys_workspace/docs/plans/2026-10-13_kickoff_sam3_stack_window_check.md`)
named the 65 imec1 stacks. The imec0 stacks of the seven subjects whose
`hypnogram_probe` is imec1 (Charles, Adrian, Santiago, Claude, Hans, Giuseppe,
Allan) are included too: their bouts moved by the imec1 per-file constants
while their imec0 timestamps did not.

## Method

Scripts and outputs live beside the migration's other scripts, under
`/Volumes/scratch/neuropixels/ecephys/prb_sync_timebase/inplace/`:

| script (`scripts/`) | output (`chronic/`) |
| --- | --- |
| `audit_sam3_stack_windows.py` | `audit_sam3_stack_windows.tsv`, one row per stack (per-subject parts and logs in `audit_sam3_stack_windows{,_logs}/`) |
| `audit_sam3_manual_labels_out_of_window.py` | `audit_sam3_manual_labels_out_of_window.tsv`, one row per manual label file |
| `audit_sam3_out_of_window_edges.py` | `audit_sam3_out_of_window_edges.tsv`, distance of out-of-window samples to the nearest window boundary, per stack |

For each stack:

- Coverage of the current window. `in` is the fraction of stack samples inside
  the current condition hypnogram
  (`cnpix_local_sleep.hyp.load_statistical_condition_hypnograms`, which reads
  `wisc_ecephys_tools.rats.cnd_hgs`); `out` is the samples (and seconds)
  outside it, and the number of chunks holding any of them.
- Simulated re-render. The writer selects samples of the whole-recording
  `processed_ap.zarr` time vector with `hg.covers_time`, after the monotonicity
  hotfix in `trace_io.open_si_zarr_recording_as_xarray`, then cuts the
  concatenated selection into 4 s chunks from index 0 and drops the last
  partial chunk (`stacks/write.py::get_stack_params`). The script repeats this
  on the current time vector and hypnogram and compares the result with the
  stack sample by sample: samples a re-render would drop or add, and how many
  leading chunks would come out identical (`prefix`).
- Validation. The same simulation on the old time vector and old hypnograms
  (`times_seg0` of `processed_ap.zarr`, `timestamps.zarr` and
  `*.condition_hypnograms.parquet` from `backup_chronic_2026-10-07/`) has to
  reproduce each stack as it was rendered.
- Per-file constants applied to the imec1 stacks come from
  `chronic/patch_sam3.tsv`.

`of` (fraction of the window's 1 s bins holding a stack sample) is in the TSV
too. Its bin rule here counts the partial bins at every bout edge, so it falls
below 0.9989 for 25 stacks that are identical before and after the fix
(minimum 0.9971, CNPIX5-Alessandro imec0 `Early.REC.NREM.Match`). It is not
comparable with the 2026-09-18 figures, and `in` plus seconds out of window are
used below instead.

## Validation

On the old timebase and old hypnograms the simulation reproduces all 182 stacks
exactly, sample for sample, and old `in` = 1.0000 for every stack. Every stack
sample, before and after the patch, matches a sample of the corresponding
`processed_ap.zarr` time vector to within 1 µs.

## Result

Of the 182 stacks, 88 would be re-rendered identically, 37 differ by at most
0.05 s of samples, and 57 have more than 0.05 s outside the current window.
No chunk of any stack lies wholly outside its window.

By scoring probe and stack probe:

| hypnogram scored on | stack probe | stacks | identical | <= 0.05 s | > 0.05 s | max out (s) | min `in` | subjects |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| imec0 | imec0 | 70 | 70 | 0 | 0 | 0 | 1.0000 | Segundo, Valentino, Doppio, Alessandro, Eugene, Luigi, Walter, Pier, Otto, Ernst |
| imec0 | imec1 | 14 | 0 | 2 | 12 | 103.92 | 0.9890 | Doppio, Pier |
| imec1 | imec0 | 49 | 5 | 5 | 39 | 42.95 | 0.9881 | Charles, Adrian, Santiago, Claude, Hans, Giuseppe, Allan |
| imec1 | imec1 | 49 | 13 | 30 | 6 | 0.33 | 0.9999 | Charles, Adrian, Santiago, Francis, Claude, Hans, Giuseppe |

By condition (26 stacks each):

| condition | identical | <= 0.05 s | > 0.05 s | max out (s) | min `in` | chunks with any out-of-window sample / all chunks |
| --- | --- | --- | --- | --- | --- | --- |
| `Early.BSL.NREM` | 16 | 6 | 4 | 0.40 | 0.9999 | 163 / 23394 |
| `Early.REC.NREM.Match` | 12 | 5 | 9 | 8.29 | 0.9978 | 545 / 29454 |
| `Early.REC.NREM` | 13 | 4 | 9 | 22.01 | 0.9939 | 334 / 23394 |
| `Late.REC.NREM` | 12 | 5 | 9 | 42.95 | 0.9881 | 363 / 23393 |
| `Early.NOD.Wake` | 14 | 6 | 6 | 39.52 | 0.9890 | 252 / 23395 |
| `Late.NOD.Wake` | 10 | 6 | 10 | 17.63 | 0.9951 | 530 / 23391 |
| `NOD` | 11 | 5 | 10 | 103.92 | 0.9945 | 1411 / 121399 |

The 17 stacks with `in` < 0.9989:

| subject | probe | condition | `in` | out (s) | chunks with out samples | manual label files | model filings |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CNPIX10-Charles | imec0 | `Early.REC.NREM` | 0.9988 | 4.30 | 73 / 900 | v1, v2 | 3 |
| CNPIX11-Adrian | imec0 | `Early.REC.NREM` | 0.9982 | 6.64 | 43 / 899 | v1 | 3 |
| CNPIX11-Adrian | imec0 | `Late.NOD.Wake` | 0.9970 | 10.85 | 82 / 900 | | 3 |
| CNPIX11-Adrian | imec0 | `Late.REC.NREM` | 0.9986 | 5.19 | 36 / 900 | | |
| CNPIX15-Claude | imec0 | `Early.REC.NREM` | 0.9978 | 7.91 | 22 / 900 | v1, v2 | 3 |
| CNPIX15-Claude | imec0 | `Early.REC.NREM.Match` | 0.9983 | 8.29 | 105 / 1226 | | 2 |
| CNPIX15-Claude | imec0 | `Late.NOD.Wake` | 0.9953 | 17.07 | 53 / 900 | v1 | 3 |
| CNPIX15-Claude | imec0 | `Late.REC.NREM` | 0.9977 | 8.41 | 21 / 900 | | |
| CNPIX15-Claude | imec0 | `NOD` | 0.9984 | 32.60 | 103 / 5193 | | |
| CNPIX18-Pier | imec1 | `Early.NOD.Wake` | 0.9890 | 39.52 | 104 / 899 | | |
| CNPIX18-Pier | imec1 | `Early.REC.NREM` | 0.9939 | 22.01 | 45 / 900 | v1, v2 | 3 |
| CNPIX18-Pier | imec1 | `Early.REC.NREM.Match` | 0.9978 | 7.59 | 72 / 860 | | 2 |
| CNPIX18-Pier | imec1 | `Late.NOD.Wake` | 0.9951 | 17.63 | 41 / 899 | v1 | 3 |
| CNPIX18-Pier | imec1 | `Late.REC.NREM` | 0.9932 | 24.38 | 62 / 899 | | |
| CNPIX18-Pier | imec1 | `NOD` | 0.9945 | 103.92 | 260 / 4753 | | |
| CNPIX7-Giuseppe | imec0 | `Early.REC.NREM` | 0.9983 | 6.29 | 33 / 900 | | 3 |
| CNPIX8-Allan | imec0 | `Late.REC.NREM` | 0.9881 | 42.95 | 43 / 900 | | |

At `in` < 0.998 the list is 11 stacks; at `in` < 0.995 it is 5 (Pier imec1
`Early.NOD.Wake`, `Early.REC.NREM`, `Late.REC.NREM`, `NOD`; Allan imec0
`Late.REC.NREM`).

### Where the out-of-window samples are

89 stacks have at least one sample outside the current window
(`audit_sam3_out_of_window_edges.tsv`). In 88 of them every such sample lies
within 1 s of a current window boundary, and the largest distance per subject
does not exceed, to that table's rounding, the subject's largest imec1
per-file constant in section 1 of
`gfys_workspace/docs/plans/2026-09-24_prb_sync_timebase_migration.md` (Pier
0.556 s, largest constant 780 ms; Claude 0.429 s, 601 ms; Adrian 0.217 s,
231 ms; Charles 0.072 s, 116 ms; Hans 0.054 s, 55 ms; Doppio 0.022 s, 22.8 ms;
Santiago 0.013 s, 12.6 ms; Francis 0.007 s, 20.2 ms).

Two cases go further:

- CNPIX8-Allan imec0 `Late.REC.NREM`: up to 2.85 s, with 27.8 s of the
  42.95 s more than 1 s from a boundary. The window's bouts from 127825 s on
  lie in imec1 file `3-2-2021_G_g0_t0`, one of the section 5 defect files of
  the migration plan (table slope -22.3 ppm), whose bouts move by 2.854 s
  under the corrected table semantics, as Giuseppe's
  `12-12-2020_BL_24hs_b_g0_t0` moves by 7.57 s. The stack has no manual labels
  and no model filings.
- CNPIX7-Giuseppe imec0 `Early.REC.NREM`: up to 0.95 s, all at the end of the
  window. The other runs are 0.177 s at bout starts. The window is cut where
  3600 s of NREM have accumulated, so those shifts move the cut: the last bout
  now ends at 114930.05 s instead of 114933.97 s, and the stack's last chunk
  runs 0.95 s past it. The current hypnogram also drops a zero-length bout at
  109505.78 s.

Window durations are unchanged (no condition window changed by more than 1 s).

### What a re-render would do

The writer chunks the selection from index 0, so the first sample added or
dropped anywhere shifts every later chunk boundary. Of the 94 stacks that would
change, 61 would differ from their first chunk on and 74 within the first 10 %
of chunks; the median stack would keep none of its chunks unchanged. Nine
stacks would gain one 4 s chunk at the end and six would lose one, from the
last-partial-chunk rule. A re-render therefore cannot be applied as a local
edit; labels painted on the old stack would need re-annotation or a time-based
remap onto the new frames.

### Labels on the stacks that would change

Manual labels: 35 of the 69 manual label files on current stacks sit on 27 of
the 94 changed stacks. In those files, per condition:

| condition | label files | instances | instances touching out-of-window samples | label pixels out of window / all | remap: whole | split across two new images | partly dropped | lost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Early.REC.NREM` | 20 | 5071 | 5 | 4463 / 10999397 | 4968 | 98 | 2 | 3 |
| `Early.REC.NREM.Match` | 3 | 1262 | 0 | 0 / 4174297 | 1223 | 39 | 0 | 0 |
| `Late.NOD.Wake` | 12 | 850 | 8 | 1046 / 564741 | 834 | 8 | 7 | 1 |

The remap columns describe a sample-exact remap of each instance onto the
simulated re-render: kept whole in one new image, split across two, partly on
dropped samples, or wholly on dropped samples. Per file numbers are in
`audit_sam3_manual_labels_out_of_window.tsv`.

Three of the ten manual `Early.REC.NREM.Match` sets are on changed stacks:
CNPIX4-Doppio imec1 (0.13 s out of window), CNPIX7-Giuseppe imec0 (1.85 s) and
CNPIX8-Allan imec0 (0.08 s). No labelled pixel in them lies on an
out-of-window sample. Three of Doppio's 51 examined images contain
out-of-window samples, with no labels on those samples. The other six Match
sets (Segundo v1 and v2, Valentino, Doppio imec0, Alessandro, Eugene, Giuseppe
imec1) are on stacks that would be re-rendered identically.

Model labels: 109 filings on 41 of the changed stacks (2026-05-03_nrem-all 27,
2026-05-23_wake-2to1 27, 2026-08-25_nrem-ap 14, 2026-08-29_nrem-ap-lfp 14,
2026-05-26_wake-1to1 14, 2026-06-16_nrem-loso-valentino 13); 28 of them are on
the 17 stacks with `in` < 0.9989.

## Label times

`timestamps.zarr` gives the time of every label pixel column on the corrected
timebase. The imec0 arrays never changed. The 2026-10-07 patch changed the
time attached to each rendered imec1 sample, not which sample an image shows.
The detector-vs-manual scorers map detections onto the label grid through
these arrays (`cnpix_local_sleep.evaluation.grid.load_stack_times_flat`).

The morphological scorer (`morphological/full48h_eval.py`) keeps only OFFs
whose `start_time` lies in the current condition window
(`aggregate_experiment_offs.load_subset_of_48h_offs`). A manual label on a
stack sample now outside the window can therefore lack a detection to match.
At most the 13 labelled instances in the table above are concerned. The
unit-based scorer detects over spans built from the stack's own chunk times
(`unit_based/banded_eval.py::labeled_chunk_bouts`) and has no such
restriction.

## CNPIX7-Giuseppe stale directories

Against the current windows named in their `STALE.json`:

| probe | directory | `in` | chunks with out samples / wholly out |
| --- | --- | --- | --- |
| imec0 | `Late.NOD.Wake.hyp2026-03-09` | 0.0330 | 872 / 870 of 900 |
| imec1 | `Early.NOD.hyp2026-06-20` | 0.9245 | 80 / 67 of 899 |
| imec1 | `Late.NOD.Wake.hyp2026-03-09` | 0.0069 | 832 / 831 of 837 |

These match the 2026-09-18 audit (0.033, 0.925, 0.007). They stay stale; their
replacements are the current stacks above.
