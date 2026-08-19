"""Pooled, condition-agnostic unit-based OFF detection.

For a (subject, probe, structure), pool the spike trains of all units into one
multiunit train and run an ``on_off_detection`` method separately on each
macro-state (NREM, NOD-Wake). OFFs from both passes are concatenated (both in
original recording time), mapped to the shared ``Off`` schema, and written as a
single condition-agnostic ``offs.parquet``. Downstream consumers assign a
condition by tagging each OFF with the statistical condition hypnogram that
covers it.

The pooled population firing rate gates the algorithm: HMM methods (hmmem,
sticky) need ~100 Hz, so below :data:`cnpix_local_sleep.unit_based.const.MIN_POOLED_FR`
detection falls back to the threshold method. If the requested HMM method raises
a detection error it also falls back to threshold. The algorithm actually used
per pass is recorded in the detection-info file.
"""

import pickle

import numpy as np
import on_off_detection
import pandas as pd
import wisc_ecephys_tools as wet
from on_off_detection import utils as oo_utils
from on_off_detection.methods.exceptions import ALL_METHOD_EXCEPTIONS

from cnpix_local_sleep import hyp, units
from cnpix_local_sleep.unit_based import const as ub_const
from cnpix_local_sleep.unit_based import files, loading


def _pooled_fr(trains, bouts_df):
    """Pooled multiunit firing rate (Hz) within the bouts."""
    total_dur = float(bouts_df["duration"].sum())
    if total_dur <= 0:
        return 0.0
    n = sum(
        len(oo_utils.subset_sorted_train(bouts_df, np.sort(np.asarray(t))))
        for t in trains
    )
    return n / total_dur


def _gate_algo(requested_algo: str, pooled_fr: float) -> str:
    """Algorithm to actually run given the pooled FR.

    Only ``hmmem`` (Chen 2009 GLM-HMM) genuinely needs ~100 Hz; below
    :data:`cnpix_local_sleep.unit_based.const.MIN_POOLED_FR` it falls back to ``sticky``
    (robust at low FR). ``sticky`` and ``threshold`` run at any firing rate.
    (Genuine HMM *fit* failures fall back to ``threshold`` in :func:`_detect_pass`.)
    """
    if requested_algo == "hmmem" and pooled_fr < ub_const.MIN_POOLED_FR:
        return "sticky"
    return requested_algo


def macro_state_bouts(subject, probe, *, hgs=None) -> dict[str, pd.DataFrame]:
    """Canonical NREM / NOD-Wake macro-state bouts for pooled OFF detection.

    Single source of truth for the two detection passes, shared by
    :func:`do_structure` (the full-recording pipeline) and the interactive tuner
    (:mod:`cnpix_local_sleep.unit_based.interactive`). Keeping the bout definitions here
    means the tuner and the pipeline cannot drift. Pass ``hgs`` (the statistical
    condition hypnograms) to avoid re-loading them.

    Returns a dict ``{"NREM": bouts_df, "NOD-Wake": bouts_df}`` whose frames carry
    the ``start_time``/``end_time``/``duration``/``state`` columns expected by
    :class:`on_off_detection.OnOffModel`.
    """
    if hgs is None:
        hgs = hyp.load_statistical_condition_hypnograms(subject, probe)
    hg_nrem = hgs["Full.Conservative"].keep_states(ub_const.NREM_STATES)
    hg_wake = hgs["NOD.Wake"].keep_states(ub_const.WAKE_STATES)
    return {"NREM": hg_nrem._df, "NOD-Wake": hg_wake._df}


def _detect_pass(trains, bouts_df, requested_algo, *, params_override=None, verbose=True):
    """Run pooled OnOffModel for one macro-state; return (off_rows, info).

    hmmem falls back to sticky below the FR gate; any HMM *fit* failure falls
    back to threshold. ``params_override`` (a dict) is merged onto the algorithm's
    default :data:`cnpix_local_sleep.unit_based.const.UNIT_BASED_PARAMS` when the run uses the
    *requested* algorithm; it is dropped on a gate/fallback to a different method
    (its keys are algorithm-specific). This is the hook the interactive tuner uses
    to recompute at tuned parameters without duplicating detection logic.
    """
    pooled_fr = _pooled_fr(trains, bouts_df)
    algo = _gate_algo(requested_algo, pooled_fr)
    if algo != requested_algo and verbose:
        print(
            f"  pooled FR={pooled_fr:.1f}Hz < {ub_const.MIN_POOLED_FR}Hz; "
            f"using '{algo}' instead of '{requested_algo}'"
        )

    def _run(method):
        params = dict(ub_const.UNIT_BASED_PARAMS[method])
        if params_override and method == requested_algo:
            params.update(params_override)
        model = on_off_detection.OnOffModel(
            trains,
            bouts_df,
            method=method,
            params=params,
            verbose=verbose,
        )
        return model.run()

    try:
        on_off_df, info = _run(algo)
    except ALL_METHOD_EXCEPTIONS as e:
        if algo == "threshold":
            raise
        print(f"  '{algo}' raised {e!r}; falling back to 'threshold'.")
        algo = "threshold"
        on_off_df, info = _run("threshold")

    offs = on_off_df[on_off_df["state"] == "off"].copy()
    info = {
        **info,
        "algo_used": algo,
        "pooled_fr": pooled_fr,
        "n_offs": int(len(offs)),
        "n_bouts": int(len(bouts_df)),
        "bouts_duration": float(bouts_df["duration"].sum()),
    }
    return offs, info


def do_structure(
    subject,
    probe: str,
    structure: str,
    *,
    algo: str = ub_const.DEFAULT_ALGO,
    params: dict | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """Detect and persist pooled OFFs for one (subject, probe, structure).

    ``params`` is an optional dict of per-algorithm parameter overrides merged
    onto :data:`cnpix_local_sleep.unit_based.const.UNIT_BASED_PARAMS` for the chosen ``algo``
    (e.g. ``{"off_rate_max": 5.0}`` for sticky). Used by the interactive tuner's
    "Persist OFFs" action to write the tuned result through the standard pipeline.
    """
    if algo not in ub_const.ALGOS:
        raise ValueError(f"algo must be one of {ub_const.ALGOS}, got {algo!r}")

    sglx_subject = (
        wet.get_sglx_subject(subject) if isinstance(subject, str) else subject
    )
    subject = sglx_subject.name

    out_path = files.get_full_offs_path(subject, probe, structure, algo)
    if out_path.exists() and not overwrite:
        print(f"OFFs exist, skipping (use overwrite=True): {out_path}")
        return

    sorting = units.load_structure_sorting(
        subject, probe, structure, unit_quality="all"
    )
    properties = sorting.properties
    if not len(properties):
        print(f"No units for {subject} {probe} {structure}; skipping.")
        return

    depths = properties["depth"].to_numpy(dtype=float)
    depth_lo = float(np.nanmin(depths))
    depth_hi = float(np.nanmax(depths))

    # Use the statistical condition hypnograms (same source as the downstream
    # condition tagging). NREM pass = all NREM; NOD-Wake pass = the sleep-
    # deprivation wake (matches the Early/Late.NOD.Wake core conditions).
    hgs = hyp.load_statistical_condition_hypnograms(subject, probe)
    hg_full = hgs["Full.Conservative"]
    passes = macro_state_bouts(subject, probe, hgs=hgs)

    all_trains = sorting.get_cluster_trains(
        return_times=True,
        start_time=float(hg_full.start_time.min()),
        end_time=float(hg_full.end_time.max()),
    )
    trains = [np.asarray(all_trains[cid]) for cid in properties["cluster_id"]]

    off_frames = []
    infos = {}
    for pass_name, bouts_df in passes.items():
        if not len(bouts_df) or float(bouts_df["duration"].sum()) <= 0:
            print(f"  {pass_name}: no bouts; skipping pass.")
            continue
        if verbose:
            print(f"  {pass_name}: {len(bouts_df)} bouts, algo={algo}")
        offs, info = _detect_pass(trains, bouts_df, algo, params_override=params, verbose=verbose)
        info["n_units"] = len(properties)
        info["low_confidence"] = bool(
            info["pooled_fr"] < ub_const.LOW_CONFIDENCE_POOLED_FR
        )
        offs["pass"] = pass_name
        off_frames.append(offs)
        infos[pass_name] = info

    all_offs = (
        pd.concat(off_frames, ignore_index=True) if off_frames else pd.DataFrame()
    )
    off_frame = loading.on_off_df_to_off_frame(
        all_offs, depth_lo=depth_lo, depth_hi=depth_hi, binsize=ub_const.BINSIZE
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    off_frame.to_parquet(out_path, index=False)
    info_path = files.get_full_detection_info_path(subject, probe, structure, algo)
    with open(info_path, "wb") as f:
        pickle.dump(infos, f)
    print(
        f"Wrote {len(off_frame)} OFFs -> {out_path} "
        f"(passes: {[(k, v['algo_used'], v['n_offs']) for k, v in infos.items()]})"
    )


def do_experiment(
    *, algo: str = ub_const.DEFAULT_ALGO, overwrite: bool = False, verbose: bool = True
) -> None:
    """Run pooled OFF detection for every included (subject, probe, structure)."""
    from cnpix_local_sleep import sps_conf

    # Cortex only: unit-based detection targets cortical OFFs.
    spsl = sps_conf.get_subject_probe_structure_list(
        method=files.METHOD,
        exclude_thalamus=True,
        exclude_striatum=True,
        exclude_other=True,
    )
    for subject, probe, structure in spsl:
        print(f"=== {subject} {probe} {structure} (algo={algo}) ===")
        try:
            do_structure(
                subject,
                probe,
                structure,
                algo=algo,
                overwrite=overwrite,
                verbose=verbose,
            )
        except Exception as e:  # noqa: BLE001 - keep going across structures
            print(f"  FAILED {subject} {probe} {structure}: {e!r}")
