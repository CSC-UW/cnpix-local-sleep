"""Banded (spatially-resolved) unit-based OFF detection.  PROVISIONAL -- under
validation; not wired into the production CLI / aggregation path yet.

Where pooled detection (:mod:`cnpix_local_sleep.unit_based.pipeline.detect_full`) merges every
unit in a structure into one multiunit train (so its OFFs carry no depth footprint),
*banded* detection runs :class:`on_off_detection.SpatialOffModel`: it pools units
within depth bands (a multi-scale ladder by default), detects OFFs per band, and
merges them within and across bands into a single depth x time OFF catalogue. The
resulting :class:`cnpix_local_sleep.off_tables.Off` rows therefore carry real ``lo`` /
``hi`` / ``span`` / ``center_of_mass_depth`` (the morphology fields -- trace
amplitudes, onset/offset propagation, laminar areas -- stay NaN, since pooling per
band gives no per-channel traces).

On disk this is ``detection_mode=banded-{algo}`` (see
:func:`cnpix_local_sleep.unit_based.files.banded_detection_mode`), kept distinct from the pooled
mode and from ``morphological`` spatial OFFs.

Two knobs mirror the engine's two strategies:

- band definition: ``band_definition="fixed_tiled"`` (ladder of ``band_sizes`` um,
  tiled from ``tile_start``) or ``"greedy_fr"`` (grow to a pooled-FR target).
- parameters: ``param_strategy="shared"`` (one param dict for all bands, e.g.
  sticky + ``off_rate_max=0.0``) or ``"adaptive"`` (per-band off-rate cap via
  :func:`build_per_band_params`, reusing the cap-sweep ``per_unit`` formula).
"""

from __future__ import annotations

import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
import wisc_ecephys_tools as wet
from on_off_detection import SpatialOffModel
from tqdm import tqdm

from cnpix_local_sleep import hyp, units
from cnpix_local_sleep.unit_based import const as ub_const
from cnpix_local_sleep.unit_based import files, loading
from cnpix_local_sleep.unit_based.pipeline.detect_full import macro_state_bouts

# Default multi-scale ladder of band sizes (um). None = one whole-structure band.
DEFAULT_BAND_SIZES = [250.0, 500.0, None]

# Hz per unit for the adaptive "per_unit" off-rate cap (anchor from the cap-scheme
# sweep: 20 Hz / 224 units on CNPIX12 M2; script removed 2026-08-12, in git history).
R_UNIT_HZ = 0.0893

# Production rollout config. Uses fixed_tiled (a fixed-size window slid across the
# structure's depth extent), NOT greedy_fr: banded detection exists to resolve OFFs in
# depth, so bands must be uniform and comparable across structures/depths -- greedy_fr's
# FR-equalized bands are coarse and FR-dependent (they scored a higher *F1* on M2 only by
# over-detecting less, i.e. by throwing away spatial resolution) and crash on structures
# whose total pooled FR is below band_min_fr. fixed_tiled is FR-independent (tiles geometry,
# soft-drops sparse bands), so it carries uniform resolution and never hits that crash.
# sticky/cap0 ("OFF = silent") + 50 ms pre-merge + 80 ms post-merge UNION floor. NOTE: the
# duration floors were tuned with greedy bands on M2 (n=1); they may want a re-check on
# fixed_tiled, and the band window size is the obvious knob to revisit.
ROLLOUT_CONFIG: dict[str, Any] = {
    "algo": "sticky",
    "band_definition": "fixed_tiled",
    "band_sizes": [250.0],
    "tile_start": "superficial",
    "param_strategy": "shared",
    "min_band_off_duration": 0.05,
    "min_merged_off_duration": 0.08,
}

# Approx peak RSS per concurrent structure (GB), measured on CNPIX12 M2 (~21 GB). Used to
# RAM-cap structure-level concurrency in the parallel sweep (do_experiment_banded).
_BANDED_PEAK_GB = 22.0


def build_per_band_params(
    model: SpatialOffModel,
    base_params: dict,
    *,
    scheme: str = "per_unit",
    r_unit_hz: float = R_UNIT_HZ,
) -> list[dict]:
    """Per-band parameter dicts for the *adaptive* strategy (sticky ``off_rate_max``).

    Each band gets a copy of ``base_params`` with its OFF-silence cap scaled by the
    band's population:

    - ``"per_unit"``: ``off_rate_max = r_unit_hz * n_units`` (Hz) -- the cap-sweep
      per-unit scheme; a band with more units may tolerate a slightly higher OFF rate.
    - ``"cap0"``: ``off_rate_max = 0.0`` for every band (degenerate -- equals the
      shared cap0 strategy; provided for symmetry).

    Assign the result to ``model.per_band_on_off_params``. Sticky-specific (keys the
    ``threshold``/``hmmem`` methods do not accept).
    """
    out = []
    for _, row in model.bands_df.iterrows():
        n_units = int(len(row["band_cluster_indices"]))
        p = dict(base_params)
        if scheme == "cap0":
            p["off_rate_max"] = 0.0
        elif scheme == "per_unit":
            p["off_rate_max"] = float(r_unit_hz * n_units)
        else:
            raise ValueError(
                f"Unrecognized adaptive scheme={scheme!r}; use 'per_unit' or 'cap0'."
            )
        out.append(p)
    return out


def _assemble_spatial_params(
    *,
    band_definition,
    band_sizes,
    tile_start,
    min_band_off_duration,
    min_merged_off_duration,
    spatial_params,
) -> dict:
    """Build the engine ``spatial_params`` dict, plumbing the duration-cleaning knobs.

    Pure (no I/O) so it is unit-testable. ``min_band_off_duration`` (pre-merge floor on
    raw band-OFFs) and ``min_merged_off_duration`` (post-merge floor on the merged OFF's
    UNION duration = full extent) are only injected when set (else the engine defaults
    apply). An explicit ``spatial_params`` dict wins (escape hatch), matching the
    pre-existing override behaviour.
    """
    sp = {
        "band_definition": band_definition,
        "band_sizes": list(band_sizes),
        "tile_start": tile_start,
    }
    if min_band_off_duration is not None:
        sp["min_band_off_duration"] = float(min_band_off_duration)
    if min_merged_off_duration is not None:
        sp["min_merged_off_duration"] = float(min_merged_off_duration)
    if spatial_params:
        sp.update(spatial_params)
    return sp


def banded_on_off_df_to_off_frame(
    off_df: pd.DataFrame,
    *,
    binsize: float = ub_const.BINSIZE,
) -> pd.DataFrame:
    """Map a merged :class:`SpatialOffModel` OFF dataframe to the ``Off`` schema.

    Sister to :func:`cnpix_local_sleep.unit_based.loading.on_off_df_to_off_frame`, but fills the
    spatial fields with the real per-OFF depth extent instead of structure-wide
    constants. The reported ``start_time``/``end_time``/``duration`` are the merged OFF's
    union window (full extent, ``union_start_time`` -> ``union_end_time``).

    Args:
        off_df: ``SpatialOffModel.run()`` output (one row per merged OFF, with
            ``union_start_time``/``union_end_time``/``union_duration``/
            ``lo``/``hi``/``span``/``N_merged``).
        binsize: detection bin size (s); used for the ``area`` time-bin proxy.

    Returns:
        DataFrame with the ``Off`` columns plus ``max_span`` (the per-structure max of
        ``span``, so ``span_rel2max`` varies across OFFs), sorted by ``start_time``.
    """
    n = len(off_df)
    if n == 0:
        return loading.empty_off_frame()

    start = off_df["union_start_time"].to_numpy(dtype=float)
    end = off_df["union_end_time"].to_numpy(dtype=float)
    dur = off_df["union_duration"].to_numpy(dtype=float)
    lo = off_df["lo"].to_numpy(dtype=float)
    hi = off_df["hi"].to_numpy(dtype=float)
    span = off_df["span"].to_numpy(dtype=float)
    nan = np.full(n, np.nan)
    max_span = float(np.nanmax(span)) if n else np.nan

    # area: the true depth x time FOOTPRINT (union of constituent band boxes), in
    # bin*um, when the engine provides ``union_area`` (s*um) -- depth-aware, unlike the
    # bounding box. Falls back to the time-bin proxy (duration/binsize) otherwise.
    if "union_area" in off_df.columns:
        area = np.maximum(
            1, np.round(off_df["union_area"].to_numpy(dtype=float) / binsize)
        ).astype(int)
    else:
        area = np.maximum(1, np.round(dur / binsize)).astype(int)

    frame = pd.DataFrame(
        {
            "label": np.arange(n, dtype=int),
            "area": area,
            "start_time": start,
            "end_time": end,
            "duration": dur,
            # No per-channel onset/offset structure; medians equal the scalar values.
            "median_start_time": start,
            "median_end_time": end,
            "median_duration": dur,
            # REAL per-OFF depth extent (the point of banded detection).
            "lo": lo,
            "hi": hi,
            "span": span,
            "median_trace": nan,
            "min_trace": nan,
            "mad_trace": nan,
            "center_of_mass_time": (start + end) / 2.0,
            "center_of_mass_depth": (lo + hi) / 2.0,
            "onset_slope": nan,
            "onset_jitter": nan,
            "onset_r2": nan,
            "onset_mad": nan,
            "offset_slope": nan,
            "offset_jitter": nan,
            "offset_r2": nan,
            "offset_mad": nan,
            "supra_area": nan,
            "infra_area": nan,
            "max_supra_nchans": nan,
            "max_infra_nchans": nan,
            "max_span": np.full(n, max_span),
        }
    )
    return frame.sort_values("start_time").reset_index(drop=True)


def load_structure_inputs(subject, probe: str, structure: str) -> dict | None:
    """Load the band-definition-INDEPENDENT inputs for one structure.

    Returns ``{"trains", "depths", "cluster_ids", "hgs"}`` -- the expensive sorting +
    spike-train load (~30-40 s/structure on NFS) -- or ``None`` if the structure has no
    units. Pass the result as ``detect_structure_banded(..., preloaded=...)`` to run
    several band definitions on one structure without reloading: the spike-train load
    dominates, while detection bounded to the labeled chunks is only a few seconds, so
    loading once and reusing across passes is ~one full load cheaper per structure.
    """
    sglx_subject = (
        wet.get_sglx_subject(subject) if isinstance(subject, str) else subject
    )
    subject = sglx_subject.name
    sorting = units.load_structure_sorting(
        subject, probe, structure, unit_quality="all"
    )
    properties = sorting.properties
    if not len(properties):
        return None
    depths = properties["depth"].to_numpy(dtype=float)
    hgs = hyp.load_statistical_condition_hypnograms(subject, probe)
    hg_full = hgs["Full.Conservative"]
    all_trains = sorting.get_cluster_trains(
        return_times=True,
        start_time=float(hg_full.start_time.min()),
        end_time=float(hg_full.end_time.max()),
    )
    cluster_ids = list(properties["cluster_id"])
    trains = [np.asarray(all_trains[cid]) for cid in cluster_ids]
    return {
        "trains": trains,
        "depths": depths,
        "cluster_ids": cluster_ids,
        "hgs": hgs,
    }


def detect_structure_banded(
    subject,
    probe: str,
    structure: str,
    *,
    algo: str = "sticky",
    band_definition: str = "fixed_tiled",
    band_sizes=DEFAULT_BAND_SIZES,
    tile_start: str = "superficial",
    min_band_off_duration: float | None = None,
    min_merged_off_duration: float | None = None,
    spatial_params: dict | None = None,
    param_strategy: str = "shared",
    adaptive_scheme: str = "per_unit",
    params: dict | None = None,
    bouts_by_pass: dict | None = None,
    preloaded: dict | None = None,
    return_artifacts: bool = False,
    persist: bool = False,
    overwrite: bool = False,
    verbose: bool = True,
) -> tuple:
    """Run banded OFF detection for one (subject, probe, structure).

    Mirrors :func:`cnpix_local_sleep.unit_based.pipeline.detect_full.do_structure` for data
    loading and the two macro-state passes (NREM, NOD-Wake), but uses
    :class:`SpatialOffModel`. Returns ``(off_frame, infos)`` in the ``Off`` schema
    (real ``lo``/``hi``/``span``). With ``persist=True`` also writes the parquet under
    ``detection_mode=banded-{algo}``.

    ``bouts_by_pass`` optionally overrides the macro-state bouts with a
    ``{pass_name: bouts_df}`` dict (e.g. bouts clipped to the manually-labeled stack
    chunks), to bound detection to the evaluation domain instead of the full 48 h.

    Morphological duration cleaning (both optional; ``None`` -> engine defaults):
    ``min_band_off_duration`` (s) drops short band-OFFs *before* merging (engine
    default 0.03); ``min_merged_off_duration`` (s) drops *merged* OFFs whose union
    duration (full extent) is shorter than this, *after* merging.

    ``preloaded`` (a dict from :func:`load_structure_inputs`) injects the
    band-definition-independent inputs (spike trains, depths, cluster ids, hypnograms)
    so several band definitions can be scored on one structure without reloading the
    (dominant) spike trains; when ``None`` they are loaded here.

    With ``return_artifacts=True`` returns ``(off_frame, infos, artifacts)`` where
    ``artifacts[pass_name]`` is ``{"off_df", "all_bands_on_off_df"}`` -- the raw merged
    OFFs (with ``merged_band_offs_indices``) and per-band OFFs needed to rasterize the
    true union-of-band-boxes footprint (see :func:`banded_eval.rasterize_banded_union`).
    """
    if algo not in ub_const.ALGOS:
        raise ValueError(f"algo must be one of {ub_const.ALGOS}, got {algo!r}")
    if param_strategy not in ("shared", "adaptive"):
        raise ValueError("param_strategy must be 'shared' or 'adaptive'")

    sglx_subject = (
        wet.get_sglx_subject(subject) if isinstance(subject, str) else subject
    )
    subject = sglx_subject.name

    def _ret(off, infos, artifacts):
        return (off, infos, artifacts) if return_artifacts else (off, infos)

    out_path = files.get_full_banded_offs_path(subject, probe, structure, algo)
    if persist and out_path.exists() and not overwrite:
        print(f"Banded OFFs exist, skipping (use overwrite=True): {out_path}")
        return _ret(
            loading.load_full_banded_offs(subject, probe, structure, algo), {}, {}
        )

    if preloaded is None:
        preloaded = load_structure_inputs(subject, probe, structure)
    if preloaded is None:
        print(f"No units for {subject} {probe} {structure}; skipping.")
        return _ret(loading.empty_off_frame(), {}, {})
    trains = preloaded["trains"]
    depths = preloaded["depths"]
    cluster_ids = preloaded["cluster_ids"]
    passes = bouts_by_pass if bouts_by_pass is not None else macro_state_bouts(
        subject, probe, hgs=preloaded["hgs"]
    )

    base_params = dict(ub_const.UNIT_BASED_PARAMS[algo])
    if params:
        base_params.update(params)

    # Morphological duration cleaning (engine-side; geometry-agnostic). Pre-merge floor
    # drops short band-OFFs; the optional post-merge floor drops short merged OFFs.
    sp = _assemble_spatial_params(
        band_definition=band_definition,
        band_sizes=band_sizes,
        tile_start=tile_start,
        min_band_off_duration=min_band_off_duration,
        min_merged_off_duration=min_merged_off_duration,
        spatial_params=spatial_params,
    )

    off_frames = []
    infos: dict = {}
    artifacts: dict = {}
    for pass_name, bouts_df in passes.items():
        if not len(bouts_df) or float(bouts_df["duration"].sum()) <= 0:
            print(f"  {pass_name}: no bouts; skipping pass.")
            continue
        if verbose:
            print(f"  {pass_name}: {len(bouts_df)} bouts, algo={algo}, "
                  f"band_definition={band_definition}, param_strategy={param_strategy}")

        model = SpatialOffModel(
            trains,
            depths,
            bouts_df,
            cluster_ids=cluster_ids,
            on_off_method=algo,
            on_off_params=base_params,
            spatial_params=sp,
            verbose=verbose,
        )
        if param_strategy == "adaptive":
            model.per_band_on_off_params = build_per_band_params(
                model, base_params, scheme=adaptive_scheme
            )

        off_df = model.run()
        off_frame = banded_on_off_df_to_off_frame(
            off_df, binsize=base_params["binsize"]
        )
        off_frame["pass"] = pass_name
        off_frames.append(off_frame)
        artifacts[pass_name] = {
            "off_df": model.off_df,
            "all_bands_on_off_df": model.all_bands_on_off_df,
        }
        infos[pass_name] = {
            "algo": algo,
            "band_definition": band_definition,
            "param_strategy": param_strategy,
            "n_bands": int(len(model.bands_df)),
            "bands": model.bands_df[["band_lo", "band_hi", "band_span", "band_scale"]]
            .to_dict("records"),
            "n_offs": int(len(off_frame)),
            "n_bouts": int(len(bouts_df)),
            "bouts_duration": float(bouts_df["duration"].sum()),
        }

    if off_frames:
        all_offs = pd.concat(off_frames, ignore_index=True)
        # Recompute max_span over the full (both-pass) frame.
        if len(all_offs):
            all_offs["max_span"] = float(np.nanmax(all_offs["span"].to_numpy()))
    else:
        all_offs = loading.empty_off_frame()

    if persist:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        all_offs.to_parquet(out_path, index=False)
        info_path = files.get_full_banded_detection_info_path(
            subject, probe, structure, algo
        )
        with open(info_path, "wb") as f:
            pickle.dump(infos, f)
        print(f"Wrote {len(all_offs)} banded OFFs -> {out_path}")

    return _ret(all_offs, infos, artifacts)


# Production rollout drivers (full 48 h, persisted)
def do_structure_banded(
    subject,
    probe: str,
    structure: str,
    *,
    overwrite: bool = False,
    verbose: bool = True,
    **config,
) -> None:
    """Detect and persist banded OFFs for one (subject, probe, structure), full 48 h.

    Thin production wrapper over :func:`detect_structure_banded` that pins the rollout
    configuration (:data:`ROLLOUT_CONFIG`) and always persists. Runs both macro-state
    passes (NREM, NOD-Wake) over the whole recording -- no ``bouts_by_pass`` override.
    ``config`` overrides individual rollout keys (e.g. ``algo``, ``band_definition``).
    """
    cfg = {**ROLLOUT_CONFIG, **config}
    detect_structure_banded(
        subject,
        probe,
        structure,
        persist=True,
        overwrite=overwrite,
        verbose=verbose,
        **cfg,
    )


def _worker_init(threads_per_job: int) -> None:
    """Per-worker setup for the parallel sweep: crash diagnostics + capped BLAS/OpenMP
    threads (mirrors :func:`cnpix_local_sleep.harding.gmm._worker_init`).

    Each structure already runs its bands serially (``SpatialOffModel`` n_jobs=1) and the
    sticky method's hot path is sequential numba ``@njit``, so the only thread spawning is
    BLAS/OpenMP oversubscription -- capped here so K concurrent structures do not fight
    over cores.
    """
    import faulthandler

    faulthandler.enable()
    t = str(int(threads_per_job))
    for var in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
    ):
        os.environ[var] = t


def _detect_structure_banded_worker(spc, overwrite, config, threads_per_job=1):
    """Top-level (picklable) worker: detect+persist one structure, never raise.

    Returns ``(spc, status, err)`` with status ``"ok"`` / ``"failed"``. Verbose is forced
    off so K concurrent workers don't interleave per-band logs.

    ``threadpool_limits`` caps BLAS/OpenMP threads at runtime (the reliable cap --
    setting env vars in the pool initializer is too late under ``spawn``, since BLAS pools
    are already sized when the worker imports numpy), so K concurrent structures each use
    ~``threads_per_job`` cores instead of oversubscribing.
    """
    import threadpoolctl

    subject, probe, structure = spc
    try:
        with threadpoolctl.threadpool_limits(limits=int(threads_per_job)):
            do_structure_banded(
                subject, probe, structure, overwrite=overwrite, verbose=False, **config
            )
        return (spc, "ok", None)
    except Exception as e:  # noqa: BLE001 - report and keep the pool alive
        return (spc, "failed", repr(e))


def _available_resources():
    """``(n_cpus, mem_avail_gb)`` for the host (psutil -> /proc/meminfo -> fallback)."""
    n_cpus = os.cpu_count() or 1
    mem_avail_gb = None
    try:
        import psutil

        mem_avail_gb = psutil.virtual_memory().available / 1e9
    except Exception:  # noqa: BLE001 - fall back to /proc
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        mem_avail_gb = int(line.split()[1]) / 1e6  # kB -> GB
                        break
        except OSError:
            mem_avail_gb = None
    return n_cpus, (mem_avail_gb if mem_avail_gb is not None else 64.0)


def _plan_jobs(
    n_pending: int,
    requested_jobs: int,
    threads_per_job: int,
    *,
    mem_avail_gb: float,
    n_cpus: int,
    per_job_gb: float = _BANDED_PEAK_GB,
    cpu_headroom: float = 0.15,
    mem_headroom: float = 0.15,
) -> tuple[int, str]:
    """Clamp the requested concurrency to be a good citizen on the shared host.

    Caps by available RAM (~``per_job_gb``/structure), cores (``jobs*threads`` within
    ``n_cpus*(1-cpu_headroom)``), and the number of pending structures. Returns
    ``(jobs, note)``; ``jobs >= 1``.
    """
    cpu_cap = max(1, int(n_cpus * (1.0 - cpu_headroom)) // max(1, threads_per_job))
    mem_cap = max(1, int(mem_avail_gb * (1.0 - mem_headroom) // per_job_gb))
    jobs = max(1, min(requested_jobs, cpu_cap, mem_cap, max(1, n_pending)))
    note = (
        f"requested={requested_jobs} cpu_cap={cpu_cap} mem_cap={mem_cap} "
        f"pending={n_pending} -> jobs={jobs}"
    )
    return jobs, note


def _pending_banded_structures(spsl, algo, overwrite):
    """Structures still needing detection (banded ``offs.parquet`` absent), unless
    ``overwrite`` (then all)."""
    if overwrite:
        return list(spsl)
    return [
        spc
        for spc in spsl
        if not files.get_full_banded_offs_path(spc[0], spc[1], spc[2], algo).exists()
    ]


def do_experiment_banded(
    *,
    overwrite: bool = False,
    verbose: bool = True,
    n_jobs: int = 1,
    threads_per_job: int = 1,
    **config,
) -> None:
    """Detect and persist banded OFFs for every included cortical (subject, probe,
    structure), mirroring :func:`detect_full.do_experiment`.

    Idempotent: structures whose banded ``offs.parquet`` already exists are skipped
    unless ``overwrite=True``. Keeps going across structures on error.

    ``n_jobs > 1`` runs structures concurrently in a ``spawn`` process pool (each capped to
    ``threads_per_job`` BLAS/OpenMP threads), the established
    :func:`cnpix_local_sleep.harding.detect` pattern. The requested concurrency is clamped by
    available RAM (~22 GB/structure) and cores (leaving headroom on the shared host); see
    :func:`_plan_jobs`. ``n_jobs == 1`` keeps the simple sequential loop.
    """
    import concurrent.futures
    import multiprocessing

    from cnpix_local_sleep import sps_conf

    # Cortex only -- unit-based detection targets cortical OFFs.
    spsl = sps_conf.get_subject_probe_structure_list(
        method=files.METHOD,
        exclude_thalamus=True,
        exclude_striatum=True,
        exclude_other=True,
    )
    algo = {**ROLLOUT_CONFIG, **config}["algo"]
    pending = _pending_banded_structures(spsl, algo, overwrite)
    n_done = len(spsl) - len(pending)
    if not pending:
        print(
            f"All {len(spsl)} cortical structures already have banded-{algo} OFFs; "
            f"nothing to do (use overwrite=True to redo)."
        )
        return

    n_cpus, mem_avail_gb = _available_resources()
    jobs, note = _plan_jobs(
        len(pending), n_jobs, threads_per_job, mem_avail_gb=mem_avail_gb, n_cpus=n_cpus
    )
    print(
        f"Banded sweep (banded-{algo}): {len(pending)} pending / {len(spsl)} cortical "
        f"({n_done} done) | host {n_cpus} cpus, {mem_avail_gb:.0f} GB avail | {note} | "
        f"~{jobs * _BANDED_PEAK_GB:.0f} GB peak, {threads_per_job} thr/job"
    )

    if jobs == 1:
        for subject, probe, structure in pending:
            print(f"=== {subject} {probe} {structure} (banded-{algo}) ===")
            try:
                do_structure_banded(
                    subject, probe, structure,
                    overwrite=overwrite, verbose=verbose, **config,
                )
            except Exception as e:  # noqa: BLE001 - keep going across structures
                print(f"  FAILED {subject} {probe} {structure}: {e!r}")
        return

    # Cap threads BEFORE spawning so children inherit the env at process creation (their
    # numpy/BLAS import then reads the capped value); the worker also applies a runtime
    # threadpool_limits as the authoritative cap.
    for var in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
    ):
        os.environ[var] = str(int(threads_per_job))

    n_ok = n_fail = 0
    mp_context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=mp_context,
        initializer=_worker_init,
        initargs=(threads_per_job,),
    ) as executor:
        futs = {
            executor.submit(
                _detect_structure_banded_worker, spc, overwrite, config, threads_per_job
            ): spc
            for spc in pending
        }
        for fut in tqdm(
            concurrent.futures.as_completed(futs),
            total=len(futs),
            desc=f"banded sweep ({jobs} workers)",
        ):
            try:
                spc, status, err = fut.result()
            except Exception as e:  # noqa: BLE001 - worker process died (OOM/segfault)
                spc, status, err = futs[fut], "failed", repr(e)
            if status == "ok":
                n_ok += 1
            else:
                n_fail += 1
                print(f"  FAILED {spc[0]} {spc[1]} {spc[2]}: {err}")
    print(
        f"Banded sweep done: {n_ok} ok, {n_fail} failed, {n_done} pre-existing "
        f"(of {len(spsl)} cortical)."
    )
