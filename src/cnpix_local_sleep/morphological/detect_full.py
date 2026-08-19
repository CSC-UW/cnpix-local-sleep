"""Full-recording OFF detection with state-dependent thresholds.

Detects OFF periods across the entire recording rather than one
condition at a time. Computes absolute NREM and Wake thresholds from
all data in the respective states, then applies them based on the
hypnogram state at each time point.

State-to-threshold mapping:
- NREM threshold -> NREM, IS, REM
- Wake threshold -> Wake, MA, Other
- Artifact, NoData, uncovered -> excluded (never OFF)

Results are saved condition-agnostically. Downstream filtering by
condition can use OFF timestamps against condition hypnograms.
"""

import numpy as np
import pandas as pd
import scipy.ndimage
import xarray as xr

from cnpix_local_sleep import const, hyp
from cnpix_local_sleep.morphological import detect as morphological_detect
from cnpix_local_sleep.morphological.common import MorphologicalSourceConfig
from cnpix_local_sleep.morphological.types import DetectionOpts
from cnpix_local_sleep.morphological import morphology
from cnpix_local_sleep.morphological.pipeline import utils
from cnpix_local_sleep.off_tables import Off


def _default_full_source_config() -> MorphologicalSourceConfig:
    """Full-recording detection is MUA-only today."""
    from cnpix_local_sleep.morphological.mua import SOURCE_CONFIG

    return SOURCE_CONFIG

# States used to *derive* per-bin thresholds: narrow.
# (Per-bin per-channel quantiles are computed over only these samples.)
NREM_DERIVATION_STATES: tuple[str, ...] = ("NREM",)
WAKE_DERIVATION_STATES: tuple[str, ...] = ("Wake",)

# States whose time points receive the NREM threshold at *application* time.
NREM_THRESHOLD_STATES: tuple[str, ...] = ("NREM", "IS", "REM")

# States whose time points receive the Wake threshold at *application* time.
WAKE_THRESHOLD_STATES: tuple[str, ...] = ("Wake", "MA", "Other")

# States (and the default empty string for uncovered times) that are
# excluded from detection: always False in the binary mask.
EXCLUDED_STATES: tuple[str, ...] = ("Artifact", "NoData", "")


# -------------------- Threshold computation --------------------


def _get_threshold_quantiles(
    subject: str,
    probe: str,
    structure: str,
    source_config: MorphologicalSourceConfig,
) -> tuple[float, float]:
    """Read NREM and Wake quantile values from the variant's thresholds CSV.

    Returns:
        Tuple of (nrem_quantile, wake_quantile).
    """
    # Delegate rather than re-filter the frame: the shared lookup also rejects a
    # NaN quantile, and a NaN threshold fails silently rather than loudly:
    # `values < nan` is False everywhere, so detection would find nothing.
    return (
        source_config.get_quantile_threshold(subject, probe, structure, "NREM"),
        source_config.get_quantile_threshold(subject, probe, structure, "Wake"),
    )


def _compute_state_thresholds(
    da: xr.DataArray,
    state_labels: np.ndarray,
    bin_boundaries: np.ndarray,
    nrem_quantile: float,
    wake_quantile: float,
    ndimage_filter_type: str | None,
    ndimage_filter_kwargs: dict | None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute per-bin per-channel thresholds from NREM and Wake data.

    Derivation is narrow: NREM thresholds use only ``state == "NREM"``
    samples; Wake thresholds use only ``state == "Wake"`` samples.
    Application broadens later (NREM threshold -> ``{NREM, IS, REM}``,
    Wake threshold -> ``{Wake, MA, Other}``).

    Args:
        da: Full-recording DataArray (time, channel), numpy-backed.
        state_labels: Per-sample state label array from
            ``hg.get_states()``.
        bin_boundaries: Sample-index boundaries of shape (n_bins+1,);
            captured from ``da.chunks`` before materialization.
        nrem_quantile: Quantile for the NREM threshold.
        wake_quantile: Quantile for the Wake threshold.
        ndimage_filter_type: Filter type (provenance metadata).
        ndimage_filter_kwargs: Filter kwargs (provenance metadata).

    Returns:
        Tuple of (nrem_thresholds, wake_thresholds), each a 2-D
        DataArray with dims ``(bin, channel)``.
    """
    nrem_mask = np.isin(state_labels, NREM_DERIVATION_STATES)
    wake_mask = np.isin(state_labels, WAKE_DERIVATION_STATES)

    nrem_thresholds = morphological_detect.compute_per_bin_thresholds(
        da,
        nrem_quantile,
        bin_boundaries,
        derivation_mask=nrem_mask,
        threshold_method="from_value",
        ndimage_filter_type=ndimage_filter_type,
        ndimage_filter_kwargs=ndimage_filter_kwargs,
    )
    wake_thresholds = morphological_detect.compute_per_bin_thresholds(
        da,
        wake_quantile,
        bin_boundaries,
        derivation_mask=wake_mask,
        threshold_method="from_value",
        ndimage_filter_type=ndimage_filter_type,
        ndimage_filter_kwargs=ndimage_filter_kwargs,
    )

    return nrem_thresholds, wake_thresholds


def _compute_per_condition_reference_thresholds(
    da: xr.DataArray,
    bin_boundaries: np.ndarray,
    subject: str,
    probe: str,
    structure: str,
    hgs: dict,
    ndimage_filter_type: str | None,
    ndimage_filter_kwargs: dict | None,
    source_config: MorphologicalSourceConfig,
) -> dict[str, xr.DataArray]:
    """Compute per-condition reference thresholds for comparison.

    For each core condition, computes a per-bin per-channel quantile
    threshold using the appropriate quantile (NREM or Wake) from the
    variant's thresholds CSV. The condition's hypnogram coverage acts
    as the per-sample derivation mask, so each bin's quantile is
    computed from the samples that fall within the condition.
    """
    nrem_q, wake_q = _get_threshold_quantiles(
        subject, probe, structure, source_config
    )

    ref_thresholds = {}
    for condition in const.CORE_CONDITIONS:
        if condition not in hgs:
            continue
        hg = hgs[condition]
        mask = hg.covers_time(da.time.values)
        if not mask.any():
            continue

        quantile = wake_q if "NOD" in condition else nrem_q
        ref_thresholds[condition] = morphological_detect.compute_per_bin_thresholds(
            da,
            quantile,
            bin_boundaries,
            derivation_mask=mask,
            threshold_method="from_value",
            ndimage_filter_type=ndimage_filter_type,
            ndimage_filter_kwargs=ndimage_filter_kwargs,
        )

    return ref_thresholds


# -------------------- Binary mask construction --------------------


def _build_state_aware_binary_mask(
    da: xr.DataArray,
    state_labels: np.ndarray,
    nrem_thresholds: xr.DataArray,
    wake_thresholds: xr.DataArray,
) -> np.ndarray:
    """Build binary OFF mask with state- and bin-dependent thresholds.

    For each time sample:
    - State in NREM_THRESHOLD_STATES -> OFF if value < nrem_threshold[bin, ch]
    - State in WAKE_THRESHOLD_STATES -> OFF if value < wake_threshold[bin, ch]
    - Otherwise (Artifact, NoData, uncovered) -> not OFF

    The bin lookup uses the ``bin_boundaries`` attr stored on each
    threshold DataArray (sample-index boundaries of shape (n_bins+1,)).

    Args:
        da: Full-recording DataArray (time, channel), numpy-backed.
        state_labels: Per-sample state labels.
        nrem_thresholds: Per-bin per-channel NREM threshold values.
        wake_thresholds: Per-bin per-channel Wake threshold values.

    Returns:
        Boolean array of shape (n_time, n_channel).
    """
    values = da.values
    n_time = values.shape[0]
    nrem_bin_mask = _per_bin_below_threshold(values, nrem_thresholds)
    wake_bin_mask = _per_bin_below_threshold(values, wake_thresholds)

    nrem_time_mask = np.isin(state_labels, NREM_THRESHOLD_STATES)
    wake_time_mask = np.isin(state_labels, WAKE_THRESHOLD_STATES)

    off_mask = np.zeros(values.shape, dtype=bool)
    off_mask[nrem_time_mask] = nrem_bin_mask[nrem_time_mask]
    off_mask[wake_time_mask] = wake_bin_mask[wake_time_mask]

    assert off_mask.shape[0] == n_time
    return off_mask


def _per_bin_below_threshold(
    values: np.ndarray, thresholds: xr.DataArray
) -> np.ndarray:
    """Per-(time, channel) ``values < thresholds[bin_idx]`` lookup."""
    boundaries = np.asarray(
        thresholds.attrs["bin_boundaries"], dtype=np.int64
    )
    thresh_arr = thresholds.values
    n_bins = thresh_arr.shape[0]
    out = np.empty(values.shape, dtype=bool)
    for bi in range(n_bins):
        lo, hi = int(boundaries[bi]), int(boundaries[bi + 1])
        out[lo:hi] = values[lo:hi] < thresh_arr[bi]
    return out


# -------------------- State annotation --------------------


def _annotate_off_states(
    offs: pd.DataFrame,
    lbl_ixs: dict[int, tuple[np.ndarray, np.ndarray]],
    state_labels: np.ndarray,
) -> None:
    """Add ``state`` column to the OFF DataFrame in-place.

    For each OFF, the predominant state is the mode of
    ``state_labels`` across the OFF's time-sample pixels.
    """
    if offs.empty:
        offs["state"] = pd.Series(dtype=str)
        return

    states = []
    for label in offs["label"]:
        time_ixs, _ = lbl_ixs[label]
        pixel_states = state_labels[time_ixs]
        unique, counts = np.unique(pixel_states, return_counts=True)
        states.append(unique[counts.argmax()])

    offs["state"] = states


# -------------------- Main detection function --------------------


def detect_offs_full(
    subject: str,
    probe: str,
    structure: str,
    opts: DetectionOpts,
    *,
    overwrite: bool = False,
    source_config: MorphologicalSourceConfig | None = None,
) -> None:
    """Detect OFF periods across the full recording and save results.

    Workflow:
    1. Load full trace (no condition filter)
    2. Get per-sample state labels from Full.Conservative hypnogram
    3. Compute NREM and Wake thresholds from state-specific data
    4. Build state-aware binary mask
    5. Apply morphological cleaning and connected-component labeling
    6. Extract OFF properties, annotate states, add laminar areas
    7. Compute per-condition reference thresholds
    8. Save all outputs

    Args:
        subject: Subject identifier.
        probe: Probe identifier.
        structure: Brain structure name.
        opts: Detection algorithm configuration. See ``DetectionOpts``.
        overwrite: Whether to overwrite existing results.
        source_config: Variant plumbing. Defaults to morphological; the
            full-recording flow does not currently support Tom-Bugnon.
    """
    cfg = source_config or _default_full_source_config()
    ndimage_filter_type = opts.get("ndimage_filter_type")
    ndimage_filter_kwargs = opts.get("ndimage_filter_kwargs")

    # -------------------- Check for existing results --------------------
    offs_path = cfg.files_module.get_full_offs_path(subject, probe, structure)
    lbls_path = cfg.files_module.get_full_off_label_indices_path(
        subject, probe, structure
    )
    if not overwrite and offs_path.exists() and lbls_path.exists():
        utils.log_step(
            "Skipping (results exist)", structure=structure
        )
        return

    # -------------------- Step 1: Load full trace into memory --------------------
    utils.log_step("Loading full trace", structure=structure)
    from cnpix_local_sleep.morphological.mua.readers import bin_boundaries_from_chunks

    da_lazy = cfg.open_traces_as_xarray(
        subject,
        probe,
        structure,
        condition=None,
        apply_detection_channel_mask=True,
        ndimage_filter_type=ndimage_filter_type,
        ndimage_filter_kwargs=ndimage_filter_kwargs,
    )
    # Capture chunk-aligned bin boundaries before .compute() clears
    # ``da.chunks``. Full-recording detection is MUA-only; the bin
    # partition matches the zarr's native chunk layout.
    bin_boundaries = bin_boundaries_from_chunks(da_lazy)
    da = da_lazy.compute()
    del da_lazy

    # -------------------- Step 2: Get per-sample state labels --------------------
    utils.log_step("Labeling states", structure=structure)
    hgs = hyp.load_statistical_condition_hypnograms(subject, probe)
    full_hg = hgs["Full.Conservative"]
    state_labels = full_hg.get_states(da.time.values)

    # -------------------- Step 3: Compute NREM and Wake thresholds --------------------
    utils.log_step("Computing thresholds", structure=structure)
    nrem_quantile, wake_quantile = _get_threshold_quantiles(
        subject, probe, structure, cfg
    )
    nrem_thresholds, wake_thresholds = _compute_state_thresholds(
        da,
        state_labels,
        bin_boundaries,
        nrem_quantile,
        wake_quantile,
        ndimage_filter_type,
        ndimage_filter_kwargs,
    )

    # -------------------- Step 4: Build state-aware binary mask --------------------
    utils.log_step("Building binary mask", structure=structure)
    off_mask = _build_state_aware_binary_mask(
        da, state_labels, nrem_thresholds, wake_thresholds
    )

    # Step 5: Morphological cleaning and labeling
    if not len(da.channel):
        lbl_ixs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        offs = pd.DataFrame(columns=list(Off.__annotations__.keys()))
    else:
        utils.log_step("Cleaning binary mask", structure=structure)
        if opts.get("clean_binary_mask", True):
            off_mask = morphology.clean_binary_mask(
                off_mask,
                n_samples_connect=opts["n_samples_connect"],
                n_samples_clean=opts["n_samples_clean"],
                n_channels_clean=opts["n_channels_clean"],
                n_channels_connect=opts["n_channels_connect"],
            )

        utils.log_step("Labeling components", structure=structure)
        lbl_img, _ = scipy.ndimage.label(off_mask)
        del off_mask

        lbl_ixs = scipy.ndimage.value_indices(lbl_img, ignore_value=0)
        del lbl_img

        utils.log_step("Extracting properties", structure=structure)
        offs = morphology.get_off_properties(
            da.y.values,
            da.time.values,
            da.attrs["fs"],
            lbl_ixs,
            values=da.values,
        )

    offs["max_span"] = da.y.max().item() - da.y.min().item()

    # Step 6: Annotate states and add laminar areas
    _annotate_off_states(offs, lbl_ixs, state_labels)
    morphological_detect.add_laminar_areas(
        offs, da, lbl_ixs, subject, probe, structure
    )

    # Step 7: Compute per-condition reference thresholds
    utils.log_step(
        "Computing reference thresholds", structure=structure
    )
    ref_thresholds = _compute_per_condition_reference_thresholds(
        da,
        bin_boundaries,
        subject,
        probe,
        structure,
        hgs,
        ndimage_filter_type,
        ndimage_filter_kwargs,
        cfg,
    )

    # -------------------- Step 8: Save results --------------------
    utils.log_step("Saving results", structure=structure)

    # Save OFFs
    offs_path.parent.mkdir(parents=True, exist_ok=True)
    offs.to_parquet(offs_path, index=False)

    # Save label indices
    if lbl_ixs:
        lbl_ixs_df = pd.DataFrame(
            [
                {
                    "label": label,
                    "time_ixs": time_ixs.tolist(),
                    "chan_ixs": chan_ixs.tolist(),
                }
                for label, (time_ixs, chan_ixs) in lbl_ixs.items()
            ]
        )
    else:
        lbl_ixs_df = pd.DataFrame(
            columns=["label", "time_ixs", "chan_ixs"]
        )
    lbls_path.parent.mkdir(parents=True, exist_ok=True)
    lbl_ixs_df.to_parquet(lbls_path, index=False)

    # Save NREM and Wake thresholds
    nrem_thresh_path = cfg.files_module.get_full_channel_thresholds_path(
        subject, probe, structure, "nrem"
    )
    nrem_thresh_path.parent.mkdir(parents=True, exist_ok=True)
    nrem_thresholds.to_zarr(nrem_thresh_path, mode="w")

    wake_thresh_path = cfg.files_module.get_full_channel_thresholds_path(
        subject, probe, structure, "wake"
    )
    wake_thresh_path.parent.mkdir(parents=True, exist_ok=True)
    wake_thresholds.to_zarr(wake_thresh_path, mode="w")

    # Save per-condition reference thresholds
    for condition, thresh in ref_thresholds.items():
        ref_path = cfg.files_module.get_full_channel_thresholds_path(
            subject, probe, structure, condition
        )
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        thresh.to_zarr(ref_path, mode="w")

    n_offs = len(offs)
    utils.log_step(
        "Done",
        structure=structure,
        n_offs=n_offs,
    )


# -------------------- Structure-level orchestration --------------------


def do_structure(
    subject: str,
    probe: str,
    structure: str,
    opts: DetectionOpts,
    *,
    overwrite: bool = False,
    source_config: MorphologicalSourceConfig | None = None,
) -> None:
    """Run full-recording OFF detection for a structure.

    Args:
        subject: Subject identifier.
        probe: Probe identifier.
        structure: Brain structure name.
        opts: Detection algorithm configuration. See ``DetectionOpts``.
        overwrite: Whether to overwrite existing results.
        source_config: Variant plumbing. See ``detect_offs_full()``.
    """
    detect_offs_full(
        subject,
        probe,
        structure,
        opts,
        overwrite=overwrite,
        source_config=source_config,
    )