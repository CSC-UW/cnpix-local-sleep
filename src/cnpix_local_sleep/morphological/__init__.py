"""Morphological OFF period detection for Neuropixels data.

Implements OFF period detection using manually set quantile-based thresholds on
preprocessed AP-band traces. Detection operates on a binary mask derived from
thresholding, cleaned via image-morphology operations, and labeled via
connected-component analysis.

This method was developed by Tom Bugnon at WISC.

Path outputs use the ``method=morphological`` on-disk segment. See
``cnpix_local_sleep.morphological.mua.files`` for path construction.

The legacy ``tom-bugnon`` variant (detection on ``processed_ap.zarr``) and
the ``tom_match`` cross-variant sweep were removed on 2026-08-11. The
annotation *grid* those traces define outlives them, because the napari
stacks and every manual OFF label are pinned to it; see
``cnpix_local_sleep.files.get_preprocessed_ap_path`` and
``cnpix_local_sleep.trace_io.open_preprocessed_traces_as_xarray``.
MUA trace preprocessing lives in ``cnpix.mua``.

Submodules:
    mua: files module, trace reader, SOURCE_CONFIG
    common: MorphologicalSourceConfig
    detect: Threshold computation and OFF detection orchestration
    detect_full: full-recording (48 h) detection
    morphology: the binary-mask image-morphology kernels
    manual_validation: per-structure scoring against manual OFF labels
    full48h_eval: experiment-wide scoring of the full-48h OFFs vs manual labels
    detection_opts: Load the YAML detection parameter template
    types: DetectionOpts TypedDict and validation
    pipeline: postprocessing, aggregation and the manuscript export drivers
    cli, analysis_cli: Click CLI entry points

Cross-method scoring (this detector against another's output rather than
against manual labels) lives in ``cnpix_local_sleep.evaluation``.

Heavy submodules must be imported directly to avoid slow import times:
    from cnpix_local_sleep.morphological import detect
"""

__all__: list[str] = []
