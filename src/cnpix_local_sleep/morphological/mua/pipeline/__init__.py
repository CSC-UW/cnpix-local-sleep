"""Morphological-specific pipeline modules.

Pipeline steps that are ``morphological``-only, i.e. that cannot be run against
other morphological variants because they depend on data products only ``morphological``
produces. Currently this is the full-recording (48h) OFF analysis
(:mod:`cnpix_local_sleep.morphological.mua.pipeline.full48h`), which consumes the whole-recording
``offs.parquet`` files written by ``morphological-offs detect-offs-full``
(``tom-bugnon`` has no whole-recording OFFs at all).

Method-neutral AP-band pipeline steps (postprocessing, aggregation, plotting,
bandpower) live in :mod:`cnpix_local_sleep.morphological.pipeline` instead.

``full48h`` is heavy (matplotlib, xarray, ecephys.plot), so it is not imported
at package level. Import it directly:

    from cnpix_local_sleep.morphological.mua.pipeline import full48h
"""
