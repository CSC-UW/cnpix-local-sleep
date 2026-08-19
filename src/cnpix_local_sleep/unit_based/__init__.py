"""Unit-based (pooled spike-train) OFF period detection.

Detection pools the spike trains of all units in a (subject, probe, structure)
into a single multiunit train and runs an ``on_off_detection`` method
(``threshold``, ``hmmem``, or ``sticky``) per macro-state. Results are written in
the shared ``Off`` schema under ``method=unit_based``. The *banded* variant runs
the same detector within depth bands, so its OFFs carry real depth footprints.

Submodules:
    files: ``method=unit_based`` path construction.
    const: detection constants and per-algorithm parameters.
    loading: Off-schema mapping and result loading.
    pipeline.detect_full: the pooled detection entry point.
    banded: the spatially-resolved (banded) detector.
    banded_eval, banded_plots: scoring and diagnostic plots vs manual labels.
    interactive: backend for the ``unit-based-off-tuner``.

Scoring banded OFFs against the *morphological* detector rather than against
manual labels lives in ``cnpix_local_sleep.evaluation.{head_to_head,
banded_vs_morphological}``.
"""

from . import const, files, loading

__all__ = ["const", "files", "loading"]
