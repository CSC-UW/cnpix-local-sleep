"""Morphological: quantile detection on shared MUA traces.

Primary path for new runs. Trace source is ``mua_traces.zarr`` from the
harding pipeline; path outputs use the ``method=morphological`` on-disk segment.

The ``SOURCE_CONFIG`` attribute is the canonical handle for shared
morphological workflow code; see ``cnpix_local_sleep.morphological.common.MorphologicalSourceConfig``.
"""

from __future__ import annotations

from cnpix_local_sleep.morphological.common import MorphologicalSourceConfig
from cnpix_local_sleep.morphological.mua import files, readers

SOURCE_CONFIG = MorphologicalSourceConfig(
    variant="morphological",
    files_module=files,
    open_traces_as_xarray=readers.open_traces_as_xarray,
    thresholds_package="cnpix_local_sleep.morphological.mua",
)

__all__ = ["SOURCE_CONFIG", "files", "readers"]