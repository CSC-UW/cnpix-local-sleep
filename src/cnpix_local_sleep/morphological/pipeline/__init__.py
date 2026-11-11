"""Post-detection pipeline modules for the morphological OFFs.

Postprocessing kernels, full-48h aggregation, plotting, bandpower analysis and
the ``export-*`` implementations.

Only lightweight modules (utils) are imported at package level.
All other pipeline modules must be imported directly to avoid slow import
times:
    from cnpix_local_sleep.morphological.pipeline import postprocess_offs
"""

from . import utils

__all__ = ["utils"]