"""Shared unit-free OFF detection pipeline modules.

This subpackage contains shared pipeline modules for unit-free (AP-band) OFF
period detection, including postprocessing, aggregation, plotting, and
bandpower analysis.

Morphological-specific modules (preprocessing, detection, detection options) have
moved to ``cnpix_local_sleep.morphological``.

Only lightweight modules (utils) are imported at package level.
All other pipeline modules must be imported directly to avoid slow import
times:
    from cnpix_local_sleep.morphological.pipeline import postprocess_offs
"""

from . import utils

__all__ = ["utils"]