"""Morphological detection-opts template loading.

The YAML template lives with the detection method, at
``cnpix_local_sleep.morphological.mua.data/spatial_detection_opts.yml``. This module provides a
thin helper for loading and validating it.
"""

from importlib import resources

import yaml

from cnpix_local_sleep.morphological.types import DetectionOpts, validate_detection_opts


def _load(package: str, filename: str) -> DetectionOpts:
    with resources.path(package, filename) as f:
        with open(f, "r") as file:
            opts = yaml.safe_load(file)
    validate_detection_opts(opts)
    return opts


def get_mua_spatial_detection_opts() -> DetectionOpts:
    """Morphological spatial detection opts (500Hz traces)."""
    return _load("cnpix_local_sleep.morphological.mua.data", "spatial_detection_opts.yml")
